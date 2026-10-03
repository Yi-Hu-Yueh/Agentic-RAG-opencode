"""手動驗證自動化腳本（run_manual_test.py）

一鍵執行完整手動驗證流程（請在專案根目錄執行）：
    python run_manual_test.py

【流程】
    1. 局部建庫：若 data/knowledge.db 未達 10 批次（約 320 chunks），
       執行 `python -m src.spark_processor --db ... --batches 10 --progress`。
    2. 測試 A（簡體單步）：`python -m src.agent_router --question "数学是什么？"`。
    3. 測試 B（繁體前綴＋Fallback）：`python -m src.agent_router --question "探討數學的關係"`。
    4. 分析：捕獲兩次提問的 INFO 日誌與答案，並以函式庫級探針驗證
       「探討」是否剝離、繁體是否觸發 Fallback 撈出資料。

全檔繁體中文註解；子進程統一以 UTF-8 捕獲輸出，避免 Windows 終端亂碼。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────
# 常數：路徑與門檻（與 CLI 預設對齊）
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent  # 專案根目錄（腳本須放根目錄）
DB_PATH = "data/knowledge.db"  # 知識庫路徑（與 CLI 預設一致）
BATCH_SIZE = 32  # 建庫 CLI 預設每批 Chunk 數（src/spark_processor.py）
REQUIRED_BATCHES = 10  # 本次驗證要求的批次数
MIN_CHUNKS = REQUIRED_BATCHES * BATCH_SIZE  # 換算為 chunks 門檻（約 320）

BUILD_TIMEOUT = 1500  # 建庫子進程逾時（秒）：2GB 掃描＋規則抽取需數分鐘
QUERY_TIMEOUT = 300  # 問答子進程逾時（秒）


def _run(cmd: list[str], timeout: int, heartbeat: int = 0,
         merge_stderr: bool = False) -> subprocess.CompletedProcess:
    """執行子命令並以 UTF-8 捕獲輸出（Windows 終端預設 cp950 會亂碼，故強制 UTF-8）。

    Args:
        cmd: 命令列（以 sys.executable 開頭，確保用同一個 Python）。
        timeout: 逾時秒數。
        heartbeat: >0 時每 N 秒印一次存活訊息（長任務防呆）。
        merge_stderr: True 時把 stderr 併入 stdout（INFO 日誌走 stderr，
            問答步驟需開啟才能一併捕獲日誌與答案）。

    Returns:
        CompletedProcess（含 stdout / stderr / returncode）。
    """
    import os

    # 【關鍵】子進程強制 UTF-8，否則中文 INFO 日誌在 Windows 下變亂碼
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    print(f"$ {' '.join(cmd)}", flush=True)
    if not heartbeat and not merge_stderr:
        return subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=timeout,
        )
    if merge_stderr:
        # 短任務合併輸出：一次取回 stdout＋stderr（日誌與答案都在同一串）
        return subprocess.run(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout,
        )
    # 長任務：背景執行＋定期心跳，避免看起來像卡死
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    start = time.time()
    last_beat = start
    while True:
        line = proc.stdout.readline()
        if line:
            lines.append(line)
        elif proc.poll() is not None:
            break
        now = time.time()
        if now - last_beat >= heartbeat:
            print(f"… 執行中（已過 {int(now - start)} 秒）", flush=True)
            last_beat = now
    rest = proc.stdout.read()
    if rest:
        lines.append(rest)
    proc.wait(timeout=30)
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(lines), "")


def _db_chunks() -> int:
    """讀取 knowledge.db 的 chunks 筆數（缺檔/缺表視為 0，交給建庫步驟處理）。"""
    import sqlite3

    db = ROOT / DB_PATH
    if not db.exists():
        return 0
    try:
        with sqlite3.connect(str(db)) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    except sqlite3.Error as e:
        print(f"讀庫失敗（{e}），視為空庫處理。")
        return 0


def step_build() -> None:
    """步驟 1：局部建庫（不足 10 批才執行，upsert 語義可重跑）。"""
    print("\n========== 步驟 1：局部建庫 ==========")
    before = _db_chunks()
    print(f"建庫前 chunks：{before} 筆（門檻 {MIN_CHUNKS} 筆 ≈ {REQUIRED_BATCHES} 批）")
    if before >= MIN_CHUNKS:
        print("已達門檻，跳過建庫。")
        return
    # 【精確採用任務指定的建庫指令】（sys.executable 保證同一個 Python）
    result = _run(
        [sys.executable, "-m", "src.spark_processor", "--db", DB_PATH,
         "--batches", str(REQUIRED_BATCHES), "--progress"],
        timeout=BUILD_TIMEOUT, heartbeat=60,
    )
    # 只印關鍵尾段（tqdm 進度行太多，只留最後 15 行＋統計行）
    tail = (result.stdout or "").strip().splitlines()[-15:]
    print("\n".join(tail))
    if result.returncode != 0:
        raise RuntimeError(f"建庫失敗（exit={result.returncode}）")
    after = _db_chunks()
    print(f"建庫後 chunks：{after} 筆")
    if after < MIN_CHUNKS:
        print("⚠ 筆數仍不足門檻（可能語料提前耗盡），繼續驗證但結果僅供參考。")


def step_ask(label: str, question: str) -> str:
    """步驟 2/3 共用：跑一次問答 CLI，回傳完整終端輸出（含 INFO 日誌與答案）。"""
    print(f"\n========== {label} ==========")
    result = _run(
        [sys.executable, "-m", "src.agent_router",
         "--question", question, "--db", DB_PATH],
        timeout=QUERY_TIMEOUT, merge_stderr=True,  # 日誌走 stderr，必須合併捕獲
    )
    output = result.stdout or ""
    print(output)
    if result.returncode != 0:
        print(f"⚠ 問答進程異常退出（exit={result.returncode}），stderr：{result.stderr}")
    return output


def step_analyze_b() -> None:
    """步驟 4：函式庫級探針，驗證測試 B 的兩項關鍵行為。

    1. 「探討」是否剝離：直接呼叫 _extract_keywords / _sanitize_keywords，
       確認污染詞還原為純實詞。
    2. 繁體是否觸發 Fallback：直接跑 answer()，檢查 contexts 的 source
       是否含 *-fallback 標記、反思是否通過。
    """
    print("\n========== 步驟 4：測試 B 深度分析 ==========")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.agent_router import (  # 延遲匯入，保持腳本頂部乾淨
        AgentConfig,
        AgentRouter,
        _extract_keywords,
        _sanitize_keywords,
    )
    from src.spark_processor import SparkConfig, SparkProcessor

    question_b = "探討數學的關係"

    # ── 4.1 前綴剝離驗證 ──
    raw_kws = _extract_keywords(question_b)
    clean_kws = _sanitize_keywords(["探討數學"])
    print(f"原始問句關鍵字：{raw_kws}")
    print(f"污染詞消毒示範：['探討數學'] → {clean_kws}")
    stripped_ok = not any("探討" in kw for kw in raw_kws) and clean_kws == ["數學"]
    print(("✅「探討」已成功剝離，最終關鍵字為純粹的「數學」。"
           if stripped_ok else "❌「探討」剝離失敗，需檢查虛詞表。"))

    # ── 4.2 Fallback 驗證（同一個 DB，程式化跑完整鏈路）──
    with SparkProcessor(SparkConfig(db_path=DB_PATH)) as sp:
        router = AgentRouter(spark_processor=sp, config=AgentConfig(mode="rule"))
        res = router.answer(question_b)
    sources = [c["source"] for c in res["contexts"]]
    fallback_hit = [s for s in sources if "fallback" in s]
    print(f"子問題：{[(s['text'], s['route']) for s in res['sub_questions']]}")
    print(f"命中來源：{sources[:6]}")
    print(f"反思：{[(x['sub'], x['passed'], x['score']) for x in res['reflections']]}")
    if res["contexts"] and fallback_hit:
        print(f"✅ 繁體觸發 Fallback 並撈出 {len(res['contexts'])} 筆資料 "
              f"（降級路徑：{sorted(set(fallback_hit))}）。")
    elif res["contexts"]:
        print(f"ℹ 直接命中 {len(res['contexts'])} 筆，未走 Fallback（精準命中亦為正常）。")
    else:
        print("❌ 仍 0 命中：路由已逐級放寬（relation→keyword）且 Fallback 全文查已執行，"
              "但庫內全是簡體資料、繁體關鍵字無法跨字形命中。"
              "建議後續加 OpenCC 正規化或簡繁雙寫索引。")

    # ── 4.3 簡繁對照探針：同義不同字形的查詢命中對比 ──
    # 目的：若簡體命中而繁體落空，即證明瓶頸在字形而非前綴清理。
    with SparkProcessor(SparkConfig(db_path=DB_PATH)) as sp:
        simp = sp.search_by_entity("数学", limit=3)
        trad = sp.search_by_entity("數學", limit=3)
    print(f"簡體「数学」實體命中：{len(simp)} 筆；繁體「數學」實體命中：{len(trad)} 筆")
    if simp and not trad:
        print("ℹ 結論：前綴清理正常，瓶頸是簡繁字形差異（完全比對無法跨字形）。"
              "解法：OpenCC 正規化或簡繁雙寫索引。")
    elif trad:
        print("✅ 結論：繁體可直接命中，無字形瓶頸。")


def main() -> int:
    """主流程：建庫 → 測試 A → 測試 B → 分析（任一步基礎建設失敗回傳 1）。"""
    try:
        step_build()
    except Exception as e:  # noqa: BLE001 — 建庫失敗就沒有驗證基礎，直接中止
        print(f"❌ 建庫階段失敗：{e}")
        return 1
    step_ask("測試 A（簡體單步）", "数学是什么？")
    step_ask("測試 B（繁體前綴與 Fallback）", "探討數學的關係")
    step_analyze_b()
    print("\n========== 驗證完成 ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
