"""Agentic RAG 一鍵串聯啟動腳本（main.py）

串起完整管線：loader（串流讀取）→ spark_processor（雙寫建庫）→
agent_router（無向量檢索問答），執行方式：

    單次提問：
        python main.py --question "數學是什麼？"
    互動問答（無限循環，輸入 exit / quit 離開）：
        python main.py
    局部建庫＋提問（只建前 10 批）：
        python main.py --batches 10 --question "數學是什麼？"
    全量建庫（預設 --batches 0 即全量，2GB 需較長時間）：
        python main.py --batches 0

建庫狀態規則：空庫一定建庫；非空庫只在有指定 --batches（>0）時才追加建庫。
"""

# ──────────────────────────────────────────────
# 環境防護（必須在所有 import / IO 之前執行）
# ──────────────────────────────────────────────
import os
import sys

# 【關鍵】os.environ.setdefault 在進程啟動後已無法改變直譯器自身的 stdio 編碼，
# 因此雙管齊下：寫入環境變數（惠及後續子進程）＋ 直接重設 stdio 編碼（本次生效）。
os.environ.setdefault("PYTHONUTF8", "1")
for _stream_name in ("stdout", "stderr"):
    try:
        getattr(sys, _stream_name).reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 終端不支援時維持原狀即可
        pass

import argparse
import logging
import sqlite3
from pathlib import Path

# 專案模組匯入（main.py 位於根目錄，直接以套件路徑引用）
from src.agent_router import AgentConfig, AgentRouter
from src.spark_processor import SparkConfig, SparkProcessor

# ──────────────────────────────────────────────
# 常數：與各模組預設值對齊
# ──────────────────────────────────────────────
DEFAULT_DATA = "data/wikipedia-zh-cn-20240901.json"  # 維基 JSONL 原始檔
DEFAULT_DB = "data/knowledge.db"  # SQLite 知識庫
DEFAULT_CHROMA_DIR = "data/chroma"  # ChromaDB 持久化目錄
PIPELINE_BATCH_SIZE = 32  # 建庫每批 Chunk 數（與 CLI 預設一致）

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 一、參數解析
# ══════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數（規格固定的六個參數，不另加隱藏開關）。"""
    parser = argparse.ArgumentParser(description="Agentic RAG 一鍵串聯啟動腳本（無向量＋雙寫索引）")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始資料路徑")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite 資料庫路徑")
    parser.add_argument("--mode", default="rule", choices=["rule", "api"], help="抽取模式")
    parser.add_argument("--batches", type=int, default=0,
                        help="限制處理批次數（0=全量；>0 傳給 pipeline 限制批次）")
    parser.add_argument("--question", default="",
                        help="提問內容（未帶則進入互動式 CLI 問答模式）")
    parser.add_argument("--no-dual", action="store_true", help="關閉雙寫索引")
    return parser.parse_args(argv)


# ══════════════════════════════════════════════
# 二、串聯管線：建庫狀態檢查 → 建庫 → 統計
# ══════════════════════════════════════════════


def _db_chunks(db_path: str) -> int:
    """讀取知識庫 chunks 筆數（缺檔/缺表/讀取失敗一律視為 0＝空庫）。"""
    try:
        with sqlite3.connect(db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    except Exception:  # noqa: BLE001 — 缺檔缺表都是空庫語義，無需中斷流程
        return 0


def ensure_database(args: argparse.Namespace) -> SparkProcessor:
    """檢查並建立資料庫，回傳已連線的處理器（呼叫端負責關閉）。

    建庫規則：
        - 空庫（chunks=0）→ 一定建庫（全量或 --batches 上限）。
        - 非空庫＋有指定 --batches（>0）→ 追加建庫（UPSERT 冪等，可重跑）。
        - 非空庫＋未指定 → 跳過建庫，直接沿用。
    """
    # 【目錄保證】SQLite 開檔前父目錄必須存在，否則 connect 直接報錯
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    processor = SparkProcessor(SparkConfig(
        mode=args.mode,
        db_path=args.db,
        chroma_dir=DEFAULT_CHROMA_DIR,
        dual_script=not args.no_dual,
    ))

    chunks = _db_chunks(args.db)
    need_build = chunks == 0 or args.batches > 0
    if not need_build:
        logger.info("知識庫已有 %d 個 Chunk，跳過建庫。", chunks)
        return processor

    # 【建庫觸發原因】打在日誌，方便事後對照除錯
    reason = "空庫" if chunks == 0 else f"指定 --batches={args.batches}"
    max_batches = args.batches if args.batches > 0 else None  # 0 即全量（None）
    logger.info("開始建庫（原因：%s，上限：%s）。",
                reason, "全量" if max_batches is None else f"{max_batches} 批")
    processor.run_pipeline(
        args.data,
        batch_size=PIPELINE_BATCH_SIZE,
        max_batches=max_batches,
        show_progress=True,  # 缺 tqdm 時 run_pipeline 內部會自動降級
    )
    return processor


def print_stats(processor: SparkProcessor) -> None:
    """印出知識庫現況：總 Chunk 數與實體數（另附關係與 Chroma 筆數）。"""
    stats = processor.stats()
    print(f"知識庫現況：Chunks={stats.get('chunks', 0):,} 筆，"
          f"實體={stats.get('entities', 0):,} 筆，"
          f"關係={stats.get('relations', 0):,} 筆，"
          f"Chroma 文件={stats.get('chroma_docs', 0):,} 筆")


# ══════════════════════════════════════════════
# 三、智能檢索回答：單次問答／互動模式
# ══════════════════════════════════════════════


def answer_once(router: AgentRouter, question: str) -> None:
    """單次提問：印出答案與整體自評，結束後由呼叫端退出。"""
    result = router.answer(question)
    print(f"\n答案：\n{result['answer']}")
    overall = result.get("overall", {})
    print(f"\n[自評] 通過={overall.get('passed')} "
          f"分數={overall.get('score')}（共嘗試 {result.get('total_attempts', 0)} 次檢索）")


def interactive_loop(router: AgentRouter) -> None:
    """互動模式：無限循環讀取提問，直到輸入 exit / quit 才安全退出。

    輸入端防護：EOF（Ctrl+Z/Ctrl+D）與中斷（Ctrl+C）皆視為正常離開；
    空行直接忽略，不送檢索浪費資源。
    """
    print("進入互動問答模式（輸入 exit 或 quit 離開）。")
    while True:
        try:
            question = input("\n請提問> ").strip()
        except EOFError:
            print("\n偵測到輸入結束，離開問答。")  # Ctrl+Z / Ctrl+D
            break
        except KeyboardInterrupt:
            print("\n收到中斷，離開問答。")  # Ctrl+C
            break
        except UnicodeError as e:
            print(f"輸入編碼異常，已略過（{e}）。")  # Windows 終端偶發，略過即可
            continue
        # 【離開指令】大小寫皆可，spec 固定為 exit / quit
        if question.lower() in ("exit", "quit"):
            print("再見！")
            break
        if not question:  # 空行不檢索
            continue
        try:
            answer_once(router, question)
        except Exception as e:  # noqa: BLE001 — 單題失敗不掐斷整個互動迴圈
            logger.warning("本次問答失敗（%s），請換個問法再試。", e)


# ══════════════════════════════════════════════
# 四、主進入點
# ══════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    """主流程：參數 → 建庫 → 統計 → 單次問答或互動問答。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)

    try:
        processor = ensure_database(args)
    except FileNotFoundError as e:
        # 資料檔不存在是最常見的啟動失敗，直接指引而非噴 traceback
        print(f"❌ 找不到資料檔：{e}\n→ 請用 --data 指定正確的 JSONL 路徑。")
        return 1
    except Exception as e:  # noqa: BLE001 — 建庫失敗就沒有問答基礎，直接中止
        logger.error("建庫失敗：%s", e)
        return 1

    try:
        print_stats(processor)
        router = AgentRouter(processor, config=AgentConfig(mode=args.mode))
        if args.question.strip():
            answer_once(router, args.question.strip())  # 單次問答後退出
        else:
            interactive_loop(router)  # 無參數則進入互動模式
        return 0
    finally:
        processor.close()  # with 外手動管理：確保任何路徑都釋放 SQLite 連線


if __name__ == "__main__":
    raise SystemExit(main())
