"""知識庫診斷腳本（diagnose_db.py）

獨立執行（不依賴 src 套件，只用標準函式庫）：
    python diagnose_db.py
    python diagnose_db.py --db data/knowledge.db

檢查項目：
    1. 資料庫檔案是否存在。
    2. chunks / entities / relations 三表是否存在、各有多少筆。
    3. 若 chunks 或 entities 為 0 → 提示建庫指令。
    4. 額外抽查：一筆 entities 樣本，確認抽取管線曾正常寫入。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# 【核心三表】與 src/spark_processor.py 的 _SCHEMA_SQL 對齊
CORE_TABLES = ("chunks", "entities", "relations")


def diagnose(db_path: str) -> int:
    """執行診斷並列印報告。回傳值：0=健康，1=需建庫，2=檔案/結構異常。"""
    db = Path(db_path)
    print(f"資料庫路徑：{db.resolve()}")

    # ── 1. 檔案是否存在 ──
    if not db.exists():
        print("❌ 找不到資料庫檔案。")
        print("→ 請先執行 python -m src.spark_processor 建庫")
        return 2

    print(f"檔案大小：{db.stat().st_size:,} bytes")
    conn = sqlite3.connect(str(db))
    try:
        # ── 2. 三表是否存在 ──
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing = [t for t in CORE_TABLES if t not in existing]
        if missing:
            print(f"❌ 缺少資料表：{missing}（可能是舊版或損壞的 DB）。")
            print("→ 請先執行 python -m src.spark_processor 建庫")
            return 2

        # ── 3. 各表筆數 ──
        counts = {}
        for table in CORE_TABLES:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"表 {table}：{counts[table]:,} 筆")

        # ── 4. 空庫判斷 ──
        if counts["chunks"] == 0 or counts["entities"] == 0:
            print("❌ 知識庫為空（chunks 或 entities 為 0 筆），問答必定 0 命中。")
            print("→ 請先執行 python -m src.spark_processor 建庫")
            return 1

        # ── 5. 抽查一筆實體樣本（確認管線曾正常寫入）──
        sample = conn.execute(
            "SELECT entity_text, entity_type, freq FROM entities LIMIT 3"
        ).fetchall()
        print(f"實體樣本：{[tuple(r) for r in sample]}")
        print("✅ 知識庫健康（雙寫索引：繁簡任一字形皆可命中）。"
              "若問答仍 0 命中，請檢查關鍵字是否含前綴污染（如「查詢數學」）。")
        return 0
    except sqlite3.Error as e:
        print(f"❌ SQLite 讀取失敗：{e}")
        return 2
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    """命令列入口：解析 --db 參數後執行診斷。"""
    # 【終端韌性】Windows 預設 cp950 終端印中文會崩潰，優先切 UTF-8，失敗則忽略
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 終端不支援時維持原狀即可
        pass
    parser = argparse.ArgumentParser(description="Agentic RAG 知識庫診斷工具")
    parser.add_argument("--db", default="data/knowledge.db", help="SQLite 知識庫路徑")
    args = parser.parse_args(argv)
    return diagnose(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
