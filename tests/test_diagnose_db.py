"""diagnose_db.py 的輕量測試：空庫告警與健康庫判斷。"""

import importlib.util
import sqlite3
import sys
import tempfile
from pathlib import Path

# 【獨立腳本載入】diagnose_db.py 在專案根目錄（非套件），用路徑載入
_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("diagnose_db", _ROOT / "diagnose_db.py")
diagnose_db = importlib.util.module_from_spec(_SPEC)
sys.modules["diagnose_db"] = diagnose_db
_SPEC.loader.exec_module(diagnose_db)


def _make_db(path: Path, chunks: int = 0) -> None:
    """建立最小結構的測試 DB（含三表，可選灌 chunks 筆假資料）。"""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        # 與真實 _SCHEMA_SQL 對齊的最小欄位（diagnose 的樣本查詢需要它們）
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY);"
        "CREATE TABLE entities(id INTEGER PRIMARY KEY, entity_text TEXT,"
        " entity_type TEXT DEFAULT '其他', freq INTEGER DEFAULT 1);"
        "CREATE TABLE relations(id INTEGER PRIMARY KEY);"
    )
    for i in range(chunks):
        conn.execute("INSERT INTO chunks VALUES (?)", (f"c_{i}",))
        conn.execute("INSERT INTO entities(entity_text) VALUES (?)", ("數學",))
    conn.commit()
    conn.close()


def test_empty_db_reports_need_build():
    """空庫 → 回傳 1（需建庫）。"""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "empty.db"
        _make_db(db, chunks=0)
        assert diagnose_db.diagnose(str(db)) == 1


def test_missing_db_reports_error(capsys=None):
    """缺檔 → 回傳 2。"""
    with tempfile.TemporaryDirectory() as tmp:
        assert diagnose_db.diagnose(str(Path(tmp) / "nope.db")) == 2


def test_healthy_db_reports_ok():
    """有資料 → 回傳 0。"""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "ok.db"
        _make_db(db, chunks=2)
        assert diagnose_db.diagnose(str(db)) == 0
