"""spark_processor.py 的輕量測試：規則抽取、SQLite 三表、精準檢索。"""

import tempfile
from pathlib import Path

from src.loader import batched_chunks
from src.spark_processor import SparkConfig, SparkProcessor


def _make_processor(tmp_path) -> SparkProcessor:
    """建立隔離測試用的處理器（SQLite :memory: 概念改用臨時檔，Chroma 用臨時目錄）。"""
    tmp = Path(str(tmp_path)) if tmp_path is not None else Path(tempfile.mkdtemp())
    cfg = SparkConfig(
        mode="rule",
        db_path=str(tmp / "test_knowledge.db"),
        chroma_dir=str(tmp / "test_chroma"),
        max_entities_per_chunk=10,
        max_relations_per_chunk=10,
    )
    return SparkProcessor(cfg)


def test_rule_extract_entities_and_relations():
    cfg = SparkConfig(mode="rule", db_path=":memory:")
    sp = SparkProcessor(cfg)
    try:
        text = "孔子是儒家學派的創始人，出生於魯國。孔子提出了仁的思想，影響了中國歷史。"
        entities = sp.extract_entities(text)
        assert any(e["text"] == "孔子" for e in entities)
        relations = sp.extract_relations(text, entities)
        assert isinstance(relations, list)
    finally:
        sp.close()


def test_enrich_batch_writes_three_tables(tmp_path=None):
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        batch = [
            {
                "id": "1",
                "title": "孔子",
                "chunk_index": 0,
                "chunk_id": "1_0",
                "text": "孔子是儒家學派的創始人，出生於魯國，著作有《論語》。",
                "char_len": 50,
            }
        ]
        enriched = sp.enrich_batch(batch)
        assert len(enriched) == 1
        assert enriched[0].chunk_id == "1_0"
        assert len(enriched[0].entities) > 0
        stats = sp.stats()
        assert stats["chunks"] == 1
        assert stats["entities"] > 0
        # 精準檢索：實體 / 關鍵字 / 關係
        assert len(sp.search_by_entity("孔子")) >= 1
        assert len(sp.search_by_keyword("儒家")) >= 1
        assert len(sp.search_by_title("孔子")) == 1
    finally:
        sp.close()


def test_enrich_batch_idempotent(tmp_path=None):
    """重跑同一批不應產生重複資料（UPSERT 語義，雙寫列亦同）。"""
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        batch = [
            {
                "id": "9",
                "title": "測試",
                "chunk_index": 0,
                "chunk_id": "9_0",
                "text": "測試文字。" * 60,
                "char_len": 300,
            }
        ]
        sp.enrich_batch(batch)
        first = sp.stats()
        sp.enrich_batch(batch)
        second = sp.stats()
        assert first == second
    finally:
        sp.close()


def _make_dual_batch():
    """簡體測試資料（含必定產生繁簡差異的 数学 一詞）。"""
    return [
        {
            "id": "100",
            "title": "数学",
            "chunk_index": 0,
            "chunk_id": "100_0",
            "text": "数学是研究数量的学科。数学家研究数学理论，数学应用于科学。",
            "char_len": 60,
        }
    ]


def test_dual_entities_written():
    """雙寫：同一實體同時擁有簡體原文列與繁體轉換列。"""
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        sp.enrich_batch(_make_dual_batch())
        rows = sp._conn.execute(
            "SELECT entity_text, script FROM entities WHERE chunk_id='100_0'"
            " AND entity_text IN ('数学', '數學')"
        ).fetchall()
        texts = {r[0] for r in rows}
        assert texts == {"数学", "數學"}, f"缺雙寫列，現有：{texts}"
        scripts = {r[1] for r in rows}
        assert scripts == {"orig", "conv"}
    finally:
        sp.close()


def test_cross_script_entity_search():
    """跨字形命中：繁體查實體命中轉換列，簡體命中原文列。"""
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        sp.enrich_batch(_make_dual_batch())
        assert len(sp.search_by_entity("數學")) >= 1  # 繁體→轉換列
        assert len(sp.search_by_entity("数学")) >= 1  # 簡體→原文列
    finally:
        sp.close()


def test_cross_script_relations():
    """跨字形關係：繁體端點同樣查到三元組。"""
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        sp.enrich_batch(_make_dual_batch())
        simp = sp.get_relations("数学")
        trad = sp.get_relations("數學")
        assert len(simp) >= 1 and len(trad) >= 1
    finally:
        sp.close()


def test_cross_script_keyword_search():
    """跨字形關鍵字：繁體經 Chroma 轉換文件命中（去重後單列）。"""
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        sp.enrich_batch(_make_dual_batch())
        hits = sp.search_by_keyword("數學")
        assert len(hits) >= 1
        assert len({h["chunk_id"] for h in hits}) == len(hits)  # 已去重
    finally:
        sp.close()


def test_dual_idempotent():
    """雙寫冪等：重跑後三表＋Chroma 筆數完全不變。"""
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    sp = _make_processor(base)
    try:
        sp.enrich_batch(_make_dual_batch())
        first = sp.stats()
        assert first["entities"] > 1  # 確認雙寫確實發生（不只一列）
        sp.enrich_batch(_make_dual_batch())
        assert sp.stats() == first
    finally:
        sp.close()


def test_migration_old_db_without_script_col():
    """舊庫遷移：無 script 欄的舊表自動補欄，且可正常雙寫。"""
    import sqlite3
    import tempfile as _tf

    base = Path(_tf.mkdtemp())
    db = str(base / "old.db")
    conn = sqlite3.connect(db)  # 手工建舊 schema（無 script 欄）
    conn.executescript(
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, article_id TEXT, title TEXT,"
        " chunk_index INTEGER, text TEXT, char_len INTEGER, created_at TEXT);"
        "CREATE TABLE entities(id INTEGER PRIMARY KEY, chunk_id TEXT, article_id TEXT,"
        " entity_text TEXT, entity_type TEXT, freq INTEGER,"
        " UNIQUE(chunk_id, entity_text, entity_type));"
        "CREATE TABLE relations(id INTEGER PRIMARY KEY, chunk_id TEXT, subj TEXT,"
        " predicate TEXT, obj TEXT, weight REAL, evidence TEXT,"
        " UNIQUE(chunk_id, subj, predicate, obj));"
    )
    conn.commit()
    conn.close()
    cfg = SparkConfig(mode="rule", db_path=db, chroma_dir=str(base / "chroma"))
    sp = SparkProcessor(cfg)
    try:
        cols = {r[1] for r in sp._conn.execute("PRAGMA table_info(entities)")}
        assert "script" in cols  # 遷移已補欄
        sp.enrich_batch(_make_dual_batch())
        assert len(sp.search_by_entity("數學")) >= 1  # 回填成功
    finally:
        sp.close()
