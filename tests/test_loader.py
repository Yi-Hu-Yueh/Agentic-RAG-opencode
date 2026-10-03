"""loader.py 的輕量測試：驗證清洗、切片、串流皆正常。"""

import json
import tempfile
from pathlib import Path

from src.loader import (
    batched_chunks,
    clean_text,
    split_text_to_chunks,
    stream_chunks,
    stream_wiki_articles,
)


def _make_sample_file() -> str:
    """建立臨時 JSONL 樣本檔（含 2 篇條目），回傳路徑。"""
    samples = [
        {"id": "1", "title": "測試條目一", "text": "這是第一篇。" * 200},
        {"id": "2", "title": "測試條目二", "text": "這是第二篇，含 [[連結|顯示文字]] 與 <ref>引用</ref> {{模板}}。\n" * 100},
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for obj in samples:
        tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


def test_clean_text_removes_markup():
    cleaned = clean_text("你好 [[人工智慧|AI]]，見 [https://example.com 官網] <ref>註</ref> {{模板}}")
    assert "人工智慧" not in cleaned or "AI" in cleaned
    assert "ref" not in cleaned
    assert "{{" not in cleaned


def test_split_text_overlap():
    text = "測" * 1500  # 無標點的長文，測試固定視窗邏輯
    chunks = split_text_to_chunks(text, chunk_size=600, overlap_ratio=0.1)
    assert len(chunks) >= 2
    assert all(100 <= len(c) <= 900 for c in chunks)


def test_stream_and_batch(tmp_path: Path = None):
    fp = _make_sample_file()
    articles = list(stream_wiki_articles(fp))
    assert len(articles) == 2
    assert articles[0]["title"] == "測試條目一"

    chunks = list(stream_chunks(fp, chunk_size=600, overlap_ratio=0.1))
    assert len(chunks) >= 2
    assert all("chunk_id" in c and "text" in c for c in chunks)

    batches = list(batched_chunks(fp, batch_size=2))
    assert all(isinstance(b, list) for b in batches)
    assert sum(len(b) for b in batches) == len(chunks)
