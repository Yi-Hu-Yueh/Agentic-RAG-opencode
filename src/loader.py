"""資料預處理模組（loader.py）

負責以「低記憶體佔用」方式串流讀取 2GB 中文維基百科 JSON 檔案，
並完成基礎清洗與文字切片（Chunking），供後續向量化 / 知識圖譜使用。

設計重點：
    1. 嚴禁一次性 ``json.load()`` 整檔讀入記憶體，避免 OOM。
    2. 預設檔案為 JSONL（每行一個 JSON 物件，含 id / title / text）。
       若偵測到檔頭為 ``[``（標準 JSON 陣列），則嘗試改用 ijson 做串流解析。
    3. 所有對外讀取介面皆為 Generator，逐條 / 逐批吐出資料。
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

# ──────────────────────────────────────────────
# 日誌設定：讓呼叫端可直接看到跳過壞行等警告
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)

# 檔案路徑型別別名：支援字串或 Path 物件
PathLike = Union[str, Path]

# 切片後的單一文字區塊結構（以 Dict 表示，方便直接送向量庫）
Chunk = Dict[str, Any]


# ══════════════════════════════════════════════
# 一、基礎資料清洗
# ══════════════════════════════════════════════

# 【正則表達式預編譯】放在模組層級，避免每次呼叫重複編譯，提升效能
_RE_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")  # [[連結|顯示文字]] -> 顯示文字
_RE_EXTERNAL_LINK = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")  # [http://... 顯示文字]
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")  # {{模板}} 直接移除
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)  # <ref> 引用標註
_RE_HTML_TAG = re.compile(r"<[^>]+>")  # 其餘 HTML 標籤
_RE_SPECIAL_SYMBOL = re.compile(
    r"[▓─━│┌┐└┘├┤┬┴┼═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬★☆◇◆□■"
    r"○●◎§※©®™°′″＊＃＆＠＾＿｀｜￣―─…—\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+"
)  # 罕見特殊符號 / 控制字元（保留中日韓、標點、emoji 以外的常見字元）
_RE_MULTI_BLANK_LINE = re.compile(r"\n\s*\n\s*\n+")  # 連續 3+ 空行壓成雙換行
_RE_MULTI_SPACE = re.compile(r"[ \t\u3000]{2,}")  # 連續半形/全形空白壓成一個


def clean_text(text: str) -> str:
    """清洗單篇維基百科內文。

    Args:
        text: 原始內文字串（可能含 wiki markup / HTML / 特殊符號）。

    Returns:
        清洗後的純文字字串。若輸入為空則回傳空字串。
    """
    # 【防呆】非字串（None / 數字）一律視為空字串，避免下游報錯
    if not isinstance(text, str):
        return ""
    if not text:
        return ""

    # 1. 移除 <ref> 引用與 HTML 標籤（先處理 ref，因為內含換行需 DOTALL）
    text = _RE_REF.sub("", text)
    text = _RE_HTML_TAG.sub("", text)

    # 2. 移除 {{模板}}（可能巢狀，這裡用迴圈多次移除直到乾淨）
    for _ in range(5):  # 最多剝 5 層，足夠應付一般條目
        new_text, n = _RE_TEMPLATE.subn("", text)
        text = new_text
        if n == 0:
            break

    # 3. 還原內部連結與外部連結的「顯示文字」
    #    例：[[人工智慧|AI]] -> AI，[https://... 官網] -> 官網
    text = _RE_WIKI_LINK.sub(r"\1", text)
    text = _RE_EXTERNAL_LINK.sub(r"\1", text)

    # 4. 移除殘留的特殊符號與控制字元（保留中文標點，。！？、；：「」『』等）
    text = _RE_SPECIAL_SYMBOL.sub("", text)

    # 5. 殘留的 wiki 符號（== 標題 ==、'''粗體'''、部の表格符號）做輕量清理
    text = text.replace("'''", "").replace("''", "")
    # 行首的 == 標題記號 / * 列表 / # 編號 / ; 定義 / : 縮排，保留文字本身
    text = re.sub(r"(?m)^\s*(==+|[*#:;]+)\s*", "", text)
    text = text.replace("==", "")
    # 表格殘留的 {| |} |- || 一律移除
    text = re.sub(r"\{\||\|\}|\|-|\\|\\|", "", text)

    # 6. 正規化空白與換行：全形空白轉半形、多空白壓縮、多空行壓縮
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_BLANK_LINE.sub("\n\n", text)

    # 7. 逐行 strip，丟掉純空行，保留段落結構（段落以 \n\n 分隔）
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    return text.strip()


# ══════════════════════════════════════════════
# 二、文字切片（含 10% Overlap）
# ══════════════════════════════════════════════

# 【中文句尾標點】用於「盡量在句號處斷開」，避免把句子切一半
_SENTENCE_END = ("。", "！", "？", "!", "?", "\n", "；", ";")


def split_text_to_chunks(
    text: str,
    chunk_size: int = 600,
    overlap_ratio: float = 0.1,
    min_chunk_len: int = 100,
    max_chunk_size: int = 800,
) -> List[str]:
    """將長文本切成 500~800 字的區塊，並保留 Overlap。

    策略：
        - 以字元數為單位的滑動視窗（中文以字為單位最直觀）。
        - 視窗大小預設 600（落在 500~800 區間內）。
        - Overlap 預設為 chunk_size 的 10%（600 -> 60 字），保留上下文連貫。
        - 每個視窗會往前/往後微調，盡量在句尾標點處斷開。
        - 最後不足 min_chunk_len 的尾巴，會併入前一塊，避免產生過短碎片。

    Args:
        text: 已清洗過的純文字。
        chunk_size: 目標區塊大小（字元數），建議 500~800，預設 600。
        overlap_ratio: 重疊比例，預設 0.1（即 10%）。
        min_chunk_len: 可接受的最小區塊長度，過短尾巴會併入前一塊。
        max_chunk_size: 區塊大小上限保護，避免異常長段落爆量。

    Returns:
        切好的文字區塊串列。每塊長度大致落在 [min_chunk_len, max_chunk_size]。
    """
    # 【參數鉗制】避免呼叫端誤傳超界數值
    chunk_size = max(500, min(int(chunk_size), max_chunk_size))
    overlap_ratio = max(0.0, min(float(overlap_ratio), 0.5))
    overlap = int(chunk_size * overlap_ratio)  # 重疊字數，如 600 * 0.1 = 60
    step = max(1, chunk_size - overlap)  # 每次前進的步伐 = 視窗 - 重疊

    text = (text or "").strip()
    if not text:
        return []
    # 短文直接回傳，不需切片
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # 1. 先取一個固定視窗
        end = min(start + chunk_size, text_len)

        # 2. 若非最後一塊，嘗試在視窗尾端附近找句尾標點，微調斷點
        #    搜尋範圍：視窗尾端往前 100 字內，避免斷點漂移太多
        if end < text_len:
            search_start = max(start + min_chunk_len, end - 100)
            best_break = -1
            for i in range(end, search_start - 1, -1):
                if text[i - 1] in _SENTENCE_END:
                    best_break = i
                    break
            # 只有找到合理的句尾才調整，否則維持固定視窗（避免無限迴圈）
            if best_break != -1:
                end = best_break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        # 3. 已到文末則結束；否則以前進 step 的方式滑動（自然形成 Overlap）
        if end >= text_len:
            break
        start += step
        # 【安全護欄】若斷點微調導致 start 超過 end，強制推進避免死迴圈
        if start >= end:
            start = end - overlap if end - overlap > 0 else end

    # 4. 尾巴過短則併入前一塊（避免最後一塊只有幾個字）
    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk_len:
        chunks[-2] = (chunks[-2] + "\n" + chunks[-1]).strip()
        # 合併後若超過上限則截斷，保證不爆量
        if len(chunks[-2]) > max_chunk_size + overlap:
            chunks[-2] = chunks[-2][:max_chunk_size]
        chunks.pop()

    return chunks


# ══════════════════════════════════════════════
# 三、記憶體安全的串流讀取（Generator）
# ══════════════════════════════════════════════


def _open_text_stream(file_path: PathLike) -> Iterable[str]:
    """依副檔名開啟文字串流（支援 .json / .jsonl / .gz）。

    使用 ``yield from`` 逐行吐出，檔案控制代碼由呼叫端統一關閉。
    本函數為內部輔助函數，不直接對外暴露。
    """
    path = Path(file_path)
    # 【gzip 支援】若為 .gz 結尾，改用 gzip 開啟（仍為串流，不會整檔解壓到記憶體）
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = open(path, "r", encoding="utf-8")
    with fh:
        yield from fh


def _looks_like_json_array(file_path: PathLike) -> bool:
    """窺視檔頭第一個非空白字元，判斷是否為標準 JSON 陣列（'[' 開頭）。

    JSONL 每行都是獨立物件（'{' 開頭），JSON 陣列則是整個檔案一個大 list。
    只讀取前 1KB 做判斷，記憶體開銷極小。
    """
    path = Path(file_path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        head = f.read(1024).lstrip()
        return head.startswith("[")


def stream_wiki_articles(
    file_path: PathLike,
    min_text_len: int = 50,
    show_progress: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """逐條串流吐出維基百科條目（核心 Generator，記憶體安全）。

    預設處理 JSONL：每行 ``json.loads`` 一筆，讀完即丟，記憶體只保留當前行。
    若偵測為 JSON 陣列格式，則優先使用 ijson 做串流解析；
    若未安裝 ijson 則退回逐行解析並記錄警告。

    每筆吐出的字典格式：
        ``{"id": ..., "title": ..., "text": ...}``

    Args:
        file_path: 資料檔路徑（.json / .jsonl，可含 .gz）。
        min_text_len: 內文短於此長度視為空條目，直接跳過。
        show_progress: 是否顯示 tqdm 進度條（需先 pip install tqdm）。

    Yields:
        清洗前的原始條目字典。注意：此處尚未做 clean/split，
        讓呼叫端可自行決定清洗策略，保持彈性。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔：{path.resolve()}")

    skipped = 0  # 損壞行 / 過短條目的計數器

    # ── 分支一：標準 JSON 陣列格式，嘗試用 ijson 串流 ──
    if _looks_like_json_array(path):
        try:
            import ijson  # type: ignore  # 延遲匯入，沒裝也不影響 JSONL 主流程

            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rb") as fb:  # ijson 需二進位串流
                # "item" 表示陣列底下每個元素，逐個吐出不載入整個 list
                for obj in ijson.items(fb, "item"):
                    if not isinstance(obj, dict):
                        skipped += 1
                        continue
                    text = obj.get("text", "") or ""
                    if len(text) < min_text_len:
                        skipped += 1
                        continue
                    yield {
                        "id": str(obj.get("id", "")),
                        "title": str(obj.get("title", "") or ""),
                        "text": text,
                    }
            if skipped:
                logger.warning("已跳過 %d 筆無效/過短條目（JSON-array 模式）", skipped)
            return
        except ImportError:
            logger.warning("偵測到 JSON 陣列格式但未安裝 ijson，將退回逐行解析。建議：pip install ijson")

    # ── 分支二：JSONL 主流程（逐行讀取，記憶體只佔一行）──
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm  # 延遲匯入，沒裝就靜默關閉進度條

            pbar = tqdm(desc="讀取維基條目", unit="篇")
        except ImportError:
            logger.warning("tqdm 未安裝，show_progress 自動停用。建議：pip install tqdm")

    try:
        for line_no, line in enumerate(_open_text_stream(path), start=1):
            line = line.strip()
            # 跳過空行與陣列符號殘留（相容「偽 JSONL」檔案）
            if not line or line in ("[", "]", ","):
                continue
            # 移除行尾逗號（相容把 JSON array 拆成多行的檔案）
            if line.endswith(","):
                line = line[:-1]
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 5:  # 只印前 5 筆，避免刷屏
                    logger.warning("第 %d 行 JSON 解析失敗，已跳過", line_no)
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            text = obj.get("text", "") or ""
            if len(text) < min_text_len:
                skipped += 1
                continue
            if pbar is not None:
                pbar.update(1)
            yield {
                "id": str(obj.get("id", "")),
                "title": str(obj.get("title", "") or ""),
                "text": text,
            }
    finally:
        if pbar is not None:
            pbar.close()

    if skipped:
        logger.warning("已跳過 %d 筆無效/過短/損壞條目（JSONL 模式）", skipped)


def stream_chunks(
    file_path: PathLike,
    chunk_size: int = 600,
    overlap_ratio: float = 0.1,
    batch_size: Optional[int] = None,
    min_text_len: int = 50,
    show_progress: bool = False,
) -> Generator[Union[Chunk, List[Chunk]], None, None]:
    """一站式管線：讀取 → 清洗 → 切片 →（可選）批次吐出。

    這是下游（向量化 / Spark 知識圖譜）最常使用的介面：
        - ``batch_size=None``：逐個吐出 Chunk（Dict）。
        - ``batch_size=N``：每 N 個 Chunk 包成一個 List 吐出，
          方便做批次 embedding，減少 I/O 往返。

    Chunk 字典欄位：
        - ``id``: 原始條目 id
        - ``title``: 原始條目標題
        - ``chunk_index``: 該條目底下的第幾個區塊（從 0 起算）
        - ``chunk_id``: 全域唯一編號 ``"{id}_{chunk_index}"``
        - ``text``: 清洗切片後的文字
        - ``char_len``: 該區塊字元數

    Args:
        file_path: 資料檔路徑。
        chunk_size: 目標區塊大小（字元數），建議 500~800。
        overlap_ratio: 重疊比例，預設 0.1（10%）。
        batch_size: 批次大小。None 表示逐個吐出，設數字則批次吐出。
        min_text_len: 過短條目閾值，直接跳過。
        show_progress: 是否顯示進度條。

    Yields:
        單個 Chunk（Dict）或一批 Chunk（List[Dict]），視 batch_size 而定。
    """
    buffer: List[Chunk] = []  # 批次模式的暫存區，集滿 batch_size 即吐出

    for article in stream_wiki_articles(
        file_path, min_text_len=min_text_len, show_progress=show_progress
    ):
        # 1. 清洗內文
        cleaned = clean_text(article["text"])
        if len(cleaned) < min_text_len:
            continue

        # 2. 切片（含 Overlap）
        pieces = split_text_to_chunks(
            cleaned,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
        )

        # 3. 包裝成 Chunk（含溯源資訊，方便除錯與引用出處）
        for idx, piece in enumerate(pieces):
            chunk: Chunk = {
                "id": article["id"],
                "title": article["title"],
                "chunk_index": idx,
                "chunk_id": f"{article['id']}_{idx}",
                "text": piece,
                "char_len": len(piece),
            }
            # 逐個吐出模式：直接 yield
            if batch_size is None:
                yield chunk
            else:
                buffer.append(chunk)
                # 集滿一批即吐出，並清空暫存（記憶體只保留一個 batch）
                if len(buffer) >= batch_size:
                    yield buffer
                    buffer = []

    # 4. 批次模式收尾：吐出不滿一批的殘留
    if batch_size is not None and buffer:
        yield buffer


# 與 stream_chunks(batch_size=N) 等價的語意別名，讓呼叫端意圖更明確
def batched_chunks(
    file_path: PathLike,
    batch_size: int = 32,
    chunk_size: int = 600,
    overlap_ratio: float = 0.1,
    min_text_len: int = 50,
    show_progress: bool = False,
) -> Generator[List[Chunk], None, None]:
    """批次吐出 Chunk 的便捷包裝（內部轉呼叫 stream_chunks）。

    Args:
        file_path: 資料檔路徑。
        batch_size: 每批幾個 Chunk，預設 32（embedding 常見批次量）。
        chunk_size: 目標區塊大小（字元數）。
        overlap_ratio: 重疊比例，預設 0.1（10%）。
        min_text_len: 過短條目閾值。
        show_progress: 是否顯示進度條。

    Yields:
        每批一個 List[Chunk]。
    """
    yield from stream_chunks(  # type: ignore[misc]
        file_path,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
        min_text_len=min_text_len,
        show_progress=show_progress,
    )


# ══════════════════════════════════════════════
# 四、命令列快速驗證（python -m src.loader）
# ══════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    import argparse

    # 【簡易 CLI】不用額外依賴即可快速抽查前 N 篇效果
    parser = argparse.ArgumentParser(description="維基百科 loader 快速驗證工具")
    parser.add_argument("--file", default="data/wikipedia-zh-cn-20240901.json", help="資料檔路徑")
    parser.add_argument("--limit", type=int, default=3, help="只顯示前 N 篇條目")
    parser.add_argument("--chunk-size", type=int, default=600, help="目標區塊大小")
    parser.add_argument("--batch-size", type=int, default=0, help=">0 則以批次模式顯示")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    bs = args.batch_size if args.batch_size > 0 else None
    shown = 0
    for item in stream_chunks(args.file, chunk_size=args.chunk_size, batch_size=bs):  # type: ignore[arg-type]
        # 批次模式 item 是 List，逐個模式 item 是 Dict，統一轉成 List 方便列印
        chunks: List[Chunk] = item if isinstance(item, list) else [item]
        for c in chunks:
            print(f"\n=== {c['title']} [{c['chunk_id']}] ({c['char_len']}字) ===")
            print(c["text"][:300] + ("…" if c["char_len"] > 300 else ""))
            shown += 1
            if shown >= args.limit:
                raise SystemExit(0)
