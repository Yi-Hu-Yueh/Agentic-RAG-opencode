"""知識圖譜處理模組（spark_processor.py）

負責串接 Muse Spark 1.3，將 ``loader.py`` 切好的 Chunk
做「實體萃取 → 關係建構 → 結構化儲存」，供後續精準檢索使用。

核心約束（開發者明確要求）：
    1. 【不要使用向量化】全模組絕不計算 Embedding、不做向量相似度檢索。
    2. ChromaDB 僅作為「結構化標籤 / 文字索引儲存器」：
       只用 ``add / upsert / get``，檢索只走 ``where``（Metadata 精準比對）
       與 ``where_document``（關鍵字 $contains），絕不呼叫 ``query()``。
    3. SQLite3 存放 3 張核心資料表，為關聯圖譜的權威來源（source of truth）。

儲存端雙寫索引（Dual-Indexed）架構：
    維基庫為簡體、用戶常用繁體提問，單字形索引必定漏檢。本模組在「寫入層」
   （非抽取層）用 OpenCC 自動產生對偶字形並雙寫：
    - SQLite：entities / relations 每筆知識存「原文列（script='orig'）」＋
      「轉換列（script='conv'）」兩個字形，完全比對天然跨字形命中。
    - ChromaDB：每個 Chunk 存「原文文件」＋「轉換文件（id 後綴 -trad）」，
      ``where_document`` 的 $contains 子字串比對跨字形百分之百命中。
    抽取層（extract_*/EnrichedChunk）保持原文不轉換，轉換只發生在寫入邊界，
    故舊程式讀取抽取結果的行為完全不受影響。

資料流向：
    loader.batched_chunks() → enrich_batch() → [SQLite3 3表(雙字形) + ChromaDB(雙文件)]
                                                     ↓
                                    agent_router 精準檢索（search_* 方法）
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ──────────────────────────────────────────────
# 日誌與型別別名
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)

# 檔案路徑型別別名：支援字串或 Path 物件
PathLike = Union[str, Path]

# 與 loader.py 的 Chunk 字典格式對齊：
# {"id", "title", "chunk_index", "chunk_id", "text", "char_len"}
Chunk = Dict[str, Any]


# ══════════════════════════════════════════════
# 一、設定與回傳資料結構
# ══════════════════════════════════════════════


@dataclass
class SparkConfig:
    """Muse Spark 1.3 與儲存後端設定。

    Attributes:
        model: 模型名稱，預設 "muse-spark-1.3"。
        api_base: API 位址（OpenAI 相容的 chat/completions 端點前綴）。
            為空字串表示【離線規則模式】，不對外發請求，開箱即用。
            也可由環境變數 ``SPARK_API_BASE`` 提供。
        api_key: API 金鑰，建議由環境變數 ``SPARK_API_KEY`` 提供，不要寫死。
        timeout: 單次 API 請求逾時（秒）。
        max_retries: API 失敗時重試次數，耗盡後自動降級為規則抽取。
        mode: "rule"（離線規則抽取，預設）或 "api"（呼叫 Spark 1.3 API）。
            設為 "api" 但連線失敗時，會自動降級為規則抽取並記錄警告。
        db_path: SQLite3 檔案路徑，父目錄會自動建立。
        chroma_dir: ChromaDB 持久化目錄，父目錄會自動建立。
        chroma_collection: ChromaDB 集合名稱（需符合 3~512 字元命名規範）。
        max_entities_per_chunk: 每個 Chunk 最多保留幾個實體。
        max_relations_per_chunk: 每個 Chunk 最多保留幾條關係。
        dual_script: 是否啟用簡繁雙寫（預設 True）。關閉則回到單字形寫入
            （行為與舊版完全一致）；若 opencc 未安裝也會自動降級為單寫。
    """

    model: str = "muse-spark-1.3"
    api_base: str = ""
    api_key: str = ""
    timeout: int = 60
    max_retries: int = 2
    mode: str = "rule"
    db_path: str = "data/knowledge.db"
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "wiki_chunks"
    max_entities_per_chunk: int = 20
    max_relations_per_chunk: int = 30
    dual_script: bool = True

    def __post_init__(self) -> None:
        """以環境變數補齊未填的連線參數（環境變數優先級低於顯式傳參）。"""
        # 【環境變數 fallback】沒顯式設定才讀環境，避免覆蓋程式傳入值
        if not self.api_base:
            self.api_base = os.getenv("SPARK_API_BASE", "")
        if not self.api_key:
            self.api_key = os.getenv("SPARK_API_KEY", "")
        env_model = os.getenv("SPARK_MODEL", "")
        if env_model and self.model == "muse-spark-1.3":
            self.model = env_model


@dataclass
class EnrichedChunk:
    """經知識圖譜擴充後的 Chunk。

    Attributes:
        chunk_id: 對應 loader 產出的 chunk_id，方便回溯原文。
        entities: 實體清單，如 [{"text": "孔子", "type": "人物", "freq": 3}]。
        relations: 關係三元組，如
            [{"s": "孔子", "p": "出生於", "o": "曲阜", "weight": 2.0}]。
        summary: 短摘要（規則模式取首句；API 模式由模型生成）。
        meta: 透傳欄位（title、article_id、char_len 等）。
    """

    chunk_id: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════
# 二、離線規則式抽取（無需連線，開箱即用）
# ══════════════════════════════════════════════

# 【中文常用停用字】單字成詞或包在候選詞內時過濾，避免抽出"的/是/在"等雜訊
_STOP_CHARS = set("的了是在有和與或及等之其這那你我他她它為被把將對就都也而於以從到比過個些種們為著那這會可所能應但若如因果當每各另別先後內外上下左右大小多少很非常最更太又再還才僅")

# 【常見姓氏】用於辨識「姓氏 + 1~2 字」的人名候選（如 王小明、歐陽修）
_SURNAMES = (
    "王李張劉陳楊黃趙吳周徐孫馬朱胡郭何高林羅鄭梁謝宋唐許韓馮董蕭程曹袁鄧傅沈曾彭呂蘇盧蔣蔡賈丁魏薛葉閻余潘杜戴夏鍾汪田任姜范方石姚譚廖鄒熊金陸郝孔白崔康毛邱秦江史顧侯邵孟龍萬段雷錢湯尹黎易武戴喬賀賴龔文歐陽司馬上官諸葛"
)

# 【地名後綴（元組，每項為完整後綴）】命中則判為「地點」
# 注意：必須用元組逐項比對，不可用長字串（逐字迭代會把每個字都當後綴，造成誤判）
_PLACE_SUFFIXES = (
    "市", "省", "縣", "區", "州", "國", "洲", "洋", "海", "湖", "河", "江",
    "山", "島", "嶼", "鎮", "村", "路", "街", "橋", "站", "港", "灣",
    "都", "京", "郡", "府",
)

# 【組織後綴（元組）】命中則判為「組織」（多字後綴在前，單字在後）
_ORG_SUFFIXES = (
    "大學", "學院", "學校", "公司", "集團", "企業", "銀行", "政府", "法院",
    "議會", "協會", "組織", "機構", "研究院", "研究所", "中心", "基金會",
    "寺廟", "教會", "聯盟", "報社",
    "黨", "局", "處", "署", "廳", "部", "委", "會", "社", "館",
)

# 【術語後綴（元組）】命中則判為「術語」（技術名詞 / 學科 / 事件）
_TERM_SUFFIXES = (
    "理論", "定理", "定律", "公式", "演算法", "算法", "模型", "網路", "網絡",
    "系統", "語言", "文字", "主義", "思想", "學派", "學科", "科學", "技術",
    "工程", "方法", "方程", "函數", "矩陣", "向量", "戰爭", "戰役", "條約",
    "朝代", "帝國", "王朝", "節日", "電影", "小說", "作品", "效應", "原則",
)

# 【候選詞尾字黑名單】以這些字結尾的多半是句子碎片（如「學的」「是中」），直接丟棄
_BAD_TAIL_CHARS = set("的了是和與在有著過地得")

# 【正則預編譯】放在模組層級，避免每次抽取重複編譯
_RE_BOOK = re.compile(r"《([^《》]{1,20})》")  # 書名號：作品 / 專有名詞
_RE_QUOTE = re.compile(r"[「“\"]([^「」“”\"]{2,10})[」”\"]")  # 引號：概念 / 別名
_RE_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,8}")  # 連續中文串，候選詞來源
_RE_SENTENCE_SPLIT = re.compile(r"[。！？\n]+")  # 句子切分（關係抽取用）

# 【關係觸發詞 → 謂語】句子中出現觸發詞即推斷該關係；皆無則回退為「相關」
_PREDICATE_TRIGGERS: List[Tuple[str, Tuple[str, ...]]] = [
    ("出生於", ("出生", "生於", "誕生")),
    ("位於", ("位於", "坐落", "地處")),
    ("創立", ("創立", "創建", "建立", "創辦")),
    ("發現", ("發明", "發現")),
    ("提出", ("提出", "主張", "倡導")),
    ("屬於", ("屬於", "是一種", "係")),
    ("包含", ("包括", "包含", "涵蓋")),
    ("擊敗", ("戰勝", "擊敗", "打敗")),
    ("統治", ("統治", "管轄", "治理")),
    ("著作", ("撰寫", "著有", "寫下", "出版")),
    ("任職於", ("擔任", "出任", "任職")),
    ("畢業於", ("畢業",)),
    ("師從", ("師從", "師承", "拜師")),
]


def _classify_entity(word: str) -> str:
    """依後綴與姓氏規則判斷實體型別（人物 / 地點 / 組織 / 術語 / 作品 / 概念 / 其他）。

    判斷順序：書名號殘留 → 後綴命中（術語/組織/地點）→ 姓氏開頭（人物）→ 其他。
    """
    # 1. 後綴判斷（多字後綴優先：元組已按長度由長到短排列的語義分組）
    for suffix in _TERM_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return "術語"
    for suffix in _ORG_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return "組織"
    for suffix in _PLACE_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return "地點"
    # 2. 姓氏開頭 + 總長 2~4 字 → 很可能是人名
    if 2 <= len(word) <= 4 and word[0] in _SURNAMES:
        return "人物"
    return "其他"


def _rule_extract_entities(
    text: str,
    title: str = "",
    max_entities: int = 20,
) -> List[Dict[str, Any]]:
    """規則式實體抽取（離線可用）：書名號 → 引號 → 後綴詞 → 高頻中文詞。

    Args:
        text: Chunk 純文字。
        title: 條目標題，一律加入為「主題」實體，確保圖譜連通性。
        max_entities: 上限，超量時按頻率截斷。

    Returns:
        [{"text", "type", "freq"}]，按 freq 由高到低排序。
    """
    if not text:
        return []

    freq: Dict[str, int] = {}  # 候選詞 → 出現次數
    etype: Dict[str, str] = {}  # 候選詞 → 型別（先標記者優先保留）

    def _add(word: str, kind: str) -> None:
        """加入候選詞（去雜訊：過短 / 純停用字 / 糟糕尾字，否則只累計次數）。"""
        word = word.strip(" ，。、；：！？「」『』《》“”\"'()（）")
        if len(word) < 2 or len(word) > 8:
            return
        # 全由停用字組成則丟棄（如 "的是"）
        if all(ch in _STOP_CHARS for ch in word):
            return
        # 尾字是虛詞多半是句子碎片（如「學的」「是中」），丟棄
        if word[-1] in _BAD_TAIL_CHARS:
            return
        freq[word] = text.count(word)
        # 先標記者優先（如書名號標的「作品」不被後綴規則覆蓋）
        etype.setdefault(word, kind)

    # 1. 書名號《》→ 作品（如《史記》）
    for m in _RE_BOOK.findall(text):
        _add(m, "作品")

    # 2. 引號「」→ 概念 / 別名
    for m in _RE_QUOTE.findall(text):
        _add(m, "概念")

    # 3. 後綴詞：在連續中文串中尋找「前缀 + 完整後綴」結構
    #    多字後綴優先比對（元組順序已保證長後綴先被命中，避免「學」類單字誤判）
    all_suffixes = _TERM_SUFFIXES + _ORG_SUFFIXES + _PLACE_SUFFIXES
    for run in _RE_CJK_RUN.findall(text):
        for i in range(1, len(run)):
            for suffix in all_suffixes:
                if run.startswith(suffix, i):
                    # 往前取最多 4 字組成候選，總長不超過 6 字
                    start = max(0, i - 4)
                    word = run[start : i + len(suffix)]
                    if len(word) <= 6:
                        _add(word, _classify_entity(word))
                    break

    # 4. 高頻中文詞：連續 2~4 字串出現 ≥2 次 → 依規則分類
    for run in _RE_CJK_RUN.findall(text):
        # 從長串中滑動取出 2~4 字子串
        for size in (4, 3, 2):
            for i in range(len(run) - size + 1):
                word = run[i : i + size]
                if word in freq:
                    continue
                if word[0] in _STOP_CHARS:
                    continue  # 首字是停用字則跳過（如"的X"）
                if text.count(word) >= 2:
                    _add(word, _classify_entity(word))

    # 5. 條目標題一律收錄為「主題」實體（freq 給高權重，確保排序靠前）
    if title and 1 < len(title) <= 20:
        freq[title] = max(freq.get(title, 0), text.count(title) + 1000)
        etype[title] = "主題"

    # 6. 按頻率排序並截斷
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:max_entities]
    return [{"text": w, "type": etype.get(w, "其他"), "freq": f} for w, f in ranked]


def _infer_predicate(sentence: str) -> str:
    """依觸發詞推斷句子內實體關係的謂語；無命中則回退為「相關」。"""
    for predicate, triggers in _PREDICATE_TRIGGERS:
        if any(t in sentence for t in triggers):
            return predicate
    return "相關"


def _rule_extract_relations(
    text: str,
    entities: List[Dict[str, Any]],
    max_relations: int = 30,
) -> List[Dict[str, Any]]:
    """規則式關係抽取（離線可用）：同句共現 → 實體對，謂語由觸發詞推斷。

    權重 = 共現句數（同一對在多句出現則累加），代表關聯強度。

    Args:
        text: Chunk 純文字。
        entities: 已抽出的實體清單（取其 text 參與共現統計）。
        max_relations: 上限，超量時按 weight 截斷。

    Returns:
        [{"s", "p", "o", "weight", "evidence"}]，按 weight 由高到低排序。
    """
    if not text or not entities:
        return []

    names = [e["text"] for e in entities]
    # 權重累加器：(s, p, o) → 共現次數；evidence 保留首次出現的句子
    pair_weight: Dict[Tuple[str, str, str], float] = {}
    pair_evidence: Dict[Tuple[str, str, str], str] = {}

    # 逐句掃描：只考慮「同句出現 ≥2 個實體」的句子
    for sentence in _RE_SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if len(sentence) < 4:
            continue
        present = [n for n in names if n and n in sentence]
        # 去重並保持穩定順序（避免 A-B 與 B-A 重複計算）
        present = sorted(set(present))
        if len(present) < 2:
            continue
        predicate = _infer_predicate(sentence)
        # 兩兩配對（同一句實體過多時只取前 6 個，避免組合爆炸）
        for i in range(min(len(present), 6)):
            for j in range(i + 1, min(len(present), 6)):
                key = (present[i], predicate, present[j])
                pair_weight[key] = pair_weight.get(key, 0.0) + 1.0
                pair_evidence.setdefault(key, sentence[:120])

    ranked = sorted(pair_weight.items(), key=lambda kv: kv[1], reverse=True)[:max_relations]
    return [
        {"s": s, "p": p, "o": o, "weight": w, "evidence": pair_evidence[(s, p, o)]}
        for (s, p, o), w in ranked
    ]


def _rule_summarize(text: str, limit: int = 120) -> str:
    """規則式摘要：取首個長度適中的句子，超長則截斷（離線可用）。"""
    for sentence in _RE_SENTENCE_SPLIT.split(text or ""):
        sentence = sentence.strip()
        if len(sentence) >= 20:
            return sentence[:limit]
    return (text or "").strip()[:limit]


# ══════════════════════════════════════════════
# 三、簡繁雙寫轉換層（Dual-Indexed 核心）
# ══════════════════════════════════════════════
#
# 本層是「儲存端雙寫」的唯一轉換點：抽取層保持原文，寫入層在此產生對偶字形。
# 轉換器採【模組級單例＋延遲初始化】：首次使用才載入 OpenCC 詞庫；
# 若 opencc 未安裝則自動降級為單字形寫入（管線不中斷，僅記警告）。

# 【簡轉繁轉換器】任務指定的模組級轉換器（延遲初始化，見 _get_s2t）。
# 直接使用：cc.convert("数学") → "數學"；繁體輸入近乎恆等映射。
cc: Any = None  # 實際為 opencc.OpenCC，延遲初始化以避免 import 期載入詞庫
_cc_t2s: Any = None  # 反向轉換器（僅 SQLite LIKE 兜底查詢用）
_cc_failed = False  # 載入失敗後不再重試，避免每次寫入都卡住


def _get_s2t() -> Optional[Any]:
    """取回簡轉繁轉換器（首次呼叫才初始化並賦值給模組級 ``cc``）。"""
    global cc
    if cc is not None or _cc_failed:
        return cc
    try:
        from opencc import OpenCC  # 延遲匯入，沒裝也不影響 import 本模組

        cc = OpenCC("s2t")
        return cc
    except Exception as e:  # noqa: BLE001 — 缺套件/詞庫損壞皆降級單寫
        globals()["_cc_failed"] = True
        logger.warning("OpenCC 載入失敗（%s），自動降級為單字形寫入", e)
        return None


def _get_t2s() -> Optional[Any]:
    """取回繁轉簡轉換器（僅查詢端 LIKE 兜底用，寫入端不需要）。"""
    global _cc_t2s
    if _cc_t2s is not None or _cc_failed:
        return _cc_t2s
    try:
        from opencc import OpenCC

        _cc_t2s = OpenCC("t2s")
        return _cc_t2s
    except Exception as e:  # noqa: BLE001 — 同上，降級不中斷查詢
        globals()["_cc_failed"] = True
        logger.warning("OpenCC 載入失敗（%s），查詢端僅用原始字形", e)
        return None


def _dual_variants(text: str, enable: bool = True) -> List[Tuple[str, str]]:
    """產生某字串的雙寫變體：[(原文, 'orig'), (轉換文, 'conv')...]。

    規則：原文一律保留；轉換文為空、與原文相同、或轉換器不可用時只回傳原文。
    script 標籤（'orig' / 'conv'）會寫入 SQLite 的 script 欄，方便稽核雙寫。

    Args:
        text: 原始字串（實體名 / 關係端點 / 謂語）。
        enable: 總開關（對應 SparkConfig.dual_script），關閉即單寫。

    Returns:
        去重後的 (文字, script) 串列，原文永遠在第一位。
    """
    text = text or ""
    if not text:
        return []
    if not enable:
        return [(text, "orig")]
    converter = _get_s2t()
    if converter is None:
        return [(text, "orig")]  # 降級：單寫原文
    try:
        converted = converter.convert(text)
    except Exception:  # noqa: BLE001 — 單串轉換失敗不應拖垮整批
        return [(text, "orig")]
    if not converted or converted == text:
        return [(text, "orig")]  # 恆等映射（如純繁體/英文/數字）無需雙寫
    return [(text, "orig"), (converted, "conv")]


def _query_variants(keyword: str) -> List[str]:
    """產生查詢關鍵字的跨字形變體（SQLite LIKE / title 兜底查詢用）。

    回傳去重後的 [原文, s2t(原文), t2s(原文)]（恆等/空值自動剔除），
    呼叫端以 OR 串接即可跨字形命中。轉換器不可用時只回傳原文。
    """
    keyword = keyword or ""
    if not keyword:
        return []
    variants = [keyword]
    for converter in (_get_s2t(), _get_t2s()):
        if converter is None:
            continue
        try:
            converted = converter.convert(keyword)
        except Exception:  # noqa: BLE001 — 單串失敗就跳過該變體
            continue
        if converted and converted not in variants:
            variants.append(converted)
    return variants


# ══════════════════════════════════════════════
# 四、SQLite3 三張核心表（關聯圖譜權威來源）
# ══════════════════════════════════════════════

# 【建表 SQL】冪等設計（IF NOT EXISTS），重複初始化不會報錯；
# 唯一鍵 + UPSERT 保證重跑管線不會產生重複資料。
#
# 【雙寫與唯一鍵的設計說明】
# entities 唯一鍵 (chunk_id, entity_text, entity_type) 與
# relations 唯一鍵 (chunk_id, subj, predicate, obj)「保持不變」：
# 雙字形即不同列值（如 数学/人物 與 數學/人物），天然共存、無需放寬；
# 保留原約束反而讓重跑冪等性對繁簡兩列同時成立。
# 本次 schema 調整只有一處：兩表各新增 ``script`` 欄（'orig'/'conv'），
# 標註該列來源字形，方便稽核雙寫完整性；舊庫由 _ensure_script_columns()
# 以 ALTER TABLE 補欄（可空，舊資料不受影響），重跑管線即自動回填繁體列。
_SCHEMA_SQL = """
-- 表 1：文章 / Chunk 表（loader 切片的存放處，chunk_id 主鍵；單列，不雙寫）
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    article_id  TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    text        TEXT NOT NULL DEFAULT '',
    char_len    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(title);

-- 表 2：實體表（每列 = 某 Chunk 提到某實體，含型別、次數與字形來源）
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT NOT NULL REFERENCES chunks(chunk_id),
    article_id  TEXT NOT NULL DEFAULT '',
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT '其他',
    freq        INTEGER NOT NULL DEFAULT 1,
    script      TEXT NOT NULL DEFAULT '',
    UNIQUE (chunk_id, entity_text, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

-- 表 3：關係權重表（每列 = 一條 s-p-o 三元組，weight 表關聯強度）
CREATE TABLE IF NOT EXISTS relations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id  TEXT NOT NULL REFERENCES chunks(chunk_id),
    subj      TEXT NOT NULL,
    predicate TEXT NOT NULL,
    obj       TEXT NOT NULL,
    weight    REAL NOT NULL DEFAULT 1.0,
    evidence  TEXT NOT NULL DEFAULT '',
    script    TEXT NOT NULL DEFAULT '',
    UNIQUE (chunk_id, subj, predicate, obj)
);
CREATE INDEX IF NOT EXISTS idx_relations_subj ON relations(subj);
CREATE INDEX IF NOT EXISTS idx_relations_obj ON relations(obj);
CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(chunk_id);
"""


# ══════════════════════════════════════════════
# 五、SparkProcessor 主類別
# ══════════════════════════════════════════════


class SparkProcessor:
    """Muse Spark 1.3 知識圖譜處理器（儲存端雙寫索引版）。

    使用範例：
        cfg = SparkConfig()  # 預設離線規則模式＋雙寫，開箱即用
        with SparkProcessor(cfg) as sp:
            for batch in batched_chunks("data/wikipedia-zh-cn-20240901.json", batch_size=32):
                sp.enrich_batch(batch)   # 抽取 → 雙寫 SQLite + Chroma
            print(sp.stats())

    檢索範例（無向量、純精準檢索，繁簡皆可命中）：
        sp.search_by_entity("孔子")      # 實體 → 含該實體的 Chunk
        sp.search_by_entity("數學")      # 繁體同樣命中（雙寫列）
        sp.search_by_keyword("儒家")      # 關鍵字 → 內文含關鍵字的 Chunk
        sp.get_relations("孔子")         # 實體 → 相關三元組
    """

    def __init__(self, config: Optional[SparkConfig] = None) -> None:
        """初始化：保存設定 → 建 SQLite 三表 → 舊庫補 script 欄 → 延遲初始化 Chroma。

        Chroma 採【延遲初始化】：真正要寫入時才連線，
        讓沒裝 chromadb 的環境仍可使用 SQLite 模式（降級不報錯）。
        """
        self.config = config or SparkConfig()

        # ── 初始化 SQLite3（建檔 + 建三表 + 開 WAL 提升併發讀）──
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)  # 自動建父目錄
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # 查回來可用欄名取值
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")  # 讀寫併發更順
        except sqlite3.Error:
            pass  # 部分環境（如 :memory:）不支援 WAL，忽略即可
        self._conn.executescript(_SCHEMA_SQL)
        self._ensure_script_columns()  # 舊庫遷移：補 script 欄（新庫已有則跳過）
        self._conn.commit()

        # ── Chroma 延遲初始化旗標（首次寫入時才連線）──
        self._chroma_client: Any = None
        self._collection: Any = None
        self._chroma_failed = False  # 失敗後不再重試，避免每次寫入都卡住

    def _ensure_script_columns(self) -> None:
        """舊庫遷移：若 entities / relations 缺 script 欄，以 ALTER 補上。

        背景：雙寫重構前建的庫沒有 script 欄；補欄後舊列 script 為 ''，
        照常可查（簡體原文），重跑管線會自動回填繁體列（冪等 UPSERT）。
        """
        for table in ("entities", "relations"):
            try:
                cols = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            except sqlite3.Error:
                continue  # 表不存在（極端情況），交給 executescript 處理即可
            if "script" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN script TEXT NOT NULL DEFAULT ''")
                logger.info("舊庫遷移：%s 表已補上 script 欄", table)

    # ── 上下文管理器：with 語法自動關閉 SQLite 連線 ──
    def __enter__(self) -> "SparkProcessor":
        """進入 with 區塊時回傳自身。"""
        return self

    def __exit__(self, *exc: Any) -> None:
        """離開 with 區塊時關閉 SQLite 連線。"""
        self.close()

    def close(self) -> None:
        """關閉 SQLite 連線（ChromaDB 為持久化客戶端，無需顯式關閉）。"""
        try:
            self._conn.commit()
            self._conn.close()
        except sqlite3.Error:
            pass

    # ══════════════════════════════════════════
    # 5.1 Muse Spark 1.3 API 呼叫（api 模式專用）
    # ══════════════════════════════════════════

    def _call_spark_api(self, system_prompt: str, user_prompt: str) -> str:
        """呼叫 Spark 1.3（OpenAI 相容 chat/completions），回傳模型原文。

        含重試機制；最終失敗則拋出例外，由上層捕捉並降級為規則抽取。

        Raises:
            RuntimeError: 未設定 api_base / 未安裝 requests / 重試耗盡。
        """
        if not self.config.api_base:
            raise RuntimeError("未設定 api_base，無法使用 api 模式")
        try:
            import requests  # 延遲匯入，規則模式完全不需要它
        except ImportError as e:
            raise RuntimeError("未安裝 requests，無法使用 api 模式（pip install requests）") from e

        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # 抽取任務用低溫，追求穩定可解析
        }

        last_err: Optional[Exception] = None
        for _ in range(max(1, self.config.max_retries + 1)):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                resp.raise_for_status()
                return str(resp.json()["choices"][0]["message"]["content"])
            except Exception as e:  # noqa: BLE001 — 重試迴圈內統一捕捉，耗盡才拋出
                last_err = e
        raise RuntimeError(f"Spark API 呼叫失敗（已重試）：{last_err}")

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """從模型回文中擷取 JSON 陣列（容忍前後夾雜說明文字）。

        策略：取第一個 '[' 到最後一個 ']' 之間的子字串做 json.loads，
        失敗則回傳空串列，由上層降級為規則抽取。
        """
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            parsed = json.loads(raw[start:end])
            return [d for d in parsed if isinstance(d, dict)]
        except (ValueError, json.JSONDecodeError):
            logger.warning("Spark 回傳無法解析為 JSON，已忽略本次 API 結果")
            return []

    # ══════════════════════════════════════════
    # 5.2 實體 / 關係抽取（api 優先，失敗降級 rule）
    # ══════════════════════════════════════════

    def extract_entities(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """從單段文字抽取實體。

        api 模式：先調 Muse Spark 1.3，失敗自動降級為規則抽取；
        rule 模式：直接使用離線規則，零延遲、零費用。

        注意：此處回傳【原文】實體（不轉換）；簡繁雙寫發生在寫入層
        （_store_entity_rows），故呼叫端行為與舊版一致。

        Args:
            text: Chunk 純文字。
            title: 條目標題（可選）；提供時會收錄為「主題」實體，確保圖譜連通性。
        """
        if not text:
            return []
        if self.config.mode == "api":
            try:
                raw = self._call_spark_api(
                    "你是中文知識圖譜標註器。只輸出 JSON 陣列，每個元素為"
                    '{"text": 實體名, "type": 型別}，型別限：人物/地點/組織/術語/作品/概念/其他。'
                    "不要輸出任何說明文字。",
                    f"抽取以下文字中的實體：\n{text[:2000]}",
                )
                parsed = self._parse_json_array(raw)
                if parsed:
                    return [
                        {"text": str(d.get("text", "")), "type": str(d.get("type", "其他") or "其他"), "freq": 1}
                        for d in parsed
                        if str(d.get("text", "")).strip()
                    ][: self.config.max_entities_per_chunk]
                # API 回傳空 → 降級規則
                logger.warning("Spark 實體抽取回傳為空，降級為規則抽取")
            except Exception as e:  # noqa: BLE001 — 降級邏輯需捕捉所有 API 異常
                logger.warning("Spark 實體抽取失敗（%s），降級為規則抽取", e)
        # 規則模式 / 降級路徑（標題一律收錄為「主題」，見 _rule_extract_entities）
        return _rule_extract_entities(
            text, title=title, max_entities=self.config.max_entities_per_chunk
        )

    def extract_relations(
        self, text: str, entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """從單段文字抽取關係三元組（需先有實體清單，未提供則先抽實體）。

        同 extract_entities：api 優先、失敗降級為同句共現規則。
        回傳為原文三元組；雙寫轉換在 _store_relation_rows 完成。
        """
        if not text:
            return []
        # 未提供實體則先抽（避免呼叫端遺漏參數導致空結果）
        if not entities:
            entities = self.extract_entities(text)
        if not entities:
            return []
        if self.config.mode == "api":
            try:
                names = "、".join(e["text"] for e in entities[:20])
                raw = self._call_spark_api(
                    "你是中文知識圖譜標註器。只輸出 JSON 陣列，每個元素為"
                    '{"s": 主語, "p": 關係, "o": 賓語}。主語賓語必須出自給定的實體清單。'
                    "不要輸出任何說明文字。",
                    f"實體：{names}\n文字：\n{text[:2000]}",
                )
                parsed = self._parse_json_array(raw)
                if parsed:
                    return [
                        {
                            "s": str(d.get("s", "")),
                            "p": str(d.get("p", "相關")),
                            "o": str(d.get("o", "")),
                            "weight": 1.0,
                            "evidence": "",
                        }
                        for d in parsed
                        if str(d.get("s", "")) and str(d.get("o", ""))
                    ][: self.config.max_relations_per_chunk]
                logger.warning("Spark 關係抽取回傳為空，降級為規則抽取")
            except Exception as e:  # noqa: BLE001 — 同上，降級需捕捉所有異常
                logger.warning("Spark 關係抽取失敗（%s），降級為規則抽取", e)
        return _rule_extract_relations(
            text, entities, max_relations=self.config.max_relations_per_chunk
        )

    # ══════════════════════════════════════════
    # 5.3 知識豐富化：enrich_chunk / enrich_batch
    # ══════════════════════════════════════════

    def enrich_chunk(self, chunk: Chunk) -> EnrichedChunk:
        """擴充單個 Chunk：實體 + 關係 + 摘要（純抽取，不寫庫）。

        寫庫由 enrich_batch 統一處理（含批次交易），大量匯入請用 enrich_batch。
        """
        text = str(chunk.get("text", "") or "")
        title = str(chunk.get("title", "") or "")
        entities = self.extract_entities(text, title=title)
        # 【保險】規則路徑已含標題主題；API 路徑若缺失則手動補上，
        # 若已存在但型別不是主題則升級，確保圖譜以條目為中心連通
        if title:
            found = next((e for e in entities if e["text"] == title), None)
            if found is None:
                entities.insert(0, {"text": title, "type": "主題", "freq": 1})
                entities = entities[: self.config.max_entities_per_chunk]
            else:
                found["type"] = "主題"
        relations = self.extract_relations(text, entities)
        summary = _rule_summarize(text)
        return EnrichedChunk(
            chunk_id=str(chunk.get("chunk_id", "")),
            entities=entities,
            relations=relations,
            summary=summary,
            meta={
                "article_id": str(chunk.get("id", "")),
                "title": title,
                "chunk_index": int(chunk.get("chunk_index", 0) or 0),
                "char_len": int(chunk.get("char_len", len(text)) or 0),
            },
        )

    def enrich_batch(self, batch_chunks: List[Chunk]) -> List[EnrichedChunk]:
        """批次豐富化（核心入口）：抽取 → 雙寫 SQLite 三表 → 雙寫 Chroma。

        接收 loader.batched_chunks() 吐出的 Chunk 列表；
        SQLite 採【單次交易提交】，一批僅一次 commit（含繁簡雙列），匯入速度快；
        Chroma 採 upsert（重跑不重複，原文/轉換文件 id 皆確定性）。

        Args:
            batch_chunks: Chunk 字典列表（含 id/title/chunk_id/text 等欄位）。

        Returns:
            與輸入一一對應的 EnrichedChunk 列表（原文，未轉換）。
        """
        if not batch_chunks:
            return []

        # 1. 逐個抽取（目前為循序呼叫；API 模式如需併發可後續加線程池）
        enriched: List[EnrichedChunk] = [self.enrich_chunk(c) for c in batch_chunks]

        # 2. 寫入 SQLite 三表（單一交易，失敗整批 rollback 不留髒資料）
        try:
            with self._conn:  # with 連線＝交易區塊，異常自動 rollback
                for chunk, en in zip(batch_chunks, enriched):
                    self._store_chunk_row(chunk)
                    self._store_entity_rows(en)
                    self._store_relation_rows(en)
        except sqlite3.Error as e:
            logger.error("SQLite 寫入失敗：%s", e)
            raise

        # 3. 寫入 ChromaDB（原文文件＋轉換文件，無向量）
        self._store_chroma_batch(batch_chunks, enriched)

        return enriched

    # ── SQLite 寫入輔助（三表各一函數，職責分明）──

    def _store_chunk_row(self, chunk: Chunk) -> None:
        """寫入 chunks 表（INSERT OR REPLACE，重跑可覆蓋更新；單列不雙寫）。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, article_id, title, chunk_index, text, char_len)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(chunk.get("chunk_id", "")),
                str(chunk.get("id", "")),
                str(chunk.get("title", "") or ""),
                int(chunk.get("chunk_index", 0) or 0),
                str(chunk.get("text", "") or ""),
                int(chunk.get("char_len", 0) or 0),
            ),
        )

    def _store_entity_rows(self, en: EnrichedChunk) -> None:
        """雙寫 entities 表：原文列（script='orig'）＋轉換列（script='conv'）。

        UPSERT 語義：唯一鍵 (chunk_id, entity_text, entity_type) 對繁簡兩列
        各自成立（字形不同即不同列），已存在則取較大 freq，重跑不膨脹。
        轉換恆等（如英文/數字）時只寫原文列，避免無意義重複。
        """
        dual = bool(self.config.dual_script)
        for e in en.entities:
            text = str(e.get("text", ""))
            if not text:
                continue
            etype = str(e.get("type", "其他") or "其他")
            freq = int(e.get("freq", 1) or 1)
            for variant, script in _dual_variants(text, enable=dual):
                self._conn.execute(
                    """INSERT INTO entities (chunk_id, article_id, entity_text, entity_type, freq, script)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT (chunk_id, entity_text, entity_type)
                       DO UPDATE SET freq = max(entities.freq, excluded.freq)""",
                    (en.chunk_id, en.meta.get("article_id", ""), variant, etype, freq, script),
                )

    def _store_relation_rows(self, en: EnrichedChunk) -> None:
        """雙寫 relations 表：原文三元組＋轉換三元組（s/p/o 逐項轉換）。

        evidence 保留原文（忠實溯源）；轉換三元組與原文完全相同時只寫一列；
        唯一鍵 (chunk_id, subj, predicate, obj) 對兩列各自成立，重跑冪等。
        """
        dual = bool(self.config.dual_script)
        for r in en.relations:
            s = str(r.get("s", ""))
            p = str(r.get("p", "相關") or "相關")
            o = str(r.get("o", ""))
            if not s or not o:
                continue
            weight = float(r.get("weight", 1.0) or 1.0)
            evidence = str(r.get("evidence", "") or "")
            # s/p/o 各自產生變體；笛卡兒組合中只取「全原文」與「全轉換」兩種，
            # 避免混合字形三元組污染圖譜語義
            s_variants = _dual_variants(s, enable=dual)
            p_variants = _dual_variants(p, enable=dual)
            o_variants = _dual_variants(o, enable=dual)
            n = max(len(s_variants), len(p_variants), len(o_variants))
            seen = set()
            for i in range(n):
                sv = s_variants[min(i, len(s_variants) - 1)][0]
                pv = p_variants[min(i, len(p_variants) - 1)][0]
                ov = o_variants[min(i, len(o_variants) - 1)][0]
                script = "orig" if (sv, pv, ov) == (s, p, o) else "conv"
                if (sv, pv, ov) in seen:
                    continue  # 恆等轉換去重（s/p/o 任一無變體時的重疊組合）
                seen.add((sv, pv, ov))
                self._conn.execute(
                    """INSERT INTO relations (chunk_id, subj, predicate, obj, weight, evidence, script)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (chunk_id, subj, predicate, obj)
                       DO UPDATE SET weight = max(relations.weight, excluded.weight)""",
                    (en.chunk_id, sv, pv, ov, weight, evidence, script),
                )

    # ── ChromaDB 寫入（無向量：只用 upsert + get，不過 query）──

    def _ensure_chroma(self) -> bool:
        """延遲建立 Chroma 持久化客戶端與集合；成功回傳 True。

        失敗（未安裝 / 目錄無權限）則記錄警告並回傳 False，
        上層自動降級為純 SQLite 模式，管線不中斷。
        """
        if self._collection is not None:
            return True
        if self._chroma_failed:
            return False
        try:
            import chromadb  # 延遲匯入，SQLite 模式不需要它
        except ImportError:
            logger.warning("未安裝 chromadb，將以純 SQLite 模式運作（pip install chromadb）")
            self._chroma_failed = True
            return False
        try:
            chroma_dir = Path(self.config.chroma_dir)
            chroma_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
            # 【關鍵】不傳 embedding_function 也不提供 embeddings：
            # 實測 chromadb 1.5 不啟用 embedding 也可 add/get，
            # 檢索只走 where / where_document，完全無向量運算。
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.config.chroma_collection
            )
            return True
        except Exception as e:  # noqa: BLE001 — 降級邏輯需捕捉所有初始化異常
            logger.warning("ChromaDB 初始化失敗（%s），降級為純 SQLite 模式", e)
            self._chroma_failed = True
            return False

    def _store_chroma_batch(
        self, batch_chunks: List[Chunk], enriched: List[EnrichedChunk]
    ) -> None:
        """雙寫 Chroma：原文文件（id=chunk_id）＋轉換文件（id=chunk_id-trad）。

        upsert 語義（id 確定性，重跑不重複）；轉換文件與原文恆等時只寫原文。
        Metadata 雙字形標註：entities 欄在原文列存原文名、轉換列存轉換名，
        另有 script（'orig'/'conv'）與 base_id（回指原 chunk）供去重與稽核。
        Metadata 值限 str/int/float/bool，實體串以 "|" 連接成單一字串。
        """
        if not self._ensure_chroma():
            return  # 降級：SQLite 已寫入成功，Chroma 跳過即可
        dual = bool(self.config.dual_script)
        ids, docs, metas = [], [], []
        for chunk, en in zip(batch_chunks, enriched):
            text = str(chunk.get("text", "") or "")
            if not text.strip():
                continue  # 空文本跳過 Chroma（SQLite 仍保留該列）
            base_meta = {
                "article_id": en.meta.get("article_id", ""),
                "title": en.meta.get("title", ""),
                "chunk_index": int(en.meta.get("chunk_index", 0) or 0),
                "char_len": int(en.meta.get("char_len", 0) or 0),
                "base_id": en.chunk_id,
                "entity_count": len(en.entities),
                "relation_count": len(en.relations),
            }
            # 列 1：原文文件
            ids.append(en.chunk_id)
            docs.append(text)
            metas.append({
                **base_meta,
                "script": "orig",
                # 實體名單壓成字串，方便 where 過濾與除錯閱覽
                "entities": "|".join(e["text"] for e in en.entities[:20]),
            })
            # 列 2：轉換文件（恆等時跳過，避免無意義重複）
            for variant, script in _dual_variants(text, enable=dual):
                if script == "orig":
                    continue
                conv_names = []
                for e in en.entities[:20]:
                    conv_names.extend(v[0] for v in _dual_variants(e["text"], enable=True) if v[1] == "conv")
                # 實體無任何變體時沿用原文名單（文件正文已是轉換文）
                if not conv_names:
                    conv_names = [e["text"] for e in en.entities[:20]]
                ids.append(f"{en.chunk_id}-trad")
                docs.append(variant)
                metas.append({**base_meta, "script": "conv", "entities": "|".join(conv_names)})
                break  # _dual_variants 最多一個 conv，取首個即離開
        if not ids:
            return
        try:
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception as e:  # noqa: BLE001 — Chroma 寫入失敗不應拖垮整批 SQLite 成果
            logger.warning("ChromaDB 寫入失敗（%s），本批僅保留 SQLite 結果", e)

    # ══════════════════════════════════════════
    # 5.4 精準檢索（無向量，供 agent_router 使用）
    # ══════════════════════════════════════════

    def search_by_entity(
        self, entity_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """依【實體名】精準檢索：entities 表 → JOIN chunks 取原文。

        雙寫加持：繁簡任一字形皆有獨立列，完全比對天然跨字形命中
        （如查「數學」命中轉換列，查「数学」命中原文列）。

        Args:
            entity_text: 實體名（完全比對，大小寫敏感）。
            limit: 最多回傳幾筆。

        Returns:
            [{chunk_id, title, text, entity_type, freq, script}...]，按 freq 排序。
        """
        cur = self._conn.execute(
            """SELECT c.chunk_id, c.title, c.text, e.entity_type, e.freq, e.script
               FROM entities e JOIN chunks c ON e.chunk_id = c.chunk_id
               WHERE e.entity_text = ? ORDER BY e.freq DESC LIMIT ?""",
            (entity_text, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    @staticmethod
    def _escape_like(keyword: str) -> str:
        """跳脫 LIKE 通配字（% _ \\)，避免關鍵字含特殊字時語義走樣或報錯。"""
        return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def search_by_keyword(
        self, keyword: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """依【關鍵字】檢索內文：優先 Chroma where_document $contains，
        Chroma 不可用時退回 SQLite LIKE，行為一致。

        雙寫加持：
        - Chroma 路徑：原文/轉換雙文件並存，$contains 任一字形皆命中，
          回傳前按 base_id 去重（原文列優先）。
        - LIKE 路徑：關鍵字展開為跨字形變體 OR 查詢，保底方案同樣跨字形。

        Args:
            keyword: 關鍵字子字串。
            limit: 最多回傳幾筆。
        """
        # 路徑一：Chroma 文件索引（$contains 子字串比對，無向量）
        if self._ensure_chroma():
            try:
                res = self._collection.get(
                    where_document={"$contains": keyword}, limit=limit
                )
                merged: Dict[str, Dict[str, Any]] = {}
                for i, cid in enumerate(res.get("ids", [])):
                    meta = (res.get("metadatas") or [{}])[i] or {}
                    docs = res.get("documents") or [""]
                    # 【去重】轉換列 id 形如 chunk_id-trad，以 base_id 歸戶；
                    # 同 chunk 優先保留原文列（script='orig'）
                    base = str(meta.get("base_id", "") or cid)
                    row = {
                        "chunk_id": base,
                        "title": meta.get("title", ""),
                        "text": docs[i] if i < len(docs) else "",
                    }
                    if base not in merged or meta.get("script") == "orig":
                        merged[base] = row
                if merged:
                    return list(merged.values())[:limit]
            except Exception as e:  # noqa: BLE001 — 退回 SQLite 即可，不中斷查詢
                logger.warning("Chroma 關鍵字檢索失敗（%s），改用 SQLite LIKE", e)
        # 路徑二：SQLite LIKE（永遠可用的保底方案，變體 OR 保證跨字形）
        variants = _query_variants(keyword)
        conds = " OR ".join(["text LIKE ? ESCAPE '\\'"] * len(variants))
        params: List[Any] = [f"%{self._escape_like(v)}%" for v in variants] + [limit]
        cur = self._conn.execute(
            f"SELECT chunk_id, title, text FROM chunks WHERE {conds} LIMIT ?",
            params,
        )
        return [dict(row) for row in cur.fetchall()]

    def search_by_title(self, title: str, limit: int = 20) -> List[Dict[str, Any]]:
        """依【條目標題】取回該條目所有 Chunk（ERP 導向最常用的精準查詢）。

        雙寫加持：標題同時比對跨字形變體（如查「數學」命中「数学」條目）。
        """
        variants = _query_variants(title)
        conds = " OR ".join(["title = ?"] * len(variants))
        cur = self._conn.execute(
            f"""SELECT chunk_id, title, chunk_index, text, char_len FROM chunks
               WHERE {conds} ORDER BY chunk_index LIMIT ?""",
            [*variants, limit],
        )
        return [dict(row) for row in cur.fetchall()]

    def get_relations(
        self, entity_text: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """取回與某實體相關的三元組（主語或賓語命中即回傳，按 weight 排序）。

        雙寫加持：三元組端點同樣雙字形儲存，任一字形皆可命中。
        """
        cur = self._conn.execute(
            """SELECT chunk_id, subj, predicate, obj, weight, evidence, script FROM relations
               WHERE subj = ? OR obj = ? ORDER BY weight DESC LIMIT ?""",
            (entity_text, entity_text, limit),
        )
        return [dict(row) for row in cur.fetchall()]

    def stats(self) -> Dict[str, int]:
        """回傳三表筆數與 Chroma 筆數（Chroma 不可用時該欄為 -1）。

        雙寫啟用時 entities / relations / chroma_docs 約為單寫的 2 倍，
        屬正常現象（同一知識的兩種字形各一列）。
        """
        out = {}
        for table in ("chunks", "entities", "relations"):
            cur = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(cur.fetchone()["n"])
        if self._ensure_chroma():
            try:
                out["chroma_docs"] = int(self._collection.count())
            except Exception:  # noqa: BLE001 — 統計失敗不影響主流程
                out["chroma_docs"] = -1
        else:
            out["chroma_docs"] = -1
        return out

    # ══════════════════════════════════════════
    # 5.5 端到端管線：loader → 本模組（一鍵執行）
    # ══════════════════════════════════════════

    def run_pipeline(
        self,
        file_path: PathLike,
        batch_size: int = 32,
        max_batches: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, int]:
        """一鍵執行 loader → enrich_batch 全流程（串流低記憶體）。

        Args:
            file_path: 維基 JSONL 檔案路徑。
            batch_size: 每批 Chunk 數（同時是 SQLite 單次交易筆數）。
            max_batches: 最多處理幾批（None 表全量；測試時設小數字）。
            show_progress: 是否顯示 tqdm 進度條。

        Returns:
            處理完畢後的 stats() 三表筆數統計。
        """
        # 【延遲匯入】避免循環依賴（loader 不依賴本模組，單向依賴保持乾淨）
        from .loader import batched_chunks

        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(desc="知識圖譜建構", unit="批")
            except ImportError:
                logger.warning("tqdm 未安裝，show_progress 自動停用")

        try:
            for i, batch in enumerate(
                batched_chunks(file_path, batch_size=batch_size), start=1
            ):
                self.enrich_batch(batch)
                if pbar is not None:
                    pbar.update(1)
                if max_batches is not None and i >= max_batches:
                    break
        finally:
            if pbar is not None:
                pbar.close()
        return self.stats()


# ══════════════════════════════════════════════
# 六、命令列快速驗證（python -m src.spark_processor）
# ══════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    import argparse

    # 【建庫 CLI】--batches 0 表全量建庫（預設）；傳正整數則只建前 N 批（快速驗證用）
    parser = argparse.ArgumentParser(description="Spark 知識圖譜建庫 / 驗證工具")
    parser.add_argument("--file", default="data/wikipedia-zh-cn-20240901.json", help="資料檔路徑")
    parser.add_argument("--batches", type=int, default=0, help="最多處理幾批（0=全量建庫）")
    parser.add_argument("--batch-size", type=int, default=32, help="每批 Chunk 數")
    parser.add_argument("--db", default="data/knowledge.db", help="SQLite 輸出路徑")
    parser.add_argument("--chroma-dir", default="data/chroma", help="Chroma 持久化目錄")
    parser.add_argument("--mode", default="rule", choices=["rule", "api"], help="抽取模式")
    parser.add_argument("--progress", action="store_true", help="顯示進度條")
    parser.add_argument("--no-dual", action="store_true", help="關閉簡繁雙寫（回到單字形寫入）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = SparkConfig(
        mode=args.mode, db_path=args.db, chroma_dir=args.chroma_dir,
        dual_script=not args.no_dual,
    )
    with SparkProcessor(cfg) as sp:
        result = sp.run_pipeline(
            # batches<=0 轉為 None（全量）；2GB 全量需較長時間，建議加 --progress 觀察
            args.file, batch_size=args.batch_size,
            max_batches=args.batches if args.batches > 0 else None,
            show_progress=args.progress,
        )
        print(f"\n入庫統計：{result}")
