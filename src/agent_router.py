"""智能體調度模組（agent_router.py）

實作「Agentic RAG（多步驟規劃 + 混合檢索 + 自我反思與修正）」，
檢索層為 RRF 混合檢索：稀疏（SQLite 關鍵字 / 實體 / 關係）＋
密集（ChromaDB bge-m3 向量語意）雙軌並行，再做排名融合。

三個核心機制：
    1. Planner（任務拆解）：Ollama 本地 LLM 輸出繁中 JSON 步驟，
       失敗降級規則拆解。
    2. Router（自適應路由＋混合執行）：每個子問題先判定稀疏路由
       （TITLE / ENTITY / KEYWORD / RELATION / HYBRID），再與密集向量
       檢索並行，最後 RRF 融合取前 5 筆。
    3. Critic（反思修正迴圈）：跨字形覆蓋率評分；不足時先關係擴張，
       再請 LLM 換句話說改寫關鍵字後重試，最多 ``max_attempts`` 次。

運作模式：
    - ``mode="rule"``（預設）：全規則，零連線、零費用，開箱即用。
    - ``mode="llm"``：Planner / 改寫 / 合成改調本機 Ollama
     （qwen3:8b 優先，llama3.1:latest 備援），任一環節失敗即降級規則；
      密集檢索不可用時退回純稀疏五路由（原無向量機制完整保留）。
    - ``mode="api"``：舊版別名，自動視為 ``llm``（相容 main.py / CLI）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────
# 日誌設定
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 一、路由目標與資料結構
# ══════════════════════════════════════════════


class RouteTarget(str, Enum):
    """子問題的檢索路由目標（稀疏腿的 API 選擇；密集腿一律並行）。

    - TITLE_LOOKUP: 書名號 / 條目名 → ``search_by_title``
    - ENTITY_LOOKUP: 人名 / 地名 / 術語 → ``search_by_entity``
    - KEYWORD_SEARCH: 一般關鍵字 → ``search_by_keyword``
    - RELATION_QUERY: 查兩實體關係 → ``get_relations``（+ 實體查補）
    - HYBRID_SEARCH: 綜合型查詢 → 關鍵字稀疏腿 ＋ 密集腿（比較 / 綜述專用）
    - DIRECT_ANSWER: 寒暄 / 常識問，無需檢索（唯一不走混合檢索者）
    """

    TITLE_LOOKUP = "title_lookup"
    ENTITY_LOOKUP = "entity_lookup"
    KEYWORD_SEARCH = "keyword_search"
    RELATION_QUERY = "relation_query"
    HYBRID_SEARCH = "hybrid_search"
    DIRECT_ANSWER = "direct_answer"


@dataclass
class SubQuestion:
    """拆解後的子問題（子查詢步驟）。

    Attributes:
        text: 子問題文字（人類可讀，同時作為檢索 query 來源）。
        route: 建議路由（Router 可再覆寫）。
        depends_on: 依賴的其他子問題索引（先查被依賴者，結果可供後續用）。
        keywords: 檢索關鍵字（Planner 預填；重試時會被改寫）。
        kind: 步驟類型：lookup（查資料）/ relation（查關係）/
            compare（比較）/ summarize（綜合）。
    """

    text: str = ""
    route: RouteTarget = RouteTarget.KEYWORD_SEARCH
    depends_on: List[int] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    kind: str = "lookup"


@dataclass
class ReflectResult:
    """自我反思結果。

    Attributes:
        passed: 是否通過自檢（可進入答案合成）。
        score: 自評分數（0~1，規則模式 = 關鍵字覆蓋率）。
        comment: 反思意見（缺了什麼、建議怎麼修正）。
        need_retry: 是否需要重查（passed 為 False 且仍有嘗試次數時為 True）。
        coverage: 問題關鍵字被檢索內容覆蓋的比例。
        missing: 未被覆蓋的關鍵字（重試改寫的依據）。
    """

    passed: bool = False
    score: float = 0.0
    comment: str = ""
    need_retry: bool = False
    coverage: float = 0.0
    missing: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """AgentRouter 設定（Ollama 本地推理＋RRF 混合檢索）。

    Attributes:
        mode: "rule"（預設，全規則）或 "llm"（Ollama 推理，失敗降級）；
            舊值 "api" 自動視為 "llm"（相容舊 CLI / main.py）。
        api_base / api_key / model / timeout / max_retries: 舊版 Spark 參數
            （保留相容，現行 LLM 改走 ollama_base_url，不再使用）。
        llm_model: 推理主模型（預設 "qwen3:8b"）。
        llm_fallback_model: 推理備援模型（預設 "llama3.1:latest"，主模型
            失敗時試一次，再失敗才降級規則）。
        ollama_base_url: Ollama 位址（預設 "http://127.0.0.1:11434"）。
        llm_timeout: 單次 LLM 呼叫逾時（秒，預設 120）。
        max_attempts: 單個子問題最多檢索幾次（含首次，預設 3 次）。
        top_k: 稀疏腿每關鍵字取回筆數。
        dense_top_k: 密集腿取回筆數（預設 10，餵 RRF 用，多取無妨）。
        hybrid_top_n: RRF 融合後保留筆數（規格：前 5 筆）。
        rrf_k: RRF 平滑常數（規格公式分母 60）。
        max_workers: 檢索執行緒池大小（稀疏＋密集各一，預設 2）。
        min_coverage: 反思通過門檻（關鍵字覆蓋率，預設 0.4）。
    """

    mode: str = "rule"
    api_base: str = ""
    api_key: str = ""
    model: str = "muse-spark-1.3"
    timeout: int = 60
    max_retries: int = 2
    llm_model: str = "qwen3:8b"
    llm_fallback_model: str = "llama3.1:latest"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_timeout: int = 120
    max_attempts: int = 3
    top_k: int = 5
    dense_top_k: int = 10
    hybrid_top_n: int = 5
    rrf_k: int = 60
    max_workers: int = 2
    min_coverage: float = 0.4

    def __post_init__(self) -> None:
        """正規化模式並補齊舊環境變數（顯式傳參優先）。"""
        # 【相容】舊版 "api"（Spark）已退役，一律視為本地 LLM 模式
        if self.mode == "api":
            self.mode = "llm"
        if not self.api_base:
            self.api_base = os.getenv("SPARK_API_BASE", "")
        if not self.api_key:
            self.api_key = os.getenv("SPARK_API_KEY", "")
        env_model = os.getenv("SPARK_MODEL", "")
        if env_model and self.model == "muse-spark-1.3":
            self.model = env_model


# ══════════════════════════════════════════════
# 二、問題前處理：關鍵字 / 書名號擷取（規則，零依賴）
# ══════════════════════════════════════════════

# 【正則預編譯】模組層級只編一次
_RE_BOOK = re.compile(r"《([^《》]{1,20})》")  # 書名號 → 條目候選
_RE_QUOTED = re.compile(r"[「“\"]([^「」“”\"]{2,10})[」”\"]")  # 引號 → 關鍵字候選
_RE_CJK = re.compile(r"[\u4e00-\u9fff]{2,6}")  # 連續中文串 → 關鍵字候選

# 【問句虛詞】改寫關鍵字時要剝掉的疑問 / 語氣 / 動作成分
# 注意：含「查詢/介紹」等動作詞，讓子問題文字（「查詢X」）也能還原出乾淨關鍵字；
# 簡繁並列（什麼/什麼、怎么/怎麼…），因維基庫為簡體、用戶可能用繁體提問
_QUESTION_PARTICLES = (
    "是什麼", "是什么", "是甚麼", "什麼", "什么", "甚麼",
    "如何", "怎麼", "怎么", "怎樣", "怎样", "為什麼", "为什么",
    "為何", "为何", "是誰", "是谁", "何人", "何處", "何处", "何時", "何时",
    "多少", "幾個", "几个", "哪裡", "哪里", "哪個", "哪个",
    "是否", "有哪些", "有什麼", "有什么",
    "請問", "请问", "請告訴我", "请告诉我", "告訴我", "告诉我",
    "查詢", "查询", "搜尋", "搜索", "尋找", "寻找", "檢索", "检索",
    "介紹", "介绍", "簡介", "简介",
    "探討", "探讨", "探索", "闡述", "阐述", "說明", "说明", "解釋", "解释",
    "了解", "談談", "谈谈", "講講", "讲讲", "描述",
    "嗎", "吗", "呢", "吧",
    "？", "?", "！", "!", "。", "，", ",", "、",
)

# 【動作前綴】檢索前必須從關鍵字「開頭」剝除的查詢動詞（前綴比對，簡繁並列）。
# 注意：只剝開頭、不做全域取代，避免誤傷「研究生」這類以動詞起首的實體；
# 「研究/分析/討論」因高度歧義（常是實體本身）刻意不列入，前綴污染時由
# _sanitize_keywords 搭配 _extract_keywords 的全域虛詞表兜底。
# 排序：長詞在前，確保「查詢一下」優先於「查詢」被剝除。
_ACTION_PREFIXES = (
    "請告訴我", "请告诉我", "告訴我", "告诉我", "請問", "请问",
    "幫我", "帮我", "查詢一下", "查询一下",
    "查詢", "查询", "檢索", "检索", "探討", "探讨", "探索",
    "闡述", "阐述", "說明", "说明", "解釋", "解释", "了解",
    "搜尋", "搜索", "尋找", "寻找", "介紹", "介绍", "簡介", "简介",
    "談談", "谈谈", "講講", "讲讲", "描述", "請", "请",
)

# 【關鍵字尾字】收尾出現這些虛字一律剝除（如「孔子是」→「孔子」；簡繁並列）
_KEYWORD_TAIL_STRIP = "的是了在有著過地得嗎呢吧么麼个個于於为為时時对對还還"

# 【關係提問觸發詞】出現即視為「問 A 與 B 的關係」（簡繁並列）
_RELATION_TRIGGERS = ("關係", "关系", "關聯", "关联", "關連", "关连",
                      "相關", "相关", "淵源", "渊源", "糾葛", "纠葛")

# 【比較提問觸發詞】出現即視為「比較 A 與 B」（簡繁並列）
_COMPARE_TRIGGERS = ("差別", "差别", "差異", "差异", "區別", "区别",
                     "不同", "比較", "比较", "對比", "对比", "對照", "对照")

# 【Cardi 監督觸發詞】關係＋比較＋額外對比詞（簡繁並列）。
# Planner 輸出邊界的審查清單：命中任一、但步驟缺 RELATION_QUERY 即強制補步。
_CARDI_TRIGGERS = _RELATION_TRIGGERS + _COMPARE_TRIGGERS + (
    "相比", "異同", "异同",
)

# 【雲端環境變數黑名單】_call_ollama 執行前強制 pop 隔離（呼叫結束歸還）。
# 前六個是各家 LLM 雲端 Key／Endpoint（本專案 requests 只用顯式參數，
# 從不需要它們，殘留只會造成雲端誤判）；代理變數 pop 是為了保證直連本機。
_CLOUD_ENV_VARS = (
    "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
    "OPENAI_ORG_ID", "OPENAI_PROJECT",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
)
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)

# 【本機迴圈白名單】Ollama Base URL 的 hostname 只允許這三者，其餘一律拒絕。
# 這是 fail-closed 設計：任何指向外部的主機名在發包前就被擋下。
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")

# 【實體分隔符】「A與B」「A和B」「A、B」中的分隔字（簡繁並列）
_ENTITY_SEPS = ("與", "与", "和", "跟", "及", "、", "以及")

# 【寒暄】命中則直接回答，無需檢索（簡繁並列）
_SMALLTALK = ("你好", "您好", "哈囉", "哈喽", "嗨", "謝謝", "谢谢", "感謝", "感谢",
              "再見", "再见", "拜拜", "早安", "午安", "晚安")

# 【停用字】關鍵字首字過濾用（避免抽出「的是」類碎片；簡繁並列）
_STOP_FIRST = set("的了是在有和與与或及等之其這这那你我他她它為为被把將将對对就都也而於于以從从到比過过個个些種种們们會会可所能應应但若如因果當当每各另別别先後后內内")  # noqa: E501

# 【合法步驟類型】LLM 拆解回傳的 kind 白名單，命中以外一律視為 lookup
_VALID_KINDS = ("lookup", "relation", "compare", "summarize")


def _extract_titles(question: str) -> List[str]:
    """擷取問句中的書名號《》內容（→ TITLE_LOOKUP 的關鍵字）。"""
    return [t.strip() for t in _RE_BOOK.findall(question or "") if t.strip()]


def _extract_keywords(question: str, max_n: int = 6) -> List[str]:
    """從問句擷取檢索關鍵字：書名號 → 引號 → 中文實詞（去虛詞、去重、保序）。

    Args:
        question: 用戶原始提問。
        max_n: 最多回傳幾個。

    Returns:
        關鍵字串列（長度 2~6 字，順序按在問句中出現先後）。
    """
    if not question:
        return []
    cands: List[str] = []
    # 1. 書名號最優先（精準條目名）
    cands.extend(_extract_titles(question))
    # 2. 引號內容次之（使用者強調的概念）
    cands.extend(t.strip() for t in _RE_QUOTED.findall(question) if t.strip())
    # 3. 連續中文串：先整句去虛詞，再切出實詞
    cleaned = question
    for p in _QUESTION_PARTICLES:
        cleaned = cleaned.replace(p, " ")
    for m in _RE_CJK.findall(cleaned):
        word = m.strip().rstrip(_KEYWORD_TAIL_STRIP)  # 剝尾字虛詞（「孔子是」→「孔子」）
        if len(word) < 2 or word[0] in _STOP_FIRST:
            continue
        cands.append(word)
    # 去重並保序
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:max_n]


def _strip_action_prefix(text: str) -> str:
    """剝除字串開頭的查詢動作前綴（可連續多層，如「請幫我查詢」）。

    只比對開頭、不做全域取代，避免誤傷「研究生」這類實體；
    並順手清掉前綴後殘留的標點（「查詢：數學」→「數學」）。

    Args:
        text: 待清理的關鍵字或子問題文字。

    Returns:
        去前綴後的字串（若原本全是虛詞則回傳空字串）。
    """
    s = (text or "").strip()
    while s:
        stripped = False
        for prefix in _ACTION_PREFIXES:
            if s.startswith(prefix) and len(s) > len(prefix):
                s = s[len(prefix):].lstrip(" ：:，,、。.？?！!")
                stripped = True
                break
        if not stripped:
            break
    return s.strip()


def _sanitize_keywords(keywords: List[str], max_n: int = 6) -> List[str]:
    """關鍵字最終消毒（檢索邊界的最後一道防線）。

    每個關鍵字先去動作前綴，再走一次 _extract_keywords 做全套清洗；
    保證進入 SQLite LIKE / ChromaDB 的只會是「數學」這類純實詞，
    絕不會是「查詢數學」「檢索數學」這類污染詞。

    Args:
        keywords: 待消毒的關鍵字串列（可能含子問題包裝文字）。
        max_n: 最多回傳幾個。

    Returns:
        乾淨關鍵字串列（去重保序；全滅時回傳空串列）。
    """
    out: List[str] = []
    seen = set()
    for kw in keywords or []:
        cleaned = _strip_action_prefix(kw)
        if not cleaned:
            continue
        # 重走一次正規擷取（含書名號/引號/虛詞/尾字全套規則）
        for word in _extract_keywords(cleaned, max_n=2):
            if word not in seen:
                seen.add(word)
                out.append(word)
                if len(out) >= max_n:
                    return out
    return out


def _keyword_variants(keyword: str) -> List[str]:
    """取關鍵字的繁簡雙字形變體（覆蓋率比對專用）。

    沿用 ``spark_processor`` 的 ``_query_variants``（OpenCC s2t＋t2s），
    回傳去重後的 [原文, 簡體形, 繁體形]；轉換器不可用時只回傳原文，
    保證降級行為與舊版完全一致。

    Args:
        keyword: 單個關鍵字。

    Returns:
        變體串列（至少含原文；空輸入回傳空串列）。
    """
    if not keyword:
        return []
    try:
        from .spark_processor import _query_variants  # 延遲匯入，避免循環依賴
    except Exception:  # noqa: BLE001 — 匯入失敗就退回單字形比對
        return [keyword]
    try:
        return _query_variants(keyword) or [keyword]
    except Exception:  # noqa: BLE001 — 轉換異常同樣退回單字形
        return [keyword]


def _strip_trailing_operators(word: str) -> str:
    """剝除關鍵字尾端的提問運算詞（覆蓋率計分專用）。

    「數學的關係」這類關鍵字尾端的「關係」是提問殘留、不是內容，
    覆蓋率若要求內文出現整串將永遠 miss；剝成核心「數學」再比對才合理。
    只剝尾端、不動中間：內容詞如「關係型資料庫」（運算詞在首）會原樣保留，
    而「血緣關係」剝成「血緣」仍是原文子字串，不影響命中。

    Args:
        word: 單個關鍵字。

    Returns:
        去尾端運算詞後的核心詞（最短保留 2 字，避免剝光）。
    """
    s = word or ""
    while len(s) > 2:
        dropped = False
        for trigger in _RELATION_TRIGGERS + _COMPARE_TRIGGERS:
            if s.endswith(trigger) and len(s) > len(trigger):
                s = s[: -len(trigger)].rstrip(_KEYWORD_TAIL_STRIP)
                dropped = True
                break
        if not dropped:
            break
    return s


def _clean_segment_words(segment: str) -> List[str]:
    """清理半句並取出候選實體詞：去虛詞 → 中文串 → 去首停用字/尾虛字。

    供 _split_two_entities 使用，避免「物理的关系是」這類污染詞。
    """
    cleaned = segment or ""
    for p in _QUESTION_PARTICLES:
        cleaned = cleaned.replace(p, " ")
    # 關係 / 比較觸發詞本身不是實體（如「物理的关系」中的「关系」），一併剝除
    for t in _RELATION_TRIGGERS + _COMPARE_TRIGGERS:
        cleaned = cleaned.replace(t, " ")
    out: List[str] = []
    for m in _RE_CJK.findall(cleaned):
        word = m.strip().rstrip(_KEYWORD_TAIL_STRIP)
        if len(word) < 2 or word[0] in _STOP_FIRST:
            continue
        out.append(word)
    return out


def _split_two_entities(question: str) -> List[str]:
    """拆出「A與B」式問句中的兩個實體名（關係 / 比較題專用）。

    策略：找分隔字（與/和/跟/及/、）左右各取最近的中文詞；
    找不到分隔字則退回關鍵字前兩個。
    """
    for sep in _ENTITY_SEPS:
        if sep in question:
            left, _, right = question.partition(sep)
            l_words = _clean_segment_words(left)
            r_words = _clean_segment_words(right)
            if l_words and r_words:
                # 左取最後一個（靠分隔字最近），右取第一個
                return [l_words[-1], r_words[0]]
    kws = _extract_keywords(question, max_n=2)
    return kws if len(kws) == 2 else []


def _rewrite_keywords(keywords: List[str], attempt: int) -> List[str]:
    """反思重試用的規則關鍵字改寫（attempt 從 1 起算，越大改寫越激進）。

    - 第 1 次重試：去尾字（「孔子的」→「孔子」），拆「的」前段。
    - 第 2 次重試：只留前 2 字（放寬為前綴比對，recall 優先）。
    - 之後：逐字拆成單字查詢（最後手段）。
    LLM 可用時由 _rewrite_for_retry 優先請模型換句話說，本函數為其保底。
    """
    out: List[str] = []
    for kw in keywords:
        kw = kw.strip().rstrip("的之")
        if "的" in kw:
            kw = kw.split("的")[0]  # 取「的」前段（通常是核心實體）
        if attempt >= 2:
            kw = kw[:2]  # 放寬為前綴
        if len(kw) >= 2:
            out.append(kw)
        elif kw:
            out.append(kw)  # 最後手段：單字也查
    # 去重保序
    seen, dedup = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return dedup


def _rrf_fuse(
    sparse: List[Dict[str, Any]],
    dense: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """RRF 相互排名融合：Score = 1/(k+r_sparse) + 1/(k+r_dense)。

    雙軌名次（1-based，落榜視為無窮大即 0 分）加總後排序，chunk_id 去重，
    取前 top_n 筆。每筆保留雙軌名次與來源（source 標 rrf:分數），
    供前端軌跡面板展示融合依據。

    Args:
        sparse: 稀疏腿上下文串列（已按該腿排名排序）。
        dense: 密集腿上下文串列（同上；為空即退化為純稀疏排序）。
        k: 平滑常數（規格 60，壓低首名壟斷，放大中段差異）。
        top_n: 最多回傳幾筆（規格：前 5 筆）。

    Returns:
        融合後的上下文串列（每筆多帶 rrf / sparse_rank / dense_rank）。
    """
    scores: Dict[str, float] = {}  # chunk_id → 融合分數
    ranks: Dict[str, Dict[str, Optional[int]]] = {}  # chunk_id → 雙軌名次
    payload: Dict[str, Dict[str, Any]] = {}  # chunk_id → 代表內容（稀疏優先）
    # 【腿內去重】同腿重複列只取最佳名次，避免重複計分扭曲融合
    seen_sparse, seen_dense = set(), set()
    for rank, ctx in enumerate(sparse, start=1):
        cid = str(ctx.get("chunk_id", ""))
        if not cid or cid in seen_sparse:
            continue
        seen_sparse.add(cid)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        ranks.setdefault(cid, {"sparse": None, "dense": None})["sparse"] = rank
        payload.setdefault(cid, ctx)  # 稀疏內容優先（通常帶完整 title）
    for rank, ctx in enumerate(dense, start=1):
        cid = str(ctx.get("chunk_id", ""))
        if not cid or cid in seen_dense:
            continue
        seen_dense.add(cid)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        ranks.setdefault(cid, {"sparse": None, "dense": None})["dense"] = rank
        payload.setdefault(cid, ctx)
    fused = []
    for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        row = dict(payload[cid])
        row["rrf"] = round(score, 4)
        row["sparse_rank"] = ranks[cid]["sparse"]
        row["dense_rank"] = ranks[cid]["dense"]
        row["source"] = f"rrf:{score:.3f}|{row.get('source', '')}"
        fused.append(row)
    return fused


def _scrub_llm_env() -> Dict[str, str]:
    """隔離雲端環境變數：pop 黑名單鍵，回傳備份（呼叫端負責歸還）。

    目的：徹底斷開與外部雲端 Provider 的關聯，保證本次 Ollama 呼叫
    100% 走本機迴圈（Console 誤判為雲端免費算力的根治手段之一）。
    注意：pop 是進程全域變更，故呼叫端必須在 finally 歸還；
    且本專案 LLM 呼叫為循序執行（Planner→改寫→合成不同時併發），無競態之虞。

    Returns:
        被移除的 {變數名: 原值} 備份字典（為空表環境本來就乾淨）。
    """
    removed: Dict[str, str] = {}
    for var in _CLOUD_ENV_VARS + _PROXY_ENV_VARS:
        if var in os.environ:
            removed[var] = os.environ.pop(var)
    if removed:
        logger.info("LLM 連線防污染：已隔離環境變數 %s（呼叫結束歸還）",
                    sorted(removed))
    return removed


# ══════════════════════════════════════════════
# 三、AgentRouter 主類別
# ══════════════════════════════════════════════


class AgentRouter:
    """OpenCode 智能體路由器：Planner → 混合檢索 → Critic 重試 → 合成答案。

    使用範例：
        from src.spark_processor import SparkConfig, SparkProcessor
        with SparkProcessor(SparkConfig()) as sp:  # 需先 run_pipeline 建庫
            router = AgentRouter(spark_processor=sp)  # rule 模式開箱即用
            print(router.answer("孔子與孟子的關係是什麼？")["answer"])
            llm_router = AgentRouter(sp, config=AgentConfig(mode="llm"))
            print(llm_router.answer("孔子與孟子的關係是什麼？")["answer"])
    """

    def __init__(self, spark_processor: Any | None = None, config: Optional[AgentConfig] = None) -> None:
        """初始化路由器。

        Args:
            spark_processor: 已建庫的 SparkProcessor 實例；為 None 時，
                首次檢索才延遲建立預設實例（方便純規則測試）。
            config: 路由器設定（重試次數、覆蓋率門檻、Ollama 參數）。
        """
        self.spark_processor = spark_processor
        self.config = config or AgentConfig()
        # 【LLM 就緒探測】僅 llm 模式做一次輕量 ping（timeout 短，不拖慢 rule 測試）；
        # 失敗就整程走規則，呼叫期不再重試連線（重試交給各環節的 try/except）。
        self._llm_ok = self._ping_ollama() if self._use_llm_mode() else False

    # ── 延遲取得 processor（避免 __init__ 產生建庫副作用）──
    def _processor(self) -> Any:
        """取回 SparkProcessor；未提供則延遲建立預設實例。"""
        if self.spark_processor is None:
            from .spark_processor import SparkConfig, SparkProcessor

            logger.info("未提供 SparkProcessor，延遲建立預設實例（rule 模式）")
            self.spark_processor = SparkProcessor(SparkConfig())
        return self.spark_processor

    def _use_llm_mode(self) -> bool:
        """是否處於 LLM 模式（mode 經正規化，僅 'llm' 為真）。"""
        return self.config.mode == "llm"

    def _use_llm(self) -> bool:
        """本輪呼叫是否可用 LLM（模式＋啟動期探測皆通過）。"""
        return self._use_llm_mode() and self._llm_ok

    # ══════════════════════════════════════════
    # 3.1 本地 Ollama 推理（ planner / 改寫 / 合成共用，失敗降級 rule）
    # ══════════════════════════════════════════

    def _ping_ollama(self, timeout: int = 5) -> bool:
        """啟動期輕量探測：Ollama 存活＋推理模型至少其一已下載。

        不在此下載模型（缺模型只 WARNING，呼叫期再逐個試錯）；
        rule 模式不呼叫本函數，測試零負擔。
        """
        base = (self.config.ollama_base_url or "").rstrip("/")
        try:
            import requests  # 延遲匯入，rule 模式不需要它
        except ImportError:
            logger.warning("未安裝 requests，LLM 停用，整程走規則")
            return False
        try:
            resp = requests.get(f"{base}/api/tags", timeout=timeout)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            wanted = [self.config.llm_model, self.config.llm_fallback_model]
            if not any(w in names for w in wanted if w):
                logger.warning("Ollama 存活但缺推理模型（要 %s，現有 %s），LLM 停用",
                               wanted, names)
                return False
            return True
        except Exception as e:  # noqa: BLE001 — 連不上就整程規則，不擋啟動
            logger.warning("Ollama 不可用（%s），LLM 停用，整程走規則", e)
            return False

    def _call_ollama(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
        """調用本地 Ollama（主模型優先，備援模型墊後），回傳模型原文。

        本機迴圈三重防護（Console 雲端誤判的根治手段）：
            1. 環境隔離：先 ``_scrub_llm_env()`` pop 雲端 Key／代理變數，
               呼叫結束於 finally 歸還（可審計：INFO 日誌列出隔離名單）。
            2. 位址鎖定：hostname 非 127.0.0.1／localhost／::1 即拒絕發包
               （fail-closed，外部主機名連線從未建立）。
            3. 會話隔離：``trust_env=False`` 的獨立 Session，連 requests
               預設的環境代理讀取一併關閉；代理變數不靠 pop 永久刪除，
               避免波及其他需代理的網路路徑。

        Args:
            system_prompt: 系統指令（含輸出格式約束）。
            user_prompt: 用戶提示（含問題與上下文）。
            json_mode: True 走 /api/chat format=json（規劃/改寫用）；
                False 回傳自由文字（答案合成用）。

        Raises:
            RuntimeError: 位址非本機／兩模型皆失敗（上層捕捉後降級規則）。
        """
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("未安裝 requests") from e
        from urllib.parse import urlparse  # 標準庫，解析 hostname 做白名單比對

        # 【防護1＋2】位址鎖定先行：非本機迴圈連線從未建立，直接拒絕
        base = (self.config.ollama_base_url or "").rstrip("/")
        host = (urlparse(base).hostname or "").lower()
        if host not in _LOCAL_HOSTS:
            raise RuntimeError(
                f"Ollama 位址被拒：僅允許本機迴圈 {list(_LOCAL_HOSTS)}，收到 {base!r}")
        models = [m for m in (self.config.llm_model, self.config.llm_fallback_model) if m]
        # 【防護3】環境隔離：pop 黑名單（finally 歸還，零 lasting 副作用）
        scrubbed = _scrub_llm_env()
        try:
            # 獨立 Session＋trust_env=False：連環境代理讀取一併關閉，請求極其純粹
            session = requests.Session()
            session.trust_env = False
            last_err: Optional[Exception] = None
            for name in models:
                try:
                    payload: Dict[str, Any] = {
                        "model": name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1},  # 推理鏈用低溫求穩
                    }
                    if json_mode:
                        payload["format"] = "json"  # 強制 JSON，省去清洗雜訊
                    resp = session.post(f"{base}/api/chat", json=payload,
                                        timeout=self.config.llm_timeout)
                    resp.raise_for_status()
                    return str(resp.json()["message"]["content"])
                except Exception as e:  # noqa: BLE001 — 換下一個模型再試
                    last_err = e
                    logger.warning("Ollama 模型 %s 呼叫失敗（%s），嘗試下一個", name, e)
            raise RuntimeError(f"Ollama 推理失敗（已試 {models}）：{last_err}")
        finally:
            os.environ.update(scrubbed)  # 【歸還】環境恢復原狀，不污染其餘網路路徑

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """從模型回文擷取 JSON 陣列（容忍物件包裝與說明文字），失敗回傳空串列。

        接受三種形態：頂層陣列 ``[...]``、物件包裝 ``{"steps": [...]}``、
        單步裸物件 ``{"text": ...}``（qwen 實測三種都出過，最後一種自動包成單步）。
        """
        candidates: List[str] = []
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            candidates.append(raw[start:end])
        except ValueError:
            pass
        candidates.append(raw)  # 退路：整段原文可能就是單一步驟物件
        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                # 物件包裝：取其內第一個「元素為字典」的陣列值
                for value in parsed.values():
                    if isinstance(value, list) and any(isinstance(x, dict) for x in value):
                        return [d for d in value if isinstance(d, dict)]
                # 單步裸物件：含步驟鍵即包成單步（鍵名含 text/子問題/kind 任一）
                if any(k in parsed for k in ("text", "子問題", "问题", "kind", "keywords")):
                    return [parsed]
                continue
            if isinstance(parsed, list):
                dicts = [d for d in parsed if isinstance(d, dict)]
                if dicts:
                    return dicts
        logger.warning("LLM 回傳無法解析為 JSON，降級為規則（首 120 字：%s）", raw[:120])
        return []

    @staticmethod
    def _parse_json_strings(raw: str) -> List[str]:
        """從模型回文擷取 JSON 字串陣列（改寫關鍵字用），失敗回傳空串列。

        同樣容忍 ``{"keywords": [...]}`` 式物件包裝。
        """
        candidates: List[str] = []
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            candidates.append(raw[start:end])
        except ValueError:
            pass
        candidates.append(raw)
        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                for value in parsed.values():
                    if isinstance(value, list) and any(isinstance(x, str) for x in value):
                        return [str(s).strip() for s in value if str(s).strip()]
                continue
            if isinstance(parsed, list):
                strings = [str(s).strip() for s in parsed if str(s).strip()]
                if strings:
                    return strings
        logger.warning("LLM 改寫回傳無法解析，降級為規則改寫")
        return []

    # ══════════════════════════════════════════
    # 3.2 Planner：任務拆解（LLM 優先，失敗降級規則）
    # ══════════════════════════════════════════

    def decompose_question(self, question: str) -> List[SubQuestion]:
        """將複雜提問拆解為子查詢步驟（LLM 優先，失敗降級規則）。

        例：「孔子與孟子的關係是什麼？」
        → [查孔子(0), 查孟子(1), 查兩者關係(2, 依賴 0,1)]
        """
        question = (question or "").strip()
        if not question:
            return []
        if self._use_llm():
            try:
                raw = self._call_ollama(
                    "你是繁體中文 RAG 任務規劃器。只輸出頂層 JSON 陣列，"
                    "每個元素為 "
                    '{"text": 子問題, "kind": lookup/relation/compare/summarize, '
                    '"keywords": [檢索關鍵字]}。關係題拆成查A、查B、查兩者關係三步；'
                    "關鍵字用繁體中文實詞。不要輸出說明文字，不要包物件外殼。"
                    '範例：[{"text": "查詢孔子", "kind": "lookup", "keywords": ["孔子"]}]',
                    f"拆解以下問題：{question}",
                )
                subs = self._subs_from_llm(self._parse_json_array(raw))
                if subs:
                    # 【Cardi】LLM 步驟先過監督卡點再放行（缺關係步即強制補步）
                    return self._cardi_guard(question, subs)
                logger.warning("LLM 拆解回傳為空，降級為規則拆解")
            except Exception as e:  # noqa: BLE001 — 降級需捕捉所有 LLM 異常
                logger.warning("LLM 拆解失敗（%s），降級為規則拆解", e)
        return self._rule_decompose(question)

    # 【LLM 鍵名別名】模型常把 text/kind/keywords 寫成中文鍵，照單全收
    _LLM_KEY_ALIASES = (
        (("text", "子問題", "问题", "question", "query"), "text"),
        (("kind", "類型", "类型", "type"), "kind"),
        (("keywords", "關鍵字", "关键词", "key_words"), "keywords"),
    )

    def _subs_from_llm(self, parsed: List[Dict[str, Any]]) -> List[SubQuestion]:
        """把 LLM 的 JSON 步驟轉為 SubQuestion（校驗 kind/關鍵字並補路由依賴）。

        LLM 的 keywords 照收但強制消毒；kind 非法視為 lookup；
        relation 步驟自動依賴之前所有步驟（先查實體再查關係）。
        """
        subs: List[SubQuestion] = []
        for d in parsed:
            # 鍵名正規化：text/子問題、kind/類型、keywords/關鍵字互通
            norm = {}
            for aliases, canon in self._LLM_KEY_ALIASES:
                for a in aliases:
                    if d.get(a) not in (None, ""):
                        norm[canon] = d.get(a)
                        break
            text = str(norm.get("text", "") or "").strip()
            if not text:
                continue
            kind = str(norm.get("kind", "lookup") or "lookup")
            kind = kind if kind in _VALID_KINDS else "lookup"
            raw_kws = norm.get("keywords") or []
            kws = _sanitize_keywords([str(k) for k in raw_kws if str(k).strip()])
            if not kws:  # LLM 沒給可用關鍵字就地提煉，避免空轉
                kws = _extract_keywords(text)
            subs.append(SubQuestion(text=text, keywords=kws, kind=kind))
        for i, s in enumerate(subs):
            if s.kind == "relation" and i > 0:
                s.depends_on = list(range(i))
            s.route = self._route_text(s.text, s.keywords)
        return subs

    def _cardi_guard(self, question: str, subs: List[SubQuestion]) -> List[SubQuestion]:
        """Cardi 監督卡點：關係／比較題缺 RELATION_QUERY 步即強制補步。

        目的：防 LLM 隨機性漏掉最重要的 SQLite3 relations 圖譜檢索。
        攔截邏輯：問句命中 _CARDI_TRIGGERS（關係、關聯、相比、比較、差別、
        不同、異同等簡繁詞），但步驟清單無 RELATION_QUERY 路由 → 硬編碼介入。
        補步內容：能拆出兩實體即針對該對下精準關係步，否則用問題關鍵字兜底；
        depends_on 指向此前所有步驟（關係綜合需先備齊實體上下文）。

        Args:
            question: 原始提問（觸發詞偵測對象）。
            subs: LLM 解析出的步驟清單（原地附加，不改既有步驟）。

        Returns:
            審查後（必要時已補步）的同一串列。
        """
        if not subs or not any(t in (question or "") for t in _CARDI_TRIGGERS):
            return subs  # 無觸發詞：直接放行，零干預
        if any(s.route == RouteTarget.RELATION_QUERY for s in subs):
            return subs  # 已有關係步：放行，避免重複檢索
        pair = _split_two_entities(question)
        if len(pair) == 2:
            text = f"查詢{pair[0]}與{pair[1]}的關係"
            keywords = pair
        else:  # 拆不出兩實體：用問題關鍵字兜底，寧可多查不可漏查
            keywords = _sanitize_keywords(_extract_keywords(question))
            text = question
        subs.append(SubQuestion(
            text=text, route=RouteTarget.RELATION_QUERY,
            keywords=keywords, kind="relation",
            depends_on=list(range(len(subs))),
        ))
        logger.warning("Cardi 介入：問題含關係／比較觸發詞但 LLM 漏拆關係步，"
                       "已強制追加 RELATION_QUERY（關鍵字=%s）", keywords)
        return subs

    def _rule_decompose(self, question: str) -> List[SubQuestion]:
        """規則式拆解：關係題 → 3 步；比較題 → 3 步；書名號 → 專查；其餘 → 關鍵字查。"""
        # 1. 書名號優先：每個《》獨立成一個精準條目查詢
        titles = _extract_titles(question)
        if titles and len(titles) == 1 and len(question) < 30:
            # 短問句只問一個條目（如「介紹《史記》」）→ 單步直查
            return [SubQuestion(text=question, route=RouteTarget.TITLE_LOOKUP,
                                keywords=titles, kind="lookup")]
        subs: List[SubQuestion] = []
        for t in titles:
            subs.append(SubQuestion(text=f"查詢條目{t}", route=RouteTarget.TITLE_LOOKUP,
                                    keywords=[t], kind="lookup"))

        # 2. 關係題：「A與B的關係」→ 查A、查B、查兩者關係（依賴前兩步）
        if any(t in question for t in _RELATION_TRIGGERS):
            pair = _split_two_entities(question)
            if len(pair) == 2:
                base = len(subs)
                subs.append(SubQuestion(text=f"查詢{pair[0]}", route=RouteTarget.ENTITY_LOOKUP,
                                        keywords=[pair[0]], kind="lookup"))
                subs.append(SubQuestion(text=f"查詢{pair[1]}", route=RouteTarget.ENTITY_LOOKUP,
                                        keywords=[pair[1]], kind="lookup"))
                subs.append(SubQuestion(
                    text=f"查詢{pair[0]}與{pair[1]}的關係", route=RouteTarget.RELATION_QUERY,
                    keywords=pair, kind="relation", depends_on=[base, base + 1]))
                return subs
            # 觸發詞存在但拆不出兩實體 → 退回關鍵字查 + 關係查
            kws = _extract_keywords(question)
            subs.append(SubQuestion(text=question, route=RouteTarget.RELATION_QUERY,
                                    keywords=kws, kind="relation"))
            return subs

        # 3. 比較題：「A與B的差別」→ 查A、查B、綜合比較（依賴前兩步，綜述走混合）
        if any(t in question for t in _COMPARE_TRIGGERS):
            pair = _split_two_entities(question)
            if len(pair) == 2:
                base = len(subs)
                subs.append(SubQuestion(text=f"查詢{pair[0]}", route=RouteTarget.ENTITY_LOOKUP,
                                        keywords=[pair[0]], kind="lookup"))
                subs.append(SubQuestion(text=f"查詢{pair[1]}", route=RouteTarget.ENTITY_LOOKUP,
                                        keywords=[pair[1]], kind="lookup"))
                subs.append(SubQuestion(text=f"比較{pair[0]}與{pair[1]}",
                                        route=RouteTarget.HYBRID_SEARCH,
                                        keywords=pair, kind="compare",
                                        depends_on=[base, base + 1]))
                return subs

        # 4. 一般問：單關鍵字走精準路由，多關鍵字合併為混合綜述（避免發散）
        if not subs:
            kws = _extract_keywords(question, max_n=3)
            if len(kws) == 1:
                subs.append(SubQuestion(text=f"查詢{kws[0]}", keywords=kws, kind="lookup",
                                        route=self._route_text(kws[0], kws)))
            elif kws:
                subs.append(SubQuestion(text=question, keywords=kws, kind="summarize",
                                        route=RouteTarget.HYBRID_SEARCH))
        if not subs:  # 極端情況（問句無中文實詞）→ 用整句做關鍵字查
            subs.append(SubQuestion(text=question, keywords=[question], kind="lookup",
                                    route=RouteTarget.KEYWORD_SEARCH))
        return subs

    # ══════════════════════════════════════════
    # 3.3 Router：自適應路由 ＋ RRF 混合執行
    # ══════════════════════════════════════════

    def route(self, question: str) -> RouteTarget:
        """判斷整個問題的路由（相容舊簽名；多步查詢請用 answer 內部分派）。"""
        return self._route_text(question or "", _extract_keywords(question or ""))

    def _route_text(self, text: str, keywords: List[str]) -> RouteTarget:
        """規則式路由核心：書名號 → 條目；關係詞 → 關係；寒暄 → 直答；其餘按型別分流。"""
        if any(s in text for s in _SMALLTALK) and len(text) < 20:
            return RouteTarget.DIRECT_ANSWER
        if _extract_titles(text):
            return RouteTarget.TITLE_LOOKUP
        if any(t in text for t in _RELATION_TRIGGERS):
            return RouteTarget.RELATION_QUERY
        # 單一關鍵字且像實體名（2~6 字、非句子）→ 實體精準查
        if len(keywords) == 1 and 2 <= len(keywords[0]) <= 6:
            return RouteTarget.ENTITY_LOOKUP
        return RouteTarget.KEYWORD_SEARCH

    def _dense_available(self) -> bool:
        """密集腿是否可用：processor 存在、Ollama 連線正常、具備語意查詢方法。

        FakeProcessor（單元測試替身）無此屬性 → 自動走純稀疏，舊測試零影響。
        """
        sp = self.spark_processor  # 不觸發延遲建立；無處理器即無密集可用
        return bool(
            sp is not None
            and getattr(sp, "_ollama_ok", False) is True
            and callable(getattr(sp, "semantic_search", None))
        )

    # ── 稀疏腿：按 route 分派到 SparkProcessor 對應 API（原五路由邏輯完整保留）──
    def _sparse_leg(self, sub: SubQuestion, kws: List[str]) -> List[Dict[str, Any]]:
        """執行稀疏檢索（單腿、無重試），回傳正規化上下文串列。

        正規化格式：[{chunk_id, title, text, source}]，
        source 標註命中哪個 API，方便除錯、引用與 RRF 融合溯源。
        內部異常自行吞掉回空串列，不讓單腿故障拖垮混合檢索。
        """
        try:
            return self._sparse_leg_inner(sub, kws)
        except Exception as e:  # noqa: BLE001 — 腿級容錯，空結果交給 RRF/另一腿
            logger.warning("稀疏檢索異常（%s），本腿回空", e)
            return []

    def _sparse_leg_inner(self, sub: SubQuestion, kws: List[str]) -> List[Dict[str, Any]]:
        """稀疏腿本體：原 _retrieve_once 的路由分派邏輯（逐字保留）。"""
        sp = self._processor()
        top_k = self.config.top_k
        contexts: List[Dict[str, Any]] = []

        def _push(rows: List[Dict[str, Any]], source: str) -> None:
            """把檢索列轉為上下文（去重：同 chunk_id 只保留首次來源）。"""
            seen = {c["chunk_id"] for c in contexts}
            for r in rows:
                cid = str(r.get("chunk_id", ""))
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                contexts.append({
                    "chunk_id": cid,
                    "title": str(r.get("title", "")),
                    "text": str(r.get("text", "") or r.get("evidence", "")),
                    "source": source,
                })

        if sub.route == RouteTarget.TITLE_LOOKUP:
            for kw in kws:  # 書名號通常精準命中，一個命中即夠
                rows = sp.search_by_title(kw, limit=top_k)
                _push(rows, f"title:{kw}")
                if rows:
                    break
            if not contexts:
                # 【兜底】條目名差一字即完全 miss，退回關鍵字全文查搶救 recall
                logger.info("TITLE miss，降級為關鍵字全文查：%s", kws)
                for kw in kws:
                    _push(sp.search_by_keyword(kw, limit=top_k), f"title-fallback:{kw}")
        elif sub.route == RouteTarget.ENTITY_LOOKUP:
            for kw in kws:
                _push(sp.search_by_entity(kw, limit=top_k), f"entity:{kw}")
            if not contexts:
                # 【兜底】實體完全比對 miss（如簡繁差異），退回關鍵字全文查
                logger.info("ENTITY miss，降級為關鍵字全文查：%s", kws)
                for kw in kws:
                    _push(sp.search_by_keyword(kw, limit=top_k), f"entity-fallback:{kw}")
        elif sub.route == RouteTarget.RELATION_QUERY:
            # 先查三元組（結構化關係），再用實體查補原文，兩路互補
            for kw in kws:
                for rel in sp.get_relations(kw, limit=top_k):
                    cid = str(rel.get("chunk_id", ""))
                    triple = f"{rel.get('subj','')}—{rel.get('predicate','')}—{rel.get('obj','')}"
                    ev = str(rel.get("evidence", ""))
                    contexts.append({
                        "chunk_id": cid or f"rel:{triple}",
                        "title": "",
                        "text": f"【關係】{triple}。{ev}",
                        "source": f"relation:{kw}",
                    })
            for kw in kws:
                _push(sp.search_by_entity(kw, limit=top_k), f"entity:{kw}")
            if not contexts:
                # 【兜底】關係＋實體皆落空時，退回關鍵字全文查做最後搶救
                logger.info("RELATION miss，降級為關鍵字全文查：%s", kws)
                for kw in kws:
                    _push(sp.search_by_keyword(kw, limit=top_k), f"relation-fallback:{kw}")
        else:  # KEYWORD_SEARCH / HYBRID_SEARCH（含 compare / summarize 的綜合查）
            for kw in kws:
                _push(sp.search_by_keyword(kw, limit=top_k), f"keyword:{kw}")
        return contexts

    # ── 密集腿：bge-m3 向量語意檢索（Ollama 即時編碼 query）──
    def _dense_leg(self, sub: SubQuestion, kws: List[str]) -> List[Dict[str, Any]]:
        """執行密集檢索（單腿、無重試）：query 轉向量 → 語意近鄰。

        查询文本用消毒後關鍵字（無則退回子問題文字）；processor 無向量能力
        （Fake替身 / Ollama 斷線 / --no-vector）直接回空，退回純稀疏。
        """
        if not self._dense_available():
            return []
        query_text = " ".join(kws) if kws else sub.text
        if not query_text.strip():
            return []
        try:
            rows = self.spark_processor.semantic_search(
                query_text, limit=max(1, int(self.config.dense_top_k)))
            out = []
            for r in rows:
                out.append({
                    "chunk_id": str(r.get("chunk_id", "")),
                    "title": str(r.get("title", "")),
                    "text": str(r.get("text", "")),
                    "source": f"dense:{query_text[:12]}",
                })
            return [c for c in out if c["chunk_id"]]
        except Exception as e:  # noqa: BLE001 — 腿級容錯，空結果交給稀疏腿
            logger.warning("密集檢索異常（%s），本腿回空", e)
            return []

    # ── 混合執行：雙腿並行 ＋ RRF 融合 ──
    def _retrieve_once(self, sub: SubQuestion) -> List[Dict[str, Any]]:
        """執行一次混合檢索（單輪、無重試），回傳 RRF 融合後的上下文串列。

        稀疏腿（SQLite）與密集腿（Chroma 向量）經 ThreadPoolExecutor 並行；
        密集可用時 RRF 取前 hybrid_top_n 筆，否則退回純稀疏（原行為）。
        """
        kws = [k for k in sub.keywords if k]
        if sub.route == RouteTarget.DIRECT_ANSWER or not kws:
            return []  # 直答 / 無關鍵字：無需檢索

        # 【檢索邊界消毒】子問題文字常帶「查詢/檢索/探討」包裝（如「查詢數學」），
        # 此處統一剝除，保證進入 SQLite LIKE / Chroma 的只有純實詞（如「數學」）。
        # 消毒前後不一致時記 INFO 日誌，方便對照除錯。
        clean_kws = _sanitize_keywords(kws)
        if clean_kws != kws:
            logger.info("關鍵字消毒：%s → %s", kws, clean_kws)
        kws = clean_kws
        if not kws:
            return []  # 消毒後無可用關鍵字（如全是虛詞），視同無需檢索

        # 【雙軌並行】稀疏（精準）與密集（語意）同時開火，取慢者時間即總耗時
        with ThreadPoolExecutor(max_workers=max(1, int(self.config.max_workers))) as ex:
            future_sparse = ex.submit(self._sparse_leg, sub, kws)
            future_dense = ex.submit(self._dense_leg, sub, kws)
            sparse, dense = future_sparse.result(), future_dense.result()

        if dense:
            fused = _rrf_fuse(sparse, dense, k=int(self.config.rrf_k),
                              top_n=int(self.config.hybrid_top_n))
            logger.info("混合檢索「%s」：稀疏 %d 筆＋密集 %d 筆 → RRF 融合 %d 筆",
                        sub.text, len(sparse), len(dense), len(fused))
            return fused
        # 【保底】無密集可用時退回純稀疏（原五路由行為逐字保留）
        return sparse[: self.config.top_k * len(kws)]

    # ══════════════════════════════════════════
    # 3.4 Critic：自我反思與修正迴圈（含 LLM 改寫）
    # ══════════════════════════════════════════

    def self_reflect(
        self, question: str, draft_answer: str, contexts: List[str]
    ) -> ReflectResult:
        """檢查檢索內容是否足以回答問題（規則覆蓋率；LLM 只用於改寫不做評判）。

        評判維持確定性規則（跨字形變體比對），避免 LLM 評分抖動；
        智能改寫統一收斂到 _rewrite_for_retry（重試迴圈內呼叫）。

        Args:
            question: 原始提問（或子問題文字）。
            draft_answer: 草稿答案（保留簽名相容，規則模式不使用）。
            contexts: 檢索到的純文字串列。

        Returns:
            ReflectResult（含覆蓋率、缺失關鍵字、是否需重試）。
        """
        texts = [c for c in (contexts or []) if c and c.strip()]
        if not texts:
            return ReflectResult(passed=False, score=0.0, need_retry=True,
                                 comment="檢索結果為空，需改寫關鍵字或關係擴張後重查")
        return self._rule_reflect(question, texts)

    def _rule_reflect(self, question: str, texts: List[str]) -> ReflectResult:
        """規則式反思：問題關鍵字在上下文中的覆蓋率 ≥ 門檻即通過。

        跨字形比對：每個關鍵字展開繁簡變體後再比對，任一字形命中即算覆蓋
        （如繁體「數學」對簡體內文「数学」）；尾端提問運算詞（如「關係」）
        先剝除再比對，避免殘留詞造成恆性 miss。
        """
        # 【反思前消毒】避免「查詢數學」這類包裝文字污染覆蓋率計算
        raw_kws = _sanitize_keywords(_extract_keywords(question or ""))
        # 【核心詞還原】剝尾端運算詞（「數學的關係」→「數學」），去重保序
        kws: List[str] = []
        for kw in raw_kws:
            core = _strip_trailing_operators(kw)
            if core and core not in kws:
                kws.append(core)
        if not kws:  # 無關鍵字可驗（如寒暄）→ 有內容即通過
            return ReflectResult(passed=True, score=1.0, comment="無需驗證關鍵字", coverage=1.0)
        joined = "\n".join(texts)
        # 【變體命中】任一字形出現即算覆蓋，終結簡繁不一致造成的 0% 誤判
        hit = [k for k in kws if any(v in joined for v in _keyword_variants(k))]
        # 單字關鍵字放寬：任一字（含變體）命中即算覆蓋（避免過度重試）
        missing = [k for k in kws if k not in hit and not (
            len(k) == 1 and any(v in joined for v in _keyword_variants(k)))]
        coverage = len(hit) / max(1, len(kws))
        passed = coverage >= self.config.min_coverage
        comment = ("覆蓋充足" if passed
                   else f"覆蓋率 {coverage:.0%} 不足，缺失：{missing}，建議改寫或關係擴張")
        return ReflectResult(passed=passed, score=coverage, comment=comment,
                             need_retry=not passed, coverage=coverage, missing=missing)

    def _rewrite_for_retry(self, keywords: List[str], question_hint: str = "") -> List[str]:
        """重試改寫：LLM 換句話說優先，失敗降級規則改寫。

        請模型把缺失關鍵字換 3 種說法（近義詞 / 簡繁對偶 / 上下位詞），
        回傳消毒後的乾淨關鍵字；LLM 不可用或回傳無效時用規則改寫保底。
        """
        if self._use_llm() and keywords:
            try:
                raw = self._call_ollama(
                    "你是繁體中文檢索關鍵字改寫器。只輸出 JSON 字串陣列，"
                    "把給定關鍵字換句話說成 3 個 2~6 字的中文檢索關鍵字"
                    "（可用近義詞、簡繁對偶、上位詞），不要輸出說明文字。",
                    f"關鍵字：{keywords}\n問題背景：{question_hint[:200]}",
                )
                rewritten = _sanitize_keywords(self._parse_json_strings(raw))
                if rewritten:
                    logger.info("LLM 改寫關鍵字：%s → %s", keywords, rewritten)
                    return rewritten
                logger.warning("LLM 改寫回傳為空，降級為規則改寫")
            except Exception as e:  # noqa: BLE001 — 改寫失敗就用規則，不中斷重試
                logger.warning("LLM 改寫失敗（%s），降級為規則改寫", e)
        return _rewrite_keywords(keywords, attempt=1)

    def _expand_via_relations(self, keywords: List[str], limit: int = 10) -> List[str]:
        """關係擴張：透過 get_relations 找鄰居實體，作為下一輪的替代關鍵字。

        例：查「孔子」無結果 → 關係表有（孔子—創立—儒家）→ 改查「儒家」。
        回傳與原關鍵字不重複的鄰居實體名（按出現次數排序）。
        """
        sp = self._processor()
        # 【擴張前消毒】get_relations 是完全比對，污染詞必定落空，先清乾淨
        keywords = _sanitize_keywords(keywords)
        neighbors: Dict[str, int] = {}
        for kw in keywords:
            try:
                for rel in sp.get_relations(kw, limit=limit):
                    for end in (str(rel.get("subj", "")), str(rel.get("obj", ""))):
                        if end and end != kw and len(end) >= 2:
                            neighbors[end] = neighbors.get(end, 0) + 1
            except Exception as e:  # noqa: BLE001 — 擴張失敗不應中斷主迴圈
                logger.warning("關係擴張失敗（%s）", e)
        known = set(keywords)
        return sorted((n for n in neighbors if n not in known),
                      key=lambda n: neighbors[n], reverse=True)

    def _answer_sub(self, sub: SubQuestion) -> Dict[str, Any]:
        """單個子問題的混合檢索 + 反思重試迴圈（最多 max_attempts 次）。

        重試策略依序升級：
            第 1 次重試：LLM 換句話說改寫（不可用則規則去虛詞），路由不變。
            第 2 次重試：關係擴張（鄰居實體替代）；
                無鄰居則規則前綴放寬＋路由降級為關鍵字查（書名號直查除外）。
        """
        if sub.route == RouteTarget.DIRECT_ANSWER:
            return {"sub": sub, "contexts": [], "attempts": 0,
                    "reflection": ReflectResult(True, 1.0, "直答，無需檢索", False, 1.0)}

        keywords = list(sub.keywords)
        base_keywords = list(sub.keywords)  # 原始關鍵字：反思與改寫的基準
        # 反思比對用的文字：用原始關鍵字而非「查詢X」包裝的子問題文字，
        # 避免「查詢」前綴污染覆蓋率計算與缺失關鍵字
        reflect_q = " ".join(base_keywords) if base_keywords else sub.text
        contexts: List[Dict[str, Any]] = []
        reflection = ReflectResult(False, 0.0, "尚未檢索", True)
        attempts = 0

        for attempt in range(max(1, self.config.max_attempts)):
            attempts = attempt + 1
            # 後輪次：先改寫關鍵字 / 切換策略再查（基準一律用原始關鍵字）
            if attempt == 1:
                keywords = self._rewrite_for_retry(
                    reflection.missing or base_keywords, sub.text)
            elif attempt >= 2:
                expanded = self._expand_via_relations(base_keywords)
                if expanded:
                    # 擴張成功：用鄰居實體查，保持原路由的精準度
                    keywords = expanded[:3]
                else:
                    # 無鄰居可擴：前綴放寬＋路由降級為關鍵字查（書名號直查除外）
                    keywords = _rewrite_keywords(base_keywords, attempt=2)
                    if sub.route != RouteTarget.TITLE_LOOKUP:
                        sub.route = RouteTarget.KEYWORD_SEARCH
            sub.keywords = keywords
            round_sub = SubQuestion(text=sub.text, route=sub.route,
                                    depends_on=sub.depends_on, keywords=keywords, kind=sub.kind)
            contexts = self._retrieve_once(round_sub)
            reflection = self._rule_reflect(
                reflect_q, [c["text"] for c in contexts])
            logger.info("子問題「%s」第 %d 次檢索：%d 筆，覆蓋率 %.0f%%",
                        sub.text, attempts, len(contexts), reflection.coverage * 100)
            if reflection.passed or attempts >= self.config.max_attempts:
                reflection.need_retry = False if reflection.passed else attempts < self.config.max_attempts
                break
        return {"sub": sub, "contexts": contexts, "attempts": attempts,
                "reflection": reflection}

    # ══════════════════════════════════════════
    # 3.5 答案合成與端到端入口（含 LLM 合成）
    # ══════════════════════════════════════════

    def _synthesize(self, question: str, results: List[Dict[str, Any]]) -> str:
        """合成最終答案（LLM 優先用模型改寫，失敗降級規則摘錄）。

        規則摘錄保證【零幻覺】：只引用檢索原文 + 標註出處【標題·chunk_id】；
        關係步驟直接列出三元組。
        """
        if self._use_llm():
            try:
                ctx_text = "\n---\n".join(
                    f"[{c['title']}·{c['chunk_id']}] {c['text'][:400]}"
                    for r in results for c in r["contexts"][:2])
                raw = self._call_ollama(
                    "你是用繁體中文回答的問答助手。只能根據提供的上下文回答，"
                    "禁止編造；引用時標註【標題·編號】；上下文不足就明說不知道。",
                    f"問題：{question}\n上下文：\n{ctx_text[:4000]}",
                    json_mode=False,
                )
                if raw.strip():
                    return raw.strip()
            except Exception as e:  # noqa: BLE001 — 合成失敗降級摘錄，不可無答案
                logger.warning("LLM 答案合成失敗（%s），降級為規則摘錄", e)
        # ── 規則摘錄 ──
        # 寒暄 / 直答題：無需檢索，直接回應（避免「查無資料」的生硬回覆）
        if results and all(r["sub"].route == RouteTarget.DIRECT_ANSWER for r in results):
            return ("你好！我是知識庫問答助手（混合檢索 RAG）。"
                    "請輸入想查詢的維基百科條目或問題，例如："
                    "「孔子與孟子的關係是什麼？」。")
        parts = []
        for r in results:
            sub = r["sub"]
            ctxs = r["contexts"]
            if not ctxs:
                parts.append(f"【{sub.text}】查無相關資料。")
                continue
            if sub.kind == "relation":
                # 關係步驟：優先列出【關係】三元組行
                triples = [c["text"] for c in ctxs if c["text"].startswith("【關係】")]
                body = "\n".join(triples[:3]) or ctxs[0]["text"][:200]
                parts.append(f"【{sub.text}】\n{body}")
            else:
                top = ctxs[0]
                cite = f"【{top['title']}·{top['chunk_id']}】" if top["title"] else ""
                parts.append(f"【{sub.text}】{cite}\n{top['text'][:300]}")
        if not parts:
            return "查無相關資料，無法回答。"
        return f"針對「{question}」的查詢結果：\n\n" + "\n\n".join(parts)

    def answer(self, question: str) -> Dict[str, Any]:
        """端到端問答入口：拆解 → 混合檢索 → 反思重試 → 合成。

        Returns:
            {"question", "sub_questions", "contexts", "answer",
             "reflections", "total_attempts"}（皆可 JSON 序列化；
             contexts 每筆另帶 rrf 融合分數，供前端軌跡展示）。
        """
        question = (question or "").strip()
        if not question:
            return {"question": "", "sub_questions": [], "contexts": [],
                    "answer": "請輸入問題。", "reflections": [], "total_attempts": 0}

        subs = self.decompose_question(question)
        # 依依賴排序：被依賴者先執行（depends_on 索引較小者已在前，拓撲確認即可）
        results: List[Dict[str, Any]] = []
        for sub in subs:
            sub.route = self._route_text(sub.text, sub.keywords) \
                if sub.route == RouteTarget.KEYWORD_SEARCH and sub.kind == "lookup" \
                else sub.route
            results.append(self._answer_sub(sub))

        final = self._synthesize(question, results)
        # 最終反思：用全文覆蓋率給整體評分（不觸發重試，只記錄）
        all_texts = [c["text"] for r in results for c in r["contexts"]]
        overall = self._rule_reflect(question, all_texts) if all_texts else \
            ReflectResult(any(s.route == RouteTarget.DIRECT_ANSWER for s in subs),
                          1.0 if any(s.route == RouteTarget.DIRECT_ANSWER for s in subs) else 0.0,
                          "直答" if any(s.route == RouteTarget.DIRECT_ANSWER for s in subs) else "無上下文")
        return {
            "question": question,
            "sub_questions": [
                {"text": s.text, "route": s.route.value, "kind": s.kind,
                 "keywords": s.keywords, "depends_on": s.depends_on} for s in subs],
            "contexts": [
                {"chunk_id": c["chunk_id"], "title": c["title"],
                 "text": c["text"][:500], "source": c["source"],
                 "rrf": c.get("rrf")} for r in results for c in r["contexts"]],
            "answer": final,
            "reflections": [
                {"sub": r["sub"].text, "passed": r["reflection"].passed,
                 "score": round(r["reflection"].score, 2),
                 "comment": r["reflection"].comment, "attempts": r["attempts"]}
                for r in results],
            "total_attempts": sum(r["attempts"] for r in results),
            "overall": {"passed": overall.passed, "score": round(overall.score, 2),
                        "comment": overall.comment},
        }


# ══════════════════════════════════════════════
# 四、命令列快速驗證（python -m src.agent_router）
# ══════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    import argparse

    # 【簡易 CLI】需先用 spark_processor 建庫，再提問驗證整條混合檢索鏈路
    parser = argparse.ArgumentParser(description="Agentic RAG 問答驗證工具（RRF 混合檢索）")
    parser.add_argument("--question", default="孔子與孟子的關係是什麼？", help="用戶提問")
    parser.add_argument("--db", default="data/knowledge.db", help="SQLite 知識庫路徑")
    parser.add_argument("--chroma-dir", default="data/chroma", help="Chroma 持久化目錄")
    parser.add_argument("--mode", default="rule", choices=["rule", "llm", "api"],
                        help="運作模式（api 為舊別名，視為 llm）")
    parser.add_argument("--llm-model", default="qwen3:8b", help="Ollama 推理主模型")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from .spark_processor import SparkConfig, SparkProcessor

    cfg = SparkConfig(mode=args.mode, db_path=args.db, chroma_dir=args.chroma_dir)
    with SparkProcessor(cfg) as sp:
        router = AgentRouter(spark_processor=sp,
                             config=AgentConfig(mode=args.mode, llm_model=args.llm_model))
        result = router.answer(args.question)
        print(f"\n問題：{result['question']}")
        print(f"子問題：{[s['text'] for s in result['sub_questions']]}")
        print(f"反思：{result['reflections']}")
        # 【命中來源】列出前 5 筆上下文的檢索路徑（rrf: 開頭即混合融合命中）
        for c in result["contexts"][:5]:
            print(f"命中：[{c['source']}] {c['title']}·{c['chunk_id']} (rrf={c.get('rrf')})")
        print(f"\n答案：\n{result['answer']}")
