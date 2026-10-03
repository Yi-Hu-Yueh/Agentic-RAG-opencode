"""智能體調度模組（agent_router.py）

實作「Agentic RAG（多步驟規劃 + 路由 + 自我反思與修正）」，
且全程【不使用 Embedding】：檢索只調用 ``SparkProcessor`` 的
無向量 API（``search_by_*`` / ``get_relations``）。

三個核心機制：
    1. Planner（任務拆解）：複雜提問 → 多個子查詢步驟（含依賴順序）。
    2. Router（自適應路由）：每個子問題 → 最合適的檢索 API。
    3. Critic（反思修正迴圈）：結果不足 → 改寫關鍵字 / 關係擴張 → 重查，
       最多 ``max_attempts`` 次（預設 3 次），再交給模型生成最終答案。

運作模式（與 spark_processor 一致）：
    - ``mode="rule"``（預設）：本地規則完成拆解 / 路由 / 反思 / 摘要，
      零連線、零費用，開箱即用。
    - ``mode="api"``：上述環節改調 Muse Spark 1.3，失敗自動降級為規則。
"""

from __future__ import annotations

import json
import logging
import os
import re
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
    """子問題的檢索路由目標（皆為無向量 API）。

    - TITLE_LOOKUP: 書名號 / 條目名 → ``search_by_title``
    - ENTITY_LOOKUP: 人名 / 地名 / 術語 → ``search_by_entity``
    - KEYWORD_SEARCH: 一般關鍵字 → ``search_by_keyword``
    - RELATION_QUERY: 查兩實體關係 → ``get_relations``（+ 實體查補）
    - DIRECT_ANSWER: 寒暄 / 常識問，無需檢索
    """

    TITLE_LOOKUP = "title_lookup"
    ENTITY_LOOKUP = "entity_lookup"
    KEYWORD_SEARCH = "keyword_search"
    RELATION_QUERY = "relation_query"
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
    """AgentRouter 設定（欄位語義與 SparkConfig 的 API 部分對齊）。

    Attributes:
        mode: "rule"（預設，本地規則）或 "api"（調 Muse Spark 1.3，失敗降級）。
        api_base / api_key / model / timeout / max_retries: API 連線參數，
            為空時自動讀環境變數 SPARK_API_BASE / SPARK_API_KEY / SPARK_MODEL。
        max_attempts: 單個子問題最多檢索幾次（含首次，預設 3 次）。
        top_k: 每次檢索取回幾筆上下文。
        min_coverage: 反思通過門檻（關鍵字覆蓋率，預設 0.4）。
    """

    mode: str = "rule"
    api_base: str = ""
    api_key: str = ""
    model: str = "muse-spark-1.3"
    timeout: int = 60
    max_retries: int = 2
    max_attempts: int = 3
    top_k: int = 5
    min_coverage: float = 0.4

    def __post_init__(self) -> None:
        """以環境變數補齊未填的連線參數（顯式傳參優先）。"""
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

# 【實體分隔符】「A與B」「A和B」「A、B」中的分隔字（簡繁並列）
_ENTITY_SEPS = ("與", "与", "和", "跟", "及", "、", "以及")

# 【寒暄】命中則直接回答，無需檢索（簡繁並列）
_SMALLTALK = ("你好", "您好", "哈囉", "哈喽", "嗨", "謝謝", "谢谢", "感謝", "感谢",
              "再見", "再见", "拜拜", "早安", "午安", "晚安")

# 【停用字】關鍵字首字過濾用（避免抽出「的是」類碎片；簡繁並列）
_STOP_FIRST = set("的了是在有和與与或及等之其這这那你我他她它為为被把將将對对就都也而於于以從从到比過过個个些種种們们會会可所能應应但若如因果當当每各另別别先後后內内")  # noqa: E501


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
    """反思重試用的關鍵字改寫（attempt 從 1 起算，越大改寫越激進）。

    - 第 1 次重試：去尾字（「孔子的」→「孔子」），拆「的」前段。
    - 第 2 次重試：只留前 2 字（放寬為前綴比對，recall 優先）。
    - 之後：逐字拆成單字查詢（最後手段）。
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


# ══════════════════════════════════════════════
# 三、AgentRouter 主類別
# ══════════════════════════════════════════════


class AgentRouter:
    """OpenCode 智能體路由器：Planner → Router → 檢索 → Critic 重試 → 合成答案。

    使用範例：
        from src.spark_processor import SparkConfig, SparkProcessor
        with SparkProcessor(SparkConfig()) as sp:  # 需先 run_pipeline 建庫
            router = AgentRouter(spark_processor=sp)
            result = router.answer("孔子與孟子的關係是什麼？")
            print(result["answer"])
    """

    def __init__(self, spark_processor: Any | None = None, config: Optional[AgentConfig] = None) -> None:
        """初始化路由器。

        Args:
            spark_processor: 已建庫的 SparkProcessor 實例；為 None 時，
                首次檢索才延遲建立預設實例（方便純規則測試）。
            config: 路由器設定（重試次數、覆蓋率門檻、API 參數）。
        """
        self.spark_processor = spark_processor
        self.config = config or AgentConfig()

    # ── 延遲取得 processor（避免 __init__ 產生建庫副作用）──
    def _processor(self) -> Any:
        """取回 SparkProcessor；未提供則延遲建立預設實例。"""
        if self.spark_processor is None:
            from .spark_processor import SparkConfig, SparkProcessor

            logger.info("未提供 SparkProcessor，延遲建立預設實例（rule 模式）")
            self.spark_processor = SparkProcessor(SparkConfig())
        return self.spark_processor

    # ══════════════════════════════════════════
    # 3.1 Muse Spark 1.3 呼叫（api 模式專用，失敗降級 rule）
    # ══════════════════════════════════════════

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """呼叫 Spark 1.3（OpenAI 相容），含重試；失敗拋例外由上層降級。"""
        if not self.config.api_base:
            raise RuntimeError("未設定 api_base，無法使用 api 模式")
        try:
            import requests  # 延遲匯入，規則模式不需要它
        except ImportError as e:
            raise RuntimeError("未安裝 requests（pip install requests）") from e
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
            "temperature": 0.1,  # 規劃/判斷任務用低溫求穩
        }
        last_err: Optional[Exception] = None
        for _ in range(max(1, self.config.max_retries + 1)):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                resp.raise_for_status()
                return str(resp.json()["choices"][0]["message"]["content"])
            except Exception as e:  # noqa: BLE001 — 重試迴圈統一捕捉，耗盡才拋出
                last_err = e
        raise RuntimeError(f"Spark API 呼叫失敗（已重試）：{last_err}")

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """從模型回文擷取 JSON 陣列（容忍前後夾雜說明文字），失敗回傳空串列。"""
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            parsed = json.loads(raw[start:end])
            return [d for d in parsed if isinstance(d, dict)]
        except (ValueError, json.JSONDecodeError):
            logger.warning("Spark 回傳無法解析為 JSON，降級為規則")
            return []

    # ══════════════════════════════════════════
    # 3.2 Planner：任務拆解
    # ══════════════════════════════════════════

    def decompose_question(self, question: str) -> List[SubQuestion]:
        """將複雜提問拆解為子查詢步驟（api 優先，失敗降級規則）。

        例：「孔子與孟子的關係是什麼？」
        → [查孔子(0), 查孟子(1), 查兩者關係(2, 依賴 0,1)]
        """
        question = (question or "").strip()
        if not question:
            return []
        if self.config.mode == "api":
            try:
                raw = self._call_llm(
                    "你是 RAG 任務規劃器。只輸出 JSON 陣列，每個元素為"
                    '{"text": 子問題, "kind": lookup/relation/compare/summarize}。'
                    "關係題要拆成：查A、查B、查兩者關係三步。不要輸出說明文字。",
                    f"拆解以下問題：{question}",
                )
                parsed = self._parse_json_array(raw)
                if parsed:
                    subs = [
                        SubQuestion(
                            text=str(d.get("text", "")),
                            keywords=_extract_keywords(str(d.get("text", ""))),
                            kind=str(d.get("kind", "lookup") or "lookup"),
                        )
                        for d in parsed
                        if str(d.get("text", "")).strip()
                    ]
                    # 關係步標依賴：讓「查關係」排在實體查詢之後
                    for i, s in enumerate(subs):
                        if s.kind == "relation" and i > 0:
                            s.depends_on = list(range(i))
                        s.route = self._route_text(s.text, s.keywords)
                    if subs:
                        return subs
                logger.warning("Spark 拆解回傳為空，降級為規則拆解")
            except Exception as e:  # noqa: BLE001 — 降級需捕捉所有 API 異常
                logger.warning("Spark 拆解失敗（%s），降級為規則拆解", e)
        return self._rule_decompose(question)

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

        # 3. 比較題：「A與B的差別」→ 查A、查B、綜合比較（依賴前兩步）
        if any(t in question for t in _COMPARE_TRIGGERS):
            pair = _split_two_entities(question)
            if len(pair) == 2:
                base = len(subs)
                subs.append(SubQuestion(text=f"查詢{pair[0]}", route=RouteTarget.ENTITY_LOOKUP,
                                        keywords=[pair[0]], kind="lookup"))
                subs.append(SubQuestion(text=f"查詢{pair[1]}", route=RouteTarget.ENTITY_LOOKUP,
                                        keywords=[pair[1]], kind="lookup"))
                subs.append(SubQuestion(text=f"比較{pair[0]}與{pair[1]}",
                                        route=RouteTarget.KEYWORD_SEARCH,
                                        keywords=pair, kind="compare",
                                        depends_on=[base, base + 1]))
                return subs

        # 4. 一般問：每個關鍵字一步（上限 3 步，避免發散）
        if not subs:
            for kw in _extract_keywords(question, max_n=3):
                subs.append(SubQuestion(text=f"查詢{kw}", keywords=[kw], kind="lookup",
                                        route=self._route_text(kw, [kw])))
        if not subs:  # 極端情況（問句無中文實詞）→ 用整句做關鍵字查
            subs.append(SubQuestion(text=question, keywords=[question], kind="lookup",
                                    route=RouteTarget.KEYWORD_SEARCH))
        return subs

    # ══════════════════════════════════════════
    # 3.3 Router：自適應路由
    # ══════════════════════════════════════════

    def route(self, question: str) -> RouteTarget:
        """判斷整個問題的路由（相容舊骨架簽名；多步查詢請用 answer 內部分派）。"""
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

    # ── 檢索執行：按 route 分派到 SparkProcessor 對應 API ──
    def _retrieve_once(self, sub: SubQuestion) -> List[Dict[str, Any]]:
        """執行一次檢索（單輪、無重試），回傳正規化上下文串列。

        正規化格式：[{chunk_id, title, text, source}]，
        source 標註命中哪個 API，方便除錯與引用。
        """
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
        else:  # KEYWORD_SEARCH（含 compare / summarize 的綜合查）
            for kw in kws:
                _push(sp.search_by_keyword(kw, limit=top_k), f"keyword:{kw}")
        return contexts[: top_k * len(kws)]

    # ══════════════════════════════════════════
    # 3.4 Critic：自我反思與修正迴圈
    # ══════════════════════════════════════════

    def self_reflect(
        self, question: str, draft_answer: str, contexts: List[str]
    ) -> ReflectResult:
        """檢查檢索內容是否足以回答問題（api 優先，失敗降級規則）。

        Args:
            question: 原始提問（或子問題文字）。
            draft_answer: 草稿答案（規則模式不使用，保留供 api 模式幻覺檢查）。
            contexts: 檢索到的純文字串列。

        Returns:
            ReflectResult（含覆蓋率、缺失關鍵字、是否需重試）。
        """
        texts = [c for c in (contexts or []) if c and c.strip()]
        if not texts:
            return ReflectResult(passed=False, score=0.0, need_retry=True,
                                 comment="檢索結果為空，需改寫關鍵字或關係擴張後重查")
        if self.config.mode == "api":
            try:
                raw = self._call_llm(
                    "你是 RAG 品質檢查器。只輸出 JSON 陣列單一元素："
                    '{"passed": true/false, "score": 0~1, "comment": 意見}。'
                    "上下文能回答問題才算通過。不要輸出說明文字。",
                    f"問題：{question}\n草稿：{draft_answer[:500]}\n上下文：\n"
                    + "\n---\n".join(texts[:5])[:3000],
                )
                parsed = self._parse_json_array(raw)
                if parsed:
                    d = parsed[0]
                    passed = bool(d.get("passed", False))
                    return ReflectResult(
                        passed=passed, score=float(d.get("score", 0.0) or 0.0),
                        comment=str(d.get("comment", "")),
                        need_retry=not passed, coverage=1.0 if passed else 0.0)
                logger.warning("Spark 反思回傳為空，降級為規則反思")
            except Exception as e:  # noqa: BLE001 — 降級需捕捉所有 API 異常
                logger.warning("Spark 反思失敗（%s），降級為規則反思", e)
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
        """單個子問題的檢索 + 反思重試迴圈（最多 max_attempts 次）。

        重試策略依序升級：
            第 1 次重試：改寫關鍵字（去虛詞 / 取「的」前段），路由不變。
            第 2 次重試：關係擴張（鄰居實體替代）+ 前綴放寬，
                並放寬路由為 KEYWORD_SEARCH（書名號直查除外）。
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
                keywords = _rewrite_keywords(reflection.missing or base_keywords, attempt=1)
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
    # 3.5 答案合成與端到端入口
    # ══════════════════════════════════════════

    def _synthesize(self, question: str, results: List[Dict[str, Any]]) -> str:
        """合成最終答案（api 優先用模型改寫，失敗降級規則摘錄）。

        規則摘錄保證【零幻覺】：只引用檢索原文 + 標註出處【標題·chunk_id】；
        關係步驟直接列出三元組。
        """
        if self.config.mode == "api":
            try:
                ctx_text = "\n---\n".join(
                    f"[{c['title']}·{c['chunk_id']}] {c['text'][:400]}"
                    for r in results for c in r["contexts"][:2])
                raw = self._call_llm(
                    "你是問答助手。只能根據提供的上下文回答，禁止編造；"
                    "引用時標註【標題·編號】；上下文不足就明說不知道。",
                    f"問題：{question}\n上下文：\n{ctx_text[:4000]}",
                )
                if raw.strip():
                    return raw.strip()
            except Exception as e:  # noqa: BLE001 — 合成失敗降級摘錄，不可無答案
                logger.warning("Spark 答案合成失敗（%s），降級為規則摘錄", e)
        # ── 規則摘錄 ──
        # 寒暄 / 直答題：無需檢索，直接回應（避免「查無資料」的生硬回覆）
        if results and all(r["sub"].route == RouteTarget.DIRECT_ANSWER for r in results):
            return ("你好！我是知識庫問答助手（無向量 Agentic RAG）。"
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
        """端到端問答入口：拆解 → 路由 → 檢索 → 反思重試 → 合成。

        Returns:
            {"question", "sub_questions", "contexts", "answer",
             "reflections", "total_attempts"}（皆可 JSON 序列化）。
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
                 "text": c["text"][:500], "source": c["source"]}
                for r in results for c in r["contexts"]],
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

    # 【簡易 CLI】需先用 spark_processor 建庫，再提問驗證整條 Agentic RAG 鏈路
    parser = argparse.ArgumentParser(description="Agentic RAG 問答驗證工具（無向量）")
    parser.add_argument("--question", default="孔子與孟子的關係是什麼？", help="用戶提問")
    parser.add_argument("--db", default="data/knowledge.db", help="SQLite 知識庫路徑")
    parser.add_argument("--chroma-dir", default="data/chroma", help="Chroma 持久化目錄")
    parser.add_argument("--mode", default="rule", choices=["rule", "api"], help="運作模式")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from .spark_processor import SparkConfig, SparkProcessor

    cfg = SparkConfig(mode=args.mode, db_path=args.db, chroma_dir=args.chroma_dir)
    with SparkProcessor(cfg) as sp:
        router = AgentRouter(spark_processor=sp, config=AgentConfig(mode=args.mode))
        result = router.answer(args.question)
        print(f"\n問題：{result['question']}")
        print(f"子問題：{[s['text'] for s in result['sub_questions']]}")
        print(f"反思：{result['reflections']}")
        # 【命中來源】列出前 3 筆上下文的檢索路徑，方便驗證是否走 Fallback
        for c in result["contexts"][:3]:
            print(f"命中：[{c['source']}] {c['title']}·{c['chunk_id']}")
        print(f"\n答案：\n{result['answer']}")
