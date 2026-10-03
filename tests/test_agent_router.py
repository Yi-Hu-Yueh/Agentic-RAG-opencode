"""agent_router.py 的輕量測試：拆解、路由、反思重試、端到端問答。"""

from src.agent_router import (
    AgentConfig,
    AgentRouter,
    ReflectResult,
    RouteTarget,
    SubQuestion,
    _keyword_variants,
    _sanitize_keywords,
    _strip_action_prefix,
    _strip_trailing_operators,
)


class FakeProcessor:
    """假檢索後端：可控回傳，驗證路由分派與重試邏輯（不碰真實 DB）。"""

    def __init__(self):
        self.calls = []  # 記錄 (api, key) 呼叫軌跡

    def search_by_title(self, title, limit=20):
        self.calls.append(("title", title))
        if title == "史記":
            return [{"chunk_id": "t_0", "title": "史記", "text": "史記是司馬遷所著的通史。"}]
        return []

    def search_by_entity(self, entity, limit=10):
        self.calls.append(("entity", entity))
        if entity in ("孔子", "孟子"):
            return [{"chunk_id": f"e_{entity}", "title": entity,
                     "text": f"{entity}是儒家代表人物。", "entity_type": "人物", "freq": 5}]
        return []

    def search_by_keyword(self, keyword, limit=10):
        self.calls.append(("keyword", keyword))
        # 模擬「長關鍵字查無，改寫放寬後命中」：只有短關鍵字回傳
        if len(keyword) <= 2 and keyword in ("孔子", "儒家"):
            return [{"chunk_id": f"k_{keyword}", "title": "儒家",
                     "text": "儒家是孔子的學派，孔子出生於魯國。"}]
        return []

    def get_relations(self, entity, limit=20):
        self.calls.append(("relation", entity))
        if entity in ("孔子", "孟子"):
            return [{"chunk_id": "r_0", "subj": "孔子", "predicate": "相關",
                     "obj": "孟子", "weight": 2.0, "evidence": "兩者皆為儒家人物。"}]
        return []


def _router(**kw) -> AgentRouter:
    cfg = AgentConfig(mode="rule", **kw)
    return AgentRouter(spark_processor=FakeProcessor(), config=cfg)


def test_decompose_relation_question():
    r = _router()
    subs = r.decompose_question("孔子與孟子的關係是什麼？")
    assert len(subs) == 3
    assert subs[2].kind == "relation"
    assert subs[2].depends_on == [0, 1]
    assert subs[2].route == RouteTarget.RELATION_QUERY


def test_route_targets():
    r = _router()
    assert r.route("介紹《史記》") == RouteTarget.TITLE_LOOKUP
    assert r.route("孔子與孟子的關係是什麼？") == RouteTarget.RELATION_QUERY
    assert r.route("你好") == RouteTarget.DIRECT_ANSWER
    assert r.route("孔子") == RouteTarget.ENTITY_LOOKUP


def test_self_reflect_empty_and_coverage():
    r = _router()
    bad = r.self_reflect("孔子是誰？", "", [])
    assert bad.passed is False and bad.need_retry is True
    good = r.self_reflect("孔子是誰？", "", ["孔子是儒家創始人，出生於魯國。"])
    assert good.passed is True


def test_answer_end_to_end_with_retry():
    r = _router(max_attempts=3)
    # 「孔子的學派」：首查長關鍵字落空 → 改寫為「孔子」後命中
    result = r.answer("孔子的學派是什麼？")
    assert len(result["contexts"]) > 0
    assert "孔子" in result["answer"]
    assert result["total_attempts"] >= 1
    assert all("text" in s and "route" in s for s in result["sub_questions"])


def test_answer_title_lookup():
    r = _router()
    result = r.answer("介紹《史記》")
    assert any(c["title"] == "史記" for c in result["contexts"])
    assert "史記" in result["answer"]


def test_answer_empty_question():
    r = _router()
    result = r.answer("   ")
    assert result["answer"] == "請輸入問題。"


def test_strip_action_prefix():
    """前綴剝除：多層包裝與殘留標點都要清乾淨，實體本身不受傷。"""
    assert _strip_action_prefix("查詢數學") == "數學"
    assert _strip_action_prefix("請幫我檢索孔子") == "孔子"
    assert _strip_action_prefix("探討：儒家思想") == "儒家思想"
    assert _strip_action_prefix("數學") == "數學"  # 無前綴原樣回傳
    assert _strip_action_prefix("研究生") == "研究生"  # 歧義實體不誤傷


def test_sanitize_keywords():
    """消毒：污染關鍵字還原為純實詞，全虛詞回傳空串列。"""
    assert _sanitize_keywords(["查詢數學", "檢索孔子", "探討儒家"]) == ["數學", "孔子", "儒家"]
    assert _sanitize_keywords(["查詢", "請問"]) == []


def test_retrieve_uses_clean_keyword():
    """送入檢索 API 的必須是純關鍵字（Fake 記錄呼叫軌跡驗證）。"""
    r = _router()
    sub = SubQuestion(text="查詢數學", route=RouteTarget.ENTITY_LOOKUP,
                      keywords=["查詢數學"], kind="lookup")
    r._retrieve_once(sub)
    entity_calls = [kw for api, kw in r.spark_processor.calls if api == "entity"]
    assert entity_calls, "應呼叫過 search_by_entity"
    assert all("查詢" not in kw for kw in entity_calls)
    assert "數學" in entity_calls


def test_entity_miss_falls_back_to_keyword():
    """實體完全比對落空時，同輪自動退回關鍵字全文查（source 標記 fallback）。"""
    r = _router()
    sub = SubQuestion(text="查詢儒家", route=RouteTarget.ENTITY_LOOKUP,
                      keywords=["儒家"], kind="lookup")
    ctxs = r._retrieve_once(sub)
    assert any(c["source"].startswith("entity-fallback") for c in ctxs)
    assert any("儒家" in c["text"] for c in ctxs)


def test_keyword_variants_cover_both_scripts():
    """變體展開：繁體關鍵字同時產出簡體變體（opencc 缺席時至少含原文）。"""
    variants = _keyword_variants("數學")
    assert "數學" in variants
    try:
        import opencc  # noqa: F401 — 有裝才斷言跨字形

        assert "数学" in variants
    except ImportError:
        pass


def test_strip_trailing_operators():
    """尾端運算詞剝除：殘留詞剝、內容詞保、中間詞不動。"""
    assert _strip_trailing_operators("數學的關係") == "數學"
    assert _strip_trailing_operators("血緣關係") == "血緣"
    assert _strip_trailing_operators("關係型資料庫") == "關係型資料庫"  # 首端保留
    assert _strip_trailing_operators("數學") == "數學"


def test_rule_reflect_cross_script():
    """核心 Bug 修復驗證：繁體關鍵字對簡體內文必須判定覆蓋通過。"""
    r = _router()
    res = r._rule_reflect("數學是什麼？", ["数学是研究数量的学科。"])
    assert res.passed is True
    assert res.coverage >= r.config.min_coverage
    assert res.need_retry is False


def test_rule_reflect_relation_residue():
    """殘留詞情境：測試 B 的「數學的關係」對簡體上下文應通過。"""
    r = _router()
    res = r._rule_reflect(
        "探討數學的關係",
        ["數學—相關—研究。", "数学是研究数量的学科。"],
    )
    assert res.passed is True
    assert res.missing == []
