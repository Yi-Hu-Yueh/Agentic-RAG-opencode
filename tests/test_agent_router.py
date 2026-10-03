"""agent_router.py 的輕量測試：拆解、路由、反思重試、端到端問答。"""

from src.agent_router import (
    AgentConfig,
    AgentRouter,
    ReflectResult,
    RouteTarget,
    SubQuestion,
    _keyword_variants,
    _rrf_fuse,
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


def _ctx(cid, source="keyword:x"):
    """構造測試用上下文（RRF 融合的最小輸入）。"""
    return {"chunk_id": cid, "title": "測", "text": f"內文{cid}", "source": source}


def test_rrf_fuse_math():
    """RRF 數學驗證：雙軌第一名融合居首，單軌 hits 亦保留。"""
    sparse = [_ctx("a"), _ctx("b"), _ctx("c")]
    dense = [_ctx("c", "dense"), _ctx("a", "dense"), _ctx("d", "dense")]
    fused = _rrf_fuse(sparse, dense, k=60, top_n=5)
    order = [c["chunk_id"] for c in fused]
    # a: 1/61+1/62, c: 1/63+1/61 → a 居首（稀疏首名＋密集次名勝過雙三名）
    assert order[0] == "a"
    assert set(order) == {"a", "b", "c", "d"}
    assert all(c["source"].startswith("rrf:") for c in fused)
    # 單軌退化：只有稀疏時按原序回傳
    assert [c["chunk_id"] for c in _rrf_fuse(sparse, [], top_n=2)] == ["a", "b"]


def test_rrf_fuse_top_n_and_dedup():
    """RRF 截斷與去重：同 chunk 多腿只留一列，超量截斷。"""
    sparse = [_ctx("a"), _ctx("a"), _ctx("b")]
    dense = [_ctx("b", "dense")]
    fused = _rrf_fuse(sparse, dense, top_n=2)
    assert [c["chunk_id"] for c in fused] == ["b", "a"]  # b 雙腿加成居首
    assert len(fused) == 2


class HybridFake:
    """帶密集能力的假後端：稀疏沿用 canned 資料，密集回固定語意命中。"""

    _ollama_ok = True  # 類別屬性：讓 router 判定密集可用

    def __init__(self):
        self.calls = []

    def search_by_keyword(self, keyword, limit=10):
        self.calls.append(("keyword", keyword))
        if keyword == "數學":
            return [{"chunk_id": "k1", "title": "数学", "text": "数学是研究数量的学科。"}]
        return []

    def search_by_entity(self, entity, limit=10):
        self.calls.append(("entity", entity))
        return []

    def search_by_title(self, title, limit=20):
        self.calls.append(("title", title))
        return []

    def get_relations(self, entity, limit=20):
        self.calls.append(("relation", entity))
        return []

    def semantic_search(self, query, limit=10):
        self.calls.append(("dense", query))
        return [{"chunk_id": "k1", "title": "数学",
                 "text": "数学是研究数量的学科。", "source": "vector", "distance": 0.5},
                {"chunk_id": "k2", "title": "物理", "text": "物理是自然科學。",
                 "source": "vector", "distance": 0.9}]


def test_hybrid_fusion_uses_both_legs():
    """混合執行：稀疏＋密集並行，RRF 融合去重（k1 雙腿命中只留一列）。"""
    r = AgentRouter(spark_processor=HybridFake(), config=AgentConfig(mode="rule"))
    sub = SubQuestion(text="查詢數學", route=RouteTarget.HYBRID_SEARCH,
                      keywords=["數學"], kind="summarize")
    ctxs = r._retrieve_once(sub)
    apis = [api for api, _ in r.spark_processor.calls]
    assert "keyword" in apis and "dense" in apis  # 雙腿皆開火
    assert [c["chunk_id"] for c in ctxs] == ["k1", "k2"]  # k1 融合居首
    assert all(c["source"].startswith("rrf:") for c in ctxs)


def test_dense_unavailable_falls_back_to_sparse():
    """密集不可用（舊 Fake 無 semantic_search）→ 純稀疏保底，行為不變。"""
    r = _router()
    sub = SubQuestion(text="查詢儒家", route=RouteTarget.HYBRID_SEARCH,
                      keywords=["儒家"], kind="summarize")
    # _router 的 FakeProcessor 只有 keyword 命中短關鍵字
    ctxs = r._retrieve_once(sub)
    assert len(ctxs) >= 1
    assert all(not c["source"].startswith("rrf:") for c in ctxs)


def test_mode_api_alias_to_llm():
    """相容：舊 mode='api' 自動視為 llm（main.py / CLI 不需改）。"""
    assert AgentConfig(mode="api").mode == "llm"
    assert AgentConfig(mode="rule").mode == "rule"


def _llm_router():
    """LLM 模式路由器（需本機 Ollama＋qwen3:8b，缺席時由呼叫端跳過）。"""
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
            import json

            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        return None
    if "qwen3:8b" not in names and "llama3.1:latest" not in names:
        return None
    return AgentRouter(spark_processor=HybridFake(),
                       config=AgentConfig(mode="llm", llm_timeout=180))


def test_llm_planner_decomposes():
    """LLM 規劃器實測：走真實模型通路，回傳結構有效的步驟即可
    （三步拆解由規則測試釘住；LLM 回傳步數具隨機性，只驗結構有效性）。"""
    r = _llm_router()
    if r is None:
        return
    subs = r.decompose_question("孔子與孟子的關係是什麼？")
    assert len(subs) >= 1
    assert all(s.text and s.keywords for s in subs)
    assert all(s.kind in ("lookup", "relation", "compare", "summarize") for s in subs)
    assert all(isinstance(s.route, RouteTarget) for s in subs)


def test_llm_rewrite_paraphrases():
    """LLM 改寫實測：缺失關鍵字應換出非空說法，失敗則保底規則。"""
    r = _llm_router()
    if r is None:
        return
    out = r._rewrite_for_retry(["孔子的學派"], "孔子的學派是什麼？")
    assert isinstance(out, list) and len(out) >= 1
    assert all(isinstance(w, str) and w for w in out)


def test_llm_end_to_end_hybrid():
    """LLM 端到端實測（HybridFake 後端）：答案非空、軌跡完整可序列化。"""
    import json

    r = _llm_router()
    if r is None:
        return
    res = r.answer("數學是什麼？")
    assert res["answer"] and len(res["contexts"]) > 0
    assert res["overall"]["passed"] is True
    json.dumps(res, ensure_ascii=False)  # 回傳必須可 JSON 序列化（API 直接可用）


def test_subs_from_llm_tolerates_wrapped_object():
    """迴歸鎖定：qwen 實測曾回傳 {"問題拆解": [{"子問題": ...}]} 包裝＋中文鍵，
    解析器必須照收並就地提煉關鍵字，不可整批丟棄。"""
    r = _router()
    raw = '{"問題拆解": [{"子問題": "數學是什麼？", "說明": "定義"}]}'
    subs = r._subs_from_llm(r._parse_json_array(raw))
    assert len(subs) == 1
    assert subs[0].text == "數學是什麼？"
    assert subs[0].keywords  # 就地提煉，不可為空


def test_parse_json_strings_tolerates_object():
    """改寫解析同理容忍 {"keywords": [...]} 包裝。"""
    r = _router()
    assert r._parse_json_strings('{"keywords": ["數學", "算術"]}') == ["數學", "算術"]
    assert r._parse_json_strings('["甲", "乙"]') == ["甲", "乙"]
    assert r._parse_json_strings("不是 json") == []


def test_parse_json_array_tolerates_bare_object():
    """單步裸物件 {"text": ...}（qwen 第二種實測形態）自動包成單步。"""
    r = _router()
    parsed = r._parse_json_array('{"text": "查詢數學", "kind": "lookup", "keywords": ["數學"]}')
    subs = r._subs_from_llm(parsed)
    assert len(subs) == 1 and subs[0].text == "查詢數學"
    assert subs[0].keywords == ["數學"] and subs[0].kind == "lookup"


def _cardi_subs():
    """Cardi 測試基底：兩個 lookup 步、無關係步（模擬 LLM 漏拆）。"""
    return [
        SubQuestion(text="查詢孔子", route=RouteTarget.ENTITY_LOOKUP,
                    keywords=["孔子"], kind="lookup"),
        SubQuestion(text="查詢孟子", route=RouteTarget.ENTITY_LOOKUP,
                    keywords=["孟子"], kind="lookup"),
    ]


def test_cardi_appends_relation_step():
    """Cardi 介入：關係觸發詞＋缺關係步 → 強制追加 RELATION_QUERY。"""
    r = _router()
    out = r._cardi_guard("孔子與孟子的關係是什麼？", _cardi_subs())
    assert len(out) == 3
    added = out[-1]
    assert added.route == RouteTarget.RELATION_QUERY
    assert added.kind == "relation"
    assert added.keywords == ["孔子", "孟子"]  # 精準鎖定兩實體
    assert added.depends_on == [0, 1]  # 依賴此前所有步驟


def test_cardi_noop_when_present():
    """Cardi 放行：已有關係步即原樣返回，不重複追加。"""
    r = _router()
    subs = _cardi_subs()
    subs.append(SubQuestion(text="查兩者關係", route=RouteTarget.RELATION_QUERY,
                            keywords=["孔子", "孟子"], kind="relation",
                            depends_on=[0, 1]))
    assert r._cardi_guard("孔子與孟子的關係是什麼？", subs) is subs
    assert len(subs) == 3


def test_cardi_compare_triggers():
    """Cardi 比較題：差別／不同／異同同樣觸發強制補步。"""
    r = _router()
    for question in ("孔子與孟子的差別是什麼？", "孔子和孟子有何不同？",
                     "孔子與孟子的異同？", "孔子相比孟子如何？"):
        out = r._cardi_guard(question, _cardi_subs())
        assert out[-1].route == RouteTarget.RELATION_QUERY, question
        assert out[-1].depends_on == [0, 1]


def test_cardi_single_entity_fallback():
    """Cardi 兜底：觸發詞在但拆不出兩實體，用問題關鍵字補步不斷流。"""
    r = _router()
    out = r._cardi_guard("數學的關係是什麼？",
                         [SubQuestion(text="查數學", route=RouteTarget.ENTITY_LOOKUP,
                                      keywords=["數學"], kind="lookup")])
    assert out[-1].route == RouteTarget.RELATION_QUERY
    assert out[-1].keywords  # 關鍵字不可空
    assert out[-1].depends_on == [0]


def test_cardi_no_trigger_noop():
    """Cardi 不誤傷：無觸發詞問句原樣放行。"""
    r = _router()
    subs = _cardi_subs()
    assert r._cardi_guard("數學是什麼？", subs) is subs
    assert len(subs) == 2


def test_llm_env_scrubbed_and_restored():
    """連線防污染：黑名單變數呼叫期隔離、結束歸還（零 lasting 副作用）。"""
    import os

    from src.agent_router import _scrub_llm_env

    os.environ["OPENAI_API_KEY"] = "sk-fake-cloud-key"
    os.environ["HTTP_PROXY"] = "http://proxy:8080"
    try:
        removed = _scrub_llm_env()
        assert removed["OPENAI_API_KEY"] == "sk-fake-cloud-key"
        assert "HTTP_PROXY" in removed
        assert "OPENAI_API_KEY" not in os.environ  # 隔離期確實移除
    finally:
        os.environ.update(removed)  # 測試內歸還（函數本體亦在 finally 歸還）
    assert os.environ["OPENAI_API_KEY"] == "sk-fake-cloud-key"
    del os.environ["OPENAI_API_KEY"]
    del os.environ["HTTP_PROXY"]


def test_ollama_base_url_locked_to_localhost():
    """位址鎖定：非本機主機名在發包前即被拒絕（fail-closed，不耗網路）。"""
    r = AgentRouter(spark_processor=None,
                    config=AgentConfig(mode="rule",
                                       ollama_base_url="http://example.com:11434"))
    try:
        r._call_ollama("sys", "hi")
        raise AssertionError("外部主機名應被拒絕")
    except RuntimeError as e:
        assert "本機迴圈" in str(e)
    assert AgentRouter(spark_processor=None).config.ollama_base_url.startswith(
        "http://127.0.0.1")  # 預設即本地迴圈
