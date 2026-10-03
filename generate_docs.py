"""專案架構文件生成腳本（generate_docs.py）

把本地端 Hybrid Search Agentic RAG 系統自動導出成 Word 文件：
    python generate_docs.py
    → 產出 generated/專案架構文件.docx

【設計說明】
- 純代碼生成、不連網：架構圖與提示框一律用 Word 表格實作
  （框線＋底色即視覺區隔），不依賴外部圖片。
- 全繁體中文註解；執行期自動建立 generated/ 目錄。
"""

# ──────────────────────────────────────────────
# 環境防護（必須在所有 import / IO 之前執行）
# ──────────────────────────────────────────────
import os
import sys

# 【關鍵】子進程沿用 UTF-8＋本次 stdio 重設，Windows cp950 終端不崩潰
os.environ.setdefault("PYTHONUTF8", "1")
for _stream_name in ("stdout", "stderr"):
    try:
        getattr(sys, _stream_name).reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 終端不支援時維持原狀即可
        pass

import datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

# ──────────────────────────────────────────────
# 全域常數：品牌色票與文件資訊（規格指定色碼）
# ──────────────────────────────────────────────
NAVY = "1B365D"  # 標題深藍色（規格指定）
BODY_GRAY = "333333"  # 內文深灰色（規格指定）
LIGHT_BLUE = "DEEAF6"  # 提示框／步驟框淺藍底
LIGHT_GRAY = "EDF2F9"  # 表格斑馬紋淺灰藍
AMBER = "FFF2CC"  # 並行節點鵝黃底
BORDER_GRAY = "BFBFBF"  # 表格框線灰
FONT_NAME = "微軟正黑體"  # 內文字型（規格指定）
FONT_SIZE_BODY = 11  # 內文字級（規格指定 11pt）
LINE_SPACING = 1.25  # 內文行距（規格指定 1.25 倍）

AUTHOR = "Yi-Hu Yueh"  # 作者（與 Git 身份一致）
VERSION = "v1.0.0"  # 文件版本（規格指定）
DOC_DATE = datetime.date.today().isoformat()  # 出版日期（執行當天自動填入）
OUTPUT_PATH = Path("generated/專案架構文件.docx")  # 輸出路徑（規格指定）


# ══════════════════════════════════════════════
# 一、底層排版工具（字型／底色／框線）
# ══════════════════════════════════════════════


def set_run_font(run, name=FONT_NAME, size=FONT_SIZE_BODY,
                 color=BODY_GRAY, bold=False):
    """設定單個文字 run 的字型（含中日韓 eastAsia，否則 Word 用 fallback 字）。"""
    run.font.name = name  # 同時寫入 ascii / hAnsi
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)  # 【關鍵】CJK 字型否則標題中文變新細明體
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def config_styles(doc):
    """配置全文件樣式：內文 11pt 深灰 1.25 倍行距，標題深藍。"""
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = Pt(FONT_SIZE_BODY)
    normal.font.color.rgb = RGBColor.from_string(BODY_GRAY)
    normal.paragraph_format.line_spacing = LINE_SPACING  # 【規格】1.25 倍行距
    normal.paragraph_format.space_after = Pt(6)
    # 標題 1/2：深藍＋段前空行（規格：章節標題前加空行）
    for style_name, size in (("Heading 1", 18), ("Heading 2", 14)):
        style = doc.styles[style_name]
        style.font.name = FONT_NAME
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(NAVY)
        style.paragraph_format.space_before = Pt(18)  # 【規格】標題前空行
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing = 1.0


def add_body(doc, text, bold=False, size=FONT_SIZE_BODY):
    """加入一段內文（自動套用全域字型色與行距）。"""
    para = doc.add_paragraph()
    run = para.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return para


def shade_cell(cell, fill):
    """設定儲存格底色（標題列深藍／斑馬紋／提示框皆用此函數）。"""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_table_borders(table, color=BORDER_GRAY, size="6"):
    """設定表格六邊框線（上下左右＋內部格線，精美表格必備）。"""
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)
        borders.append(element)
    tbl_pr.append(borders)


def set_col_widths(table, widths):
    """固定各欄寬（關閉自動適應，中文表格不爆版的關鍵）。"""
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths):
            if idx < len(row.cells):
                row.cells[idx].width = width


def fill_cell(cell, lines, align=WD_ALIGN_PARAGRAPH.LEFT, size=10,
              color=BODY_GRAY, shading=None, bold_first=False):
    """填儲存格內容：lines 為 [(文字, 粗體)] 或純字串串列。

    Args:
        cell: 目標儲存格（會先清空）。
        lines: 每行內容；元組表 (文字, 是否粗體)，字串表純內文。
        align: 對齊方式；表格內文預設 10pt 以容納多欄。
        shading: 底色 Hex（如 NAVY / LIGHT_BLUE），None 表不填色。
        bold_first: 首行是否粗體（標題列用）。
    """
    cell.text = ""  # 清空預設空段落，避免首行空白
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if shading:
        shade_cell(cell, shading)
    for i, item in enumerate(lines):
        text, bold = item if isinstance(item, tuple) else (item, bold_first and i == 0)
        para = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        para.alignment = align
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.line_spacing = 1.0
        set_run_font(para.add_run(text), size=size, color=color, bold=bold)


def add_styled_table(doc, headers, rows, widths, header_bg=NAVY):
    """產出精美表格：深藍標題列白字＋斑馬紋＋完整框線。

    Args:
        doc: 文件物件。
        headers: 標題列字串串列。
        rows: 資料列（每列為字串串列，與 headers 等長）。
        widths: 每欄寬（Inches），總和勿超過 6.27 吋（A4 可用寬）。
        header_bg: 標題列底色（規格：深藍）。

    Returns:
        建立的 Table 物件。
    """
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    set_col_widths(table, widths)
    for j, header in enumerate(headers):
        fill_cell(table.cell(0, j), [(header, True)], align=WD_ALIGN_PARAGRAPH.CENTER,
                  size=10, color="FFFFFF", shading=header_bg)
    for i, row in enumerate(rows):
        band = LIGHT_GRAY if i % 2 == 1 else None  # 【斑馬紋】奇偶列交替底色
        for j, value in enumerate(row):
            fill_cell(table.cell(i + 1, j), [value], size=10, shading=band)
    set_table_borders(table)
    doc.add_paragraph()  # 表後空行，呼吸感
    return table


def add_callout(doc, title, body):
    """技術精華提示方塊：淺藍底單格表格，標題粗體深藍＋內文。

    （純代碼環境無法插圖，此即「圖文並茂」的文字區隔框解法。）
    """
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    fill_cell(cell, [(title, True), body], size=10, shading=LIGHT_BLUE)
    # 標題行改深藍字以呼應品牌色
    set_run_font(cell.paragraphs[0].runs[0], size=10, color=NAVY, bold=True)
    set_table_borders(table, color=NAVY)
    doc.add_paragraph()
    return table


# ══════════════════════════════════════════════
# 二、封面與頁尾
# ══════════════════════════════════════════════


def add_cover(doc):
    """封面：專案名稱＋副標＋作者/日期/版本置中，大標深藍 28pt。"""
    for _ in range(3):
        doc.add_paragraph()  # 頂部留白，大標沉底置中
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(title.add_run("本地端 Hybrid Search\nAgentic RAG 系統"),
                 size=28, color=NAVY, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(subtitle.add_run("專案架構文件"), size=16, color=NAVY, bold=True)
    doc.add_paragraph()
    for label, value in (("作　　者", AUTHOR), ("出版日期", DOC_DATE),
                         ("版　　本", VERSION), ("運行環境", "Windows／Python 3.14／Ollama 本機")):
        line = doc.add_paragraph()
        line.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(line.add_run(f"{label}：{value}"), size=12)
    doc.add_page_break()  # 封面獨立成頁


def add_footer(doc):
    """頁尾：每頁置中顯示系統名與版本（專業文件標配）。"""
    footer = doc.sections[0].footer
    para = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(para.add_run(f"本地端 Hybrid Search Agentic RAG 系統　{VERSION}"),
                 size=9, color="808080")


# ══════════════════════════════════════════════
# 三、視覺化架構圖框（表格實作，字型無關不斷位）
# ══════════════════════════════════════════════


def _arrow_row(table, row_idx, label):
    """流程圖的箭頭列：置中 ▼＋說明字，白底。"""
    fill_cell(table.cell(row_idx, 0), [f"▼ {label}"],
              align=WD_ALIGN_PARAGRAPH.CENTER, size=10, color=NAVY)


def _step_cell(cell, title, desc):
    """流程圖的步驟框：淺藍底＋深藍粗體標題＋說明內文。"""
    fill_cell(cell, [(title, True), desc], align=WD_ALIGN_PARAGRAPH.CENTER,
              size=10, shading=LIGHT_BLUE)
    set_run_font(cell.paragraphs[0].runs[0], size=11, color=NAVY, bold=True)


def add_diagram_a(doc):
    """圖 A：Loader → Processor(bge-m3) → DB 雙寫的資料管線流程圖。

    單欄五列表格：步驟框（有框線底色）與箭頭列交錯，
    純表格排版故任何字型下皆不斷位（ASCII 藝術在比例字下必歪，此法免疫）。
    """
    add_body(doc, "圖 A：資料管線流程（建庫方向）", bold=True)
    table = doc.add_table(rows=5, cols=1)
    set_col_widths(table, [Inches(6.27)])
    _step_cell(table.cell(0, 0), "① 資料擷取層　src/loader.py",
               "2GB JSONL 串流讀取（禁一次性載入）・清洗 wiki 標記・"
               "500–800 字切片＋10% 重疊")
    _arrow_row(table, 1, "batched_chunks(batch_size=32)　低記憶體逐批吐出")
    _step_cell(table.cell(2, 0), "② 知識豐富層　src/spark_processor.py",
               "Muse Spark／規則抽取實體關係 ＋ OpenCC 繁簡變體 ＋ "
               "Ollama bge-m3 1024 維向量編碼")
    _arrow_row(table, 3, "單一交易 UPSERT（失敗整批 rollback，冪等可重跑）")
    _step_cell(table.cell(4, 0), "③ 雙寫儲存層　SQLite＋ChromaDB",
               "SQLite 3 表（orig／conv 雙字形列）＋ Chroma 原文／-trad 雙文件向量")
    set_table_borders(table)
    doc.add_paragraph()


def add_diagram_b(doc):
    """圖 B：提問 → 雙軌併發 → RRF 融合的分流架構圖。

    三欄表格＋合併儲存格實作分流／合流：中欄為並行節點（鵝黃底），
    合併列為主幹流程（淺藍底），箭頭列置中導引視線。
    """
    add_body(doc, "圖 B：混合檢索分流架構（問答方向）", bold=True)
    table = doc.add_table(rows=7, cols=3)
    set_col_widths(table, [Inches(2.64), Inches(0.99), Inches(2.64)])
    for merge_row in (0, 1, 3, 4, 5, 6):  # 主幹列合併為通欄
        table.cell(merge_row, 0).merge(table.cell(merge_row, 2))
    _step_cell(table.cell(0, 0), "用戶提問 → AgentRouter.answer()",
               "Planner 拆解子問題（含依賴序）")
    _arrow_row(table, 1, "ThreadPoolExecutor(max_workers=2) 雙軌並行")
    _step_cell(table.cell(2, 0), "稀疏腿　SQLite 精準檢索",
               "實體／標題／關鍵字 LIKE／關係三元組")
    fill_cell(table.cell(2, 1), [("⚡ 並行", True)], align=WD_ALIGN_PARAGRAPH.CENTER,
              size=11, shading=AMBER)
    set_run_font(table.cell(2, 1).paragraphs[0].runs[0],
                 size=11, color=NAVY, bold=True)
    _step_cell(table.cell(2, 2), "密集腿　Chroma 向量語意",
               "bge-m3 即時編碼 query → 顯式向量近鄰")
    _arrow_row(table, 3, "RRF 融合　Score = 1/(60+r₁) ＋ 1/(60+r₂)　取 Top-5")
    _step_cell(table.cell(4, 0), "Critic 跨字形反思",
               "繁簡變體覆蓋率 ≥ 門檻即通過；不足則 LLM 改寫／關係擴張重試（≤3 輪）")
    _arrow_row(table, 5, "答案合成（含引用標註）")
    _step_cell(table.cell(6, 0), "回傳答案＋反思軌跡（前端手風琴展開）",
               "POST /api/chat → web/index.html")
    set_table_borders(table)
    doc.add_paragraph()


# ══════════════════════════════════════════════
# 四、文件內容資料（目錄結構／測試清單／問答集）
# ══════════════════════════════════════════════

# 【目錄結構表】路徑＋職責一行說明（data 下為不進版控的大檔，特別標註）
DIR_ROWS = [
    ("api/__init__.py", "API 套件標記"),
    ("api/app.py", "FastAPI 服務：lifespan 建處理器、POST /api/chat、GET /api/health、掛載首頁"),
    ("src/__init__.py", "核心套件匯出（loader／processor／router 全量匯出）"),
    ("src/loader.py", "2GB JSONL 串流讀取、清洗、500–800 字切片（Generator 零 OOM）"),
    ("src/spark_processor.py", "實體關係抽取、OpenCC 雙寫、Ollama bge-m3 1024 維、SQLite 三表＋Chroma"),
    ("src/agent_router.py", "Planner／五路由＋HYBRID／ThreadPoolExecutor／RRF／Critic／LLM 合成"),
    ("web/index.html", "Tailwind 對話前端：歷史框、輸入列、推理軌跡手風琴"),
    ("main.py", "一鍵串聯：參數解析、建庫、統計、單問／互動問答、UTF-8 防護"),
    ("diagnose_db.py", "知識庫健檢：三表筆數、空庫告警、cp950 終端韌性"),
    ("run_manual_test.py", "手動驗證自動化：建庫→雙組提問→深度分析（含簡繁對照探針）"),
    ("generate_docs.py", "本文件生成器（python-docx，執行即輸出 docx）"),
    ("requirements.txt", "依賴清單（chromadb／opencc／requests／fastapi／uvicorn 等）"),
    ("tests/test_loader.py", "loader 單元測試（3 項）"),
    ("tests/test_spark_processor.py", "processor 單元測試（雙寫／向量／遷移／冪等）"),
    ("tests/test_agent_router.py", "router 單元測試（路由／RRF／反思／LLM 實測）"),
    ("tests/test_diagnose_db.py", "診斷腳本測試（空庫／健康／缺檔）"),
    ("data/*.json", "2GB 維基語料（不進版控，.gitignore 排除）"),
    ("data/knowledge.db", "建庫產物 SQLite（320 chunks 實測，不進版控）"),
    ("data/chroma/", "Chroma 持久化索引（639 文件，不進版控）"),
    ("generated/", "本文件輸出目錄（執行自動建立）"),
]

# 【測試清單】（模組，案例，驗證重點）：以全綠實測為準，Cardi 合流後共 52 項
TEST_ROWS = [
    ("loader", "test_clean_text_removes_markup", "wiki 標記與 HTML 清洗正確性"),
    ("loader", "test_split_text_overlap", "切片長度區間與重疊率"),
    ("loader", "test_stream_and_batch", "串流生成器逐批吐出不斷流"),
    ("processor", "test_rule_extract_entities_and_relations", "規則實體關係抽取基本盤"),
    ("processor", "test_enrich_batch_writes_three_tables", "三表寫入＋三路精準檢索"),
    ("processor", "test_enrich_batch_idempotent", "同批重跑筆數不變"),
    ("processor", "test_dual_entities_written", "同一實體 orig／conv 雙列並存"),
    ("processor", "test_cross_script_entity_search", "繁簡任一字形實體命中"),
    ("processor", "test_cross_script_keyword_search", "關鍵字跨字形命中＋去重"),
    ("processor", "test_cross_script_relations", "關係端點跨字形命中"),
    ("processor", "test_dual_idempotent", "雙寫開啟下重跑冪等"),
    ("processor", "test_migration_old_db_without_script_col", "舊庫 ALTER 補欄＋回填"),
    ("processor", "test_no_vector_mode_writes_text_only", "--no-vector 純文字降級可用"),
    ("processor", "test_ollama_unreachable_degrades", "Ollama 斷線建庫不中斷"),
    ("processor", "test_vector_embeddings_written", "1024 維向量全文件寫入"),
    ("processor", "test_vector_dim_mismatch_recreates", "384 釘住集合自癒重建"),
    ("processor", "test_vector_idempotent", "向量開啟下重跑冪等"),
    ("router", "test_answer_empty_question", "空問題擋下提示輸入"),
    ("router", "test_answer_end_to_end_with_retry", "端到端重試鏈路有答案"),
    ("router", "test_answer_title_lookup", "書名號條目精準命中"),
    ("router", "test_decompose_relation_question", "關係題三步拆解＋依賴序"),
    ("router", "test_dense_unavailable_falls_back_to_sparse", "無密集腿退回純稀疏"),
    ("router", "test_entity_miss_falls_back_to_keyword", "實體落空關鍵字兜底"),
    ("router", "test_hybrid_fusion_uses_both_legs", "雙腿並行＋RRF 融合排序"),
    ("router", "test_keyword_variants_cover_both_scripts", "繁簡變體展開正確"),
    ("router", "test_llm_end_to_end_hybrid", "真模型全鏈路＋回傳可序列化"),
    ("router", "test_llm_planner_decomposes", "真模型規劃步驟結構有效"),
    ("router", "test_llm_rewrite_paraphrases", "真模型換句話說非空"),
    ("router", "test_mode_api_alias_to_llm", "舊 mode=api 相容對映"),
    ("router", "test_parse_json_array_tolerates_bare_object", "單步裸物件自動包步"),
    ("router", "test_parse_json_strings_tolerates_object", "改寫包裝物件容忍"),
    ("router", "test_retrieve_uses_clean_keyword", "消毒後關鍵字進檢索 API"),
    ("router", "test_route_targets", "五路由分派正確"),
    ("router", "test_rrf_fuse_math", "RRF 分數公式排序驗證"),
    ("router", "test_rrf_fuse_top_n_and_dedup", "腿內去重＋Top-N 截斷"),
    ("router", "test_rule_reflect_cross_script", "繁問簡文覆蓋判定通過"),
    ("router", "test_rule_reflect_relation_residue", "殘留詞還原後覆蓋通過"),
    ("router", "test_sanitize_keywords", "污染詞消毒成純實詞"),
    ("router", "test_self_reflect_empty_and_coverage", "空結果重試＋覆蓋計分"),
    ("router", "test_strip_action_prefix", "動作前綴剝除不傷實體"),
    ("router", "test_strip_trailing_operators", "尾端運算詞剝除保內容"),
    ("router", "test_subs_from_llm_tolerates_wrapped_object", "包裝物件＋中文鍵容忍"),
    ("router", "test_cardi_appends_relation_step", "漏關係步強制追加＋依賴正確"),
    ("router", "test_cardi_noop_when_present", "已有關係步不重複追加"),
    ("router", "test_cardi_compare_triggers", "比較／異同觸發詞同樣補步"),
    ("router", "test_cardi_single_entity_fallback", "拆不出雙實體用關鍵字兜底"),
    ("router", "test_cardi_no_trigger_noop", "無觸發詞問句原樣放行"),
    ("router", "test_llm_env_scrubbed_and_restored", "雲端變數隔離＋呼叫結束歸還"),
    ("router", "test_ollama_base_url_locked_to_localhost", "非本機主機名發包前拒絕"),
    ("diagnose", "test_empty_db_reports_need_build", "空庫回報需建庫"),
    ("diagnose", "test_healthy_db_reports_ok", "健康庫判定通過"),
    ("diagnose", "test_missing_db_reports_error", "缺檔錯誤碼正確"),
]

# 【檢索演進對照】三代架構一行對比（第一章用）
EVOLVE_ROWS = [
    ("v0 無向量", "SQLite LIKE／實體完全比對", "零依賴、精準但無語意"),
    ("v1 雙寫索引", "＋OpenCC 繁簡雙列／雙文件", "跨字形命中，仍無語意"),
    ("v2 混合檢索", "＋Ollama bge-m3 1024維／RRF Top-5", "語意＋精準雙軌融合（現行）"),
]

# 【故障排除】已實裝的防護對應表（第四章用）
TROUBLE_ROWS = [
    ("知識庫 0 筆", "diagnose_db.py 空庫告警 → python main.py --batches 10 建庫"),
    ("關鍵字 0 命中", "動作前綴消毒＋三級 Fallback（實體→關鍵字→關係擴張）"),
    ("繁簡字形 miss", "儲存端雙寫＋查詢端變體展開＋反思變體比對"),
    ("集合維度釘住", "_recreate_chroma 自癒重建＋完整重跑回填（警告明示）"),
    ("Ollama 斷線", "初始化探測＋逐環節降級規則，管線永不中斷"),
]


# ══════════════════════════════════════════════
# 五、章節組裝
# ══════════════════════════════════════════════


def add_chapter_1(doc):
    """第一章：專案背景與硬體限制（目標、資料、演進三代對照）。"""
    doc.add_heading("第一章　專案背景與硬體限制", level=1)
    add_body(doc, "本系統為本地端繁體中文知識問答系統：以 2GB 中文維基百科 JSONL "
                  "為語料，經清洗切片、實體關係抽取、雙寫索引後，透過 Agentic "
                  "多步推理（規劃→檢索→反思→合成）回答用戶提問，全程可在單機完成。")
    doc.add_heading("1.1 硬體配置與優化策略", level=2)
    add_body(doc, "目標機器：NVIDIA T1070（8GB VRAM）＋ 16GB 主記憶體。對應優化："
                  "（1）2GB 語料禁一次性載入，一律 Generator 串流；（2）向量模型選用 "
                  "Ollama bge-m3（服務常駐、與本地 LLM 共存、免 torch 顯存常駐）；"
                  "（3）建庫批次 32 為單位交易，重跑冪等，記憶體峰值可控。")
    doc.add_heading("1.2 檢索架構三代演進", level=2)
    add_body(doc, "系統從精準檢索起步，依實測痛點兩度升級（下表），現行 v2 為 "
                  "RRF 混合檢索：稀疏精準與密集語意雙軌並行融合。")
    add_styled_table(doc, ["版本", "檢索手段", "特性"],
                     EVOLVE_ROWS, [Inches(1.2), Inches(2.6), Inches(2.47)])
    add_callout(doc, "【重點】為何不用純向量？",
                "中文維基條目名、專有名詞最適合完全比對；純語意檢索易漂移。"
                "本系統以 SQLite 結構化圖譜為權威來源，向量只負責語意召回，"
                "兩者經 RRF 融合互補，這正是混合檢索的設計原點。")


def add_chapter_2(doc):
    """第二章：雙軌資料庫設計（建庫管線圖＋三表＋向量＋自癒）。"""
    doc.add_heading("第二章　雙軌資料庫設計", level=1)
    add_body(doc, "建庫管線（loader → processor → 雙寫儲存）如下圖，10 批次局部建庫實測："
                  "320 chunks、10,898 實體、17,969 關係、Chroma 639 文件。")
    add_diagram_a(doc)
    doc.add_heading("2.1 SQLite 三表（繁簡雙寫）", level=2)
    add_body(doc, "chunks（切片原文，單列不雙寫）、entities（實體＋型別＋次數）、"
                  "relations（主謂賓＋權重＋證據）。entities／relations 各帶 script 欄 "
                  "（orig／conv），同一知識存繁簡兩列；唯一鍵刻意不放寬——字形不同即不同列值，"
                  "天然共存且重跑冪等。舊庫以 ALTER TABLE 補欄遷移，重跑即回填。")
    doc.add_heading("2.2 ChromaDB 1024 維密集向量索引", level=2)
    add_body(doc, "每 Chunk 存原文文件與 -trad 轉換文件，經 Ollama bge-m3 編碼攜 1024 維向量 "
                  "upsert；Metadata 標 script／base_id，去重時原文列優先。查詢端 where／"
                  "where_document 精準路徑與顯式向量 query 並存。")
    add_callout(doc, "【重點】_recreate_chroma 自癒機制",
                "Chroma 集合維度在首次寫入即被釘住（如早期純文字時代的預設 384 維），"
                "之後不同維度 upsert 必報錯。寫入層偵測維度報錯即自動刪建重試；"
                "重建後須完整重跑管線回填（程式以 WARNING 明示），局部重建會遺失批外文件。")


def add_chapter_3(doc):
    """第三章：大腦路由與 RRF 混合檢索（分流架構圖＋公式＋防護）。"""
    doc.add_heading("第三章　大腦路由與 RRF 混合檢索演算法", level=1)
    add_body(doc, "AgentRouter 收到提問後依序執行 Planner（拆解）→ 混合檢索 → "
                  "Critic（反思重試）→ 答案合成，全程如下圖。")
    add_diagram_b(doc)
    doc.add_heading("3.1 五路由＋HYBRID 與併發雙腿", level=2)
    add_body(doc, "稀疏腿按 TITLE／ENTITY／KEYWORD／RELATION／HYBRID 分派對應 SQLite API"
                  "（附三級 Fallback）；密集腿以 bge-m3 即時編碼查詢打 Chroma 語意近鄰。"
                  "雙腿經 ThreadPoolExecutor(max_workers=2) 並行，單腿異常回空不拖垮另一腿；"
                  "密集不可用時整體退回純稀疏（原無向量機制完整保留）。")
    doc.add_heading("3.2 RRF 相互排名融合", level=2)
    add_body(doc, "雙軌名次（1-based）依公式 Score = 1／(60＋r_sparse)＋1／(60＋r_dense) "
                  "加總，chunk_id 去重（腿內重複只取最佳名次），取前 5 筆；"
                  "每筆保留雙軌名次與 rrf:分數來源標，供前端軌跡面板展示融合依據。")
    add_callout(doc, "【重點】Cardi 規則監督卡點（✅ 本地已實作完工）",
                "目的：防 LLM 隨機性漏掉最重要的 relations 圖譜檢索。Planner 輸出邊界經 "
                "_cardi_guard 審查：問句命中觸發詞（關係、關聯、相比、比較、差別、不同、"
                "異同等簡繁詞）但步驟缺 RELATION_QUERY 時，硬編碼強制追加針對兩實體的"
                "關係檢索步，depends_on 指向此前所有步驟。另 _call_ollama 具三重本機迴圈防護："
                "環境變數隔離（呼叫結束歸還）、Base URL 白名單鎖定、trust_env=False 獨立會話。")
    doc.add_heading("3.3 跨字形反思防護", level=2)
    add_body(doc, "Critic 以覆蓋率評分：關鍵字先消毒、剝尾端運算詞（如「數學的關係」→「數學」），"
                  "再展開繁簡變體比對，任一字形命中即算覆蓋，終結簡繁不一致的 0% 誤判；"
                  "不足時先關係擴張，再請 LLM 換句話說改寫，最多 3 輪。")


def add_chapter_4(doc):
    """第四章：進入點、測試報告與故障排除。"""
    doc.add_heading("第四章　全系統進入點與測試報告", level=1)
    doc.add_heading("4.1 系統目錄結構", level=2)
    add_body(doc, "全專案檔案一覽（data 下為不進版控的大檔與建庫產物）：")
    add_styled_table(doc, ["路徑", "職責說明"],
                     DIR_ROWS, [Inches(2.2), Inches(4.07)])
    doc.add_heading("4.2 進入點速查", level=2)
    add_body(doc, "一鍵串聯：python main.py（--question 單問／無參數互動，--batches 局部建庫）；"
                  "建庫：python -m src.spark_processor（--no-vector／--no-dual 降級開關）；"
                  "問答驗證：python -m src.agent_router；Web 服務：uvicorn api.app:app "
                  "（首頁對話前端＋POST /api/chat＋GET /api/health）；健檢：python diagnose_db.py；"
                  "手動驗證：python run_manual_test.py。")
    doc.add_heading("4.3 52／52 完整測試通過報告", level=2)
    add_body(doc, "以下為全量測試清單（2026-10-03 實測全綠，含 LLM 真模型實測 3 項；"
                  "無 Ollama 環境時 LLM 項自動跳過）：")
    numbered = [[str(i + 1), mod, name, focus]
                for i, (mod, name, focus) in enumerate(TEST_ROWS)]
    add_styled_table(doc, ["編號", "模組", "測試案例", "驗證重點"],
                     numbered, [Inches(0.5), Inches(1.0), Inches(2.6), Inches(2.17)])
    add_callout(doc, "【測試結論】52／52 全數通過",
                "涵蓋串流讀取、雙寫冪等、跨字形命中、向量寫入、自癒重建、"
                "混合融合數學、反思計分、真模型規劃／改寫／端到端全鏈路、"
                "Cardi 強制補步、本機迴圈三重防護。")
    doc.add_heading("4.4 故障排除速查", level=2)
    add_body(doc, "實測沉澱的已知坑與已實裝解法：")
    add_styled_table(doc, ["症狀", "解法（已實裝）"],
                     TROUBLE_ROWS, [Inches(1.8), Inches(4.47)])


def add_closing(doc):
    """結語與免責聲明。"""
    doc.add_heading("結語", level=1)
    add_body(doc, "本系統證明：單機 8GB 顯存亦可跑通 Planner→混合檢索→Critic 的完整 "
                  "Agentic RAG 閉環——關鍵不在堆硬體，而在分層降級（規則保底、純稀疏保底、"
                  "純文字保底）讓每一層失敗都不拖垮管線。後續可選方向：Cardi 卡點程式落地、"
                  "全量 2GB 建庫、前端展示 rrf 融合分數。")
    doc.add_heading("免責聲明", level=2)
    add_body(doc, "本文件由 generate_docs.py 自動生成，數據取自 10 批次局部索引實測值；"
                  "全量建庫數值將等比放大。維基語料版權歸原作者，問答引用僅供學習研究；"
                  "本地 LLM 輸出僅供參考，重要決策請查證原始條目。")


# ══════════════════════════════════════════════
# 六、主流程（組裝＋存檔閉環）
# ══════════════════════════════════════════════


def build_document():
    """組裝全文並回傳 Document 物件（存檔由主流程執行，方便單元引用）。"""
    doc = Document()
    config_styles(doc)
    add_cover(doc)
    add_chapter_1(doc)
    add_chapter_2(doc)
    add_chapter_3(doc)
    add_chapter_4(doc)
    add_closing(doc)
    add_footer(doc)
    return doc


def main():
    """主流程：建目錄 → 組裝 → 存檔 → 回報（單一入口閉環）。"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 自動建立 generated/
    doc = build_document()
    try:
        doc.save(str(OUTPUT_PATH))  # 【閉環】存檔即完成，無後續手動步驟
        final = OUTPUT_PATH
    except PermissionError:
        # 【佔用退路】舊檔在 Word 開啟中被鎖定時，改存時間戳版本，不覆寫失敗
        stamped = OUTPUT_PATH.with_stem(
            f"{OUTPUT_PATH.stem}_{datetime.datetime.now():%Y%m%d-%H%M%S}")
        doc.save(str(stamped))
        print(f"舊檔被佔用，已另存新檔（請關閉 Word 後重整）：{stamped.name}")
        final = stamped
    print(f"已產出：{final.resolve()}（{final.stat().st_size:,} bytes）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
