# 本地端 Hybrid Search Agentic RAG 系統

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Ollama](https://img.shields.io/badge/Ollama-local-green)
![Tests](https://img.shields.io/badge/tests-52%2F52-brightgreen)
![Version](https://img.shields.io/badge/version-v1.0.0-navy)

> Planner → 混合檢索 → Critic 重試 → 答案合成，全程本地閉環的繁體中文知識問答系統。

- [環境與硬體](#1-環境與硬體純本地零外網算力)
- [核心亮點](#2-核心亮點)
- [系統架構](#3-系統架構)
- [專案目錄結構](#4-專案目錄結構)
- [快速啟動](#5-快速啟動)
- [測試報告](#6-測試報告5252-全綠)
- [架構文件生成](#7-架構文件生成)
- [故障排除](#8-故障排除)

---

## 1. 環境與硬體（純本地、零外網算力）

| 項目 | 規格 |
|---|---|
| 架構 | **純本地、100% 隱私防護與無外網算力**：推理、向量、儲存全在單機完成 |
| GPU | NVIDIA T1070（8GB VRAM） |
| 記憶體 | 16GB 主記憶體 |
| 推理大腦（LLM） | 本機 Ollama `qwen3:8b`（主）／`llama3.1:latest`（備援） |
| 密集向量 | 本機 Ollama `bge-m3:latest`（1024 維，HTTP 批次編碼，不載入 torch） |
| 語料 | 2GB 中文維基百科 JSONL（串流讀取，禁一次性載入） |
| 平台 | Windows／Python 3.14／Ollama 本機服務（`http://127.0.0.1:11434`） |

> **優化策略**：2GB 語料一律 Generator 串流；向量走 Ollama 常駐服務（免 torch 顯存常駐）；建庫以 32 筆一批單一交易提交，記憶體峰值可控。

---

## 2. 核心亮點

### 2.1 本地 Ollama 雙模型分工

| 職責 | 模型 | 用途 |
|---|---|---|
| 大腦推理 | `qwen3:8b` → `llama3.1:latest` | Planner JSON 拆解、Critic 關鍵字改寫、答案合成 |
| 密集向量 | `bge-m3:latest` | 建庫批次編碼＋查詢即時向量化（1024 維） |

任一環節失敗自動降級規則／純稀疏，管線永不中斷。

### 2.2 混合檢索（Hybrid Search）＋RRF

- `ThreadPoolExecutor(2)` 同時發射**稀疏腿**（SQLite：實體／標題／關鍵字 LIKE／關係三元組）與**密集腿**（ChromaDB 向量近鄰）。
- 融合公式：`Score = 1 / (60 + r_sparse) + 1 / (60 + r_dense)`，`chunk_id` 去重後取 **Top-5**。
- 五路由（TITLE／ENTITY／KEYWORD／RELATION／HYBRID）決定稀疏腿組成，密集腿一律並行；`DIRECT_ANSWER` 不檢索。

### 2.3 簡繁雙寫自癒索引

- 延遲單例 OpenCC：每筆知識寫 `orig` 原文列＋`conv` 轉換列（UPSERT 冪等，唯一鍵不放寬）。
- ChromaDB 每 Chunk 存原文＋`-trad` 雙文件向量；舊集合被預設 384 維釘住時，`_recreate_chroma` 自動重建重試（事後須完整重跑回填）。

### 2.4 大腦防護與安全網

- **Cardi 監督卡點**：Planner 輸出邊界審查——問句含關係／比較觸發詞卻缺 `RELATION_QUERY` 步時，硬編碼強制追加（含正確 `depends_on`）。
- **本機迴圈三重防護**（`_call_ollama`）：雲端環境變數隔離（呼叫結束歸還）＋ Base URL 白名單（僅 127.0.0.1／localhost／::1）＋ `trust_env=False` 獨立 Session。

---

## 3. 系統架構

```text
建庫方向
────────
loader.py ──batched_chunks(32)──▶ spark_processor.py ──單一交易UPSERT──▶ SQLite三表＋Chroma
   2GB串流/清洗/切片              實體關係抽取＋OpenCC雙寫＋bge-m3向量

問答方向
────────
提問 → Planner拆解 → ┌─稀疏腿（SQLite精準）─┐
                      │      ⚡並行           ├─▶ RRF融合Top-5 ─▶ Critic反思/重試 ─▶ 答案合成
                      └─密集腿（Chroma向量）─┘
```

---

## 4. 專案目錄結構

| 路徑 | 職責 |
|---|---|
| `src/loader.py` | 串流讀取、清洗、500–800 字切片（Generator） |
| `src/spark_processor.py` | 抽取、雙寫、Ollama 向量、自癒重建、精準＋語意檢索 API |
| `src/agent_router.py` | Planner／路由／RRF／Critic／LLM 合成、Cardi、本機防護 |
| `api/__init__.py`、`api/app.py` | FastAPI：lifespan 建處理器、`POST /api/chat`、`GET /api/health`、掛載首頁 |
| `web/index.html` | Tailwind 對話前端：歷史框、輸入列、推理軌跡手風琴 |
| `main.py` | 一鍵串聯：參數解析、建庫、統計、單問／互動問答 |
| `diagnose_db.py` | 知識庫健檢（三表筆數、空庫告警） |
| `run_manual_test.py` | 手動驗證自動化（建庫→雙組提問→深度分析） |
| `generate_docs.py` | 本架構文件生成器（輸出 `generated/`） |
| `requirements.txt` | 依賴清單（chromadb／opencc／requests／fastapi／uvicorn 等） |
| `tests/` | `test_loader`、`test_spark_processor`、`test_agent_router`、`test_diagnose_db` |
| `data/` | 語料與建庫產物（不進版控：2GB JSON、knowledge.db、chroma/） |
| `generated/` | 產出的 Word 架構文件 |

---

## 5. 快速啟動

### 5.1 安裝與前置

```bash
pip install -r requirements.txt
ollama pull bge-m3:latest
ollama pull qwen3:8b   # 備援：ollama pull llama3.1:latest
```

### 5.2 建庫＋單次問答（外部真實終端機）

```bash
# 健檢（空庫會提示建庫指令）
python diagnose_db.py

# 局部建庫 10 批（空庫自動建；全量改 --batches 0）
python main.py --batches 10 --question "数学是什么？"

# 完整參數：--data 語料路徑／--db 知識庫路徑／--mode {rule,api}
#          --batches 批次上限／--question 提問／--no-dual 關雙寫
#          （--mode api 即 Ollama LLM 模式；--no-vector 在建庫 CLI）
```

### 5.3 互動問答模式

```bash
python main.py            # 無 --question 即進入互動迴圈
# 請提問> 孔子與孟子的關係是什麼？
# 輸入 exit 或 quit 安全離開（Ctrl+C / Ctrl+Z 亦可）
```

### 5.4 網頁 UI（含思維手風琴面板）

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
# 瀏覽器開 http://127.0.0.1:8000/
# 每則回答附「🔍 Agentic 推理軌跡」手風琴：拆解步驟 → 自我反思 → 命中來源（含 rrf 分數）
```

---

## 6. 測試報告（52/52 全綠）

```bash
python -m pytest tests/ -q   # 或各測試檔直跑；LLM 3 項需本機 Ollama，在無服務環境自動跳過
```

| 模組 | 數量 | 涵蓋 |
|---|---|---|
| `test_loader` | 3 | 清洗、切片重疊、串流批次 |
| `test_spark_processor` | 14 | 三表寫入、雙寫、跨字形、向量、遷移、自癒、冪等、降級 |
| `test_agent_router` | 32 | 路由、消毒、RRF 數學、反思、Cardi、本機防護、真模型全鏈路 |
| `test_diagnose_db` | 3 | 空庫告警、健康判定、缺檔錯誤 |

---

## 7. 架構文件生成

```bash
python generate_docs.py
# → generated/專案架構文件.docx（封面＋四章＋結語，含架構圖框與 52 項測試表）
```

---

## 8. 故障排除

| 症狀 | 解法 |
|---|---|
| 知識庫 0 筆 | `python diagnose_db.py` 確認 → `python main.py --batches 10` 建庫 |
| 關鍵字 0 命中 | 動作前綴自動消毒＋三級 Fallback（實體→關鍵字→關係擴張） |
| 繁簡字形 miss | 儲存端雙寫＋查詢端變體展開＋反思變體比對 |
| 集合維度釘住（384） | `_recreate_chroma` 自癒重建後**完整重跑**回填 |
| Ollama 斷線 | 初始化探測＋逐環節降級規則，管線永不中斷 |

---

*本地端 Hybrid Search Agentic RAG 系統　v1.0.0 — 維基語料版權歸原作者，問答引用僅供學習研究。*
