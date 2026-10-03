"""Web API 服務（api/app.py）

把無向量 Agentic RAG 包裝成 HTTP 服務：
    啟動：uvicorn api.app:app --host 127.0.0.1 --port 8000
    首頁：http://127.0.0.1:8000/（web/index.html 對話前端）
    問答：POST /api/chat {"question": "..."}
    健康：GET /api/health（知識庫筆數，供前端狀態列使用）

啟動時初始化本地 SparkProcessor（data/knowledge.db）與 AgentRouter，
關機時釋放 SQLite 連線；問答端點為同步函式（FastAPI 自動丟執行緒池），
避免阻塞事件循環。
"""

# ──────────────────────────────────────────────
# 環境防護（必須在所有 import / IO 之前執行）
# ──────────────────────────────────────────────
import os
import sys

# 【關鍵】寫入環境變數（惠及子進程）＋直接重設 stdio 編碼（本次生效），
# Windows 預設 cp950 終端印中文會崩潰，兩者缺一不可。
os.environ.setdefault("PYTHONUTF8", "1")
for _stream_name in ("stdout", "stderr"):
    try:
        getattr(sys, _stream_name).reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 終端不支援時維持原狀即可
        pass

# 【路徑相容】直接跑 python api/app.py 時，把專案根目錄補進 sys.path，
# 讓 `from src...` 與 `uvicorn api.app:app` 兩種啟動方式都能 import 成功。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent_router import AgentConfig, AgentRouter
from src.spark_processor import SparkConfig, SparkProcessor

# ──────────────────────────────────────────────
# 常數：與各模組預設值對齊
# ──────────────────────────────────────────────
DB_PATH = "data/knowledge.db"  # SQLite 知識庫（規格指定路徑）
CHROMA_DIR = "data/chroma"  # ChromaDB 持久化目錄
WEB_DIR = os.path.join(_ROOT, "web")  # 前端靜態目錄
MODE = os.getenv("AGENT_MODE", "rule")  # 抽取模式（環境變數可切 api）
CTX_PREVIEW_LEN = 200  # 回傳上下文預覽長度（控制 payload 大小）

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ══════════════════════════════════════════════
# 一、請求／回應模型
# ══════════════════════════════════════════════


class ChatRequest(BaseModel):
    """問答請求：前端送 {"question": "..."}。"""

    question: str = Field(..., min_length=1, description="用戶提問（不可為空）")


class ChatResponse(BaseModel):
    """問答回應：最終答案＋完整反思軌跡（前端 Accordion 渲染用）。"""

    question: str  # 原始提問（回音，方便除錯）
    answer: str  # 最終答案（含引用標註）
    sub_questions: List[Dict[str, Any]]  # 拆解步驟（text/route/kind/keywords）
    contexts: List[Dict[str, Any]]  # 命中上下文（chunk_id/title/source/預覽）
    reflections: List[Dict[str, Any]]  # 每步反思（passed/score/attempts）
    overall: Dict[str, Any]  # 整體自評（passed/score/comment）
    total_attempts: int  # 累計檢索次數


class HealthResponse(BaseModel):
    """健康檢查：服務存活＋知識庫筆數（前端狀態列用）。"""

    status: str  # "ok" 表服務正常
    chunks: int  # Chunk 總數（0 表空庫，問答必定落空）
    entities: int  # 實體總數（含繁簡雙列）
    relations: int  # 關係總數（含繁簡雙列）


# ══════════════════════════════════════════════
# 二、應用生命週期：啟動建處理器，關機釋放連線
# ══════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ lifespan：啟動時建 SparkProcessor＋AgentRouter，關機時 close()。

    重型初始化只做一次（ tvo 每次請求重建 DB 連線），
    空庫只記警告不擋啟動（前端 health 可見 0 筆，問答會誠實回查無資料）。
    """
    processor = SparkProcessor(SparkConfig(db_path=DB_PATH, chroma_dir=CHROMA_DIR))
    try:
        stats = processor.stats()
        logger.info("知識庫載入：%s", stats)
        if stats.get("chunks", 0) == 0:
            logger.warning("知識庫為空！請先執行 python main.py --batches 10 建庫。")
    except Exception as e:  # noqa: BLE001 — 統計失敗不擋啟動，請求時再報錯
        logger.warning("知識庫統計失敗（%s）", e)
    app.state.processor = processor
    app.state.router = AgentRouter(processor, config=AgentConfig(mode=MODE))
    logger.info("Agentic RAG API 就緒（mode=%s）。", MODE)
    yield
    # 【關機】釋放 SQLite 連線（ChromaDB 為持久化客戶端，無需顯式關閉）
    try:
        processor.close()
    except Exception:  # noqa: BLE001 — 關機期錯誤僅記錄，不影響退出碼
        logger.exception("關機釋放連線失敗")


app = FastAPI(title="Agentic RAG API（無向量＋雙寫索引）", lifespan=lifespan)


# ══════════════════════════════════════════════
# 三、API 端點（先註冊，靜態掛載放最後才不會蓋掉 /api/*）
# ══════════════════════════════════════════════


@app.get("/api/health", response_model=HealthResponse)
def health() -> Dict[str, Any]:
    """健康檢查：回傳服務狀態與知識庫三表筆數。"""
    processor: SparkProcessor = app.state.processor
    stats = processor.stats()
    return {
        "status": "ok",
        "chunks": int(stats.get("chunks", 0)),
        "entities": int(stats.get("entities", 0)),
        "relations": int(stats.get("relations", 0)),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> Dict[str, Any]:
    """問答端點：同步函式（FastAPI 自動丟執行緒池，不阻塞事件循環）。

    流程：空問題擋下 → router.answer 全鏈路 → 上下文截斷後回傳。
    檢索/反思tick 皆在 router 內完成，本層只做傳輸封裝與錯誤轉譯。
    """
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="問題不可為空。")
    router: AgentRouter = app.state.router
    try:
        result = router.answer(question)
    except Exception as e:  # noqa: BLE001 — 檢索異常轉 500，前端顯示錯誤訊息
        logger.exception("問答失敗")
        raise HTTPException(status_code=500, detail=f"問答失敗：{e}")
    # 【瘦身】上下文內文只留前 200 字預覽，避免巨型 payload 拖慢前端
    contexts = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "title": c.get("title", ""),
            "source": c.get("source", ""),
            "preview": str(c.get("text", ""))[:CTX_PREVIEW_LEN],
        }
        for c in result.get("contexts", [])
    ]
    return {
        "question": result.get("question", question),
        "answer": result.get("answer", ""),
        "sub_questions": result.get("sub_questions", []),
        "contexts": contexts,
        "reflections": result.get("reflections", []),
        "overall": result.get("overall", {}),
        "total_attempts": int(result.get("total_attempts", 0)),
    }


# ══════════════════════════════════════════════
# 四、靜態前端掛載（放最後註冊，避免蓋掉上面的 /api/*）
# ══════════════════════════════════════════════

# 【首頁】GET / 直接回傳對話前端（比 html=True 掛載更明確、可預測）
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """首頁：回傳 web/index.html（檔案缺失時直接 404，部署問題立即現形）。"""
    return FileResponse(os.path.join(WEB_DIR, "index.html"), media_type="text/html")


# 【靜態資源】首頁引用的 CSS/JS/圖檔走 /static/*（目前單檔頁面備用，未來擴充免改後端）
if Path(WEB_DIR).is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ══════════════════════════════════════════════
# 五、本機啟動入口（python api/app.py）
# ══════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # 【直接執行】等價於 uvicorn api.app:app，import 字串用 app:app
    # （因 __main__ 執行時模組名不是 api.app，必須切工作目錄到根目錄）
    os.chdir(_ROOT)
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, log_level="info")
