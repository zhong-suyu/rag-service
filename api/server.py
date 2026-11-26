import json
import uuid
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import run_agent, run_agent_stream
from storage.db import init_db, insert_feedback, insert_qa, stats
from utils.logger_handler import get_logger


logger = get_logger(__name__)

_app_state: dict = {}


# 把 main.py 建好的 Agent 存进模块级 _app_state
def set_agent_app(agent_app):
    _app_state["agent_app"] = agent_app


def new_trace_id() -> str:
    """每次请求一个短 ID，前端拿它上报反馈；不是用户标识，不含任何个人信息。"""
    return uuid.uuid4().hex[:12]


# 创建 FastAPI 实例并注册路由
def create_app() -> FastAPI:
    app = FastAPI(title="RAG Agent Service", version="1.1.0")
    init_db()


    # Pydantic 是 Python 的数据校验库—— 用一个类声明「数据长什么样」，它负责解析、校验、转换
    class ChatRequest(BaseModel):
        message: str

    class ChatResponse(BaseModel):
        reply: str
        trace_id: str
        latency_ms: int

    class FeedbackRequest(BaseModel):
        trace_id: str
        rating: Literal[1, -1]
        comment: str | None = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        """一次性返回完整回答，并回传 trace_id 供点踩/点赞时上报。"""
        agent_app = _app_state.get("agent_app")
        trace_id = new_trace_id()
        if agent_app is None:
            return ChatResponse(reply="Agent 未初始化", trace_id=trace_id, latency_ms=0)

        logger.info(f"API chat: {req.message} trace={trace_id}")
        ctx: dict = {"report": False}   # 工具会把 sources 写进这个 dict
        t0 = perf_counter()
        answer, error = "", None
        try:
            answer = run_agent(agent_app, req.message, ctx)
        except Exception as e:
            error = str(e)
            answer = f"服务异常: {e}"
            logger.error(f"chat error trace={trace_id}: {e}")
        latency_ms = int((perf_counter() - t0) * 1000)

        insert_qa(
            trace_id=trace_id,
            question=req.message,
            answer=answer,
            sources=ctx.get("rag_sources", []),
            latency_ms=latency_ms,
            tokens=None,      # 同步接口暂不取 token 用量，不编数字
            rejected=bool(ctx.get("rag_rejected", False)),
            error=error,
        )
        return ChatResponse(reply=answer, trace_id=trace_id, latency_ms=latency_ms)

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        """SSE 流式返回：首帧 {"trace_id": ...}，之后 {"delta": ...}，末帧 {"done": ...} + [DONE]。"""
        agent_app = _app_state.get("agent_app")
        trace_id = new_trace_id()
        if agent_app is None:
            return StreamingResponse(
                iter([f'data: {json.dumps({"trace_id": trace_id})}\n\n', "data: [DONE]\n\n"]),
                media_type="text/event-stream",
            )

        logger.info(f"API chat_stream: {req.message} trace={trace_id}")
        ctx: dict = {"report": False}

        def gen():
            t0 = perf_counter()
            parts: list[str] = []
            error = None
            yield f'data: {json.dumps({"trace_id": trace_id}, ensure_ascii=False)}\n\n'
            try:
                for token in run_agent_stream(agent_app, req.message, ctx):
                    parts.append(token)
                    yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"
            except Exception as e:
                error = str(e)
                logger.error(f"stream error trace={trace_id}: {e}")
                yield f"data: {json.dumps({'error': error}, ensure_ascii=False)}\n\n"
            latency_ms = int((perf_counter() - t0) * 1000)

            # 流式没有完整 usage，tokens 留空；答案在这里已经拼完整，可直接入库
            insert_qa(
                trace_id=trace_id,
                question=req.message,
                answer="".join(parts),
                sources=ctx.get("rag_sources", []),
                latency_ms=latency_ms,
                tokens=None,
                rejected=bool(ctx.get("rag_rejected", False)),
                error=error,
            )
            yield f'data: {json.dumps({"done": True, "latency_ms": latency_ms})}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        """点赞/点踩。trace_id 必须来自 /chat 或 /chat/stream 的响应。"""
        ok = insert_feedback(req.trace_id, req.rating, req.comment)
        if not ok:
            raise HTTPException(status_code=404, detail=f"trace_id 不存在: {req.trace_id}")
        logger.info(f"API feedback trace={req.trace_id} rating={req.rating}")
        return {"ok": True, "trace_id": req.trace_id, "rating": req.rating}

    @app.get("/stats")
    def get_stats():
        """调用量、好评率（分母是收到反馈的请求数）、拒答率、平均/P95 延迟、最近点踩样本。"""
        return stats()

    return app
