import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import run_agent, run_agent_stream
from utils.logger_handler import get_logger


logger = get_logger(__name__)

_app_state: dict = {}


# 把 main.py 建好的 Agent 存进模块级 _app_state
def set_agent_app(agent_app):
    _app_state["agent_app"] = agent_app

# 创建 FastAPI 实例并注册路由
def create_app() -> FastAPI:
    app = FastAPI(title="RAG Agent Service", version="1.0.0")


    # Pydantic 是 Python 的数据校验库—— 用一个类声明「数据长什么样」，它负责解析、校验、转换
    class ChatRequest(BaseModel):
        message: str

    class ChatResponse(BaseModel):
        reply: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        """一次性返回完整回答。"""
        agent_app = _app_state.get("agent_app")
        if agent_app is None:
            return ChatResponse(reply="Agent 未初始化")
        logger.info(f"API chat: {req.message}")
        return ChatResponse(reply=run_agent(agent_app, req.message))

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        """SSE 流式返回：每条 data 是 {"delta": "..."}，结束发 [DONE]。"""
        agent_app = _app_state.get("agent_app")
        if agent_app is None:
            return StreamingResponse(iter(["data: [DONE]\n\n"]), media_type="text/event-stream")

        logger.info(f"API chat_stream: {req.message}")

        def gen():
            try:
                for token in run_agent_stream(agent_app, req.message):
                    yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
