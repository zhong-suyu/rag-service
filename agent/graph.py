from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

from agent.middleware import monitor_tool, log_before_model, build_prompt_switch
from agent.tools import bind_rag_summarizer, create_tools
from utils.logger_handler import get_logger


logger = get_logger(__name__)


def build_agent_graph(
    model: ChatOpenAI,
    main_prompt: str,
    report_prompt: str,
    rag_summarizer,
):
    """用 create_agent 组装 ReAct Agent，中间件负责日志与报告提示词切换。"""
    # 绑定摘要工具
    bind_rag_summarizer(rag_summarizer)
    tools = create_tools()

    agent = create_agent(
        model=model,
        system_prompt=main_prompt,
        tools=tools,
        middleware=[
            monitor_tool,
            log_before_model,
            build_prompt_switch(main_prompt, report_prompt),
        ],
    )
    logger.info("Agent 图构建完成")
    return agent

# 从消息历史中提取最后一个 AI 回答（不包含工具调用）
def _last_ai_text(messages) -> str:
    for m in reversed(messages):    # 从最后往前找
        role = getattr(m, "type", None) or ""
        tool_calls = getattr(m, "tool_calls", None) or []
        if role == "ai" and not tool_calls and m.content:  # 第一个「不带工具调用的 AI 消息」
            return m.content
    return "无法生成有效回答"


def run_agent(app, user_input: str) -> str:
    """一次性拿到完整回答。"""
    result = app.invoke(
        {"messages":
             [
                 {"role": "user", "content": user_input}
             ]
        },
        context={"report": False},
        config={"recursion_limit": 30},   # 最多 30 步，防止无限循环
    )
    return _last_ai_text(result.get("messages", []))


def run_agent_stream(app, user_input: str):
    """流式版：逐块产出模型输出的文本增量（str）。"""
    for chunk, _metadata in app.stream(
        {"messages":
             [
                 {"role": "user", "content": user_input}
             ]
        },
        stream_mode="messages",
        context={"report": False},
        config={"recursion_limit": 30},
    ):
        # 只留 AI 块，跳过其他消息类型
        if isinstance(chunk, AIMessageChunk):
            text = chunk.content
            # 工具调用块的内容为空，跳过
            if isinstance(text, str) and text:
                yield text
