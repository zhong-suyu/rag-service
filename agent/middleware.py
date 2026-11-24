from langchain.agents import AgentState
from langchain.agents.middleware import (
    wrap_tool_call,
    before_model,
    dynamic_prompt,
    ModelRequest,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import get_logger


logger = get_logger(__name__)


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler,
) -> ToolMessage | Command:
    """工具调用前后打日志；fill_context_for_report 被调用时切到报告提示词。"""
    name = request.tool_call["name"]
    args = request.tool_call.get("args")
    logger.info(f"[tool monitor] 执行工具：{name} 入参：{args}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor] 工具 {name} 调用完成")
        if name == "fill_context_for_report":
            request.runtime.context["report"] = True
            logger.info("[tool monitor] 已切换到报告提示词上下文")
        return result
    except Exception as e:
        logger.error(f"[tool monitor] 工具 {name} 调用失败：{e}")
        raise


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """模型调用前记录一次状态。"""
    logger.info(f"[before_model] 即将调用模型，当前历史 {len(state['messages'])} 条消息")
    return None


def build_prompt_switch(main_prompt: str, report_prompt: str):
    """生成 dynamic_prompt 中间件：按 context["report"] 决定用哪套系统提示词。

    用工厂函数而不是模块级全局，是为了让提示词仍由 main.py 统一传入。
    """

    @dynamic_prompt
    def report_prompt_switch(request: ModelRequest) -> str:
        if request.runtime.context.get("report", False):
            return report_prompt
        return main_prompt

    return report_prompt_switch
