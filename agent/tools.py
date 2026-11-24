import csv
import os
import random

from langchain_core.tools import tool

from config import BASE_DIR
from utils.config_handler import load_config
from utils.logger_handler import get_logger


logger = get_logger(__name__)

# 模拟用户数据，沿用课程 demo 的取值集合
USER_IDS = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
MONTHS = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
]
CITIES = ["深圳", "北京", "广州"]

_rag_summarizer = None


def bind_rag_summarizer(rag_summarizer):
    # 绑定全局变量，用于后续调用
    global _rag_summarizer
    _rag_summarizer = rag_summarizer


@tool
def rag_summarize(query: str) -> str:
    """从向量库中检索扫地机器人/扫拖机器人的专业资料并总结。入参 query 为检索关键词。"""
    if _rag_summarizer is None:
        return "检索服务未初始化"
    logger.info(f"[tool] rag_summarize query={query}")
    try:
        return _rag_summarizer.invoke(query)
    except Exception as e:
        logger.error(f"rag_summarize error: {e}")
        return f"检索失败: {e}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气、湿度、降雨概率等环境信息。入参 city 为城市名。"""
    logger.info(f"[tool] get_weather city={city}")
    conditions = ["晴", "多云", "阴", "小雨", "中雨", "霾"]
    cond = random.choice(conditions)
    temp = random.randint(10, 35)
    humidity = random.randint(30, 90)
    rain = random.randint(0, 80)
    return (
        f"城市: {city}\n天气: {cond}\n温度: {temp}°C\n"
        f"湿度: {humidity}%\n降雨概率: {rain}%"
    )


@tool
def get_user_location() -> str:
    """获取当前用户所在的城市名称。无参数。"""
    city = random.choice(CITIES)
    logger.info(f"[tool] get_user_location → {city}")
    return city


@tool
def get_user_id() -> str:
    """获取当前用户的唯一标识 ID（数字字符串，如 1001）。无参数。"""
    user_id = random.choice(USER_IDS)
    logger.info(f"[tool] get_user_id → {user_id}")
    return user_id


@tool
def get_current_month() -> str:
    """获取系统当前月份，格式 YYYY-MM。无参数。"""
    month = random.choice(MONTHS)
    logger.info(f"[tool] get_current_month → {month}")
    return month


@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """检索指定用户在指定月份的扫地机器人使用记录。user_id 为数字字符串，month 为 YYYY-MM。"""
    logger.info(f"[tool] fetch_external_data user_id={user_id} month={month}")
    rel_path = load_config("agent").get("external_data_path", "data/external/records.csv")
    path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(path):
        return "外部数据文件不存在"
    matched: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["用户ID"] == user_id and row["时间"] == month:
                matched.append(
                    f"特征: {row['特征']}\n清洁效率: {row['清洁效率']}\n"
                    f"耗材: {row['耗材']}\n对比: {row['对比']}\n时间: {row['时间']}"
                )
    if not matched:
        return f"未找到用户 {user_id} 在 {month} 的使用记录"
    return "\n---\n".join(matched)


@tool
def fill_context_for_report() -> str:
    """报告生成场景的前置工具。无参数。用于中间件切换到报告提示词上下文。"""
    logger.info("[tool] fill_context_for_report")
    return "报告上下文已注入"


def create_tools() -> list:
    return [
        rag_summarize,
        get_weather,
        get_user_location,
        get_user_id,
        get_current_month,
        fetch_external_data,
        fill_context_for_report,
    ]