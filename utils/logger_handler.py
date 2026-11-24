import logging
import sys
import os
from config import BASE_DIR

_LOGGER_NAME = "rag_service"
_initialized = False


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """初始化根日志器（幂等），返回 rag_service 这个根 logger。

    所有子 logger 都挂在 "rag_service." 命名空间下，向父级传播时共用这里的
    两个 handler，因此模块内部日志也能同时进控制台和 logs/app.log。
    """
    global _initialized
    logger = logging.getLogger(_LOGGER_NAME)
    if _initialized:
        return logger
    logger.setLevel(level)
    logger.propagate = False  # 不再向 root 传播，避免重复输出

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _initialized = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """获取 rag_service 命名空间下的子 logger。

    - 传 __name__ 时保留模块名（如 rag_service.agent.tools），日志里能看出来源
    - 不传 name 或传 "rag_service" 时返回根 logger
    - 子 logger 不单独设 handler，靠 propagate 复用根 logger 的 handler
    """
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    full_name = name if name.startswith(f"{_LOGGER_NAME}.") else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)
