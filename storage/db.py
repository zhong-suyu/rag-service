"""调用记录与用户反馈的持久化（SQLite）。

为什么用 SQLite：
    单机、单进程、数据量小（每次问答一行），标准库自带、零部署；
    真上多实例再换 PostgreSQL——上层只依赖本文件的 4 个函数，替换成本低。

两张表的关系：
    qa_log    每次请求一行，trace_id 是主键（API 生成后回给前端）
    feedback  一次反馈一行，靠 trace_id 关联，允许一条请求多条反馈

并发注意：
    FastAPI 的同步路由跑在线程池里，SQLite 默认单写锁会报 "database is locked"。
    做法：check_same_thread=False + WAL 模式 + 进程内一把写锁。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Callable, Iterable

from config import BASE_DIR
from utils.logger_handler import get_logger

logger = get_logger(__name__)

DB_PATH = os.getenv("RAG_HISTORY_DB") or os.path.join(BASE_DIR, "logs", "app.db")

_write_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # 读写不互斥，避免长连接阻塞写入
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _run(fn: Callable[[sqlite3.Connection], object]):
    """开连接 → 执行 → 提交 → 关闭。sqlite3 的连接上下文只管事务，不管关闭。"""
    conn = _connect()
    try:
        result = fn(conn)
        conn.commit()
        return result
    finally:
        conn.close()


def init_db() -> None:
    """建表（幂等）。服务启动时调一次。"""
    with _write_lock:
        _run(
            lambda conn: (
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS qa_log (
                        trace_id   TEXT PRIMARY KEY,
                        question   TEXT NOT NULL,
                        answer     TEXT,
                        sources    TEXT,          -- JSON: [{index, source, snippet}]
                        latency_ms INTEGER,
                        tokens     INTEGER,       -- 流式拿不到，允许 NULL
                        rejected   INTEGER DEFAULT 0,
                        error      TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                ),
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        trace_id   TEXT NOT NULL,
                        rating     INTEGER NOT NULL,   -- 1 好评 / -1 点踩
                        comment    TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                ),
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id)"),
            )
        )
    logger.info(f"SQLite 就绪: {DB_PATH}")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def insert_qa(
    trace_id: str,
    question: str,
    answer: str | None,
    sources: Iterable[dict] | None = None,
    latency_ms: int | None = None,
    tokens: int | None = None,
    rejected: bool = False,
    error: str | None = None,
) -> bool:
    """写一条问答记录。失败只记 warning，绝不影响主流程返回。"""
    try:
        payload = json.dumps(list(sources or []), ensure_ascii=False)

        def _insert(conn):
            conn.execute(
                "INSERT OR REPLACE INTO qa_log "
                "(trace_id, question, answer, sources, latency_ms, tokens, rejected, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    question,
                    answer,
                    payload,
                    latency_ms,
                    tokens,
                    1 if rejected else 0,
                    error,
                    _now(),
                ),
            )

        with _write_lock:
            _run(_insert)
        return True
    except Exception as e:  # 记录失败不能拖垮业务
        logger.warning(f"写 qa_log 失败: {e}")
        return False


def insert_feedback(trace_id: str, rating: int, comment: str | None = None) -> bool:
    """写一条反馈；trace_id 不存在时返回 False（由接口转成 404）。"""
    if rating not in (1, -1):
        raise ValueError("rating 只能是 1 或 -1")
    try:

        def _insert(conn):
            exists = conn.execute(
                "SELECT 1 FROM qa_log WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT INTO feedback (trace_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
                (trace_id, rating, comment, _now()),
            )
            return True

        with _write_lock:
            return bool(_run(_insert))
    except Exception as e:
        logger.warning(f"写 feedback 失败: {e}")
        return False


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[idx])


def stats() -> dict:
    """统计口径：好评率分母是「收到反馈的请求数」，不是总请求数。"""

    def _query(conn):
        total = conn.execute("SELECT COUNT(*) AS n FROM qa_log").fetchone()["n"]
        rejected = conn.execute("SELECT COUNT(*) AS n FROM qa_log WHERE rejected = 1").fetchone()["n"]
        fb = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
        good = conn.execute("SELECT COUNT(*) AS n FROM feedback WHERE rating = 1").fetchone()["n"]
        lat = [
            r["latency_ms"]
            for r in conn.execute("SELECT latency_ms FROM qa_log WHERE latency_ms IS NOT NULL")
        ]
        recent = [
            dict(r)
            for r in conn.execute(
                "SELECT q.trace_id, q.question, f.rating, f.comment, q.created_at "
                "FROM feedback f JOIN qa_log q ON q.trace_id = f.trace_id "
                "WHERE f.rating = -1 ORDER BY f.id DESC LIMIT 5"
            )
        ]
        return total, rejected, fb, good, lat, recent

    total, rejected, fb, good, lat, recent = _run(_query)
    return {
        "total": total,
        "feedback_count": fb,
        "good_rate": round(good / fb, 3) if fb else 0.0,
        "reject_rate": round(rejected / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "p95_latency_ms": round(_p95([float(x) for x in lat]), 1),
        "recent_bad_cases": recent,
    }
