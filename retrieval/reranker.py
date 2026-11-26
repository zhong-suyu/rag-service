"""精排（rerank）：对已召回的候选块重新打分、重新排序。

为什么单独一层：
    向量、BM25 都属于"召回"——目标是别漏，宁可多捞几条参差不齐的候选；
    但它们打分是"双塔"式的（query 和 doc 各自编码、再算相似度），两者从没在模型里交互，
    排序精度有限。rerank 是"交互式"打分（query 和 doc 一起送进模型），精度高得多，
    代价是每算一篇都要跑一次模型，没法给全库打分，所以只能做精排、不能做召回。

接口事实（阿里云百炼 文本排序 API）：
    POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
    入参 model / input{query, documents} / parameters{top_n, return_documents, instruct}
    出参 output.results[{index, relevance_score}]，按分数降序
    注意：relevance_score 是"本次请求内的相对分"，不能跨请求比较，
          所以不能拿它当全局拒答阈值。
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import requests
from langchain_core.documents import Document

from utils.logger_handler import get_logger

logger = get_logger(__name__)

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

# 问答检索场景的默认排序指令：优先选"能回答问题"的块，而不是"只是主题相关"的块
QA_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."


class DashScopeReranker:
    """百炼文本排序模型封装。

    参数：
        model   模型名（qwen3.7-text-rerank / qwen3-rerank / qwen3-vl-rerank）
        top_n   精排后保留条数
        timeout 单次请求超时（秒）
        instruct 排序指令，仅 qwen3 系列生效
        retries 失败重试次数（网络抖动常见，重试一次足够）
    """

    def __init__(
        self,
        model: str = "qwen3.7-text-rerank",
        top_n: int = 3,
        timeout: float = 20.0,
        instruct: str | None = QA_INSTRUCT,
        retries: int = 1,
    ):
        self.model = model
        self.top_n = top_n
        self.timeout = timeout
        self.instruct = instruct
        self.retries = retries
        # 累计消耗，评估脚本可直接读（用于算成本）
        self.total_tokens = 0
        self.calls = 0
        self.failures = 0

    # ---------- 主入口 ----------
    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float | None]]:
        """返回 [(doc, relevance_score)]，按分数降序；失败时降级为原顺序、分数为 None。

        降级而不是抛异常，是因为精排属于"锦上添花"：API 挂了也不能让整个问答崩掉，
        用召回阶段的顺序兜底即可（这正是保证可用性的做法）。
        """
        if not docs:
            return []

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": [d.page_content for d in docs],
            },
            "parameters": {"top_n": min(self.top_n, len(docs)), "return_documents": False},
        }
        if self.instruct:
            payload["parameters"]["instruct"] = self.instruct

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("未配置 DASHSCOPE_API_KEY，跳过精排")
            return [(d, None) for d in docs[: self.top_n]]

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    RERANK_URL,
                    json=payload,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                body = resp.json()

                results = body["output"]["results"]
                usage = body.get("usage", {}) or {}
                self.total_tokens += int(usage.get("total_tokens", 0) or 0)
                self.calls += 1

                ranked = []
                for item in results:
                    idx = int(item["index"])
                    if 0 <= idx < len(docs):
                        ranked.append((docs[idx], float(item["relevance_score"])))
                logger.info(
                    f"rerank[{self.model}]: {len(docs)} -> {len(ranked)} "
                    f"tokens={usage.get('total_tokens', 0)}"
                )
                return ranked
            except Exception as e:  # 网络/限流/解析都可能出错，统一降级
                last_err = e
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))

        self.failures += 1
        logger.warning(f"rerank 失败，降级为原顺序: {last_err}")
        return [(d, None) for d in docs[: self.top_n]]
