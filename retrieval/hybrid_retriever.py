"""混合检索：向量召回 + BM25 关键词召回，用 RRF 融合排序。

为什么需要它：
    纯向量检索擅长"语义相近"，但对参数类问题不敏感 —— "地毯需要多大吸力"
    和"≥4000Pa"在语义空间上并不算近，向量模型常常召不回含答案的那一块。
    BM25 靠词面命中，能精确抓住 4000Pa / 2.5L / 3米 这类数字和单位。
    两路互补，再用 RRF（Reciprocal Rank Fusion）融合，不需要归一化两路分数。
"""

from __future__ import annotations

from typing import List

import jieba
import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from rank_bm25 import BM25Okapi

from utils.logger_handler import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """中文分词 + 转小写。jieba 会把 "3000Pa" 切成 3000 / pa，两边同样切法即可对齐。"""
    return [t.lower() for t in jieba.lcut(text) if t.strip()]


class HybridRetriever:
    """向量 + BM25 双路召回，RRF 融合。

    参数：
        store       ChromaStore（读底层 collection 建 BM25 索引）
        k           最终返回条数
        candidate_k 每一路各召回多少条候选（再融合取前 k）
        rrf_k       RRF 平滑常数，常用 60
    """

    def __init__(self, store, k: int = 3, candidate_k: int = 10, rrf_k: int = 60):
        self.store = store
        self.k = k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._docs: List[Document] = []
        self._bm25: BM25Okapi | None = None

    # ---------- 索引构建（懒加载，只在第一次检索时建一次） ----------
    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return
        coll = self.store.vectorstore._collection
        got = coll.get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=text, metadata=md or {})
            for text, md in zip(got["documents"], got["metadatas"])
        ]
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self._docs])
        logger.info(f"BM25 index built on {len(self._docs)} chunks")

    # ---------- 两路召回 ----------
    def _vector_hits(self, query: str) -> List[Document]:
        pairs = self.store.vectorstore.similarity_search_with_score(query, k=self.candidate_k)
        return [doc for doc, _score in pairs]

    def _bm25_hits(self, query: str) -> List[Document]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][: self.candidate_k]
        return [self._docs[int(i)] for i in order if scores[int(i)] > 0]

    # ---------- 融合 ----------
    def retrieve(self, query: str) -> List[Document]:
        self._ensure_index()
        fused: dict[str, float] = {}
        pool: dict[str, Document] = {}

        for rank, doc in enumerate(self._vector_hits(query), start=1):
            key = doc.page_content
            pool[key] = doc
            fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, doc in enumerate(self._bm25_hits(query), start=1):
            key = doc.page_content
            pool.setdefault(key, doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank)

        top = sorted(fused, key=fused.get, reverse=True)[: self.k]
        logger.info(f"hybrid retrieve: {len(pool)} candidates -> top{self.k}")
        return [pool[k] for k in top]

    def as_runnable(self) -> RunnableLambda:
        """包装成 LCEL 可用的 Runnable，直接接进 summarizer 的链。"""
        return RunnableLambda(self.retrieve)
