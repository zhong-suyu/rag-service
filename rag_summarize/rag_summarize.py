from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from retrieval.hybrid_retriever import HybridRetriever
from utils.logger_handler import get_logger

# 注解只写给人和静态检查器，运行时不需要真的导入 ChromaStore
if TYPE_CHECKING:
    from vector_store.chroma_store import ChromaStore


logger = get_logger(__name__)


class RagSummarizeService:
    """RAG 问答服务：检索 → 拼上下文 → 提示词 → 模型总结。

    retrieval_mode:
        vector  仅向量检索（原课程实现）
        hybrid  向量 + BM25 双路召回，RRF 融合（对参数/数字类问题更稳）
    """

    def __init__(
        self,
        vectorstore: ChromaStore,
        model: ChatOpenAI,
        summarize_prompt: str,
        k: int = 3,
        retrieval_mode: str = "vector",
        candidate_k: int = 10,
        rrf_k: int = 60,
        cited_prompt: str | None = None,
    ):
        self.vectorstore = vectorstore
        self.model = model
        self.k = k
        self.retrieval_mode = retrieval_mode

        # 检索器：两种模式都最终包装成 Runnable，方便接进 LCEL 链
        if retrieval_mode == "hybrid":
            self.hybrid = HybridRetriever(
                vectorstore, k=k, candidate_k=candidate_k, rrf_k=rrf_k
            )
            self.retriever = self.hybrid.as_runnable()
        else:
            self.hybrid = None
            self.retriever = vectorstore.as_retriever(k=k)

        prompt_template = ChatPromptTemplate.from_template(summarize_prompt)
        # 链：检索 → 拼 context → 填模板 → 模型总结 → 取纯文本
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "input": RunnablePassthrough(),
            }
            | prompt_template
            | self.model
            | StrOutputParser()
        )

        # 带出处的链（可选）：context 每块前面加 [编号] + 来源文件名
        self.cited_chain = None
        if cited_prompt:
            self.cited_chain = (
                ChatPromptTemplate.from_template(cited_prompt)
                | self.model
                | StrOutputParser()
            )

    @staticmethod
    # 把检索到的所有正文，用双换行拼成一个长字符串
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _format_docs_with_refs(docs: list[Document]) -> str:
        """给每块参考资料加 [编号] 和来源文件名，让模型能引用。"""
        blocks = []
        for i, doc in enumerate(docs, start=1):
            source = os.path.basename(str(doc.metadata.get("source", "未知来源")))
            blocks.append(f"[{i}] 来源：{source}\n{doc.page_content}")
        return "\n\n".join(blocks)

    def retrieve_docs(self, query: str) -> list[Document]:
        """单独暴露检索结果，评估脚本和 API 都用它，保证口径一致。"""
        return self.retriever.invoke(query)

    # 薄封装：调用链，生成摘要
    def invoke(self, query: str) -> str:
        result = self.chain.invoke(query)
        logger.info(f"RAG answer generated (len={len(result)})")
        return result

    def invoke_with_sources(self, query: str) -> dict:
        """返回 {answer, sources}：答案带 [编号] 标注，sources 给出编号对应的文件与片段。

        没有配置 cited_prompt 时退化成普通 invoke，sources 仍然返回（只是答案里没有标注）。
        """
        docs = self.retrieve_docs(query)
        sources = [
            {
                "index": i,
                "source": os.path.basename(str(doc.metadata.get("source", "未知来源"))),
                "snippet": doc.page_content.replace("\n", " ")[:60],
            }
            for i, doc in enumerate(docs, start=1)
        ]

        if self.cited_chain is None:
            answer = self.invoke(query)
        else:
            answer = self.cited_chain.invoke(
                {"context": self._format_docs_with_refs(docs), "input": query}
            )
        logger.info(f"RAG answer with sources generated (sources={len(sources)})")
        return {"answer": answer, "sources": sources}
