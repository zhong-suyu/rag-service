from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils.logger_handler import get_logger

# 注解只写给人和静态检查器，运行时不需要真的导入 ChromaStore
if TYPE_CHECKING:
    from vector_store.chroma_store import ChromaStore


logger = get_logger(__name__)


class RagSummarizeService:
    def __init__(
        self,
        vectorstore: ChromaStore,
        model: ChatOpenAI,
        summarize_prompt: str,
        k: int = 3,
    ):
        self.vectorstore = vectorstore
        self.model = model
        # 真正的检索器：返回 List[Document]，与 class 名区分开
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

    @staticmethod
    # 把检索到的所有正文，用双换行拼成一个长字符串
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # 薄封装：调用链，生成摘要
    def invoke(self, query: str) -> str:
        result = self.chain.invoke(query)
        logger.info(f"RAG answer generated (len={len(result)})")
        return result
