import os
from typing import List

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config import BASE_DIR
from utils.logger_handler import get_logger
from utils.path_tool import PathTool


logger = get_logger(__name__)


class ChromaStore:
    def __init__(self, collection_name, persist_directory, embedding_model, md5_store_path=None):
        # 初始化 persist_directory 目录
        self.full_persist_dir = os.path.join(BASE_DIR, persist_directory)
        os.makedirs(self.full_persist_dir, exist_ok=True)

        # 初始化 md5_store_path 目录
        self.md5_store_path = (
            os.path.join(BASE_DIR, md5_store_path) if md5_store_path else None
        )


        # 初始化 ChromaStore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=self.full_persist_dir,
        )

    # 临时开一个原生 chromadb 客户端读目录，列出集合名。调试用，跑主流程时不调用
    def list_collection(self) -> List[str]:
        client = chromadb.PersistentClient(path=self.full_persist_dir)
        return [c.name for c in client.list_collections()]



    # 从 md5_store_path 中加载已索引的文档的 md5 值
    def _load_md5s(self) -> set[str]:
        if not self.md5_store_path or not os.path.exists(self.md5_store_path):
            return set()
        with open(self.md5_store_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
        """
            result = set()
            for line in f:
                if line.strip():          # 过滤空行
                    result.add(line.strip())   # 去首尾空白
            return result
        """

    # 增量追加本次新增的 md5，避免每次都全量重写文件
    def _append_md5s(self, md5s: set[str]) -> None:
        if not self.md5_store_path or not md5s:
            return
        # 确保目录存在，避免写入失败
        parent = os.path.dirname(self.md5_store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # 老文件末尾可能没有换行符，先补一个，否则新内容会粘到最后一行上
        prefix = ""
        if os.path.exists(self.md5_store_path) and os.path.getsize(self.md5_store_path) > 0:
            with open(self.md5_store_path, "rb") as f:
                # 先读取文件末尾，判断是否有换行符，没有则补一个
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    prefix = "\n"
        with open(self.md5_store_path, "a", encoding="utf-8") as f:
            f.write(prefix + "\n".join(sorted(md5s)) + "\n")

    # 整理 md5 清单：去重 + 排序 + 重写。数据量变大或怀疑有重复行时手动调用
    def compact_md5s(self) -> None:
        if not self.md5_store_path:
            return
        md5s = self._load_md5s()
        with open(self.md5_store_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(md5s)) + "\n")
        logger.info(f"md5 store compacted: {len(md5s)} records")

    # 添加文档到 ChromaStore
    def add_documents(self, documents: List[Document]) -> None:
        existing = self._load_md5s()
        seen: set[str] = set(existing)   # 去重判断用：库内已有 + 本批已处理
        new_md5s: set[str] = set()       # 本批真正新增的，用于追加写盘

        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []


        # 遍历所有文档，过滤已索引的文档
        for doc in documents:
            content = doc.page_content
            md5 = PathTool.md5_text(content)
            # 过滤已索引的文档：seen 同时覆盖「库内已有」和「本批内重复」
            if md5 in seen:
                continue
            seen.add(md5)  # 处理完立刻登记，本批后续重复就能拦住
            texts.append(content)
            metadatas.append(doc.metadata or {})
            ids.append(md5)
            new_md5s.add(md5)

        # 如果没有新文档，直接返回
        if not texts:
            logger.info("No new documents to add (all chunks already indexed)")
            return

        logger.info(f"Adding {len(texts)} new chunks to Chroma...")
        self.vectorstore.add_texts(
            texts=texts,  # 新文档内容
            metadatas=metadatas,  # 新文档元数据
            ids=ids,  # 新文档的 id，这里使用 md5 值
        )
        self._append_md5s(new_md5s)   # 只追加本批新增的 md5
        logger.info("Chroma index updated and md5 store appended")


    # 从 vectorstore 中获取检索器
    def as_retriever(self, k: int = 3):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})