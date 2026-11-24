import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from utils.logger_handler import get_logger
from utils.path_tool import PathTool


logger = get_logger(__name__)


class DocumentLoader:
    def __init__(self, data_path: str, allowed_types: list[str]):
        self.data_path = data_path
        self.allowed_types = [t.lower().lstrip(".") for t in allowed_types]

    def collect_files(self) -> List[str]:
        # 递归收集所有允许类型的文件
        return PathTool.list_files(self.data_path, self.allowed_types)

    def _load_one(self, filepath: str) -> List[Document]:
        # 加载单个文件
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext in (".txt", ".md"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            logger.warning(f"Unsupported file type: {filepath}")
            return []
        docs = loader.load()
        fname = os.path.basename(filepath)
        for d in docs:
            # 覆盖 loader 默认塞进来的绝对路径，只保留纯文件名，便于迁移和追溯
            d.metadata["source"] = fname
        logger.info(f"Loaded {len(docs)} document(s) from {fname}")
        return docs

    def load_all(self) -> List[Document]:
        # 加载所有文件
        files = self.collect_files()   # 收集所有允许类型的文件
        all_docs: List[Document] = []
        for fp in files:
            # `_load_one` 返回的是**列表**，`extend` 把列表里的元素逐个摊进去。用 `append` 会得到一个二维列表
            all_docs.extend(self._load_one(fp))    
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs