from typing import List, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger_handler import get_logger


logger = get_logger(__name__)


def build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: Iterable[str],  # 一个能逐个吐出字符串的可迭代对象
) -> RecursiveCharacterTextSplitter:
    """按传入的切分配置构造切分器。

    三个参数都是必填，由调用方（main.py）从 config/chroma.yml 读出来显式传入，
    保证「配置」是切分参数的唯一真相源，函数内部不再保留一份可能漂移的默认值。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
        is_separator_regex=False,    # 不使用正则表达式分隔符
    )


def split_documents(
    documents: List[Document],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks")
    return chunks
