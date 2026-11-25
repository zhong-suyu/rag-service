"""条目对齐切分：先按"编号条目"边界切，再把相邻完整条目打包到目标长度。

为什么需要它：
    语料是"1. 现象：…；检测：…；修复：…"这种结构化规则。字符硬切会把一条规则
    腰斩成两块（现象在 A 块、修复在 B 块），检索时容易召回"现象块"、漏掉"答案块"。
    本模块保证每一块都由**若干条完整规则**组成，绝不从条目中间断开。

策略：
    1. 按 `数字.` / `数字、` 的起始位置切成条目
    2. 相邻条目累加，直到超过 target_size 才另起一块（累加时不切条目）
    3. 单条自身长于 1.6 倍目标长度时，按句号/分号二次切
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from utils.logger_handler import get_logger

logger = get_logger(__name__)

# 条目起点：1~3 位数字 + 点/顿号 + 空白，且紧跟中文或 **（排除 2.5L、0.3μm 这类小数）
_ITEM_START = re.compile(r"(?=\d{1,3}[.、]\s*(?=[\u4e00-\u9fff*]))")
# 二次切分点：句号、分号
_SENT_SPLIT = re.compile(r"(?<=[。；])")


def split_items(text: str) -> List[str]:
    """把一段文本按编号切成若干条目（每条已 strip）。"""
    return [p.strip() for p in _ITEM_START.split(text) if p.strip()]


def pack_items(items: List[str], target_size: int = 300) -> List[str]:
    """把相邻条目打包成块：只在条目边界处断开，单条过长才二次切。"""
    chunks: List[str] = []
    buf = ""

    for item in items:
        # 单条本身过长：先把已攒的缓冲落盘，再按句号切这一条
        if len(item) > target_size * 1.6:
            if buf:
                chunks.append(buf)
                buf = ""
            cur = ""
            for sent in _SENT_SPLIT.split(item):
                if len(cur) + len(sent) > target_size and cur:
                    chunks.append(cur)
                    cur = sent
                else:
                    cur += sent
            if cur:
                chunks.append(cur)
            continue

        # 常规情况：装得下就继续攒，装不下就把缓冲落盘、本条另起
        if buf and len(buf) + len(item) > target_size:
            chunks.append(buf)
            buf = item
        else:
            buf += item

    if buf:
        chunks.append(buf)

    return [c.strip() for c in chunks if c.strip()]


def split_documents_structural(
    documents: List[Document],
    target_size: int = 300,
) -> List[Document]:
    """对每个 Document 做条目对齐切分，保留原 metadata（source 等）。"""
    chunks: List[Document] = []
    for doc in documents:
        for text in pack_items(split_items(doc.page_content), target_size):
            chunks.append(Document(page_content=text, metadata=dict(doc.metadata)))
    logger.info(f"Structural split {len(documents)} docs into {len(chunks)} chunks (target={target_size})")
    return chunks
