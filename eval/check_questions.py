# -*- coding: utf-8 -*-
"""评估集自检：检查每条题目的关键词是否真的存在于库里、区分度是否足够。

输出：
    eval/questions_review.md   —— 给人工扫的核对表
    命令行打印问题项（关键词不存在 / 出现块数过多 / 关键词不在标注的来源文件里）
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()
from utils.config_handler import load_config
from embedding.embeddings import build_embedding_model
from vector_store.chroma_store import ChromaStore


def norm(t: str) -> str:
    return "".join(t.split()).lower()


def main() -> int:
    chroma_cfg = load_config("chroma")
    rag_cfg = load_config("rag")
    store = ChromaStore(chroma_cfg["collection_name"], chroma_cfg["persist_directory"],
                        build_embedding_model(rag_cfg["embedding_model"]), None)
    got = store.vectorstore._collection.get(include=["documents", "metadatas"])

    docs = [(norm(t), os.path.basename(str(m.get("source", ""))), t)
            for t, m in zip(got["documents"], got["metadatas"])]

    items = json.load(open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8"))
    rows, problems = [], []

    for it in items:
        if it.get("expect_no_answer"):
            rows.append(f"| {it['id']} | {it['question']} | （负样本，无关键词） | — | — | — |")
            continue

        kws = it["answer_keywords"]
        exists_all, hit_counts, preview, src_ok = True, [], "", True
        for kw in kws:
            n_kw = norm(kw)
            hits = [d for d in docs if n_kw in d[0]]
            hit_counts.append(len(hits))
            if not hits:
                exists_all = False
            if not preview and hits:
                preview = hits[0][2].replace("\n", " ")[:45]
            if hits and not any(h[1] == it["expected_source"] for h in hits):
                src_ok = False

        flag = []
        if not exists_all:
            flag.append("关键词不存在")
            problems.append((it["id"], it["question"], "关键词在库里找不到: " + " / ".join(kws)))
        for kw, c in zip(kws, hit_counts):
            if c > 8:
                flag.append(f"“{kw}”出现{c}块(区分度低)")
                problems.append((it["id"], it["question"], f"关键词“{kw}”出现 {c} 个块，区分度不足"))
        if not src_ok:
            flag.append("不在标注来源")
            problems.append((it["id"], it["question"], "关键词不在 expected_source 指定的文件里"))

        counts = ",".join(str(c) for c in hit_counts)
        rows.append(f"| {it['id']} | {it['question']} | {', '.join(kws)} | {counts} | "
                    f"{it['expected_source']} | {preview} | {'⚠ ' + '; '.join(flag) if flag else '✅'} |")

    out = ["# 评估集核对表\n",
           f"共 {len(items)} 题，其中正样本 {sum(1 for i in items if not i.get('expect_no_answer'))}、"
           f"负样本 {sum(1 for i in items if i.get('expect_no_answer'))}。\n",
           "「出现块数」= 该关键词在 326 个 chunk 里命中的数量；1~3 最理想，>8 说明区分度太低。\n",
           "| 编号 | 问题 | 关键词 | 出现块数 | 标注来源 | 命中块原文(前45字) | 自检 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    out.extend(rows)
    with open(os.path.join(ROOT, "eval", "questions_review.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"共 {len(items)} 题，发现问题 {len(problems)} 条：")
    for pid, q, reason in problems:
        print(f"  [{pid}] {q} → {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
