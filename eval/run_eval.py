# -*- coding: utf-8 -*-
"""检索效果评估脚本（离线跑，不依赖 FastAPI / Agent）

评估口径：
    hit@k     —— top-k 个块里，是否至少有一块包含答案关键词
    MRR       —— 第一块命中答案的倒数排名（越接近 1 越准，位置越靠前）
    source@k  —— top-k 里是否出现过标注的标准来源文件
    latency   —— 单题检索耗时（毫秒）
    负样本距离 —— 库里没有答案的问题，其 top1 距离分布（仅 vector 模式可测）

用法（必须用 venv 的解释器）：
    .venv\\Scripts\\python.exe eval\\run_eval.py                     # 默认按 config 的 retrieval_mode
    .venv\\Scripts\\python.exe eval\\run_eval.py --mode vector       # 只跑向量检索，做对比基线
    .venv\\Scripts\\python.exe eval\\run_eval.py --mode hybrid       # 向量 + BM25 融合
    .venv\\Scripts\\python.exe eval\\run_eval.py --with-llm          # 加评生成质量（慢）

产物：
    eval/report.md      可直接贴进 README 的报告
    eval/results.json   明细数据，便于二次分析
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable

# 允许直接 `python eval/run_eval.py` 运行，也能在项目根里 import 业务模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from utils.config_handler import load_config
from utils.logger_handler import setup_logger
from embedding.embeddings import build_embedding_model
from vector_store.chroma_store import ChromaStore


def _norm(text: str) -> str:
    """关键词匹配前的归一化：去空白 + 转小写，避免 "3000 Pa" / "3000pa" 这类差异干扰。"""
    return "".join(text.split()).lower()


def _basename(path) -> str:
    return os.path.basename(str(path or ""))


def _build_store() -> ChromaStore:
    """直接打开已持久化的 chroma_db，不重建索引（评估必须基于线上同一份数据）。"""
    rag_cfg = load_config("rag")
    chroma_cfg = load_config("chroma")
    embedding_model = build_embedding_model(model_name=rag_cfg["embedding_model"])
    return ChromaStore(
        collection_name=chroma_cfg["collection_name"],
        persist_directory=chroma_cfg["persist_directory"],
        embedding_model=embedding_model,
        md5_store_path=None,  # 评估只读，不碰 md5 清单
    )


def _make_searcher(
    store: ChromaStore,
    mode: str,
    k: int,
    rerank_model: str = "qwen3.7-text-rerank",
    rerank_candidate_k: int = 10,
) -> Callable:
    """返回 search(query) -> [(Document, score or None)]，各模式统一签名。

    mode:
        vector         纯向量
        hybrid         向量 + BM25 + RRF
        vector_rerank  向量多捞 rerank_candidate_k 条 → 精排取 k
        hybrid_rerank  混合检索多捞 rerank_candidate_k 条 → 精排取 k
    """
    do_rerank = mode.endswith("_rerank")
    base_mode = mode[: -len("_rerank")] if do_rerank else mode
    depth = rerank_candidate_k if do_rerank else k
    chroma_cfg = load_config("chroma")

    if base_mode == "hybrid":
        from retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(
            store,
            k=depth,
            candidate_k=chroma_cfg.get("hybrid_candidate_k", 10),
            rrf_k=chroma_cfg.get("rrf_k", 60),
        )
        base_search = retriever.retrieve
    else:
        base_search = lambda q: [  # noqa: E731
            doc for doc, _score in store.vectorstore.similarity_search_with_score(q, k=depth)
        ]

    if not do_rerank:
        if base_mode == "hybrid":
            return lambda q: [(doc, None) for doc in base_search(q)]  # noqa: E731
        return lambda q: store.vectorstore.similarity_search_with_score(q, k=k)  # noqa: E731

    from retrieval.reranker import DashScopeReranker

    reranker = DashScopeReranker(model=rerank_model, top_n=k)

    def search_rerank(query: str):
        return reranker.rerank(query, base_search(query))

    search_rerank.reranker = reranker  # 供汇总 token 消耗
    return search_rerank


def _evaluate_one(search: Callable, item: dict) -> dict:
    t0 = time.perf_counter()
    pairs = search(item["question"])
    latency_ms = (time.perf_counter() - t0) * 1000

    hits, first_rank = [], None
    for rank, (doc, _score) in enumerate(pairs, start=1):
        content = _norm(doc.page_content)
        if any(_norm(kw) in content for kw in item.get("answer_keywords", [])):
            hits.append(rank)
            if first_rank is None:
                first_rank = rank

    sources = [_basename(doc.metadata.get("source", "")) for doc, _ in pairs]
    top1_text = pairs[0][0].page_content.replace("\n", " ")[:80] if pairs else ""
    top1_score = pairs[0][1] if pairs and pairs[0][1] is not None else None

    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "is_negative": bool(item.get("expect_no_answer")),
        "hit": bool(hits),
        "hit_ranks": hits,
        "first_rank": first_rank,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "source_hit": item.get("expected_source") in sources,
        "expected_source": item.get("expected_source", ""),
        "top1_source": sources[0] if sources else "",
        "top1_score": round(float(top1_score), 4) if top1_score is not None else None,
        "sources": sources,
        "top1_preview": top1_text,
        "latency_ms": round(latency_ms, 1),
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def _run_llm_check(store: ChromaStore, items: list[dict], k: int, mode: str) -> list[dict]:
    """可选：调用真实生成链路，检查答案是否含关键词、是否引用了来源。"""
    from langchain_openai import ChatOpenAI
    from rag_summarize.rag_summarize import RagSummarizeService
    from retrieval.reranker import DashScopeReranker
    from utils.prompt_loader import PromptLoader

    rag_cfg = load_config("rag")
    chroma_cfg = load_config("chroma")
    prompt_loader = PromptLoader(load_config("prompts"))

    # mode 可能是 hybrid_rerank，这里拆成"基础检索模式 + 是否精排"
    base_mode = "hybrid" if "hybrid" in mode else "vector"
    do_rerank = mode.endswith("_rerank")

    model = ChatOpenAI(
        model=rag_cfg["chat_model"],
        base_url=rag_cfg["model_base_url"],
        temperature=0,
    )
    service = RagSummarizeService(
        vectorstore=store,
        model=model,
        summarize_prompt=prompt_loader.get("rag_summarize_prompt_path"),
        k=k,
        retrieval_mode=base_mode,
        candidate_k=chroma_cfg.get("hybrid_candidate_k", 10),
        rrf_k=chroma_cfg.get("rrf_k", 60),
        cited_prompt=prompt_loader.get("rag_summarize_cited_prompt_path"),
        rerank_enabled=do_rerank,
        reranker=(
            DashScopeReranker(model=chroma_cfg.get("rerank_model", "qwen3.7-text-rerank"), top_n=k)
            if do_rerank
            else None
        ),
        rerank_candidate_k=chroma_cfg.get("rerank_candidate_k", 10),
    )

    results = []
    for item in items:
        t0 = time.perf_counter()
        payload = service.invoke_with_sources(item["question"])
        cost = time.perf_counter() - t0
        answer = payload["answer"]
        results.append(
            {
                "id": item["id"],
                "answer": answer,
                "sources": payload["sources"],
                "keyword_hit": any(_norm(kw) in _norm(answer) for kw in item["answer_keywords"]),
                "has_citation": "[1]" in answer or "[2]" in answer,
                "latency_s": round(cost, 2),
            }
        )
        print(
            f"  [{item['id']}] kw={results[-1]['keyword_hit']} "
            f"cited={results[-1]['has_citation']} {cost:.1f}s"
        )
    return results


def _write_report(summary: dict, rows: list[dict], llm_rows: list[dict], items: list[dict]) -> str:
    k = summary["k"]
    score_kind = summary.get("score_kind", "distance")
    score_col = "top1 精排分" if score_kind == "rerank" else "top1 距离"
    lines = []
    lines.append("# 检索效果评估报告\n")
    lines.append(f"- 检索模式：`{summary['mode']}`（top-k={k}）")
    lines.append(f"- 评估集：{summary['total']} 题（正样本 {summary['positives']} / 负样本 {summary['negatives']}）")
    lines.append(f"- 索引：`chroma_db` 中已持久化的真实数据")
    lines.append(f"- 生成时间：{summary['timestamp']}\n")

    lines.append("## 核心指标\n")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 检索命中率 hit@{k} | {summary['hit_rate']:.1%} | top-{k} 中至少一块含答案关键词 |")
    lines.append(f"| MRR | {summary['mrr']:.3f} | 首个命中块的平均倒数排名 |")
    lines.append(f"| 来源命中率 | {summary['source_rate']:.1%} | top-{k} 中出现正确来源文件 |")
    lines.append(f"| 平均检索耗时 | {summary['latency_mean']:.1f} ms | 单题检索 |")
    lines.append(f"| P95 检索耗时 | {summary['latency_p95']:.1f} ms | 慢尾延迟 |")

    lines.append("\n## 明细\n")
    lines.append(f"| 编号 | 类型 | 问题 | 命中 | 首次命中排名 | 来源命中 | {score_col} | 耗时(ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        dist = "—" if r["top1_score"] is None else f"{r['top1_score']:.3f}"
        if r["is_negative"]:
            lines.append(f"| {r['id']} | 负样本 | {r['question']} | — | — | — | {dist} | {r['latency_ms']:.1f} |")
        else:
            mark = "✅" if r["hit"] else "❌"
            lines.append(
                f"| {r['id']} | {r['type']} | {r['question']} | {mark} | {r['first_rank'] or '—'} "
                f"| {'✅' if r['source_hit'] else '❌'} | {dist} | {r['latency_ms']:.1f} |"
            )

    if summary.get("rerank"):
        rr = summary["rerank"]
        lines.append("\n## 精排消耗\n")
        lines.append("| 项 | 数值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 模型 | `{rr['model']}` |")
        lines.append(f"| 排序候选数 | {rr['candidate_k']} |")
        lines.append(f"| 成功调用 / 失败 | {rr['calls']} / {rr['failures']} |")
        lines.append(f"| 累计 token | {rr['total_tokens']} |")

    missed = [r for r in rows if not r["is_negative"] and not r["hit"]]
    if missed:
        kw_map = {i["id"]: i["answer_keywords"] for i in items}
        lines.append("\n## 未命中样本（优化线索）\n")
        for r in missed:
            lines.append(f"**{r['id']} {r['question']}**")
            lines.append(f"- 关键词：`{', '.join(kw_map.get(r['id'], []))}`")
            lines.append(f"- top1 来源：{r['top1_source']} / 距离 {r['top1_score']}")
            lines.append(f"- top1 片段：{r['top1_preview']}\n")

    if llm_rows:
        ok = sum(1 for x in llm_rows if x["keyword_hit"])
        cited = sum(1 for x in llm_rows if x["has_citation"])
        avg = statistics.mean([x["latency_s"] for x in llm_rows])
        lines.append("\n## 生成质量（端到端）\n")
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| 答案关键词覆盖率 | {ok}/{len(llm_rows)} | 生成答案是否包含标准答案关键词 |")
        lines.append(f"| 出处标注率 | {cited}/{len(llm_rows)} | 答案是否带 [编号] 引用 |")
        lines.append(f"| 平均生成耗时 | {avg:.1f} s | 含一次大模型调用 |")

    lines.append("\n## 结论\n")
    lines.append(
        f"{summary['mode']} 模式在 {summary['positives']} 条正样本上 hit@{k} = {summary['hit_rate']:.1%}，"
        f"MRR = {summary['mrr']:.3f}，来源命中 {summary['source_rate']:.1%}，"
        f"单题检索平均 {summary['latency_mean']:.0f}ms。"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索效果评估")
    parser.add_argument("--k", type=int, default=3, help="检索 top-k，默认 3（与 config/chroma.yml 的 k 一致）")
    parser.add_argument("--mode", choices=["vector", "hybrid", "vector_rerank", "hybrid_rerank"],
                        default=None,
                        help="检索模式，不传则读 config/chroma.yml 的 retrieval_mode")
    parser.add_argument("--questions", default=os.path.join("eval", "questions.json"))
    parser.add_argument("--with-llm", action="store_true", help="额外评估生成答案质量（慢）")
    parser.add_argument("--rerank-model", default="qwen3.7-text-rerank",
                        help="精排模型：qwen3.7-text-rerank / qwen3-rerank / qwen3-vl-rerank")
    parser.add_argument("--rerank-candidate-k", type=int, default=10,
                        help="交给精排的候选条数（默认 10）")
    args = parser.parse_args()

    setup_logger()
    chroma_cfg = load_config("chroma")
    mode = args.mode or chroma_cfg.get("retrieval_mode", "vector")

    questions_path = args.questions
    if not os.path.isabs(questions_path):
        questions_path = os.path.join(PROJECT_ROOT, questions_path)
    with open(questions_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"加载 {len(items)} 条评估样本，模式 = {mode}，打开向量库...")
    store = _build_store()
    search = _make_searcher(
        store, mode, args.k,
        rerank_model=args.rerank_model,
        rerank_candidate_k=args.rerank_candidate_k,
    )

    rows = []
    for item in items:
        row = _evaluate_one(search, item)
        rows.append(row)
        flag = "—" if row["is_negative"] else ("命中" if row["hit"] else "未命中")
        dist = "-" if row["top1_score"] is None else f"{row['top1_score']:.3f}"
        print(f"[{row['id']}] {flag:<3} rank={row['first_rank']} dist={dist} "
              f"{row['latency_ms']:.0f}ms  {row['question']}")

    positives = [r for r in rows if not r["is_negative"]]
    negatives = [r for r in rows if r["is_negative"]]

    hit_rate = sum(1 for r in positives if r["hit"]) / len(positives) if positives else 0.0
    source_rate = sum(1 for r in positives if r["source_hit"]) / len(positives) if positives else 0.0
    mrr = statistics.mean([r["mrr"] for r in positives]) if positives else 0.0
    latencies = [r["latency_ms"] for r in rows]

    reranker = getattr(search, "reranker", None)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "k": args.k,
        "total": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "hit_rate": hit_rate,
        "mrr": mrr,
        "source_rate": source_rate,
        "latency_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_p95": _percentile(latencies, 0.95),
        # 分数语义不同：向量/混合是 L2 距离（越小越好），精排是相关性分（0~1，越大越好）
        "score_kind": "rerank" if reranker else "distance",
        "rerank": (
            {
                "model": reranker.model,
                "candidate_k": args.rerank_candidate_k,
                "calls": reranker.calls,
                "failures": reranker.failures,
                "total_tokens": reranker.total_tokens,
            }
            if reranker
            else None
        ),
    }

    llm_rows = []
    if args.with_llm:
        print("\n开始评估生成质量（每题一次大模型调用，较慢）...")
        llm_rows = _run_llm_check(store, positives, args.k, mode)

    out_dir = os.path.join(PROJECT_ROOT, "eval")
    os.makedirs(out_dir, exist_ok=True)

    # 精排模式下把模型名和候选深度写进文件名，便于多次对照
    suffix = ""
    if getattr(search, "reranker", None):
        suffix = f"_{args.rerank_model.replace('.', '')}_c{args.rerank_candidate_k}"

    report_path = os.path.join(out_dir, f"report_{mode}{suffix}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_write_report(summary, rows, llm_rows, items))

    json_path = os.path.join(out_dir, f"results_{mode}{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "llm": llm_rows}, f, ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    print(f"模式 {mode} | hit@{args.k}: {hit_rate:.1%} | MRR: {mrr:.3f} | 来源命中: {source_rate:.1%}")
    print(f"检索耗时: 平均 {summary['latency_mean']:.1f}ms / P95 {summary['latency_p95']:.1f}ms")
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
