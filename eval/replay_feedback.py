# -*- coding: utf-8 -*-
"""把评估集回放成"线上请求"，自动生成点赞/点踩，跑通反馈闭环。

为什么需要它：
    项目还没上线，没有真实用户流量。回放的作用不是制造统计结论，而是
    把 trace_id → qa_log → feedback → /stats → 归因 这条链路验证通、跑得出来。
    真实流量来了只需要换数据源，代码不用改。

判分规则（故意保持简单、可解释）：
    正样本：回答里包含该题任一答案关键词 → 👍；否则 👎，comment 写缺了哪个词
    负样本：回答里出现"未提及/无法/没有相关" → 👍（说明它承认不知道）；否则 👎

用前先启动 API：
    .venv\\Scripts\\python.exe main.py --mode api --port 8000
然后：
    .venv\\Scripts\\python.exe eval\\replay_feedback.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests

REFUSE_HINTS = ("未提及", "没有相关", "无法", "资料中暂无", "未找到")


def _norm(text: str) -> str:
    return "".join(text.split()).lower()


def load_items(path: str, limit: int) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    pos = [i for i in items if not i.get("expect_no_answer")]
    neg = [i for i in items if i.get("expect_no_answer")]
    # 按比例抽样：约 3/4 正样本，1/4 负样本
    n_neg = max(1, limit // 4)
    n_pos = max(1, limit - n_neg)
    return pos[:n_pos] + neg[:n_neg]


def judge(item: dict, answer: str) -> tuple[int, str | None]:
    if item.get("expect_no_answer"):
        if any(h in answer for h in REFUSE_HINTS):
            return 1, None
        return -1, "库外问题却没说明未提及，可能在硬凑"
    missing = [kw for kw in item.get("answer_keywords", []) if _norm(kw) not in _norm(answer)]
    if len(missing) < len(item.get("answer_keywords", [])):
        return 1, None
    return -1, f"未包含答案关键词: {', '.join(missing)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="回放评估集并自动生成反馈")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=20, help="回放题数（默认 20，太多会很慢）")
    parser.add_argument("--questions", default=os.path.join("eval", "questions.json"))
    parser.add_argument("--timeout", type=int, default=180, help="单题超时（秒）")
    args = parser.parse_args()

    path = args.questions if os.path.isabs(args.questions) else os.path.join(PROJECT_ROOT, args.questions)
    items = load_items(path, args.limit)
    base = args.base_url.rstrip("/")

    print(f"回放 {len(items)} 题 → {base}")
    rows = []
    for idx, item in enumerate(items, start=1):
        try:
            r = requests.post(
                f"{base}/chat", json={"message": item["question"]}, timeout=args.timeout
            )
            r.raise_for_status()
            body = r.json()
            trace_id, answer = body["trace_id"], body["reply"]
        except Exception as e:
            print(f"  [{idx}/{len(items)}] {item['id']} 请求失败: {e}")
            continue

        rating, comment = judge(item, answer)
        fb = requests.post(
            f"{base}/feedback",
            json={"trace_id": trace_id, "rating": rating, "comment": comment},
            timeout=30,
        )
        if fb.status_code != 200:
            print(f"  [{idx}/{len(items)}] {item['id']} 反馈失败: {fb.status_code} {fb.text}")

        rows.append({"id": item["id"], "rating": rating, "comment": comment})
        mark = "👍" if rating == 1 else "👎"
        print(f"  [{idx}/{len(items)}] {item['id']} {mark} {item['question']}"
              + (f"  ← {comment}" if comment else ""))

    good = sum(1 for r in rows if r["rating"] == 1)
    print(f"\n回放完成：{len(rows)} 条，好评 {good}，点踩 {len(rows) - good}")

    try:
        st = requests.get(f"{base}/stats", timeout=30).json()
        print("\n=== /stats ===")
        print(json.dumps(st, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"/stats 读取失败: {e}")

    out = os.path.join(PROJECT_ROOT, "eval", "replay_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
