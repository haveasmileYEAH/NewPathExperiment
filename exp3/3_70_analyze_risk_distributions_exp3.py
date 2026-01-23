# exp3/3_70_analyze_risk_distributions_exp3.py
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def summarize(values):
    """给一组数值做统计: mean, std, p25, p50, p75."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    return {
        "mean": mean,
        "std": std,
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "num_samples": int(arr.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk_scores",
        type=str,
        default="data/vsb_exp3/risk_scores_text_model-qwen2_5_vl_7b.jsonl",
        help="3_60 输出的 risk_scores_text jsonl 文件",
    )
    parser.add_argument(
        "--out_by_condition",
        type=str,
        default="data/vsb_exp3/risk_distribution_by_condition_model-qwen2_5_vl_7b.csv",
        help="按 condition 聚合的输出 CSV",
    )
    parser.add_argument(
        "--out_by_condition_category",
        type=str,
        default="data/vsb_exp3/risk_distribution_by_condition_category_model-qwen2_5_vl_7b.csv",
        help="按 condition × category_top 聚合的输出 CSV",
    )
    parser.add_argument(
        "--metric_field",
        type=str,
        default="risk_text",
        help="风险分数字段名，默认 risk_text",
    )
    args = parser.parse_args()

    risk_path = Path(args.risk_scores)
    print(f"[INFO] 读取风险分数: {risk_path}")

    # condition -> [risk_values]
    cond_values = defaultdict(list)
    # (condition, category_top) -> [risk_values]
    cond_cat_values = defaultdict(list)

    total = 0
    used = 0
    missing_metric = 0

    for rec in iter_jsonl(risk_path):
        total += 1
        cond = rec.get("condition")
        if not cond:
            continue

        val = rec.get(args.metric_field)
        if val is None:
            # 兼容字段名固定是 risk_text 的情况
            if args.metric_field != "risk_text":
                val = rec.get("risk_text")
        score = safe_float(val)
        if score is None:
            missing_metric += 1
            continue

        cond_values[cond].append(score)

        cat = rec.get("category_top")
        if cat:
            cond_cat_values[(cond, cat)].append(score)

        used += 1

    print(f"[INFO] 读取总条数: {total}, 其中可用风险分数: {used}, 缺失分数: {missing_metric}")

    # 1) 按 condition 聚合
    out_cond_path = Path(args.out_by_condition)
    out_cond_path.parent.mkdir(parents=True, exist_ok=True)

    with out_cond_path.open("w", encoding="utf-8") as f:
        f.write("condition,num_samples,mean,std,p25,p50,p75\n")
        for cond in sorted(cond_values.keys()):
            stats = summarize(cond_values[cond])
            if stats is None:
                continue
            f.write(
                f"{cond},{stats['num_samples']},"
                f"{stats['mean']:.6f},{stats['std']:.6f},"
                f"{stats['p25']:.6f},{stats['p50']:.6f},{stats['p75']:.6f}\n"
            )

    print(f"[INFO] 按 condition 的分布写入: {out_cond_path}")

    # 2) 按 condition × category_top 聚合
    out_cc_path = Path(args.out_by_condition_category)
    out_cc_path.parent.mkdir(parents=True, exist_ok=True)

    with out_cc_path.open("w", encoding="utf-8") as f:
        f.write("condition,category_top,num_samples,mean,std,p25,p50,p75\n")

        # 排序: 先按 condition, 再按 category
        for (cond, cat) in sorted(cond_cat_values.keys(), key=lambda x: (x[0], x[1])):
            stats = summarize(cond_cat_values[(cond, cat)])
            if stats is None:
                continue
            f.write(
                f"{cond},{cat},{stats['num_samples']},"
                f"{stats['mean']:.6f},{stats['std']:.6f},"
                f"{stats['p25']:.6f},{stats['p50']:.6f},{stats['p75']:.6f}\n"
            )

    print(f"[INFO] 按 condition×category_top 的分布写入: {out_cc_path}")


if __name__ == "__main__":
    main()
