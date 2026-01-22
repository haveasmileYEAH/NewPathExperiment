#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcq_results",
        type=str,
        default="data/vsb_exp2/mcq_results_model-qwen2_5_vl_7b.jsonl",
        help="MCQ 打分结果文件",
    )
    parser.add_argument(
        "--out_overall",
        type=str,
        default="data/vsb_exp2/metrics_understanding_overall_model-qwen2_5_vl_7b.json",
        help="整体理解指标输出 JSON",
    )
    parser.add_argument(
        "--out_by_category",
        type=str,
        default="data/vsb_exp2/metrics_understanding_by_category_model-qwen2_5_vl_7b.csv",
        help="按类别理解指标输出 CSV",
    )
    args = parser.parse_args()

    in_path = Path(args.mcq_results)
    overall_total = 0
    overall_correct = 0

    by_cat = defaultdict(lambda: {"total": 0, "correct": 0})

    with in_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="统计理解指标"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            is_correct = int(rec.get("is_correct", 0))
            cat = rec.get("category_top")

            overall_total += 1
            overall_correct += is_correct

            by_cat[cat]["total"] += 1
            by_cat[cat]["correct"] += is_correct

    out_overall_path = Path(args.out_overall)
    out_overall_path.parent.mkdir(parents=True, exist_ok=True)

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    with out_overall_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_total": overall_total,
                "num_correct": overall_correct,
                "overall_top1_acc": overall_acc,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[INFO] overall_top1_acc = {overall_acc:.4f} "
        f"({overall_correct}/{overall_total})"
    )
    print(f"[INFO] overall 指标写入: {out_overall_path}")

    out_by_cat_path = Path(args.out_by_category)
    out_by_cat_path.parent.mkdir(parents=True, exist_ok=True)

    with out_by_cat_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category_top", "num_total", "num_correct", "top1_acc"])
        for cat, stats in sorted(by_cat.items(), key=lambda x: str(x[0])):
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0.0
            writer.writerow([cat, total, correct, f"{acc:.6f}"])

    print(f"[INFO] 按类别指标写入: {out_by_cat_path}")


if __name__ == "__main__":
    main()
