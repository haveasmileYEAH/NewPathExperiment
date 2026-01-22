#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0_20_sample_vsb_subset.py

根据 vsb_category_index.json 对每个类别抽样生成子集索引。
输出:
  - data/vsb_exp0/vsb_subset_indices_seed{seed}.json
  - data/vsb_exp0/vsb_subset_stats_seed{seed}.csv
"""

import argparse
import csv
import json
import os
import random
from typing import Dict, List


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category_index_path",
        type=str,
        default="data/vsb_exp0/vsb_category_index.json",
        help="0_10 输出的 vsb_category_index.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/vsb_exp0",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=80,
        help="每个类别目标抽取数量",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced_down_to_min",
        choices=["balanced_down_to_min", "unbalanced_fill_all"],
        help=(
            "balanced_down_to_min: 所有类别统一降到最小类数量和 target_per_class 的较小值；"
            "unbalanced_fill_all: 不足 target_per_class 的类别全取，其余类别抽 target_per_class。"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"[INFO] Loading category index from {args.category_index_path}")
    with open(args.category_index_path, "r", encoding="utf-8") as f:
        category_index: Dict[str, List[dict]] = json.load(f)

    # 计算每类样本数和最小值
    sizes = {cat: len(items) for cat, items in category_index.items()}
    for cat, sz in sorted(sizes.items(), key=lambda x: x[0]):
        print(f"[INFO] Category {cat}: {sz} samples")

    min_count = min(sizes.values())
    print(f"[INFO] Minimum samples across categories: {min_count}")

    if args.mode == "balanced_down_to_min":
        effective_target = min(args.target_per_class, min_count)
        print(f"[INFO] Mode = balanced_down_to_min, effective_target_per_class = {effective_target}")
    else:
        effective_target = args.target_per_class
        print(f"[INFO] Mode = unbalanced_fill_all, target_per_class = {effective_target}")

    subset_records = []
    subset_stats_rows = []

    for cat, items in sorted(category_index.items(), key=lambda x: x[0]):
        total = len(items)

        if args.mode == "balanced_down_to_min":
            target = effective_target
            if total < target:
                print(f"[WARN] Category {cat} has only {total} samples (< {target}). Using {total}.")
                target = total
            chosen = random.sample(items, target)
        else:  # unbalanced_fill_all
            if total <= effective_target:
                print(f"[INFO] Category {cat}: total={total} <= target={effective_target}, take all.")
                chosen = items
            else:
                chosen = random.sample(items, effective_target)

            target = len(chosen)

        for entry in chosen:
            rec = {
                "video_id": entry["video_id"],
                "category_top": cat,
                "video_relpath": entry["video_relpath"],
                "split": entry.get("split", "all"),
                "seed": args.seed,
                "sampling_mode": args.mode,
            }
            subset_records.append(rec)

        subset_stats_rows.append(
            {
                "category_top": cat,
                "num_selected": target,
                "num_total": total,
                "mode": args.mode,
            }
        )

    # 写子集索引
    subset_index_path = os.path.join(
        args.out_dir, f"vsb_subset_indices_seed{args.seed}.json"
    )
    with open(subset_index_path, "w", encoding="utf-8") as f:
        json.dump(subset_records, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Subset index written to {subset_index_path}")
    print(f"[INFO] Total selected videos: {len(subset_records)}")

    # 写子集统计
    subset_stats_path = os.path.join(
        args.out_dir, f"vsb_subset_stats_seed{args.seed}.csv"
    )
    with open(subset_stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category_top", "num_selected", "num_total", "mode"])
        for row in subset_stats_rows:
            writer.writerow(
                [
                    row["category_top"],
                    row["num_selected"],
                    row["num_total"],
                    row["mode"],
                ]
            )

    print(f"[INFO] Subset stats written to {subset_stats_path}")


if __name__ == "__main__":
    main()
