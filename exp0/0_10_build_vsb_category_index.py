#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0_10_build_vsb_category_index.py

从 vsb_raw_metadata.jsonl 按 category_top 构建类别索引和统计信息。
输出:
  - data/vsb_exp0/vsb_category_index.json
  - data/vsb_exp0/vsb_category_stats.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/vsb_exp0/vsb_raw_metadata.jsonl",
        help="0_00 输出的 vsb_raw_metadata.jsonl 路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/vsb_exp0",
        help="输出目录",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    category_index: Dict[str, List[dict]] = defaultdict(list)

    print(f"[INFO] Reading metadata from {args.metadata_path}")
    num_lines = 0
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            num_lines += 1
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            cat_top = rec.get("category_top", "UNKNOWN")
            entry = {
                "video_id": rec["video_id"],
                "video_relpath": rec["video_relpath"],
                "category_sub": rec.get("category_sub"),
                "split": rec.get("split", "all"),
            }
            category_index[cat_top].append(entry)

    print(f"[INFO] Total records read: {num_lines}")
    print(f"[INFO] Unique category_top count: {len(category_index)}")

    # 写类别索引
    index_path = os.path.join(args.out_dir, "vsb_category_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(category_index, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Category index written to {index_path}")

    # 写统计信息
    stats_path = os.path.join(args.out_dir, "vsb_category_stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category_top", "num_samples"])
        for cat_top, items in sorted(category_index.items(), key=lambda x: x[0]):
            writer.writerow([cat_top, len(items)])

    print(f"[INFO] Category stats written to {stats_path}")


if __name__ == "__main__":
    main()
