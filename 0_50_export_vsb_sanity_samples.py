#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0_50_export_vsb_sanity_samples.py

从 manifest_vsb_subset_seedX.jsonl 中按类别抽取少量样本，
输出 sanity 子集 manifest_vsb_sanity_samples_seedX.jsonl
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",
        help="0_30 输出的 manifest_vsb_subset_seedX.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp0/manifest_vsb_sanity_samples_seed0.jsonl",
        help="sanity 子集输出路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    parser.add_argument(
        "--per_category",
        type=int,
        default=2,
        help="每个类别抽取多少条样本 (1~3 之间即可)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    by_cat: Dict[str, List[dict]] = defaultdict(list)

    print(f"[INFO] Loading manifest from {args.manifest_path}")
    with open(args.manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cat = rec.get("category_top", "UNKNOWN")
            by_cat[cat].append(rec)

    print(f"[INFO] Categories in manifest: {len(by_cat)}")

    sanity_records = []
    for cat, items in sorted(by_cat.items(), key=lambda x: x[0]):
        if not items:
            continue
        k = min(args.per_category, len(items))
        chosen = random.sample(items, k)
        sanity_records.extend(chosen)
        print(f"[INFO] Category {cat}: total={len(items)}, chosen={k}")

    print(f"[INFO] Total sanity samples: {len(sanity_records)}")

    with open(args.out_path, "w", encoding="utf-8") as f:
        for rec in sanity_records:
            # 可以只写关键字段，也可以原样写出
            out_rec = {
                "video_id": rec["video_id"],
                "category_top": rec.get("category_top"),
                "video_path": rec.get("video_path"),
                "Q_h": rec.get("Q_h"),
                "Q_b": rec.get("Q_b"),
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Sanity manifest written to {args.out_path}")


if __name__ == "__main__":
    main()
