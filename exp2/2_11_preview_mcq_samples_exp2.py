#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_mcq",
        type=str,
        default="data/vsb_exp2/manifest_exp2_mcq.jsonl",
        help="MCQ manifest 路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp2/mcq_sanity_samples.jsonl",
        help="输出小样本文件路径",
    )
    parser.add_argument(
        "--per_category",
        type=int,
        default=3,
        help="每个 category_top 抽样样本数",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    in_path = Path(args.manifest_mcq)
    by_cat = defaultdict(list)

    with in_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="读取 MCQ manifest"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cat = rec.get("category_top")
            by_cat[cat].append(rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for cat, items in sorted(by_cat.items(), key=lambda x: str(x[0])):
            if not items:
                continue
            k = min(args.per_category, len(items))
            chosen = random.sample(items, k)
            for rec in chosen:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"[INFO] 类别数: {len(by_cat)}")
    print(f"[INFO] 共写出样本: {total_samples}")
    print(f"[INFO] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
