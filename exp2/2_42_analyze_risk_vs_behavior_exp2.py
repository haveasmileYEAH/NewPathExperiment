#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2_42_analyze_risk_vs_behavior_exp2.py

功能：
  - 基于 2_41 的合并结果（risk_vs_behavior），按 VH-BQ 下的行为标签分组，
    分析风险理解特征（risk_keywords / video_intent）与攻击行为之间的关系。

输入：
  - risk_vs_behavior_model-*.jsonl

输出：
  - metrics_risk_by_behavior_model-*.csv
  - metrics_intent_by_behavior_model-*.csv
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import csv


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged",
        type=str,
        default="data/vsb_exp2/risk_vs_behavior_model-qwen2_5_vl_7b.jsonl",
        help="2_41 合并后的 JSONL 路径",
    )
    parser.add_argument(
        "--out_risk",
        type=str,
        default="data/vsb_exp2/metrics_risk_by_behavior_model-qwen2_5_vl_7b.csv",
        help="按行为标签汇总风险关键词指标的 CSV 输出路径",
    )
    parser.add_argument(
        "--out_intent",
        type=str,
        default="data/vsb_exp2/metrics_intent_by_behavior_model-qwen2_5_vl_7b.csv",
        help="按行为标签 × video_intent 统计的 CSV 输出路径",
    )

    args = parser.parse_args()

    print(f"[INFO] 读取合并结果: {args.merged}")
    recs = list(iter_jsonl(args.merged))
    total = len(recs)
    print(f"[INFO] 样本总数: {total}")

    # 按 final_label_VH_BQ 分组
    groups = defaultdict(list)
    for rec in recs:
        fl = rec.get("final_label_VH_BQ", "MISSING")
        groups[fl].append(rec)

    # 1) 按行为标签统计 risk_keywords 指标
    out_risk_path = Path(args.out_risk)
    out_risk_path.parent.mkdir(parents=True, exist_ok=True)

    with out_risk_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "final_label_VH_BQ",
                "num_samples",
                "num_has_risk_keywords",
                "ratio_has_risk_keywords",
                "avg_num_risk_keywords",
            ]
        )

        print("[INFO] 按行为标签统计 risk_keywords：")
        for fl, items in groups.items():
            n = len(items)
            if n == 0:
                continue

            num_has_risk = 0
            total_risk_kw = 0

            for rec in items:
                has_risk = rec.get("has_risk_keywords", False)
                if has_risk:
                    num_has_risk += 1
                num_risk = rec.get("num_risk_keywords", 0)
                total_risk_kw += num_risk

            ratio_has_risk = num_has_risk / n
            avg_num_risk = total_risk_kw / n

            print(
                f"  - {fl:10s}: n={n:4d}, has_risk={num_has_risk:3d} ({ratio_has_risk:.3f}), "
                f"avg_num_risk_keywords={avg_num_risk:.3f}"
            )

            writer.writerow(
                [
                    fl,
                    n,
                    num_has_risk,
                    f"{ratio_has_risk:.6f}",
                    f"{avg_num_risk:.6f}",
                ]
            )

    print(f"[INFO] risk_keywords 汇总写入: {out_risk_path}")

    # 2) 按行为标签 × video_intent 统计
    out_intent_path = Path(args.out_intent)
    out_intent_path.parent.mkdir(parents=True, exist_ok=True)

    with out_intent_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "final_label_VH_BQ",
                "video_intent",
                "num_samples",
                "ratio_in_group",
            ]
        )

        print("[INFO] 按行为标签 × video_intent 统计：")
        for fl, items in groups.items():
            n = len(items)
            if n == 0:
                continue

            cnt_intent = Counter(rec.get("video_intent", "unknown") for rec in items)

            print(f"  [Group: {fl}, n={n}]")
            for intent, cnt in cnt_intent.most_common():
                ratio = cnt / n
                print(f"    - {intent:24s}: {cnt:3d} ({ratio:.3f})")
                writer.writerow(
                    [
                        fl,
                        intent,
                        cnt,
                        f"{ratio:.6f}",
                    ]
                )

    print(f"[INFO] video_intent 分布写入: {out_intent_path}")


if __name__ == "__main__":
    main()
