#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import csv
import os
from typing import Any, Dict


def main():
    parser = argparse.ArgumentParser(
        description="Build behavior_results_model-qwen2_5_vl_7b.csv from Exp1 behavior_labels JSONL"
    )
    parser.add_argument(
        "--behavior_labels",
        type=str,
        default="data/vsb_exp1/behavior_labels_model-qwen2_5_vl_7b.jsonl",
        help="Path to behavior_labels_model-qwen2_5_vl_7b.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp1/behavior_results_model-qwen2_5_vl_7b.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    in_path = args.behavior_labels
    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "global_id",
            "dataset",
            "video_id",
            "condition",
            "final_label",
            "is_harmful",
        ])

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)

            # 尝试取几个常见字段名，如果没有就用空字符串占位
            gid = obj.get("global_id") or obj.get("sample_id") or ""
            dataset = obj.get("dataset") or "VSB"
            video_id = obj.get("video_id") or obj.get("vsb_video_id") or ""
            condition = (
                obj.get("condition")
                or obj.get("attack_condition")
                or obj.get("exp_condition")
                or ""
            )
            # 行为层最终标签：比如 SAFE / REFUSAL / HARMFUL 等
            final_label = (
                obj.get("final_label")
                or obj.get("behavior_label")
                or obj.get("agg_label")
                or ""
            )

            if not gid:
                # 没有 ID 的行直接跳过
                continue

            # 简单二值化 ASR：1 = HARMFUL，0 = 其他
            is_harmful = 1 if str(final_label).upper() == "HARMFUL" else 0

            writer.writerow([
                gid,
                dataset,
                video_id,
                condition,
                final_label,
                is_harmful,
            ])
            num += 1

    print(f"[INFO] Wrote {num} behavior rows to {out_path}")


if __name__ == "__main__":
    main()
