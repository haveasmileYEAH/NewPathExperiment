#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2_41_merge_behavior_and_risk_features_exp2.py

功能：
  - 将 Exp2 解析后的风险理解结果（video 级别）与 Exp1 行为标签（attack 级别）在 video_id 上做 merge。
  - 目前只关注 VH-BQ 条件下的 final_label，用于“Benign Query + Harmful Video”场景的对齐分析。

输入：
  - risk_features_parsed_model-*.jsonl（来自 2_31）
  - behavior_labels_model-*.jsonl（来自 Exp1 的 1_50/1_60 pipeline）

输出：
  - understanding_vs_behavior_model-*.jsonl（每视频一行）
"""

import argparse
import json
from pathlib import Path
from collections import Counter


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
        "--risk_features_parsed",
        type=str,
        default="data/vsb_exp2/risk_features_parsed_model-qwen2_5_vl_7b.jsonl",
        help="2_31 生成的风险理解解析结果 JSONL 路径",
    )
    parser.add_argument(
        "--behavior_labels",
        type=str,
        default="data/vsb_exp1/behavior_labels_model-qwen2_5_vl_7b.jsonl",
        help="Exp1 行为标签 JSONL 路径（包含 condition / final_label 等）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp2/risk_vs_behavior_model-qwen2_5_vl_7b.jsonl",
        help="合并后输出 JSONL 路径",
    )

    args = parser.parse_args()

    print(f"[INFO] 读取风险理解解析结果: {args.risk_features_parsed}")
    risk_recs = list(iter_jsonl(args.risk_features_parsed))
    print(f"[INFO] 风险理解样本数: {len(risk_recs)}")

    print(f"[INFO] 读取行为标签: {args.behavior_labels}")
    behavior_recs = list(iter_jsonl(args.behavior_labels))
    print(f"[INFO] 行为标签记录总数: {len(behavior_recs)}")

    # 1) 从行为标签中提取 VH-BQ 条件下的 final_label，建立 video_id -> final_label 映射
    video2label = {}
    video2cnt = Counter()

    for rec in behavior_recs:
        condition = rec.get("condition")
        if condition != "VH-BQ":
            continue

        video_id = rec.get("video_id")
        if not video_id:
            continue

        final_label = rec.get("final_label", "UNKNOWN")
        video2cnt[video_id] += 1
        video2label[video_id] = final_label

    print(f"[INFO] VH-BQ 条件下有行为标签的视频数: {len(video2label)}")
    # 如果同一个 video_id 对应多个 VH-BQ 记录，这里也统计一下，方便你后续排查
    multi_count = sum(1 for v, c in video2cnt.items() if c > 1)
    if multi_count > 0:
        print(f"[WARN] 有 {multi_count} 个 video_id 在 VH-BQ 下出现了多条记录（将采用最后一条的 final_label）")

    # 2) 合并到风险理解结果上
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = 0
    missing = 0
    label_counter = Counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in risk_recs:
            video_id = rec.get("video_id")
            if not video_id:
                rec["final_label_VH_BQ"] = "MISSING_VIDEO_ID"
                missing += 1
            else:
                fl = video2label.get(video_id)
                if fl is None:
                    rec["final_label_VH_BQ"] = "MISSING"
                    missing += 1
                else:
                    rec["final_label_VH_BQ"] = fl
                    merged += 1
                    label_counter[fl] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(risk_recs)
    print(f"[INFO] 合并完成，样本总数: {total}")
    print(f"[INFO] 找到 VH-BQ 行为标签的样本数: {merged} ({merged / total:.3f})")
    print(f"[INFO] 未找到 VH-BQ 行为标签的样本数: {missing} ({missing / total:.3f})")
    print("[INFO] 在合并子集内 final_label_VH_BQ 分布:")
    for label, cnt in label_counter.most_common():
        print(f"  - {label:10s}: {cnt:4d}")

    print(f"[INFO] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
