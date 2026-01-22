#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0_00_prepare_vsb_raw.py

从 Hugging Face 加载 BAAI/Video-SafetyBench，
将 benign / harmful 两个 split 按 video_path 合并为单条视频级元数据。
输出: data/vsb_exp0/vsb_raw_metadata.jsonl
"""

import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


def build_per_video_records(ds_benign, ds_harmful):
    """
    将 benign / harmful 按 video_path 合并成视频级记录。
    返回: list[dict]
    """
    benign_by_path = {}
    harmful_by_path = {}

    print("[INFO] Indexing benign split...")
    for row in tqdm(ds_benign, desc="benign", ncols=100):
        vp = row["video_path"]
        if vp in benign_by_path:
            # 理论上不应该重复，如果重复这里只做覆盖并提示
            print(f"[WARN] Duplicate benign video_path: {vp}, overwriting previous record.")
        benign_by_path[vp] = row

    print("[INFO] Indexing harmful split...")
    for row in tqdm(ds_harmful, desc="harmful", ncols=100):
        vp = row["video_path"]
        if vp in harmful_by_path:
            print(f"[WARN] Duplicate harmful video_path: {vp}, overwriting previous record.")
        harmful_by_path[vp] = row

    all_paths = sorted(set(list(benign_by_path.keys()) + list(harmful_by_path.keys())))
    print(f"[INFO] Unique video_path count (union benign & harmful): {len(all_paths)}")

    records = []
    num_only_benign = 0
    num_only_harmful = 0

    for idx, vp in enumerate(all_paths):
        b = benign_by_path.get(vp)
        h = harmful_by_path.get(vp)

        if b is None:
            num_only_harmful += 1
        if h is None:
            num_only_benign += 1

        # 选一个存在的行来提供 category / subcategory 等基础字段
        base = b if b is not None else h

        rec = {
            # 给一个内部 video_id，后续所有脚本用这个 ID
            "video_id": f"vsb_{idx:04d}",

            # 数据集层面的信息
            "source_dataset": "BAAI/Video-SafetyBench",
            "split": "all",  # 这里不再区分 benign/harmful，统一视为一条视频级记录

            # 类别信息
            "category_top": base["category"],
            "category_sub": base["subcategory"],

            # 文件路径：这里直接使用官方给出的 video_path 作为相对路径
            # 后续你可以将视频解压到 data/Video-SafetyBench_raw/，保证该路径在该目录下可访问
            "video_relpath": base["video_path"],

            # benign / harmful 文本
            "Q_b_raw": b["question"] if b is not None else None,
            "Q_h_raw": h["question"] if h is not None else None,

            # 对应的 question_id / harmful_intention / question_type 等额外信息
            "question_id_b": b["question_id"] if b is not None else None,
            "question_id_h": h["question_id"] if h is not None else None,
            "harmful_intention_b": b["harmful_intention"] if b is not None else None,
            "harmful_intention_h": h["harmful_intention"] if h is not None else None,
            "question_type_b": b["question_type"] if b is not None else None,
            "question_type_h": h["question_type"] if h is not None else None,
        }

        records.append(rec)

    print(f"[INFO] Videos with only benign question: {num_only_benign}")
    print(f"[INFO] Videos with only harmful question: {num_only_harmful}")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/vsb_exp0",
        help="目录用于保存 vsb_raw_metadata.jsonl",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BAAI/Video-SafetyBench",
        help="Hugging Face 数据集名",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)

    if "benign" not in ds or "harmful" not in ds:
        raise RuntimeError(f"Dataset {args.dataset_name} does not have benign/harmful splits.")

    ds_benign = ds["benign"]
    ds_harmful = ds["harmful"]

    print("[INFO] benign examples:", len(ds_benign))
    print("[INFO] harmful examples:", len(ds_harmful))

    records = build_per_video_records(ds_benign, ds_harmful)

    out_path = os.path.join(args.out_dir, "vsb_raw_metadata.jsonl")
    print(f"[INFO] Writing {len(records)} records to {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
