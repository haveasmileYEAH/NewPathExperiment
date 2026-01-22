#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0_30_build_vsb_subset_manifest.py

根据 vsb_raw_metadata.jsonl 和 vsb_subset_indices_seedX.json
构建统一子集 manifest: manifest_vsb_subset_seedX.jsonl
"""

import argparse
import json
import os
from typing import Dict


def load_raw_metadata(raw_metadata_path: str) -> Dict[str, dict]:
    """
    读取 vsb_raw_metadata.jsonl，返回 video_id -> rec 映射。
    """
    mapping: Dict[str, dict] = {}
    with open(raw_metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            if vid in mapping:
                raise RuntimeError(f"Duplicate video_id in raw metadata: {vid}")
            mapping[vid] = rec
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_metadata_path",
        type=str,
        default="data/vsb_exp0/vsb_raw_metadata.jsonl",
        help="0_00 输出的 vsb_raw_metadata.jsonl",
    )
    parser.add_argument(
        "--subset_indices_path",
        type=str,
        default="data/vsb_exp0/vsb_subset_indices_seed0.json",
        help="0_20 输出的 vsb_subset_indices_seedX.json",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="data/Video-SafetyBench_raw",
        help="视频根目录（与 video_relpath 拼接生成 video_path）",
    )
    parser.add_argument(
        "--out_manifest_path",
        type=str,
        default="data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",
        help="输出 manifest 路径",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_manifest_path), exist_ok=True)

    print(f"[INFO] Loading raw metadata from {args.raw_metadata_path}")
    raw_meta = load_raw_metadata(args.raw_metadata_path)
    print(f"[INFO] Raw metadata videos: {len(raw_meta)}")

    print(f"[INFO] Loading subset indices from {args.subset_indices_path}")
    with open(args.subset_indices_path, "r", encoding="utf-8") as f:
        subset_indices = json.load(f)

    print(f"[INFO] Subset size: {len(subset_indices)}")

    # 构建 manifest
    num_missing = 0
    with open(args.out_manifest_path, "w", encoding="utf-8") as out_f:
        for entry in subset_indices:
            vid = entry["video_id"]
            meta = raw_meta.get(vid)
            if meta is None:
                print(f"[WARN] video_id {vid} not found in raw metadata, skip.")
                num_missing += 1
                continue

            video_relpath = meta["video_relpath"]
            video_path = os.path.join(args.video_root, video_relpath)

            manifest_rec = {
                # 标识信息
                "video_id": vid,
                "category_top": meta["category_top"],
                "category_sub": meta.get("category_sub"),
                "split": meta.get("split", "all"),
                "source_dataset": meta.get("source_dataset", "BAAI/Video-SafetyBench"),

                # 文件路径
                "video_relpath": video_relpath,
                "video_path": video_path,

                # 文本字段
                "Q_h": meta.get("Q_h_raw"),
                "Q_b": meta.get("Q_b_raw"),

                # 抽样信息
                "seed": entry.get("seed"),
                "sampling_mode": entry.get("sampling_mode"),
            }

            out_f.write(json.dumps(manifest_rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Manifest written to {args.out_manifest_path}")
    if num_missing > 0:
        print(f"[WARN] {num_missing} entries in subset_indices not found in raw metadata.")


if __name__ == "__main__":
    main()
