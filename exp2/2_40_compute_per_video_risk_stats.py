#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from typing import Any, Dict

def load_video_index(path: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not gid:
                continue
            idx[gid] = obj
    return idx

def load_keyword_labels(path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kw = str(obj.get("keyword", "")).strip().lower()
            label = str(obj.get("label", "")).strip().upper()
            if kw:
                labels[kw] = label
    return labels

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-video risk statistics from keyword labels."
    )
    parser.add_argument(
        "--video_index",
        type=str,
        required=True,
        help="Path to video_index_all.jsonl",
    )
    parser.add_argument(
        "--keywords_path",
        type=str,
        required=True,
        help="Path to keywords_model-<vlm>.jsonl",
    )
    parser.add_argument(
        "--keyword_labels",
        type=str,
        required=True,
        help="Path to keyword_risk_labels_gpt4omini.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for per-video risk stats.",
    )
    args = parser.parse_args()

    video_index = load_video_index(args.video_index)
    kw_labels = load_keyword_labels(args.keyword_labels)

    per_video: Dict[str, Dict[str, Any]] = {}

    with open(args.keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not gid:
                continue
            kw_list = obj.get("keywords_list") or []
            model_name = obj.get("model_name", "")
            stats = per_video.setdefault(
                gid,
                {
                    "global_id": gid,
                    "dataset": video_index.get(gid, {}).get("dataset", ""),
                    "model_name": model_name,
                    "num_keywords": 0,
                    "num_harmful": 0,
                    "num_ambig": 0,
                    "num_nonharm": 0,
                    "num_unknown": 0,
                },
            )
            for kw in kw_list:
                kw_norm = str(kw).strip().lower()
                if not kw_norm:
                    continue
                stats["num_keywords"] += 1
                label = kw_labels.get(kw_norm)
                if label == "HARMFUL":
                    stats["num_harmful"] += 1
                elif label == "AMBIGUOUS":
                    stats["num_ambig"] += 1
                elif label == "NON_HARMFUL":
                    stats["num_nonharm"] += 1
                else:
                    stats["num_unknown"] += 1

    fieldnames = [
        "global_id",
        "dataset",
        "model_name",
        "num_keywords",
        "num_harmful",
        "num_ambig",
        "num_nonharm",
        "num_unknown",
        "risk_ratio",
    ]
    with open(args.out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for gid, s in per_video.items():
            denom = s["num_keywords"] if s["num_keywords"] > 0 else 1
            risk_ratio = s["num_harmful"] / denom
            row = dict(s)
            row["risk_ratio"] = risk_ratio
            writer.writerow(row)

    print(f"[INFO] Wrote per-video risk stats for {len(per_video)} videos to {args.out_path}")

if __name__ == "__main__":
    main()
