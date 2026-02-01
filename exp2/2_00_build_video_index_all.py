#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

def _load_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            # 如果是 {id: {...}} 这种结构，自动把 key 写回 video_id
            for k, v in data.items():
                if isinstance(v, dict):
                    v.setdefault("video_id", k)
                yield v
        else:
            raise ValueError(f"Unsupported JSON structure in {path!r}: {type(data)}")

def _guess_field(d: Dict[str, Any], candidates: List[str], default=None):
    for key in candidates:
        if key in d and d[key] is not None:
            return d[key]
    return default

def build_vsb_index(vsb_ann_path: str, vsb_video_root: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw in _load_json_or_jsonl(vsb_ann_path):
        video_id = _guess_field(raw, ["video_id", "id", "video_name", "name"])
        if video_id is None:
            raise ValueError(f"Cannot find video id in annotation item: keys={list(raw.keys())[:10]}")
        split = _guess_field(raw, ["split", "subset"], default="all")
        label_fine = _guess_field(raw, ["vsb_label_fine", "fine_label", "label", "category"])
        harmful_prompt = _guess_field(raw, ["harmful_query", "harmful_prompt", "attack_query"])
        benign_prompt = _guess_field(raw, ["benign_query", "benign_prompt", "benign_text"])
        video_rel_path = _guess_field(raw, ["video_path", "path", "video_file"], default=f"{video_id}.mp4")
        video_path = video_rel_path
        if not os.path.isabs(video_rel_path):
            video_path = os.path.join(vsb_video_root, video_rel_path)
        rec = {
            "global_id": f"VSB_{video_id}",
            "dataset": "VSB",
            "split": split,
            "video_path": video_path,
            "vsb_label_fine": label_fine,
            "vsb_label_id": raw.get("vsb_label_id"),
            "harmful_prompt": harmful_prompt,
            "benign_prompt": benign_prompt,
            "meta": {
                "source": "vsb_ann",
            },
        }
        records.append(rec)
    return records

def main():
    parser = argparse.ArgumentParser(
        description="Build unified video_index_all.jsonl for Exp2. Currently only supports Video-SafetyBench (VSB)."
    )
    parser.add_argument("--vsb_ann_path", type=str, required=True, help="Path to VSB annotation JSON/JSONL.")
    parser.add_argument("--vsb_video_root", type=str, required=True, help="Root directory containing VSB video files.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to output JSONL index.")
    args = parser.parse_args()

    records = build_vsb_index(args.vsb_ann_path, args.vsb_video_root)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote {len(records)} records to {args.out_path}")

if __name__ == "__main__":
    main()
