#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3_21_dump_summaries_teacher_vs_qwen.py

作用：
- 读取 Teacher 和 Qwen 的 100 条 summary；
- 按 global_id 对齐；
- 按顺序打印出来，方便人工扫一遍。

输入文件（写死为你当前的 debug 路径）：
- data/vsb_eval/teacher_summaries_llama90b_debug3.jsonl
- data/vsb_eval/qwen_summaries_video_debug3.jsonl
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    teacher_path = Path("data/vsb_eval/teacher_summaries_llama90b_debug3.jsonl")
    qwen_path = Path("data/vsb_eval/qwen_summaries_video_debug3.jsonl")

    teacher_recs = read_jsonl(teacher_path)
    qwen_recs = read_jsonl(qwen_path)

    print("=== Basic counts ===")
    print(f"Teacher summaries: {len(teacher_recs)}")
    print(f"Qwen summaries   : {len(qwen_recs)}")

    # 映射成 {global_id: rec}
    teacher_by_id: Dict[str, Dict[str, Any]] = {}
    for r in teacher_recs:
        gid = r.get("global_id")
        if isinstance(gid, str):
            teacher_by_id[gid] = r

    qwen_by_id: Dict[str, Dict[str, Any]] = {}
    for r in qwen_recs:
        gid = r.get("global_id")
        if isinstance(gid, str):
            qwen_by_id[gid] = r

    teacher_ids = set(teacher_by_id.keys())
    qwen_ids = set(qwen_by_id.keys())
    inter_ids = sorted(teacher_ids & qwen_ids)

    print("\n=== ID alignment ===")
    print(f"Teacher only: {len(teacher_ids - qwen_ids)}")
    print(f"Qwen only   : {len(qwen_ids - teacher_ids)}")
    print(f"Intersection: {len(inter_ids)}")

    print("\n=== Dumping summaries (Teacher vs Qwen) ===\n")

    for gid in inter_ids:
        t = teacher_by_id[gid]
        q = qwen_by_id[gid]

        teacher_summary = t.get("teacher_summary", "").strip()
        qwen_summary = q.get("model_summary", "").strip()

        print("=" * 80)
        print(f"global_id: {gid}")
        print("\n[Teacher summary]")
        print(teacher_summary or "<EMPTY>")

        print("\n[Qwen summary]")
        print(qwen_summary or "<EMPTY>")
        print()  # 空行分隔


if __name__ == "__main__":
    main()
