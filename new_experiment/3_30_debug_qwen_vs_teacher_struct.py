#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3_30_debug_qwen_vs_teacher_struct.py

对比：
- Teacher: data/vsb_eval/teacher_structured_labels_llama90b_debug.jsonl
- Qwen   : data/vsb_eval/qwen_structured_labels_video_debug3.jsonl

只看结构化 6 个字段的一致性，
并重点列出 visible_weapon == "clear_weapon" 的样本。
"""

import json
from collections import Counter, defaultdict
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
    teacher_path = Path("data/vsb_eval/teacher_structured_labels_llama90b_debug.jsonl")
    qwen_path = Path("data/vsb_eval/qwen_structured_labels_video_debug3.jsonl")

    teacher_recs = read_jsonl(teacher_path)
    qwen_recs = read_jsonl(qwen_path)

    print("=== Basic counts ===")
    print(f"Teacher records: {len(teacher_recs)}")
    print(f"Qwen records   : {len(qwen_recs)}")

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
    inter_ids = teacher_ids & qwen_ids

    print("\n=== ID alignment ===")
    print(f"Teacher only: {len(teacher_ids - qwen_ids)}")
    print(f"Qwen only   : {len(qwen_ids - teacher_ids)}")
    print(f"Intersection: {len(inter_ids)}")

    fields = [
        "num_visible_people",
        "main_environment",
        "primary_focus",
        "physical_contact",
        "visible_weapon",
        "camera_view",
    ]

    # 统计每个字段的 match / total
    match_counts = Counter()
    total_counts = Counter()

    # 顺便记录每个字段的一些 mismatch 例子
    mismatch_examples = {f: [] for f in fields}

    # 统计 parse_error
    teacher_parse = Counter(r.get("parse_error", False) for r in teacher_by_id.values())
    qwen_parse = Counter(r.get("parse_error", False) for r in qwen_by_id.values())

    print("\n=== parse_error distribution ===")
    print(f"Teacher: {dict(teacher_parse)}")
    print(f"Qwen   : {dict(qwen_parse)}")

    # 逐条比对
    for gid in sorted(inter_ids):
        t = teacher_by_id[gid]
        q = qwen_by_id[gid]

        for f in fields:
            t_val = t.get(f)
            q_val = q.get(f)
            # 如果 teacher 这个字段是 unclear，就不纳入比较（等价于“teacher 自己也不确定”）
            if t_val is None or t_val == "unclear":
                continue
            total_counts[f] += 1
            if t_val == q_val:
                match_counts[f] += 1
            else:
                if len(mismatch_examples[f]) < 10:
                    mismatch_examples[f].append((gid, t_val, q_val))

    print("\n=== Field-level agreement (Teacher vs Qwen) ===")
    for f in fields:
        tot = total_counts[f]
        match = match_counts[f]
        acc = match / tot if tot > 0 else 0.0
        print(f"{f:20s}  match={match:3d} / {tot:3d}  acc={acc:.3f}")

    # 输出每个字段前几个 mismatch 样本
    print("\n=== Example mismatches per field (up to 10 each) ===")
    for f in fields:
        if not mismatch_examples[f]:
            continue
        print(f"\n--- {f} ---")
        for gid, t_val, q_val in mismatch_examples[f]:
            print(f"{gid}: teacher={t_val!r}, qwen={q_val!r}")

    # 专门关注 visible_weapon == "clear_weapon" 的样本
    print("\n=== Samples with teacher.visible_weapon == 'clear_weapon' ===")
    clear_weapon_ids = []
    for gid in sorted(inter_ids):
        t = teacher_by_id[gid]
        if t.get("visible_weapon") == "clear_weapon":
            clear_weapon_ids.append(gid)

    print(f"Total clear_weapon samples (teacher): {len(clear_weapon_ids)}\n")

    for gid in clear_weapon_ids:
        t = teacher_by_id[gid]
        q = qwen_by_id.get(gid)
        if q is None:
            continue
        print("=" * 80)
        print(f"global_id: {gid}")
        print("[Teacher]")
        print(
            f"  num_visible_people : {t.get('num_visible_people')}\n"
            f"  main_environment   : {t.get('main_environment')}\n"
            f"  primary_focus      : {t.get('primary_focus')}\n"
            f"  physical_contact   : {t.get('physical_contact')}\n"
            f"  visible_weapon     : {t.get('visible_weapon')}\n"
            f"  camera_view        : {t.get('camera_view')}"
        )
        print("[Qwen]")
        print(
            f"  num_visible_people : {q.get('num_visible_people')}\n"
            f"  main_environment   : {q.get('main_environment')}\n"
            f"  primary_focus      : {q.get('primary_focus')}\n"
            f"  physical_contact   : {q.get('physical_contact')}\n"
            f"  visible_weapon     : {q.get('visible_weapon')}\n"
            f"  camera_view        : {q.get('camera_view')}"
        )


if __name__ == "__main__":
    main()
