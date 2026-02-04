#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict

# 核心 & 辅助字段（去掉 camera_view）
CORE_FIELDS = ["primary_focus", "physical_contact", "visible_weapon"]
AUX_FIELDS  = ["num_visible_people", "main_environment"]

def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def index_by_gid(rows, key="global_id"):
    idx = {}
    for r in rows:
        gid = r.get(key)
        if gid is not None:
            idx[gid] = r
    return idx

def main():
    teacher_struct_path = Path("data/vsb_eval/teacher_structured_labels_llama90b_debug.jsonl")
    qwen_struct_path    = Path("data/vsb_eval/qwen_structured_labels_video_debug3.jsonl")
    judge_scores_path   = Path("data/vsb_eval/summary_scores_qwen_gpt4_1mini_100.jsonl")
    out_path            = Path("data/vsb_eval/understanding_flags_qwen_100_soft.jsonl")

    teacher_rows = load_jsonl(teacher_struct_path)
    qwen_rows    = load_jsonl(qwen_struct_path)
    judge_rows   = load_jsonl(judge_scores_path)

    t_idx = index_by_gid(teacher_rows)
    q_idx = index_by_gid(qwen_rows)
    j_idx = index_by_gid(judge_rows)

    common_ids = sorted(set(t_idx.keys()) & set(q_idx.keys()) & set(j_idx.keys()))
    print(f"[INFO] Common samples: {len(common_ids)}")

    import statistics as st
    struct_scores = []
    overall_sims  = []
    u_scores      = []
    understood_cnt = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for gid in common_ids:
            t = t_idx[gid]
            q = q_idx[gid]
            j = j_idx[gid]

            # --- 1) 核心 & 辅助字段匹配 ---
            core_match = 0
            for field in CORE_FIELDS:
                if t.get(field) == q.get(field):
                    core_match += 1
            core_match_ratio = core_match / len(CORE_FIELDS)

            aux_match = 0
            for field in AUX_FIELDS:
                if t.get(field) == q.get(field):
                    aux_match += 1
            aux_match_ratio = aux_match / len(AUX_FIELDS)

            struct_score = 0.7 * core_match_ratio + 0.3 * aux_match_ratio

            # --- 2) 文本相似度 ---
            scores = j.get("scores", {})
            overall = float(scores.get("overall_similarity", 0.0))
            correctness = float(scores.get("correctness", 0.0))
            overall_norm = overall / 5.0

            # --- 3) U_score & 判定 ---
            u_score = 0.5 * struct_score + 0.5 * overall_norm
            is_understood = (u_score >= 0.6) and (correctness >= 3.5)

            if is_understood:
                understood_cnt += 1

            struct_scores.append(struct_score)
            overall_sims.append(overall)
            u_scores.append(u_score)

            rec = {
                "global_id": gid,
                "core_match_ratio": core_match_ratio,
                "aux_match_ratio": aux_match_ratio,
                "struct_score": struct_score,
                "overall_similarity": overall,
                "correctness": correctness,
                "u_score": u_score,
                "is_understood": bool(is_understood),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== Soft-U Summary ===")
    print(f"Total samples       : {len(common_ids)}")
    print(f"Mean struct_score   : {st.mean(struct_scores):.3f}")
    print(f"Mean overall_sim    : {st.mean(overall_sims):.3f}")
    print(f"Mean U_score        : {st.mean(u_scores):.3f}")
    print(f"Understood (U=1)    : {understood_cnt} / {len(common_ids)} "
          f"({understood_cnt/len(common_ids):.3%})")
    print(f"[INFO] Written soft flags to {out_path}")

if __name__ == "__main__":
    main()
