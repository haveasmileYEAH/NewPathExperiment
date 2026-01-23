# exp3/3_80_merge_risk_with_behavior_exp3.py
import argparse
import json
from pathlib import Path


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def infer_final_label(rec):
    """
    尝试从行为标签记录中推断最终标签字段:
    优先使用 'final_label'；
    否则在形如 'final_label_VH_BQ' 等字段中，根据 condition 进行匹配。
    """
    if "final_label" in rec:
        return rec["final_label"]

    cond = rec.get("condition", "")
    cond_norm = cond.replace("-", "_").lower() if cond else ""

    # 找所有可能的 final_label* 字段
    candidates = [k for k in rec.keys() if k.startswith("final_label")]
    if not candidates:
        return None

    if len(candidates) == 1:
        return rec[candidates[0]]

    # 如果有多个，就根据 condition 做匹配
    for k in candidates:
        kl = k.lower()
        if cond_norm and cond_norm in kl:
            return rec[k]
        if cond and cond.lower() in kl:
            return rec[k]

    # 实在匹配不到，就随便选一个（至少有东西），但会有提示
    return rec[candidates[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk_scores",
        type=str,
        default="data/vsb_exp3/risk_scores_text_model-qwen2_5_vl_7b.jsonl",
        help="3_60 输出的 risk_text 分数文件",
    )
    parser.add_argument(
        "--behavior_labels",
        type=str,
        default="data/vsb_exp1/behavior_labels_model-qwen2_5_vl_7b.jsonl",
        help="Exp1 行为标签文件",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp3/risk_vs_behavior_text_model-qwen2_5_vl_7b.jsonl",
        help="合并后输出文件",
    )
    args = parser.parse_args()

    risk_path = Path(args.risk_scores)
    beh_path = Path(args.behavior_labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取行为标签，建立索引
    print(f"[INFO] 读取行为标签: {beh_path}")
    by_attack = {}
    by_vid_cond = {}

    n_beh = 0
    for rec in iter_jsonl(beh_path):
        n_beh += 1
        aid = rec.get("attack_id")
        vid = rec.get("video_id")
        cond = rec.get("condition")
        if aid is not None:
            by_attack[aid] = rec
        if vid is not None and cond is not None:
            by_vid_cond[(vid, cond)] = rec

    print(f"[INFO] 行为标签记录总数: {n_beh}")
    print(f"[INFO] by_attack 大小: {len(by_attack)}, by_vid_cond 大小: {len(by_vid_cond)}")

    # 2) 读取 risk_scores，并与行为标签合并
    print(f"[INFO] 读取风险分数: {risk_path}")

    total = 0
    merged = 0
    no_label = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in iter_jsonl(risk_path):
            total += 1
            aid = rec.get("attack_id")
            vid = rec.get("video_id")
            cond = rec.get("condition")
            category_top = rec.get("category_top")
            risk_text = rec.get("risk_text")

            # 查找行为标签记录
            beh_rec = None
            if aid is not None and aid in by_attack:
                beh_rec = by_attack[aid]
            elif vid is not None and cond is not None and (vid, cond) in by_vid_cond:
                beh_rec = by_vid_cond[(vid, cond)]

            final_label = None
            if beh_rec is not None:
                final_label = infer_final_label(beh_rec)
                # 如果行为那边有 category_top 也可以顺带拿一下
                if category_top is None:
                    category_top = beh_rec.get("category_top")

            if beh_rec is None:
                no_label += 1

            merged_rec = {
                "attack_id": aid,
                "video_id": vid,
                "condition": cond,
                "category_top": category_top,
                "risk_text": risk_text,
            }
            if final_label is not None:
                merged_rec["final_label"] = final_label

            f_out.write(json.dumps(merged_rec, ensure_ascii=False) + "\n")
            merged += 1

    print(f"[INFO] 合并完成，总样本: {total}, 输出记录: {merged}, 未找到行为标签: {no_label}")
    print(f"[INFO] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
