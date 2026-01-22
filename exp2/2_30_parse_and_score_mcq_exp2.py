#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def parse_option_id(text: str, allowed: set) -> str:
    """
    从模型原始输出中解析出第一个合法的选项字母。
    allowed: 允许的选项集合，例如 {'A', 'B', ..., 'O'}
    """
    if not text:
        return "INVALID"

    upper = text.upper()
    for ch in upper:
        if "A" <= ch <= "Z" and ch in allowed:
            return ch
    return "INVALID"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_mcq",
        type=str,
        default="data/vsb_exp2/manifest_exp2_mcq.jsonl",
        help="MCQ manifest 路径",
    )
    parser.add_argument(
        "--raw_outputs",
        type=str,
        default="data/vsb_exp2/raw_mcq_outputs_model-qwen2_5_vl_7b.jsonl",
        help="模型原始输出文件",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp2/mcq_results_model-qwen2_5_vl_7b.jsonl",
        help="解析并打分后的结果文件",
    )
    args = parser.parse_args()

    # 1. 建立 video_id -> (category_top, correct_option_id, video_path, options_info)
    manifest_path = Path(args.manifest_mcq)
    mcq_index = {}
    allowed_ids = None

    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="读取 MCQ manifest"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            mcq_index[vid] = {
                "category_top": rec.get("category_top"),
                "correct_option_id": rec.get("correct_option_id"),
                "video_path": rec.get("video_path"),
                "options": rec.get("options", []),
            }
            if allowed_ids is None:
                allowed_ids = {o["option_id"] for o in rec.get("options", [])}

    if allowed_ids is None:
        raise RuntimeError("未能从 MCQ manifest 中解析出合法选项集合 allowed_ids。")

    print(f"[INFO] 从 manifest 中读取 {len(mcq_index)} 条 video_id 记录")
    print(f"[INFO] 合法选项集合: {sorted(allowed_ids)}")

    raw_path = Path(args.raw_outputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_total = 0
    num_scored = 0
    num_missing_manifest = 0
    num_invalid = 0
    num_correct = 0

    with raw_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, desc="解析模型输出"):
            line = line.strip()
            if not line:
                continue
            num_total += 1
            rec = json.loads(line)

            video_id = rec["video_id"]
            raw_answer_text = rec.get("raw_answer_text", "")
            model_id = rec.get("model_id")
            model_name = rec.get("model_name")

            if video_id not in mcq_index:
                num_missing_manifest += 1
                print(f"[WARN] video_id={video_id} 不在 MCQ manifest 中，跳过。")
                continue

            mcq_info = mcq_index[video_id]
            correct_option_id = mcq_info["correct_option_id"]
            category_top = mcq_info["category_top"]
            video_path = mcq_info["video_path"]

            pred_option_id = parse_option_id(raw_answer_text, allowed_ids)
            if pred_option_id == "INVALID":
                num_invalid += 1
                is_correct = 0
            else:
                is_correct = int(pred_option_id == correct_option_id)
                if is_correct:
                    num_correct += 1

            out_rec = {
                "video_id": video_id,
                "video_path": video_path,
                "model_id": model_id,
                "model_name": model_name,
                "category_top": category_top,
                "correct_option_id": correct_option_id,
                "pred_option_id": pred_option_id,
                "is_correct": is_correct,
                "raw_answer_text": raw_answer_text,
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            num_scored += 1

    print(f"[INFO] 原始输出总行数: {num_total}")
    print(f"[INFO] 成功打分样本数: {num_scored}")
    print(f"[INFO] manifest 缺失样本数: {num_missing_manifest}")
    print(f"[INFO] INVALID 解析样本数: {num_invalid}")
    if num_scored > 0:
        print(f"[INFO] overall_top1_acc = {num_correct / num_scored:.4f}")
    print(f"[INFO] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
