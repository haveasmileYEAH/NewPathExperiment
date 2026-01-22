#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_in",
        type=str,
        default="data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",
        help="Step0: VSB 子集 manifest 路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp2_understanding.yaml",
        help="Experiment 2 理解任务配置文件",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp2/manifest_exp2_mcq.jsonl",
        help="输出 MCQ manifest 路径",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    categories = cfg["categories"]
    extra_options = cfg.get("extra_options", [])

    # 预构建: category_id (即 category_top) -> option_id (A..M)
    catid_to_option = {c["id"]: c["option_id"] for c in categories}
    valid_cat_ids = set(catid_to_option.keys())

    # 构造完整 options 列表（13 类 + 额外选项）
    all_options = []
    for c in categories:
        all_options.append(
            {
                "option_id": c["option_id"],
                "category_id": c["id"],
                "description_zh": c["description_zh"],
                "description_en": c["description_en"],
            }
        )
    for extra in extra_options:
        all_options.append(
            {
                "option_id": extra["option_id"],
                "category_id": extra.get("id"),
                "description_zh": extra["description_zh"],
                "description_en": extra["description_en"],
            }
        )

    in_path = Path(args.manifest_in)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_in = 0
    num_out = 0
    num_skipped = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, desc="构建 MCQ manifest"):
            line = line.strip()
            if not line:
                continue
            num_in += 1
            rec = json.loads(line)

            video_id = rec.get("video_id")
            video_path = rec.get("video_path")
            category_top = rec.get("category_top")

            if category_top not in valid_cat_ids:
                num_skipped += 1
                print(
                    f"[WARN] video_id={video_id} 的 category_top={category_top} 不在配置 categories 中，跳过。"
                )
                continue

            correct_category_id = category_top
            correct_option_id = catid_to_option[correct_category_id]

            out_rec = {
                "video_id": video_id,
                "video_path": video_path,
                "category_top": category_top,
                "correct_category_id": correct_category_id,
                "correct_option_id": correct_option_id,
                "options": all_options,
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            num_out += 1

    print(f"[INFO] 读取 Step0 manifest 行数: {num_in}")
    print(f"[INFO] 写出 MCQ manifest 行数: {num_out}")
    print(f"[INFO] 因类别不匹配跳过: {num_skipped}")
    print(f"[INFO] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
