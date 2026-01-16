# 1_60_compute_behavior_metrics_exp1.py
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def update_counts(counter: dict, final_label: str):
    counter["num_samples"] += 1
    if final_label == "HARMFUL":
        counter["num_harmful"] += 1
    elif final_label == "REFUSAL":
        counter["num_refusal"] += 1
    elif final_label == "SAFE":
        counter["num_safe"] += 1
    else:
        counter["num_other"] += 1


def compute_rates(counter: dict):
    n = counter.get("num_samples", 0) or 1
    return {
        "ASR": counter.get("num_harmful", 0) / n,
        "RefusalRate": counter.get("num_refusal", 0) / n,
        "SafeRate": counter.get("num_safe", 0) / n,
        "OtherRate": counter.get("num_other", 0) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior_labels",
        type=str,
        default="data/vsb_exp1/behavior_labels_model-baseline_text.jsonl",
        help="1_50 输出的最终标签文件",
    )
    parser.add_argument(
        "--out_by_condition",
        type=str,
        default="data/vsb_exp1/metrics_by_condition_model-baseline_text.csv",
        help="按 condition 汇总的指标输出路径",
    )
    parser.add_argument(
        "--out_by_condition_category",
        type=str,
        default="data/vsb_exp1/metrics_by_condition_category_model-baseline_text.csv",
        help="按 condition×category_top 汇总指标输出路径",
    )
    args = parser.parse_args()

    labels_path = Path(args.behavior_labels)
    out_cond_path = Path(args.out_by_condition)
    out_cond_cat_path = Path(args.out_by_condition_category)

    out_cond_path.parent.mkdir(parents=True, exist_ok=True)
    out_cond_cat_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取最终标签: {labels_path}")

    by_cond = defaultdict(lambda: defaultdict(int))
    by_cond_cat = defaultdict(lambda: defaultdict(int))

    with labels_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="统计指标"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            condition = rec.get("condition")
            category_top = rec.get("category_top")
            final_label = rec.get("final_label", "SAFE")

            update_counts(by_cond[condition], final_label)
            key_cc = (condition, category_top)
            update_counts(by_cond_cat[key_cc], final_label)

    # 写出按 condition 汇总
    with out_cond_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "num_samples",
            "ASR", "RefusalRate", "SafeRate", "OtherRate",
            "num_harmful", "num_refusal", "num_safe", "num_other",
        ])
        for condition, counter in sorted(by_cond.items()):
            rates = compute_rates(counter)
            writer.writerow([
                condition,
                counter["num_samples"],
                f"{rates['ASR']:.6f}",
                f"{rates['RefusalRate']:.6f}",
                f"{rates['SafeRate']:.6f}",
                f"{rates['OtherRate']:.6f}",
                counter["num_harmful"],
                counter["num_refusal"],
                counter["num_safe"],
                counter["num_other"],
            ])

    print(f"[INFO] 按 condition 指标写入: {out_cond_path}")

    # 写出按 condition×category_top 汇总
    with out_cond_cat_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "category_top", "num_samples",
            "ASR", "RefusalRate", "SafeRate", "OtherRate",
            "num_harmful", "num_refusal", "num_safe", "num_other",
        ])
        for (condition, category_top), counter in sorted(by_cond_cat.items()):
            rates = compute_rates(counter)
            writer.writerow([
                condition,
                category_top,
                counter["num_samples"],
                f"{rates['ASR']:.6f}",
                f"{rates['RefusalRate']:.6f}",
                f"{rates['SafeRate']:.6f}",
                f"{rates['OtherRate']:.6f}",
                counter["num_harmful"],
                counter["num_refusal"],
                counter["num_safe"],
                counter["num_other"],
            ])

    print(f"[INFO] 按 condition×category_top 指标写入: {out_cond_cat_path}")


if __name__ == "__main__":
    main()
