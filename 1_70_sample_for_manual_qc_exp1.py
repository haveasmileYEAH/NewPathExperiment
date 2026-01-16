# 1_70_sample_for_manual_qc_exp1.py
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior_labels",
        type=str,
        default="data/vsb_exp1/behavior_labels_model-baseline_text.jsonl",
        help="1_50 输出的最终标签文件",
    )
    parser.add_argument(
        "--raw_outputs",
        type=str,
        default="data/vsb_exp1/raw_outputs_model-baseline_text.jsonl",
        help="1_20 输出的原始模型输出（不带 pre_label 也可以）",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp1/manual_qc_samples_model-baseline_text.jsonl",
        help="人工质检样本输出路径",
    )
    parser.add_argument(
        "--per_condition",
        type=int,
        default=30,
        help="每个 condition 抽取样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    args = parser.parse_args()

    labels_path = Path(args.behavior_labels)
    raw_path = Path(args.raw_outputs)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    print(f"[INFO] 读取最终标签: {labels_path}")
    print(f"[INFO] 读取原始输出: {raw_path}")

    # 1) attack_id → raw 输出
    raw_map = {}
    with raw_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="构建 raw_map"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw_map[rec["attack_id"]] = rec

    # 2) 按 condition 分组
    by_condition = defaultdict(list)
    with labels_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="按 condition 分组"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cond = rec.get("condition")
            by_condition[cond].append(rec)

    # 3) 抽样并写出
    with out_path.open("w", encoding="utf-8") as fout:
        for cond, lst in by_condition.items():
            n = len(lst)
            k = min(args.per_condition, n)
            print(f"[INFO] condition={cond}, 总数={n}, 抽样={k}")
            sampled = random.sample(lst, k)

            for lab in sampled:
                attack_id = lab["attack_id"]
                raw_rec = raw_map.get(attack_id, {})
                out_rec = {
                    "attack_id": attack_id,
                    "condition": cond,
                    "video_id": lab.get("video_id"),
                    "video_path": raw_rec.get("video_path"),
                    "query_text": raw_rec.get("query_text"),
                    "output_text": raw_rec.get("output_text"),
                    "final_label": lab.get("final_label"),
                    "label_source": lab.get("label_source"),
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[INFO] 人工质检样本写入: {out_path}")


if __name__ == "__main__":
    main()
