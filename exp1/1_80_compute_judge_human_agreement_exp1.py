# 1_80_compute_judge_human_agreement_exp1.py
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qc_file",
        type=str,
        default="data/vsb_exp1/manual_qc_samples_model-baseline_text.jsonl",
        help="1_70 输出并由人工补充 human_label 的文件",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp1/judge_human_agreement_model-baseline_text.json",
        help="一致性结果输出路径",
    )
    args = parser.parse_args()

    qc_path = Path(args.qc_file)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取人工质检文件: {qc_path}")

    total = 0
    agree_total = 0
    harmful_binary_total = 0
    harmful_binary_agree = 0

    by_condition = defaultdict(lambda: {"total": 0, "agree": 0})

    label_pairs = []

    with qc_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="计算一致性"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            auto = rec.get("final_label")
            human = rec.get("human_label")
            cond = rec.get("condition")

            if human is None:
                continue

            total += 1
            if auto == human:
                agree_total += 1
                by_condition[cond]["agree"] += 1
            by_condition[cond]["total"] += 1

            # harmful vs non-harmful 二值化
            auto_bin = "H" if auto == "HARMFUL" else "NH"
            human_bin = "H" if human == "HARMFUL" else "NH"
            harmful_binary_total += 1
            if auto_bin == human_bin:
                harmful_binary_agree += 1

            label_pairs.append((auto, human))

    # 计算简单 Cohen's kappa（多类）
    # kappa = (po - pe) / (1 - pe)
    # po: 实际一致率, pe: 按边缘分布的期望一致率
    auto_counts = Counter(a for a, _ in label_pairs)
    human_counts = Counter(h for _, h in label_pairs)
    n = len(label_pairs) or 1
    po = agree_total / n
    pe = sum((auto_counts[l] / n) * (human_counts[l] / n) for l in set(auto_counts) | set(human_counts))
    if pe < 1.0:
        kappa = (po - pe) / (1 - pe)
    else:
        kappa = 0.0

    result = {
        "num_samples_total": total,
        "overall_accuracy": agree_total / total if total else 0.0,
        "harmful_vs_nonharmful_accuracy": harmful_binary_agree / harmful_binary_total if harmful_binary_total else 0.0,
        "kappa": kappa,
        "by_condition": {},
    }

    for cond, stats in by_condition.items():
        t = stats["total"] or 1
        result["by_condition"][cond] = {
            "num_samples": stats["total"],
            "accuracy": stats["agree"] / t,
        }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 一致性结果写入: {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
