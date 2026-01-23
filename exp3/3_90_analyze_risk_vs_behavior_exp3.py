# exp3/3_90_analyze_risk_vs_behavior_exp3.py
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged",
        type=str,
        default="data/vsb_exp3/risk_vs_behavior_text_model-qwen2_5_vl_7b.jsonl",
        help="3_80 输出的 risk_vs_behavior jsonl 文件",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp3/risk_vs_behavior_text_stats_model-qwen2_5_vl_7b.csv",
        help="按 condition × 风险分桶 的统计输出 CSV",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="T-BQ,T-HQ,VH-BQ,VH-HQ",
        help="需要分析的条件列表，逗号分隔；填 '*' 表示所有条件",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=3,
        help="按 risk_text 分桶的数量，默认 3（low/mid/high）",
    )
    args = parser.parse_args()

    merged_path = Path(args.merged)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 处理条件过滤
    if args.conditions.strip() == "*" or args.conditions.strip() == "":
        cond_filter = None
    else:
        cond_filter = set(c.strip() for c in args.conditions.split(",") if c.strip())

    print(f"[INFO] 读取合并结果: {merged_path}")
    print(f"[INFO] 条件过滤: {cond_filter if cond_filter is not None else 'ALL'}")
    print(f"[INFO] 分桶数量: {args.num_bins}")

    # condition -> list of (risk_text, final_label)
    cond_scores = defaultdict(list)

    total = 0
    used = 0
    missing_risk = 0
    missing_label = 0

    for rec in iter_jsonl(merged_path):
        total += 1
        cond = rec.get("condition")
        if cond_filter is not None and cond not in cond_filter:
            continue

        risk = safe_float(rec.get("risk_text"))
        if risk is None:
            missing_risk += 1
            continue

        label = rec.get("final_label")
        if label is None:
            missing_label += 1
            # 仍然可以用来看风险分布，但没法做行为统计
            # 这里选择跳过，以保证统计结果可解释
            continue

        cond_scores[cond].append((risk, str(label).upper()))
        used += 1

    print(
        f"[INFO] 总记录: {total}, 参与统计: {used}, "
        f"缺失 risk_text: {missing_risk}, 缺失 final_label: {missing_label}"
    )

    # 准备分桶名
    if args.num_bins == 3:
        bin_names = ["low", "mid", "high"]
    else:
        bin_names = [f"bin{i+1}" for i in range(args.num_bins)]

    # condition × bin_name -> 收集 (risk, label)
    cond_bin_data = defaultdict(list)

    for cond, pairs in cond_scores.items():
        if not pairs:
            continue

        risks = np.asarray([p[0] for p in pairs], dtype=float)
        labels = [p[1] for p in pairs]

        # 计算分位点
        if args.num_bins > 1:
            qs = [i / args.num_bins for i in range(1, args.num_bins)]
            boundaries = np.quantile(risks, qs).tolist()
        else:
            boundaries = []

        print(f"[INFO] condition={cond} 样本数={len(risks)}, 分位边界={boundaries}")

        # 按风险值分桶
        for r, lab in zip(risks, labels):
            # 决定 bin index
            bin_idx = 0
            for j, b in enumerate(boundaries):
                if r <= b:
                    bin_idx = j
                    break
            else:
                bin_idx = len(boundaries)
            bin_name = bin_names[bin_idx]
            cond_bin_data[(cond, bin_name)].append((r, lab))

    # 写出统计结果
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "condition,risk_bin,num_samples,"
            "risk_text_mean,risk_text_std,risk_text_min,risk_text_max,"
            "RefusalRate,ASR,SafeRate,OtherRate\n"
        )

        for cond in sorted({c for c, _ in cond_bin_data.keys()}):
            for bin_name in bin_names:
                key = (cond, bin_name)
                data = cond_bin_data.get(key, [])
                if not data:
                    continue

                risks = np.asarray([x[0] for x in data], dtype=float)
                labs = [x[1] for x in data]

                n = risks.size
                mean = float(risks.mean())
                std = float(risks.std(ddof=0))
                rmin = float(risks.min())
                rmax = float(risks.max())

                n_refusal = sum(1 for l in labs if l == "REFUSAL")
                n_harmful = sum(1 for l in labs if l == "HARMFUL")
                n_safe = sum(1 for l in labs if l == "SAFE")
                n_other = n - n_refusal - n_harmful - n_safe

                refusal_rate = n_refusal / n if n > 0 else 0.0
                asr = n_harmful / n if n > 0 else 0.0
                safe_rate = n_safe / n if n > 0 else 0.0
                other_rate = n_other / n if n > 0 else 0.0

                f.write(
                    f"{cond},{bin_name},{n},"
                    f"{mean:.6f},{std:.6f},{rmin:.6f},{rmax:.6f},"
                    f"{refusal_rate:.6f},{asr:.6f},{safe_rate:.6f},{other_rate:.6f}\n"
                )

    print(f"[INFO] 统计结果写入: {out_path}")


if __name__ == "__main__":
    main()
