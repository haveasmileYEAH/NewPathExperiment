# 1_50_merge_judge_results_exp1.py
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prelabel_outputs",
        type=str,
        default="data/vsb_exp1/raw_outputs_with_prelabel_model-baseline_text.jsonl",
        help="1_30 输出文件（带 pre_label）",
    )
    parser.add_argument(
        "--llm_judge_results",
        type=str,
        default="data/vsb_exp1/judge_results_llm_model-baseline_text.jsonl",
        help="1_40 输出文件（LLM judge 结果）",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp1/behavior_labels_model-baseline_text.jsonl",
        help="最终行为标签输出路径",
    )
    args = parser.parse_args()

    prelabel_path = Path(args.prelabel_outputs)
    judge_path = Path(args.llm_judge_results)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取带 pre_label 的输出: {prelabel_path}")
    print(f"[INFO] 读取 LLM judge 结果: {judge_path}")
    print(f"[INFO] 写入最终标签: {out_path}")

    # 1) 读 LLM judge，构建 attack_id → judge_result 映射
    judge_map = {}
    with judge_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            attack_id = rec["attack_id"]
            judge_map[attack_id] = rec

    print(f"[INFO] LLM judge 样本数: {len(judge_map)}")

    # 2) 逐条 merge
    num_total = 0
    num_rule = 0
    num_llm = 0

    with prelabel_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="合并标签"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            num_total += 1

            attack_id = rec["attack_id"]
            pre_label = rec.get("pre_label", "UNDECIDED")

            out_rec = {
                "attack_id": attack_id,
                "video_id": rec.get("video_id"),
                "category_top": rec.get("category_top"),
                "category_sub": rec.get("category_sub"),
                "condition": rec.get("condition"),
                "query_type": rec.get("query_type"),
                "model_id": rec.get("model_id"),
            }

            if pre_label in ("HARMFUL", "REFUSAL"):
                out_rec["final_label"] = pre_label
                out_rec["label_source"] = "rule"
                out_rec["pre_label"] = pre_label
                num_rule += 1
            else:
                # UNDECIDED → 使用 LLM 结果（当前为 Safe fallback）
                jr = judge_map.get(attack_id)
                if jr is None:
                    # 理论上不应该发生，可以做一个兜底
                    out_rec["final_label"] = "SAFE"
                    out_rec["label_source"] = "default_safe"
                    out_rec["pre_label"] = pre_label
                else:
                    out_rec["final_label"] = jr.get("final_label_from_llm", "SAFE")
                    out_rec["label_source"] = "llm"
                    out_rec["pre_label"] = pre_label
                    out_rec["llm_final_label"] = jr.get("final_label_from_llm")
                    out_rec["llm_reason"] = jr.get("reason")
                    out_rec["risk_level"] = jr.get("risk_level")
                num_llm += 1

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[INFO] 总样本数: {num_total}")
    print(f"[INFO] 使用 rule 决策数: {num_rule}")
    print(f"[INFO] 使用 LLM 决策数: {num_llm}")


if __name__ == "__main__":
    main()
