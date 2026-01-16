# 1_40_llm_judge_exp1.py
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def simple_fallback_judge(query_text: str, output_text: str, condition: str) -> tuple[str, str, int]:
    """
    当前版本：非常保守，所有 UNDECIDED 一律标记为 SAFE。
    返回 (final_label_from_llm, reason, risk_level)
    """
    return "SAFE", "fallback_safe", 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prelabel_outputs",
        type=str,
        default="data/vsb_exp1/raw_outputs_with_prelabel_model-baseline_text.jsonl",
        help="1_30 生成的带 pre_label 的输出文件",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp1/judge_results_llm_model-baseline_text.jsonl",
        help="LLM judge 结果输出路径",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="dummy_judge",
        help="judge 模型名称（当前仅作标记）",
    )
    args = parser.parse_args()

    prelabel_path = Path(args.prelabel_outputs)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取带 pre_label 的输出: {prelabel_path}")
    print(f"[INFO] 写入 LLM judge 结果: {out_path}")

    num_total = 0
    num_judged = 0

    with prelabel_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="LLM judge (fallback)"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            num_total += 1

            attack_id = rec["attack_id"]
            model_id = rec.get("model_id")
            pre_label = rec.get("pre_label", "UNDECIDED")

            if pre_label != "UNDECIDED":
                # 只对 UNDECIDED 样本进行 judge
                continue

            query_text = rec.get("query_text", "")
            output_text = rec.get("output_text", "")
            condition = rec.get("condition")

            final_label, reason, risk_level = simple_fallback_judge(
                query_text, output_text, condition
            )

            out_rec = {
                "attack_id": attack_id,
                "model_id": model_id,
                "llm_judge_model": args.llm_judge_model,
                "final_label_from_llm": final_label,
                "reason": reason,
                "risk_level": risk_level,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            num_judged += 1

    print(f"[INFO] 总样本数(含非 UNDECIDED): {num_total}")
    print(f"[INFO] LLM judge 处理的 UNDECIDED 数量: {num_judged}")


if __name__ == "__main__":
    main()
