# exp3/3_20_run_external_inference.py
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --- 报错补丁：解决 Torch 与 Transformers 版本不兼容问题 ---
orig_is_autocast_enabled = torch.is_autocast_enabled
def patched_is_autocast_enabled(device_type=None):
    return orig_is_autocast_enabled()
torch.is_autocast_enabled = patched_is_autocast_enabled
# -------------------------------------------------------

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp3_mechanism.yaml",
        help="exp3 配置文件",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/external/manifest_external_xstest_mmsb.jsonl",
        help="外部攻击 manifest",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/external/raw_outputs_external_model-qwen2_5_vl_7b.jsonl",
        help="原始输出 JSONL",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多跑多少条（调试用，默认全部）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="每个样本生成最大 token 数",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("id", "Qwen/Qwen2.5-VL-7B-Instruct")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 加载模型: {model_id} 到 {device}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    in_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 进度条优化：计算总数 ---
    print(f"[INFO] 正在扫描输入文件以计算总行数...")
    with in_path.open("r", encoding="utf-8") as f_count:
        total_lines = sum(1 for _ in f_count if _.strip())
    
    total_to_process = min(total_lines, args.max_samples) if args.max_samples else total_lines
    # --------------------------

    num = 0
    t0 = time.time()

    with in_path.open("r", encoding="utf-8") as f_in, \
            out_path.open("w", encoding="utf-8") as f_out:

        # 传入 total 参数显示确切进度条
        for line in tqdm(f_in, desc="External Inference", total=total_to_process):
            if not line.strip():
                continue
            ex = json.loads(line)
            prompt = ex["prompt"]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            full_ids = gen_ids[0]
            prompt_len = inputs["input_ids"].shape[1]
            out_ids = full_ids[prompt_len:]
            raw_answer = processor.tokenizer.decode(
                out_ids, skip_special_tokens=True
            ).strip()

            rec = {
                **ex,
                "model_name": "qwen2_5_vl_7b",
                "model_id": model_id,
                "raw_answer_text": raw_answer,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            num += 1
            if args.max_samples is not None and num >= args.max_samples:
                break

    dt = time.time() - t0
    print(f"[INFO] 推理完成: {num} 条样本, 输出: {out_path}, 用时 {dt:.1f}s")


if __name__ == "__main__":
    main()