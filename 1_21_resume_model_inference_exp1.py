# 1_21_resume_model_inference_exp1.py
#
# 功能：在已有 raw_outputs_model-*.jsonl 的基础上“断点续跑”：
#   - 读取 manifest_exp1_attacks.jsonl（全部 attack_id）
#   - 读取已有输出文件中的 attack_id 集合
#   - 只对「尚未出现过的 attack_id」继续调用 Qwen2.5-VL-7B-Instruct 推理
#   - 新结果以 append 方式追加到同一个输出文件
#
# 不会修改你现有的 1_20_run_model_inference_exp1.py 逻辑。

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
import yaml

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def has_vision_content(messages):
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") in ("image", "video"):
                return True
    return False


def build_messages(system_prompt: str, sample: dict, use_video: bool) -> list:
    """
    构造与 1_20 中一致的 Qwen2.5-VL messages：
    - system：安全提示
    - user：
        - VH-*：video + text
        - T-*：text only
    """
    condition = sample["condition"]
    query_text = sample["query_text"]

    messages = []

    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt.strip(),
                    }
                ],
            }
        )

    # user 内容
    if use_video:
        video_path = sample.get("video_path")
        if not video_path:
            # 回退到纯文本
            user_content = [
                {
                    "type": "text",
                    "text": query_text,
                }
            ]
        else:
            abs_path = Path(video_path).resolve()
            video_uri = abs_path.as_uri()  # file:///...
            user_content = [
                {
                    "type": "video",
                    "video": video_uri,
                },
                {
                    "type": "text",
                    "text": query_text,
                },
            ]
    else:
        # 纯文本条件
        user_content = [
            {
                "type": "text",
                "text": query_text,
            }
        ]

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    return messages


def collect_existing_attack_ids(out_path: Path) -> set[str]:
    """
    从已有的 raw_outputs_model-*.jsonl 中收集所有 attack_id，
    用于 resume 时跳过这些已经处理过的样本。
    """
    existing: set[str] = set()
    if not out_path.exists():
        return existing

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = obj.get("attack_id")
            if aid is not None:
                existing.add(aid)
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/vsb_exp1/manifest_exp1_attacks.jsonl",
        help="攻击样本 manifest jsonl 路径",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/exp1_behavior.yaml",
        help="行为实验配置文件路径",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="从配置文件 models 中选择的模型 name，如不指定则使用第一个",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="输出文件路径；默认 data/vsb_exp1/raw_outputs_model-<model_name>.jsonl",
    )
    parser.add_argument(
        "--max_new_samples",
        type=int,
        default=0,
        help="这次最多新跑多少条（仅对“未完成”的样本计数）；=0 表示把剩余的全部跑完",
    )

    args = parser.parse_args()

    print(f"[INFO] 读取攻击 manifest: {args.manifest_path}")
    print(f"[INFO] 使用配置: {args.config_path}")

    cfg = load_config(args.config_path)
    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("配置文件中 models 为空，请检查 configs/exp1_behavior.yaml")

    if args.model_name is None:
        model_cfg = models_cfg[0]
    else:
        match = None
        for m in models_cfg:
            if m.get("name") == args.model_name:
                match = m
                break
        if match is None:
            raise ValueError(f"在配置文件中找不到 name={args.model_name} 的模型配置")
        model_cfg = match

    model_name = model_cfg.get("name", "qwen2_5_vl_7b")
    model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
    model_type = model_cfg.get("model_type", "qwen2_5_vl")

    if model_type != "qwen2_5_vl":
        print(
            f"[WARN] model_type={model_type} 不是 qwen2_5_vl，仍然尝试按 Qwen2.5-VL 加载"
        )

    gen_cfg = cfg.get("generation", {})
    system_prompt = cfg.get("system_prompt", "")

    if args.out_path is None:
        out_path = Path(f"data/vsb_exp1/raw_outputs_model-{model_name}.jsonl")
    else:
        out_path = Path(args.out_path)

    print(f"[INFO] 模型名称: {model_name}")
    print(f"[INFO] 模型 ID: {model_id}")
    print(f"[INFO] resume 输出文件: {out_path}")

    # 1. 加载模型和 processor（与 1_20 保持一致）
    print("[INFO] 加载 Qwen2.5-VL-7B-Instruct 模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("[INFO] 模型加载完成")

    # 2. 读取 manifest（全量 3120 条）
    all_samples = list(iter_jsonl(args.manifest_path))
    total_all = len(all_samples)
    print(f"[INFO] manifest 中攻击样本总数: {total_all}")

    # 3. 读取已有输出中的 attack_id，构造“未完成样本列表”
    existing_ids = collect_existing_attack_ids(out_path)
    num_exist = len(existing_ids)
    print(f"[INFO] 在输出文件中已发现 {num_exist} 条 attack_id（已完成样本）")

    pending_samples = [s for s in all_samples if s["attack_id"] not in existing_ids]
    num_pending = len(pending_samples)
    print(f"[INFO] 当前还剩待处理样本数: {num_pending}")

    if num_pending == 0:
        print("[INFO] 没有剩余样本，无需继续运行。")
        return

    max_new = args.max_new_samples
    if max_new > 0 and max_new < num_pending:
        print(f"[INFO] 本次仅跑剩余样本中的前 {max_new} 条")
        pending_samples = pending_samples[:max_new]
        num_pending = max_new

    print(f"[INFO] 本次实际将推理的新样本数: {num_pending}")

    # 4. 以 append 模式打开输出文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a", encoding="utf-8")

    try:
        for sample in tqdm(pending_samples, desc="resume 运行模型推理"):
            attack_id = sample["attack_id"]
            video_id = sample.get("video_id")
            condition = sample["condition"]
            query_text = sample["query_text"]
            query_type = sample.get("query_type")
            video_path = sample.get("video_path")

            use_video = condition.startswith("VH-") and bool(video_path)
            messages = build_messages(system_prompt, sample, use_video=use_video)

            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                if has_vision_content(messages):
                    # 与你在 1_20 中的修复逻辑保持一致：return_video_kwargs=True + fps 修正
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages, return_video_kwargs=True
                    )

                    fps = video_kwargs.pop("fps", None)
                    if isinstance(fps, list) and len(fps) > 0:
                        fps = float(fps[0])
                    elif fps is not None:
                        fps = float(fps)

                    proc_kwargs = dict(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    if fps is not None:
                        proc_kwargs["fps"] = fps

                    proc_kwargs.update(video_kwargs)

                    inputs = processor(**proc_kwargs)
                else:
                    # 纯文本条件：保持你目前的兼容写法
                    vision_res = process_vision_info(messages)
                    if isinstance(vision_res, (list, tuple)):
                        image_inputs = vision_res[0] if len(vision_res) > 0 else None
                        video_inputs = vision_res[1] if len(vision_res) > 1 else None
                    else:
                        image_inputs, video_inputs = None, None

                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                inputs = inputs.to(model.device)

                generate_kwargs = {
                    "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
                    "temperature": gen_cfg.get("temperature", 0.0),
                    "top_p": gen_cfg.get("top_p", 1.0),
                    "do_sample": gen_cfg.get("do_sample", False),
                }

                start_time = time.time()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generate_kwargs)
                elapsed = time.time() - start_time

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                record = {
                    "attack_id": attack_id,
                    "video_id": video_id,
                    "condition": condition,
                    "model_name": model_name,
                    "model_id": model_id,
                    "query_text": query_text,
                    "query_type": query_type,
                    "video_path": video_path,
                    "output_text": output_text,
                    "gen_time": elapsed,
                }

            except Exception as e:
                record = {
                    "attack_id": attack_id,
                    "video_id": video_id,
                    "condition": condition,
                    "model_name": model_name,
                    "model_id": model_id,
                    "query_text": query_text,
                    "query_type": query_type,
                    "video_path": video_path,
                    "output_text": f"[ERROR during generation: {repr(e)}]",
                    "gen_time": None,
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    finally:
        fout.close()

    print(f"[INFO] 本次新增写入样本数: {num_pending}")
    print(f"[INFO] 输出文件位置: {out_path}")


if __name__ == "__main__":
    main()
