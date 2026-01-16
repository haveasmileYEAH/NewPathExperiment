# 1_20_run_model_inference_exp1.py
#
# 功能：读取 1_10 生成的攻击 manifest，用 Qwen2.5-VL-7B-Instruct
#        对每一个 attack_id 做一次推理，并把原始输出写到 jsonl 文件中。
#
# 说明：
# - VH-* 条件：使用「视频 + 文本」输入；
# - T-* 条件：只使用文本输入；
# - 视频使用官方推荐的 video 接口（file:/// 形式），由 qwen_vl_utils 处理。
#
# 依赖：
#   pip install "transformers @ git+https://github.com/huggingface/transformers" accelerate "qwen-vl-utils[decord]>=0.0.8"

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
    根据条件构造 Qwen2.5-VL 的 messages 结构：
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
            video_uri = abs_path.as_uri()  # 形如 file:///home/...
            user_content = [
                {
                    "type": "video",
                    "video": video_uri,
                    # 这里不强行指定 fps，使用 qwen-vl-utils 的默认视频采样策略
                },
                {
                    "type": "text",
                    "text": query_text,
                },
            ]
    else:
        # 纯文本条件（T-HQ / T-BQ）
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
        "--max_samples",
        type=int,
        default=0,
        help="仅用于冒烟测试：>0 时只跑前 max_samples 条；=0 跑全部",
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
        # 按 name 匹配
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
        out_path = f"data/vsb_exp1/raw_outputs_model-{model_name}.jsonl"
    else:
        out_path = args.out_path

    print(f"[INFO] 模型名称: {model_name}")
    print(f"[INFO] 模型 ID: {model_id}")
    print(f"[INFO] 推理结果输出到: {out_path}")

    # 1. 加载模型和 processor
    print("[INFO] 加载 Qwen2.5-VL-7B-Instruct 模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("[INFO] 模型加载完成")

    # 2. 读取攻击 manifest
    samples = list(iter_jsonl(args.manifest_path))
    total = len(samples)
    print(f"[INFO] 攻击样本总数: {total}")

    max_samples = args.max_samples
    if max_samples > 0 and max_samples < total:
        print(f"[INFO] 仅跑前 {max_samples} 条（冒烟测试模式）")
        samples = samples[:max_samples]
        total = max_samples

    # 3. 遍历样本逐条推理并写入结果
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="运行模型推理"):
            attack_id = sample["attack_id"]
            video_id = sample.get("video_id")
            condition = sample["condition"]
            query_text = sample["query_text"]
            query_type = sample.get("query_type")
            video_path = sample.get("video_path")

            # 判断是否需要视频
            use_video = condition.startswith("VH-") and bool(video_path)

            # 构造 messages
            messages = build_messages(system_prompt, sample, use_video=use_video)

            # 准备输入
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                if has_vision_content(messages):
                    # 视频/图像条件：使用 qwen_vl_utils 提取视觉信息和 video_kwargs
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages, return_video_kwargs=True
                    )
                    
                    # --- 核心修复：处理 FPS 列表类型错误 ---
                    fps = video_kwargs.pop("fps", None)
                    if isinstance(fps, list) and len(fps) > 0:
                        fps = float(fps[0]) # 解决 [1.916...] 报错
                    elif fps is not None:
                        fps = float(fps)
                    # ------------------------------------
                
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
                    # 纯文本条件
                    # 修复：确保非视频模式下也能正确处理返回值
                    vision_res = process_vision_info(messages)
                    image_inputs = vision_res[0] if isinstance(vision_res, (list, tuple)) else None
                    video_inputs = vision_res[1] if isinstance(vision_res, (list, tuple)) and len(vision_res) > 1 else None
                    
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )


                inputs = inputs.to(model.device)

                # 生成参数
                generate_kwargs = {
                    "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
                    "temperature": gen_cfg.get("temperature", 0.0),
                    "top_p": gen_cfg.get("top_p", 1.0),
                    "do_sample": gen_cfg.get("do_sample", False),
                }

                # 推理
                start_time = time.time()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generate_kwargs)
                elapsed = time.time() - start_time

                # 截掉 prompt 部分，只保留新生成的 tokens
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
                }
                # 便于后续分析的字段
                record.update(
                    {
                        "query_text": query_text,
                        "query_type": query_type,
                        "video_path": video_path,
                        "output_text": output_text,
                        "gen_time": elapsed,
                    }
                )

            except Exception as e:
                # 出错时不让整个实验中断，记录错误信息
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

    print(f"[INFO] 写入模型输出完成: {out_path}")


if __name__ == "__main__":
    main()