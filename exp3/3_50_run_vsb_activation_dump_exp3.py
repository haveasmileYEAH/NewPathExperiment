# exp3/3_50_run_vsb_activation_dump_exp3.py
#
# 功能：
#   在 VSB 的攻击 manifest 上跑一遍前向，抽取若干文本层的 hidden states，
#   用于后续 risk_text 投影。
#
# 输出：
#   data/vsb_exp3/text_activations_model-qwen2_5_vl_7b.npz
#   包含：
#       layer_ids: [L1, L2, ...]
#       states: [N, num_layers, hidden_dim]
#       attack_ids, video_ids, conditions, category_top: 长度为 N 的数组
#
# 说明：
#   - 只 forward，不 generate（和 3_40 构建方向保持一致）
#   - 文本位置：使用最后一个 token 的 hidden（与 3_40 一致）

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# Qwen2.5-VL 的小补丁，避免 torch.is_autocast_enabled 签名问题
orig_is_autocast_enabled = torch.is_autocast_enabled
def patched_is_autocast_enabled(*args, **kwargs):
    return orig_is_autocast_enabled()
torch.is_autocast_enabled = patched_is_autocast_enabled


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
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


def build_messages_for_sample(system_prompt: str, sample: dict) -> list:
    """
    和 Exp1 一致的输入格式：
      - system：安全提示
      - user：
          - VH-*：video + text
          - T-* ：text only
    """
    condition = sample["condition"]
    query_text = sample["query_text"]
    video_path = sample.get("video_path")

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

    use_video = condition.startswith("VH-") and bool(video_path)

    if use_video:
        abs_path = Path(video_path).resolve()
        video_uri = abs_path.as_uri()
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
        "--config",
        type=str,
        default="configs/exp3_mechanism.yaml",
        help="exp3 配置文件路径",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/vsb_exp1/manifest_exp1_attacks.jsonl",
        help="Exp1 攻击 manifest 路径",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default="data/vsb_exp3/text_activations_model-qwen2_5_vl_7b.npz",
        help="输出 npz 文件路径",
    )
    parser.add_argument(
        "--text_layers",
        type=str,
        default="24,25,26",
        help="要抽取的文本层索引，逗号分隔，例如 '24,25,26'",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="T-HQ,T-BQ,VH-HQ,VH-BQ",
        help="要保留的 condition 列表，逗号分隔；留空表示使用 config 里的 conditions_for_vsb",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=">0 时仅跑前 max_samples 条（调试用）",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("id", "Qwen/Qwen2.5-VL-7B-Instruct")
    system_prompt = cfg.get("system_prompt_for_exp1", cfg.get("system_prompt", ""))

    # 解析文本层
    text_layers = [int(x) for x in args.text_layers.split(",") if x.strip()]
    text_layers = sorted(set(text_layers))
    print(f"[INFO] 将抽取文本层: {text_layers}")

    # 条件过滤
    if args.conditions:
        allowed_conditions = {c.strip() for c in args.conditions.split(",") if c.strip()}
    else:
        cond_cfg = cfg.get("conditions_for_vsb", {})
        allowed_conditions = {c for c, flag in cond_cfg.items() if flag}
    print(f"[INFO] 使用条件: {sorted(allowed_conditions)}")

    manifest_path = Path(args.manifest)
    print(f"[INFO] 读取攻击 manifest: {manifest_path}")

    samples_raw = list(iter_jsonl(str(manifest_path)))
    print(f"[INFO] manifest 原始行数: {len(samples_raw)}")

    samples = [
        ex for ex in samples_raw
        if ex.get("condition") in allowed_conditions
    ]
    print(f"[INFO] 条件过滤后样本数: {len(samples)}")

    if args.max_samples > 0 and args.max_samples < len(samples):
        print(f"[INFO] 仅保留前 {args.max_samples} 条（调试模式）")
        samples = samples[:args.max_samples]

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 加载模型到 {device}: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # 收集 meta + states
    attack_ids, video_ids, conditions, cat_tops = [], [], [], []
    states_list = []  # 每个元素 shape = [num_layers, hidden_dim]

    out_npz_path = Path(args.out_npz)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    for ex in tqdm(samples, desc="Collect VSB text activations"):
        attack_id = ex["attack_id"]
        video_id = ex.get("video_id")
        condition = ex.get("condition")
        category_top = ex.get("category_top")

        messages = build_messages_for_sample(system_prompt, ex)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if has_vision_content(messages):
            # 视/视频条件：使用 qwen_vl_utils
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
            # 纯文本
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            )

        inputs = inputs.to(device)

        with torch.no_grad():
            out = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = out.hidden_states  # tuple: [emb, layer1, ..., layerN]

        # 从指定层抽取 last-token hidden
        vecs = []
        for L in text_layers:
            if L >= len(hidden_states):
                raise ValueError(
                    f"指定层 {L} 超过 hidden_states 数量 {len(hidden_states)}"
                )
            h_L = hidden_states[L][0, -1, :]  # [dim]
            vecs.append(h_L.detach().cpu().float().numpy())

        vecs = np.stack(vecs, axis=0)  # [num_layers, dim]
        states_list.append(vecs)

        attack_ids.append(attack_id)
        video_ids.append(video_id)
        conditions.append(condition)
        cat_tops.append(category_top)

    # 汇总到 npz
    if not states_list:
        print("[WARN] 没有任何样本，退出")
        return

    states_arr = np.stack(states_list, axis=0)  # [N, num_layers, dim]
    attack_ids_arr = np.array(attack_ids, dtype=object)
    video_ids_arr = np.array(video_ids, dtype=object)
    conditions_arr = np.array(conditions, dtype=object)
    cat_tops_arr = np.array(cat_tops, dtype=object)
    layer_ids_arr = np.array(text_layers, dtype=np.int32)

    print(f"[INFO] 最终 states 形状: {states_arr.shape} (N, num_layers, hidden_dim)")

    np.savez(
        out_npz_path,
        states=states_arr,
        layer_ids=layer_ids_arr,
        attack_ids=attack_ids_arr,
        video_ids=video_ids_arr,
        conditions=conditions_arr,
        category_top=cat_tops_arr,
    )
    print(f"[INFO] 激活已写入: {out_npz_path}")


if __name__ == "__main__":
    main()
