# exp3/3_60_compute_risk_scores_from_activations.py
#
# 功能：
#   使用 3_40 得到的 text refusal directions，
#   对 3_50 dump 的 VSB 文本激活计算 risk_text 分数。
#
# 输出：
#   data/vsb_exp3/risk_scores_text_model-qwen2_5_vl_7b.jsonl
#   每行：
#       attack_id, video_id, condition, category_top,
#       risk_text, risk_text_by_layer: {layer: score}

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp3_mechanism.yaml",
        help="exp3 配置文件（主要用于记录路径，必要性不强）",
    )
    parser.add_argument(
        "--activations_npz",
        type=str,
        default="data/vsb_exp3/text_activations_model-qwen2_5_vl_7b.npz",
        help="3_50 输出的激活 npz 路径",
    )
    parser.add_argument(
        "--text_directions_npz",
        type=str,
        default="data/risk_directions_qwen2_5_vl_7b_alllayers/text_directions_selected_qwen2_5_vl_7b.npz",
        help="3_40 输出的文本拒绝方向 npz 路径",
    )
    parser.add_argument(
        "--out_scores",
        type=str,
        default="data/vsb_exp3/risk_scores_text_model-qwen2_5_vl_7b.jsonl",
        help="输出 risk score 的 jsonl 路径",
    )
    parser.add_argument(
        "--use_layers",
        type=str,
        default="",
        help="可选：仅使用这些层（逗号分隔），为空则使用 directions_npz 中的全部 selected_layers",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1) 读取 activations
    act_path = Path(args.activations_npz)
    print(f"[INFO] 读取激活: {act_path}")
    act_data = np.load(act_path, allow_pickle=True)

    states = act_data["states"]          # [N, num_layers_act, dim]
    layer_ids_act = act_data["layer_ids"]  # [num_layers_act]
    attack_ids = act_data["attack_ids"]
    video_ids = act_data["video_ids"]
    conditions = act_data["conditions"]
    category_top = act_data["category_top"]

    N, num_layers_act, dim = states.shape
    print(f"[INFO] activations.shape = {states.shape} (N, num_layers_act, dim)")
    print(f"[INFO] activations layer_ids = {layer_ids_act.tolist()}")

    # 2) 读取 text directions
    dir_path = Path(args.text_directions_npz)
    print(f"[INFO] 读取文本拒绝方向: {dir_path}")
    dir_data = np.load(dir_path, allow_pickle=True)

    selected_layers = dir_data["selected_layers"]  # [K]
    directions = dir_data["directions"]           # [K, dim]

    print(f"[INFO] directions selected_layers = {selected_layers.tolist()}")

    # 3) 决定要用的层：激活层 & direction 层 的交集，再根据 use_layers 过滤
    layers_act_set = set(int(x) for x in layer_ids_act.tolist())
    layers_dir_set = set(int(x) for x in selected_layers.tolist())
    common_layers = layers_act_set & layers_dir_set

    if args.use_layers:
        manual_layers = {int(x) for x in args.use_layers.split(",") if x.strip()}
        common_layers = common_layers & manual_layers

    common_layers = sorted(common_layers)
    if not common_layers:
        raise ValueError(
            f"activations 的层 {sorted(layers_act_set)} 与 directions 的层 {sorted(layers_dir_set)} 无交集，或与 use_layers 不匹配"
        )

    print(f"[INFO] 实际用于打分的层: {common_layers}")

    # 为方便索引，建立 layer -> index 的映射
    layer_to_idx_act = {int(L): i for i, L in enumerate(layer_ids_act.tolist())}
    layer_to_idx_dir = {int(L): i for i, L in enumerate(selected_layers.tolist())}

    # 4) 对每个样本计算 risk_text（三层均值）
    out_path = Path(args.out_scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_layers_use = len(common_layers)
    print(f"[INFO] 每个样本使用 {num_layers_use} 个层的 score 做均值")

    with out_path.open("w", encoding="utf-8") as f_out:
        for i in range(N):
            h_all = states[i]  # [num_layers_act, dim]
            h_all = h_all.astype(np.float32)

            scores_by_layer = {}
            scores_list = []

            for L in common_layers:
                idx_a = layer_to_idx_act[L]
                idx_d = layer_to_idx_dir[L]

                h = h_all[idx_a]          # [dim]
                d = directions[idx_d]     # [dim]

                h_n = l2_normalize(h[None, :])[0]
                d_n = l2_normalize(d[None, :])[0]
                score = float(np.dot(h_n, d_n))
                scores_by_layer[str(L)] = score
                scores_list.append(score)

            # 三层（或者多层）score 取均值作为 risk_text
            risk_text = float(np.mean(scores_list))

            rec = {
                "attack_id": str(attack_ids[i]),
                "video_id": str(video_ids[i]),
                "condition": str(conditions[i]),
                "category_top": str(category_top[i]),
                "risk_text": risk_text,
                "risk_text_by_layer": scores_by_layer,
                "layers_used": common_layers,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] risk_text 已写入: {out_path}")


if __name__ == "__main__":
    main()
