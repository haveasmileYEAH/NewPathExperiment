# exp3/3_00_prepare_exp3_config.py
#
# 功能：
#   为 Experiment 3 写出一个统一的配置文件 configs/exp3_mechanism.yaml
#   后续 3_10 / 3_50 / 3_60 / 3_80 / 3_90 均读取该配置。
#
# 说明：
#   - 默认模型沿用 Exp1 / Exp2 的 Qwen2.5-VL-7B-Instruct
#   - 默认只启用 T-HQ / T-BQ / VH-HQ / VH-BQ 四个条件
#   - external_safety_datasets 先留一个占位路径，之后你可以自己改
#
# 使用示例：
#   python exp3/3_00_prepare_exp3_config.py \
#       --out configs/exp3_mechanism.yaml

import argparse
import os
from pathlib import Path

import yaml


def build_default_config() -> dict:
    """
    构造一个默认的 Experiment 3 配置字典。
    后续可以通过命令行参数部分覆盖。
    """
    cfg = {
        "model": {
            # 和 Exp1 / Exp2 一致的多模态模型
            "name": "qwen2_5_vl_7b",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_type": "qwen2_5_vl",
        },

        # 外部安全数据集（用于构建文本侧风险方向）
        # 这里只给出一个占位，你可以后续直接在 yaml 里改成自己的路径列表
        "external_safety_datasets": {
            "text_attack": [
                # 示例：你可以改成 RQ1 那边的文本攻击/拒绝数据
                # "data/external_safety/text_attack_dataset.jsonl"
            ]
        },

        # 外部安全 hidden states 存放目录（3_10 输出）
        "external_states_dir": "data/external_safety_states",

        # 风险方向（文本/视觉）的统一存放目录（3_20, 3_30, 3_40 输出）
        "direction_storage_dir": "data/risk_directions",

        # 在外部文本安全数据上，拟议的候选层（供 3_10 / 3_20 / 3_30 使用）
        # 注意：Qwen2.5-VL-7B 的 transformer 层数为 32，下面只是常见几层的示例。
        "candidate_layers_for_text_direction": [10, 14, 18, 22, 26, 30],

        # 视觉方向可先留空，后续如果从 RQ1 或 HiddenDetect 导入，可以在 yaml 里补充
        "candidate_layers_for_visual_direction": [],

        # VSB 激活 dump 相关目录（3_50 / 3_60 / 3_70 / 3_80 / 3_90 使用）
        "vsb": {
            # Step0 子集 manifest（你已经有的文件）
            "subset_manifest": "data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",

            # Exp1 的攻击 manifest & 行为标签（已经存在）
            "attack_manifest": "data/vsb_exp1/manifest_exp1_attacks.jsonl",
            "behavior_labels": "data/vsb_exp1/behavior_labels_model-qwen2_5_vl_7b.jsonl",

            # Step3 专用输出目录
            "exp3_activation_dir": "data/vsb_exp3/activations_model-qwen2_5_vl_7b",
            "exp3_scores_path": "data/vsb_exp3/risk_scores_model-qwen2_5_vl_7b.jsonl",

            # VSB 条件开关（先只启用四个主条件）
            "conditions_for_vsb": {
                "T-HQ": True,
                "T-BQ": True,
                "VH-HQ": True,
                "VH-BQ": True,
                # 如后续有需要，可以在 yaml 中把下面两项改成 True
                "I-HQ": False,
                "I-BQ": False,
            },
        },

        # 文本侧风险方向结果文件（3_20 / 3_30 输出 & 3_60 使用）
        "text_direction_files": {
            # 原始方向（包含所有候选层）
            "raw": "data/risk_directions/text_directions_raw_qwen2_5_vl_7b.npz",
            # AUROC 扫描结果（各层分数）
            "auroc_csv": "data/risk_directions/text_direction_auroc_qwen2_5_vl_7b.csv",
            # 选中的主力层方向（供后续 VSB 实验使用）
            "selected": "data/risk_directions/text_directions_selected_qwen2_5_vl_7b.npz",
        },

        # 视觉侧风险方向结果文件（可选，如果你从 RQ1 导入，可以填充下面两个路径）
        "visual_direction_files": {
            # 原始视觉方向
            "raw": "data/risk_directions/visual_direction_raw_qwen2_5_vl_7b.npz",
            # 最终用于 VSB 评分的视觉方向
            "selected": "data/risk_directions/visual_direction_selected_qwen2_5_vl_7b.npz",
        },

        # 便于后续复用的随机种子设置（如有需要）
        "seed": 1234,
    }
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="configs/exp3_mechanism.yaml",
        help="输出的 Experiment 3 配置文件路径",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如目标文件已存在，是否允许覆盖写入",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)

    if out_path.exists() and not args.overwrite:
        print(f"[WARN] 目标文件已存在: {out_path}")
        print("       如需覆盖，请添加参数 --overwrite")
        return

    cfg = build_default_config()

    # 确保上级目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Experiment 3 配置已写入: {out_path}")
    print("[INFO] 你可以手工打开该文件，检查/修改：")
    print("       - external_safety_datasets")
    print("       - candidate_layers_for_text_direction")
    print("       - conditions_for_vsb 等字段")


if __name__ == "__main__":
    main()
