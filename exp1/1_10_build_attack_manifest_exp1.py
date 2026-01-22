# 1_10_build_attack_manifest_exp1.py
import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(cfg_path: Path):
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vsb_manifest",
        type=str,
        default="data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",
        help="Step0 输出的子集 manifest",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp1_behavior.yaml",
        help="Experiment 1 行为层配置",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_exp1/manifest_exp1_attacks.jsonl",
        help="行为实验攻击 manifest 输出路径",
    )
    args = parser.parse_args()

    vsb_manifest_path = Path(args.vsb_manifest)
    cfg_path = Path(args.config)
    out_path = Path(args.out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    cond_enabled = cfg.get("conditions_enabled", {})
    seed = cfg.get("attack", {}).get("seed", 0)

    print(f"[INFO] 读取 Step0 manifest: {vsb_manifest_path}")
    print(f"[INFO] 使用配置: {cfg_path}")
    print(f"[INFO] 输出攻击 manifest: {out_path}")

    records = []
    with vsb_manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"[INFO] 读取 Step0 样本数: {len(records)}")

    num_out = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="构建攻击样本"):
            video_id = rec["video_id"]
            video_path = rec["video_path"]
            category_top = rec.get("category_top")
            category_sub = rec.get("category_sub")
            q_h = rec.get("Q_h", "").strip()
            q_b = rec.get("Q_b", "").strip()

            base = {
                "video_id": video_id,
                "category_top": category_top,
                "category_sub": category_sub,
                "seed": seed,
            }

            # VH-HQ: video + harmful query
            if cond_enabled.get("VH-HQ", False):
                attack_id = f"{video_id}__VH-HQ"
                row = {
                    **base,
                    "attack_id": attack_id,
                    "condition": "VH-HQ",
                    "video_path": video_path,
                    "image_path": None,
                    "query_type": "H",
                    "query_text": q_h,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_out += 1

            # VH-BQ: video + benign query
            if cond_enabled.get("VH-BQ", False):
                attack_id = f"{video_id}__VH-BQ"
                row = {
                    **base,
                    "attack_id": attack_id,
                    "condition": "VH-BQ",
                    "video_path": video_path,
                    "image_path": None,
                    "query_type": "B",
                    "query_text": q_b,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_out += 1

            # T-HQ: text-only harmful query
            if cond_enabled.get("T-HQ", False):
                attack_id = f"{video_id}__T-HQ"
                row = {
                    **base,
                    "attack_id": attack_id,
                    "condition": "T-HQ",
                    "video_path": None,
                    "image_path": None,
                    "query_type": "H",
                    "query_text": q_h,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_out += 1

            # T-BQ: text-only benign query
            if cond_enabled.get("T-BQ", False):
                attack_id = f"{video_id}__T-BQ"
                row = {
                    **base,
                    "attack_id": attack_id,
                    "condition": "T-BQ",
                    "video_path": None,
                    "image_path": None,
                    "query_type": "B",
                    "query_text": q_b,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_out += 1

            # I-HQ / I-BQ 目前先不启用，等你有抽帧路径后再扩展

    print(f"[INFO] 共写入攻击样本数: {num_out}")


if __name__ == "__main__":
    main()
