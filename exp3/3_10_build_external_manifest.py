# exp3/3_10_build_external_manifest.py
import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp3_mechanism.yaml",
        help="exp3 的配置文件",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/external/manifest_external_xstest_mmsb.jsonl",
        help="输出 manifest 路径",
    )
    parser.add_argument(
        "--max_xstest",
        type=int,
        default=None,
        help="XSTest 最多使用多少条（默认全部）",
    )
    parser.add_argument(
        "--max_mmsb_per_config",
        type=int,
        default=None,
        help="MM-SafetyBench 每个 config 最多使用多少条（默认全部）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ext_cfg = cfg.get("external_datasets", {})

    x_cfg = ext_cfg.get("xstest")
    m_cfg = ext_cfg.get("mm_safetybench")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_total = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        # 1) XSTest
        if x_cfg is not None:
            x_id = x_cfg["hf_id"]
            x_split = x_cfg.get("split", "train")
            print(f"[INFO] 加载 XSTest: {x_id} ({x_split})")
            ds_x = load_dataset(x_id, split=x_split)

            for i, ex in enumerate(tqdm(ds_x, desc="XSTest")):
                if args.max_xstest is not None and i >= args.max_xstest:
                    break
                prompt = ex.get("prompt", "")
                sample = {
                    "sample_id": f"xstest_{ex.get('id', i)}",
                    "source": "xstest",
                    "dataset_id": x_id,
                    "source_id": ex.get("id", i),
                    "prompt": prompt,
                    "meta": {
                        "type": ex.get("type", None),
                        "final_label_dataset": ex.get("final_label", None),
                    },
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                num_total += 1

        # 2) MM-SafetyBench (Text_only)
        if m_cfg is not None:
            m_id = m_cfg["hf_id"]
            split = m_cfg.get("split", "Text_only")
            configs = m_cfg.get("configs", [])
            print(f"[INFO] 加载 MM-SafetyBench: {m_id}, configs={configs}, split={split}")

            for config_name in configs:
                print(f"[INFO] 读取 config = {config_name}")
                ds_m = load_dataset(m_id, config_name, split=split)
                for j, ex in enumerate(tqdm(ds_m, desc=f"MM-SB:{config_name}")):
                    if args.max_mmsb_per_config is not None and j >= args.max_mmsb_per_config:
                        break
                    prompt = ex.get("question", "")
                    sample = {
                        "sample_id": f"mmsb_{config_name}_{ex.get('id', j)}",
                        "source": "mmsb",
                        "dataset_id": m_id,
                        "config": config_name,
                        "source_id": ex.get("id", j),
                        "prompt": prompt,
                    }
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    num_total += 1

    print(f"[INFO] manifest 写入完成: {out_path} (总样本数: {num_total})")


if __name__ == "__main__":
    main()
