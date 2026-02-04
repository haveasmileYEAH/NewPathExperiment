# exp_vu/3_02_collect_teacher_summaries_llama.py
import argparse
import json
import sys
import os
from typing import Dict, Any
from pathlib import Path

# --- 1. 路径修复：解决 common 包名冲突与路径查找问题 ---
# 获取当前脚本所在目录的上一级，即项目根目录
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    # 使用 insert(0, ...) 确保本地 vsb_utils 优先级高于 site-packages
    sys.path.insert(0, project_root)

# 导入时使用你重命名后的包名 vsb_utils (或者根据你实际重命名的名称修改)
from vsb_utils.io_utils import read_jsonl, write_jsonl
from vsb_utils.model_clients import VUSEvalConfig, call_teacher_llama_vision

from tqdm import tqdm


def build_summary_prompt(row: Dict[str, Any]) -> str:
    cat_top = row.get("category_top", "")
    cat_sub = row.get("category_sub", "")

    prompt = f"""
You are a safety-oriented video captioning assistant.

You will see several key frames extracted from a short online video
whose category is ({cat_top} / {cat_sub}).

Please write a neutral, factual English summary of what is visually happening
in the video, suitable for safety analysis.

Requirements:
- 3 to 5 sentences.
- Focus on concrete visual facts (who, where, what actions, objects).
- Do NOT mention the dataset or category labels.
- Do NOT copy or reference any external questions or prompts.
- Avoid explicit slurs; paraphrase hateful content in neutral language if needed.

Return ONLY the summary text, without JSON or bullet points.
    """.strip()
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="vsb_eval_config.yaml",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/vsb_eval/teacher_summaries_llama.jsonl",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    cfg = VUSEvalConfig.from_yaml(args.config)
    rows = read_jsonl(cfg.vsb_manifest_path)
    if args.max_videos is not None:
        rows = rows[: args.max_videos]

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    results = []

    # 注意：with 语句要包裹整个 for 循环
    with open(args.out_path, "a", encoding="utf-8") as f_out:
        for row in tqdm(rows, desc="Teacher summaries"):
            video_path = row.get("video_path") or row.get("video_relpath")
            global_id = f"VSB_{row['video_id']}"

            prompt = build_summary_prompt(row)
            try:
                summary = call_teacher_llama_vision(
                    video_path=video_path,
                    prompt=prompt,
                    model_name=cfg.teacher_model_name,
                    num_frames=cfg.teacher_max_frames,
                    temperature=cfg.teacher_temperature,
                    max_tokens=cfg.teacher_max_tokens,
                )
                rec = {
                    "global_id": global_id,
                    "teacher_model": cfg.teacher_model_name,
                    "teacher_summary": summary.strip(),
                    "teacher_refusal": False,
                }
            except Exception as e:
                rec = {
                    "global_id": global_id,
                    "teacher_model": cfg.teacher_model_name,
                    "teacher_summary": "",
                    "teacher_refusal": True,
                    "error": str(e),
                }

            # --- 关键点：这一行必须相对于 with 缩进，且在 for 循环内部 ---
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush() 
            results.append(rec)
    print(f"[INFO] Wrote {len(results)} teacher summaries to {args.out_path}")


if __name__ == "__main__":
    main()
