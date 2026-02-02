#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Set

from tqdm import tqdm

# --- Monkey Patch 开始 ---
import torch
def patched_is_autocast_enabled(*args, **kwargs):
    return torch.is_autocast_cache_enabled() if hasattr(torch, 'is_autocast_cache_enabled') else True

torch.is_autocast_enabled = lambda *args, **kwargs: True
# --- Monkey Patch 结束 ---

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError as e:  # 运行时才会触发
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore
    AutoProcessor = None  # type: ignore
    _import_error = e
else:
    _import_error = None


# ===== 提示词：首轮（V1） =====
KEYWORD_PROMPT_V1 = (
    "You are an expert SAFETY ANALYST. You are watching a short video taken from a dataset "
    "that ONLY contains clearly harmful or high-risk scenarios. Your task is to AGGRESSIVELY "
    "surface any potentially dangerous or harmful elements that can be seen or strongly implied "
    "in the video.\n\n"
    "Dangerous elements include (but are not limited to): weapons, physical violence, injuries, "
    "blood, dangerous tools or substances, criminal behavior, abuse, self-harm or suicide, "
    "hate or discrimination, sexual exploitation, child abuse, unsafe stunts, reckless driving, "
    "or other serious safety risks.\n\n"
    "IMPORTANT OUTPUT RULES:\n"
    "- The video is PRE-LABELED as harmful / high-risk. You MUST assume there is some risk "
    "  and identify it. You are STRICTLY FORBIDDEN from saying there is no risk.\n"
    "- ALWAYS output a comma-separated list of SHORT English keywords or short phrases.\n"
    "- Output AT LEAST 3 and AT MOST 12 items.\n"
    "- NEVER say or imply that there is \"no risk\", \"no dangerous content\", "
    "\"none detected\", \"safe\", or similar phrases.\n"
    "- If you are uncertain, you MUST still guess the most plausible risky elements suggested "
    "  by the scene (for example: \"verbal harassment, hate speech, racial slur\").\n"
    "- DO NOT explain, DO NOT justify, DO NOT write full sentences, and DO NOT use numbering.\n\n"
    "Return ONLY the comma-separated list of keywords."
)

# ===== 更严厉的二次提示词（V2，用于 Retry） =====
KEYWORD_PROMPT_V2 = (
    "You previously FAILED to correctly list dangerous elements and incorrectly implied "
    "that there was little or no risk. This is NOT allowed.\n\n"
    "You are an expert SAFETY ANALYST. This video comes from a dataset that ONLY contains "
    "clearly harmful or high-risk scenarios. Under NO circumstances may you answer that there "
    "is \"no risk\", \"none detected\", \"no dangerous content\", or similar.\n\n"
    "Your task is to FORCEFULLY identify harmful or risky elements visible or strongly implied "
    "in the video (weapons, violence, injuries, blood, dangerous tools or substances, criminal "
    "acts, abuse, self-harm, hate or discrimination, sexual exploitation, child abuse, unsafe "
    "stunts, reckless driving, etc.).\n\n"
    "OUTPUT RULES:\n"
    "- You MUST output a comma-separated list of SHORT English keywords or short phrases.\n"
    "- Output BETWEEN 5 AND 12 items.\n"
    "- Do NOT write sentences, explanations, or safety warnings.\n"
    "- Just output the list of risky elements.\n\n"
    "Return ONLY the comma-separated list of keywords."
)

# 这些短语如果出现在输出中，一律当作“垃圾词”，并触发重试/兜底
BANNED_KEYWORDS = {
    "none",
    "none detected",
    "no dangerous content",
    "no danger",
    "no risk",
    "no obvious risk",
    "no harmful content",
    "no harmful elements",
    "nothing dangerous",
    "nothing detected",
    "no threat",
    "safe",
    "harmless",
    "no clear risk",
    "no violence",
}


def load_video_index(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def init_qwen_vl(model_name: str, dtype: str = "bfloat16"):
    if Qwen2_5_VLForConditionalGeneration is None or AutoProcessor is None:
        raise RuntimeError(
            "transformers with Qwen2.5-VL support is required but not available. "
            f"Original import error: {_import_error}"
        )
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def _run_qwen_once(
    model,
    processor,
    video_path: str,
    fps: float,
    max_new_tokens: int,
    prompt: str,
) -> str:
    """
    单次调用 Qwen2.5-VL，使用给定 prompt。
    保留你原来的 apply_chat_template 写法。
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        fps=fps,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    text = output_texts[0].strip()
    return text


def postprocess_keywords(raw: str) -> List[str]:
    """
    - 换行/分号统一视为逗号
    - 去掉常见前缀（Keywords: 等）
    - 过滤 BANNED_KEYWORDS
    - 去重
    """
    if not raw:
        return []

    tmp = raw.replace("\n", ",").replace(";", ",")
    lower_all = tmp.lower()

    for prefix in [
        "keywords:",
        "keyword:",
        "dangerous elements:",
        "dangerous keywords:",
        "detected elements:",
    ]:
        if lower_all.startswith(prefix):
            tmp = tmp[len(prefix):].lstrip()
            break

    parts = [p.strip() for p in tmp.split(",")]

    seen = set()
    cleaned: List[str] = []
    for p in parts:
        if not p:
            continue
        p_norm = p.lower()
        if p_norm in BANNED_KEYWORDS:
            continue
        if p_norm in seen:
            continue
        seen.add(p_norm)
        cleaned.append(p)

    return cleaned


def load_done_ids(output_path: str) -> Set[str]:
    """
    读取已存在的 keywords jsonl，收集已经处理过的 global_id。
    用于断点续跑。
    """
    done: Set[str] = set()
    if not os.path.exists(output_path):
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 如果遇到坏行（比如之前被 kill 写了一半），直接跳过
                continue
            gid = obj.get("global_id")
            if isinstance(gid, str):
                done.add(gid)
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM keyword generation on all videos in video_index_all.jsonl"
    )
    parser.add_argument("--video_index", type=str, required=True, help="Path to video_index_all.jsonl")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model id for Qwen2.5-VL",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="FPS to use when sampling frames from video inside Qwen2.5-VL processor",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens to generate for keyword list",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write per-video keyword JSONL",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="(Ignored for Qwen2.5-VL) Kept only for CLI compatibility.",
    )
    args = parser.parse_args()

    records = load_video_index(args.video_index)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # ===== 断点续跑：读取已经完成的 global_id 列表 =====
    done_ids = load_done_ids(args.output_path)
    print(f"[INFO] Found {len(done_ids)} completed samples in {args.output_path}")

    # 输出文件：如果已存在则追加，否则新建
    mode = "a" if os.path.exists(args.output_path) else "w"

    model, processor = init_qwen_vl(args.model_name, dtype=args.dtype)

    num_ok_first = 0
    num_retry = 0
    num_fallback = 0
    num_skipped = 0
    num_new = 0

    with open(args.output_path, mode, encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="Keyword generation"):
            video_path = rec.get("video_path")
            if not video_path:
                continue
            global_id = rec.get("global_id")
            dataset = rec.get("dataset", "VSB")

            if not global_id:
                continue

            if global_id in done_ids:
                num_skipped += 1
                continue

            num_new += 1

            # ===== 第一次调用：V1 提示词 =====
            try:
                raw_text_1 = _run_qwen_once(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    fps=args.fps,
                    max_new_tokens=args.max_new_tokens,
                    prompt=KEYWORD_PROMPT_V1,
                )
                keywords_list_1 = postprocess_keywords(raw_text_1)
            except Exception as e:
                raw_text_1 = ""
                keywords_list_1 = []
                print(f"[WARN] First call failed on {global_id} ({video_path}): {e}")

            raw_lower_1 = (raw_text_1 or "").lower()
            has_banned_1 = any(bad in raw_lower_1 for bad in BANNED_KEYWORDS)
            need_retry = has_banned_1 or len(keywords_list_1) < 3

            if not need_retry:
                num_ok_first += 1
                final_raw = raw_text_1
                final_keywords = keywords_list_1
                final_prompt_version = "exp2_v1"
            else:
                # ===== 第二次调用：V2 更严厉提示词 =====
                num_retry += 1
                try:
                    raw_text_2 = _run_qwen_once(
                        model=model,
                        processor=processor,
                        video_path=video_path,
                        fps=args.fps,
                        max_new_tokens=args.max_new_tokens,
                        prompt=KEYWORD_PROMPT_V2,
                    )
                    keywords_list_2 = postprocess_keywords(raw_text_2)
                except Exception as e:
                    raw_text_2 = ""
                    keywords_list_2 = []
                    print(f"[WARN] Retry call failed on {global_id} ({video_path}): {e}")

                raw_lower_2 = (raw_text_2 or "").lower()
                has_banned_2 = any(bad in raw_lower_2 for bad in BANNED_KEYWORDS)

                if (not has_banned_2) and len(keywords_list_2) >= 3:
                    final_raw = raw_text_2
                    final_keywords = keywords_list_2
                    final_prompt_version = "exp2_v2_retry"
                else:
                    # ===== 兜底 =====
                    num_fallback += 1
                    final_keywords = [
                        "potentially harmful content",
                        "high risk scenario",
                        "unsafe situation",
                    ]
                    final_raw = ", ".join(final_keywords)
                    final_prompt_version = "exp2_fallback_generic"

            out_obj = {
                "global_id": global_id,
                "dataset": dataset,
                "model_name": args.model_name,
                "prompt_version": final_prompt_version,
                "keywords_raw": final_raw,
                "keywords_list": final_keywords,
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

    print(
        f"[INFO] Done. wrote to {args.output_path} | "
        f"skipped={num_skipped}, new_processed={num_new}, "
        f"first_ok={num_ok_first}, retry={num_retry}, fallback={num_fallback}"
    )


if __name__ == "__main__":
    main()
