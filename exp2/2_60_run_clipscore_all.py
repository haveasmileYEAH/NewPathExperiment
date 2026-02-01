#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import torch

try:
    from decord import VideoReader, cpu
except ImportError as e:  # 运行时才会触发
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    _decord_import_error = e
else:
    _decord_import_error = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError as e:  # 运行时才会触发
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    _clip_import_error = e
else:
    _clip_import_error = None

from PIL import Image

def load_video_index(path: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not gid:
                continue
            idx[gid] = obj
    return idx

def load_keywords(path: str) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not gid:
                continue
            kw_list = obj.get("keywords_list") or []
            m[gid] = [str(k) for k in kw_list]
    return m

def sample_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    if VideoReader is None or cpu is None:
        raise RuntimeError(
            "decord is required for video reading but not available. "
            f"Original import error: {_decord_import_error}"
        )
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    vr = VideoReader(video_path, ctx=cpu())
    total = len(vr)
    if total == 0:
        return []
    if num_frames >= total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    batch = vr.get_batch(indices).asnumpy()  # (F,H,W,3)
    frames: List[Image.Image] = []
    for arr in batch:
        frames.append(Image.fromarray(arr))
    return frames

def init_clip(model_name: str, device: str):
    if CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError(
            "transformers with CLIP support is required but not available. "
            f"Original import error: {_clip_import_error}"
        )
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor

def compute_clip_features(
    model,
    processor,
    device: str,
    frames: List[Image.Image],
    text: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not frames:
        raise ValueError("No frames to encode.")
    # image features
    image_inputs = processor(images=frames, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)  # (F,D)
    image_features = image_features.mean(dim=0, keepdim=True)  # (1,D)
    # text features
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)  # (1,D)
    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0).cpu(), text_features.squeeze(0).cpu()

def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIPScore-style alignment between video frames and keyword text."
    )
    parser.add_argument(
        "--video_index",
        type=str,
        required=True,
        help="Path to video_index_all.jsonl",
    )
    parser.add_argument(
        "--keywords_path",
        type=str,
        required=True,
        help="Path to keywords_model-<vlm>.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for per-video CLIP scores.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for CLIP model (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample per video.",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=10,
        help="Number of negative texts per video to estimate negative CLIPScore.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_index = load_video_index(args.video_index)
    keywords_map = load_keywords(args.keywords_path)

    model, processor = init_clip(args.clip_model_name, device=args.device)

    feats: List[Dict[str, Any]] = []

    # First pass: compute features
    for gid, rec in tqdm(video_index.items(), desc="Computing CLIP features"):
        if gid not in keywords_map:
            continue
        video_path = rec.get("video_path")
        if not video_path:
            continue
        kw_list = keywords_map[gid]
        if not kw_list:
            continue
        text = ", ".join(kw_list)
        try:
            frames = sample_frames(video_path, num_frames=args.num_frames)
            v_feat, t_feat = compute_clip_features(
                model=model,
                processor=processor,
                device=args.device,
                frames=frames,
                text=text,
            )
        except Exception as e:
            print(f"[WARN] Failed CLIP features for {gid} ({video_path}): {e}")
            continue

        feats.append(
            {
                "global_id": gid,
                "dataset": rec.get("dataset", ""),
                "model_name": rec.get("model_name", ""),
                "video_feat": v_feat,
                "text_feat": t_feat,
            }
        )

    n = len(feats)
    print(f"[INFO] Collected CLIP features for {n} videos.")

    fieldnames = [
        "global_id",
        "dataset",
        "clip_model_name",
        "clipscore_pos",
        "clipscore_neg_mean",
        "num_negatives",
    ]
    with open(args.out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, fi in enumerate(feats):
            v_feat_i: torch.Tensor = fi["video_feat"]
            t_feat_i: torch.Tensor = fi["text_feat"]
            pos = float(torch.dot(v_feat_i, t_feat_i).item())

            # sample negatives
            if n <= 1 or args.num_negatives <= 0:
                neg_mean = 0.0
                num_negs = 0
            else:
                candidates = list(range(n))
                candidates.remove(i)
                if len(candidates) <= args.num_negatives:
                    neg_indices = candidates
                else:
                    neg_indices = random.sample(candidates, args.num_negatives)
                neg_scores: List[float] = []
                for j in neg_indices:
                    fj = feats[j]
                    t_feat_j: torch.Tensor = fj["text_feat"]
                    score = float(torch.dot(v_feat_i, t_feat_j).item())
                    neg_scores.append(score)
                neg_mean = float(sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0
                num_negs = len(neg_scores)

            row = {
                "global_id": fi["global_id"],
                "dataset": fi["dataset"],
                "clip_model_name": args.clip_model_name,
                "clipscore_pos": pos,
                "clipscore_neg_mean": neg_mean,
                "num_negatives": num_negs,
            }
            writer.writerow(row)

    print(f"[INFO] Wrote per-video CLIP scores to {args.out_path}")

if __name__ == "__main__":
    main()
