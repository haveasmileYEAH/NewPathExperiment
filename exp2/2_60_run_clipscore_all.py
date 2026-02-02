#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import csv
import random
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor

try:
    from torchvision.io import read_video
except ImportError as e:
    raise RuntimeError(
        "torchvision is required for reading video frames. "
        "Please install it with: pip install torchvision"
    ) from e

from PIL import Image


def load_video_index(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_keywords(path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取 2_10 的输出：
      data/vsb_exp2/keywords_model-qwen2_5_vl_7b.jsonl

    返回：
      global_id -> {"keywords_list": [...], "prompt_version": ...}
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not isinstance(gid, str):
                continue
            mapping[gid] = {
                "keywords_list": obj.get("keywords_list") or [],
                "prompt_version": obj.get("prompt_version", ""),
            }
    return mapping


def sample_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    """
    使用 torchvision.io.read_video 读取视频，并均匀采样 num_frames 帧，返回 PIL.Image 列表。
    """
    # read_video 返回 (video, audio, info)
    video, _, info = read_video(video_path, pts_unit="sec")
    # video: [T, H, W, C]，uint8
    total_frames = video.shape[0]
    if total_frames == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    if total_frames <= num_frames:
        indices = torch.arange(total_frames)
    else:
        indices_f = torch.linspace(0, total_frames - 1, steps=num_frames)
        indices = indices_f.long()

    frames = video[indices]  # [K, H, W, C]
    images: List[Image.Image] = []
    for frame in frames:
        # frame: [H, W, C], uint8
        img = Image.fromarray(frame.numpy())
        images.append(img)
    return images


def init_clip(model_name: str, device: str = "cuda") -> Tuple[CLIPModel, CLIPProcessor]:
    """
    初始化 CLIP 模型 & 处理器。
    关键点：use_safetensors=True，避免触发 torch.load 的安全检查。
    """
    print(f"[INFO] Loading CLIP model: {model_name} (use_safetensors=True)")
    model = CLIPModel.from_pretrained(
        model_name,
        use_safetensors=True,
    )
    processor = CLIPProcessor.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return model, processor


def encode_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    samples: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    gid2textfeat: Dict[str, torch.Tensor] = {}
    for rec in tqdm(samples, desc="Encode text features"):
        gid = rec["global_id"]
        keywords = rec.get("keywords_list") or []
        caption = ", ".join(keywords) if keywords else "no keywords"

        # 修复点 1: 添加 truncation=True 和 max_length=77
        inputs = processor(
            text=[caption],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            # 修复点 2: 确保获取的是张量 (Tensor)
            text_outputs = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attn_mask,
            )
            # 兼容不同版本的 Transformers 返回类型
            text_emb = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs

        # L2 归一化并移到 CPU
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        gid2textfeat[gid] = text_emb.cpu()
    return gid2textfeat


def encode_image_and_compute_scores(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    samples: List[Dict[str, Any]],
    gid2textfeat: Dict[str, torch.Tensor],
    num_frames: int,
    num_negatives: int,
    out_path: str,
) -> None:
    """
    对每个视频：
      - 抽帧 -> image features -> 平均成 1 向量
      - 与本视频 text_feature 做正配对 similarity
      - 随机采样 num_negatives 个其它视频 text_feature 作为负样本
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 方便负采样
    all_gids = [rec["global_id"] for rec in samples]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_id",
            "dataset",
            "clip_model",
            "clipscore_pos",
            "clipscore_neg_mean",
            "num_negatives",
        ])

        for rec in tqdm(samples, desc="Compute CLIPScore per video"):
            gid = rec["global_id"]
            dataset = rec.get("dataset", "")
            video_path = rec.get("video_path")
            if not video_path:
                continue
            if not os.path.exists(video_path):
                print(f"[WARN] video not found: {video_path}, skip")
                continue

            # 抽帧 -> image features
            try:
                images = sample_video_frames(video_path, num_frames)
            except Exception as e:
                print(f"[WARN] Failed to read video {gid} ({video_path}): {e}")
                continue

            inputs = processor(
                text=None,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                # 1. 获得原始输出
                raw_img_outputs = model.get_image_features(pixel_values=pixel_values)
                # 2. 确保拿到的是 Tensor
                img_emb = raw_img_outputs.pooler_output if hasattr(raw_img_outputs, "pooler_output") else raw_img_outputs

            # 3. L2 归一化 + 多帧平均
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_emb_mean = img_emb.mean(dim=0, keepdim=True)  # [1, D]

            # 正配对文本特征
            text_feat_pos = gid2textfeat.get(gid)
            if text_feat_pos is None:
                print(f"[WARN] No text feature for {gid}, skip")
                continue
            text_feat_pos = text_feat_pos.to(device)  # [1, D]

            # 余弦相似度（向量已经归一化）
            clipscore_pos = float((img_emb_mean * text_feat_pos).sum(dim=-1).item())

            # 负样本
            neg_scores: List[float] = []
            if num_negatives > 0 and len(all_gids) > 1:
                # 从其它视频中随机采样 num_negatives 个 gid
                candidates = [g for g in all_gids if g != gid]
                if len(candidates) <= num_negatives:
                    neg_ids = candidates
                else:
                    neg_ids = random.sample(candidates, num_negatives)

                for ng in neg_ids:
                    text_feat_neg = gid2textfeat.get(ng)
                    if text_feat_neg is None:
                        continue
                    text_feat_neg = text_feat_neg.to(device)  # [1, D]
                    s = float((img_emb_mean * text_feat_neg).sum(dim=-1).item())
                    neg_scores.append(s)

            clipscore_neg_mean = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0

            writer.writerow([
                gid,
                dataset,
                model.name_or_path,
                clipscore_pos,
                clipscore_neg_mean,
                len(neg_scores),
            ])
            f.flush()
            os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIPScore between videos and keyword-based text for all samples."
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
        help="Output CSV path for per-video CLIPScore.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model id for CLIP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run CLIP model on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=10,
        help="Number of negative text samples to draw per video.",
    )
    args = parser.parse_args()

    video_index = load_video_index(args.video_index)
    kw_map = load_keywords(args.keywords_path)

    # 合并 video_index 和 keywords，确保每条样本都有 video_path + keywords
    samples: List[Dict[str, Any]] = []
    for rec in video_index:
        gid = rec.get("global_id")
        if not isinstance(gid, str):
            continue
        kw_info = kw_map.get(gid)
        if kw_info is None:
            continue
        merged = {
            "global_id": gid,
            "dataset": rec.get("dataset", ""),
            "video_path": rec.get("video_path"),
            "keywords_list": kw_info.get("keywords_list") or [],
            "prompt_version": kw_info.get("prompt_version", ""),
        }
        samples.append(merged)

    print(f"[INFO] Merged {len(samples)} samples with video + keywords.")

    model, processor = init_clip(args.clip_model_name, device=args.device)

    # 第一步：只算文本特征
    gid2textfeat = encode_text_features(model, processor, device=args.device, samples=samples)

    # 第二步：视频 + 正负配对 CLIPScore
    encode_image_and_compute_scores(
        model=model,
        processor=processor,
        device=args.device,
        samples=samples,
        gid2textfeat=gid2textfeat,
        num_frames=args.num_frames,
        num_negatives=args.num_negatives,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
