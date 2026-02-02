#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import csv
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict

def tokenize(text: str) -> List[str]:
    """非常简单的英文分词：非字母数字全部当空格，转小写，长度>=3"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3]
    return tokens

def load_vsb_manifest_coarse(manifest_path: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    从 data/vsb_exp0/manifest_vsb_subset_seed0.jsonl 构建：
      - global_id -> coarse_label (用 category_top 作为 13 个大类别)
      - coarse_label -> 词表（从 category_top / category_sub / Q_h / Q_b 抽词）

    假定 manifest 有字段：
      - video_id (如 'vsb_0049')
      - category_top (大类：比如 '10_Hate')
      - category_sub (子类)
      - Q_h / Q_b
    """
    global2label: Dict[str, str] = {}
    label_vocab: Dict[str, Set[str]] = defaultdict(set)

    num_lines = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            obj = json.loads(line)

            video_id = obj.get("video_id")  # e.g. 'vsb_0049'
            if not isinstance(video_id, str):
                continue

            # 粗粒度类别：用 category_top
            coarse_label = obj.get("category_top")
            if not isinstance(coarse_label, str) or not coarse_label:
                continue

            global_id = f"VSB_{video_id}"
            global2label[global_id] = coarse_label

            # 构建大类的词表：用 top/sub/Q_h/Q_b 的所有文本
            cat_top = obj.get("category_top") or ""
            cat_sub = obj.get("category_sub") or ""
            q_h = obj.get("Q_h") or ""
            q_b = obj.get("Q_b") or ""

            text_for_vocab = " ".join([coarse_label, cat_top, cat_sub, q_h, q_b])
            tokens = tokenize(text_for_vocab)
            label_vocab[coarse_label].update(tokens)

    print(
        f"[INFO] Loaded VSB manifest (coarse) from {manifest_path}: "
        f"{num_lines} lines, {len(global2label)} videos, {len(label_vocab)} coarse labels."
    )
    return global2label, label_vocab

def load_keywords(keywords_path: str) -> Dict[str, Dict]:
    """
    读取 2_10 的输出：
      data/vsb_exp2/keywords_model-qwen2_5_vl_7b.jsonl

    返回字典：
      global_id -> { 'keywords_list': [...], 'prompt_version': ... }
    """
    gid2info: Dict[str, Dict] = {}
    num = 0
    with open(keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not isinstance(gid, str):
                continue
            num += 1
            gid2info[gid] = {
                "keywords_list": obj.get("keywords_list") or [],
                "prompt_version": obj.get("prompt_version", ""),
            }
    print(f"[INFO] Loaded {num} keyword records from {keywords_path}")
    return gid2info

def build_video_wordset(keywords_list) -> Set[str]:
    """
    把一条视频的 keywords_list 转成词袋（按照 tokenize 分词）。
    """
    if not keywords_list:
        return set()
    joined = ", ".join(str(k) for k in keywords_list)
    return set(tokenize(joined))

def analyze_alignment(
    global2label: Dict[str, str],
    label_vocab: Dict[str, Set[str]],
    gid2info: Dict[str, Dict],
    out_path: str,
) -> None:
    """
    对每个 global_id（交集部分）：
      * true_label = coarse_label (category_top)
      * video_wordset = keywords_list -> token 集
      * 对每个 coarse_label 算 overlap = |video_wordset ∩ vocab(label)|
      * argmax 取 best_label_by_keywords
      * 写出 per-video 结果 + 汇总 top-1 accuracy
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    label_list = sorted(label_vocab.keys())
    n_total = 0
    n_nonzero = 0
    n_correct = 0

    with open(out_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "global_id",
            "true_label_coarse",
            "best_label_by_keywords",
            "overlap_true",
            "overlap_best",
            "correct",
            "num_keywords",
            "prompt_version",
        ])

        for gid, info in gid2info.items():
            true_label = global2label.get(gid)
            if true_label is None:
                # 只评估有 VSB coarse 标签的
                continue

            keywords_list = info.get("keywords_list") or []
            pv = info.get("prompt_version", "")

            video_words = build_video_wordset(keywords_list)
            if not video_words:
                overlaps = {lbl: 0 for lbl in label_list}
            else:
                overlaps = {}
                for lbl in label_list:
                    vocab = label_vocab.get(lbl, set())
                    overlaps[lbl] = len(video_words & vocab)

            best_label = None
            best_overlap = -1
            for lbl in label_list:
                ov = overlaps[lbl]
                if ov > best_overlap:
                    best_overlap = ov
                    best_label = lbl

            overlap_true = overlaps.get(true_label, 0)
            correct = 1 if best_label == true_label and best_label is not None else 0

            n_total += 1
            if best_overlap > 0:
                n_nonzero += 1
            n_correct += correct

            writer.writerow([
                gid,
                true_label,
                best_label if best_label is not None else "",
                overlap_true,
                best_overlap,
                correct,
                len(keywords_list),
                pv,
            ])

    acc = n_correct / n_total if n_total > 0 else 0.0
    print(
        f"[INFO] Processed {n_total} VSB videos (coarse). "
        f"Non-zero-overlap videos: {n_nonzero}. "
        f"Keyword→coarse-class top-1 accuracy: {acc:.4f}"
    )
    print(f"[INFO] Wrote per-video coarse alignment stats to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze keyword-category alignment on VSB (13 coarse classes, category_top)."
    )
    parser.add_argument(
        "--video_index",
        type=str,
        required=False,
        help="(Unused, kept for CLI compatibility.)",
    )
    parser.add_argument(
        "--keywords_path",
        type=str,
        required=True,
        help="Path to keywords_model-<vlm>.jsonl",
    )
    parser.add_argument(
        "--vsb_manifest_path",
        type=str,
        default="data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",
        help="Path to VSB manifest (subset) JSONL.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output CSV for coarse alignment stats.",
    )
    args = parser.parse_args()

    global2label, label_vocab = load_vsb_manifest_coarse(args.vsb_manifest_path)
    gid2info = load_keywords(args.keywords_path)
    analyze_alignment(global2label, label_vocab, gid2info, args.out_path)

if __name__ == "__main__":
    main()
