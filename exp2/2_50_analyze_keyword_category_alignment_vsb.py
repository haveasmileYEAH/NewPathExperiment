#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

STOPWORDS = {
    "the","a","an","and","or","of","in","on","at","to","for","from","with",
    "is","are","was","were","be","being","been","by","as","this","that",
    "these","those","it","its","into","about","over","under","up","down",
    "out","off","than","then","so","such","can","could","may","might",
    "will","would","shall","should","do","does","did","have","has","had",
    "you","your","yours","we","our","ours","they","their","theirs","i",
}

TOKEN_RE = re.compile(r"[A-Za-z]+")

def tokenize(text: str) -> List[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

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

def build_label_vocab(video_index: Dict[str, Dict[str, Any]], top_k: int = 50) -> Dict[str, Set[str]]:
    label_counters: Dict[str, Counter] = defaultdict(Counter)
    for rec in video_index.values():
        if rec.get("dataset") != "VSB":
            continue
        label = rec.get("vsb_label_fine") or rec.get("label") or rec.get("category")
        if not label:
            continue
        label = str(label)
        harmful_prompt = rec.get("harmful_prompt") or rec.get("harmful_query") or ""
        text = f"{label} {harmful_prompt}"
        tokens = tokenize(text)
        label_counters[label].update(tokens)

    label_vocab: Dict[str, Set[str]] = {}
    for label, counter in label_counters.items():
        most_common = [w for w, _ in counter.most_common(top_k)]
        label_vocab[label] = set(most_common)
    return label_vocab

def main():
    parser = argparse.ArgumentParser(
        description="Analyze keyword-category alignment on VSB subset."
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
        help="Output CSV path for VSB keyword-category alignment.",
    )
    parser.add_argument(
        "--top_k_vocab",
        type=int,
        default=50,
        help="Top-K tokens per label to keep in label vocab.",
    )
    args = parser.parse_args()

    video_index = load_video_index(args.video_index)
    label_vocab = build_label_vocab(video_index, top_k=args.top_k_vocab)
    labels = sorted(label_vocab.keys())

    rows: List[Dict[str, Any]] = []
    total = 0
    correct = 0

    with open(args.keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("global_id")
            if not gid:
                continue
            rec = video_index.get(gid)
            if not rec or rec.get("dataset") != "VSB":
                continue

            true_label = rec.get("vsb_label_fine") or rec.get("label") or rec.get("category")
            if not true_label:
                continue
            true_label = str(true_label)

            kw_list = obj.get("keywords_list") or []
            tokens: Set[str] = set()
            for kw in kw_list:
                tokens.update(tokenize(str(kw)))

            if not tokens:
                best_label = "NONE"
                overlap_true = 0
                overlap_best = 0
                is_correct = 0
            else:
                overlaps = {}
                for lb in labels:
                    vocab = label_vocab.get(lb, set())
                    overlaps[lb] = len(tokens & vocab)
                if overlaps:
                    best_label = max(overlaps.items(), key=lambda x: x[1])[0]
                    overlap_true = overlaps.get(true_label, 0)
                    overlap_best = overlaps.get(best_label, 0)
                    is_correct = int(best_label == true_label and overlap_best > 0)
                else:
                    best_label = "NONE"
                    overlap_true = 0
                    overlap_best = 0
                    is_correct = 0

            row = {
                "global_id": gid,
                "vsb_label_fine": true_label,
                "best_label_by_keywords": best_label,
                "overlap_true": overlap_true,
                "overlap_best": overlap_best,
                "correct": is_correct,
            }
            rows.append(row)
            total += 1
            correct += is_correct

    with open(args.out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "global_id",
                "vsb_label_fine",
                "best_label_by_keywords",
                "overlap_true",
                "overlap_best",
                "correct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    acc = correct / total if total > 0 else 0.0
    print(f"[INFO] Processed {total} VSB videos. Keyword→class top-1 accuracy: {acc:.4f}")
    print(f"[INFO] Wrote per-video alignment stats to {args.out_path}")

if __name__ == "__main__":
    main()
