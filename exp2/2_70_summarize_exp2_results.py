#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict

import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Experiment 2 results into a single JSON file."
    )
    parser.add_argument(
        "--video_index",
        type=str,
        required=True,
        help="Path to video_index_all.jsonl (not heavily used, but kept for completeness).",
    )
    parser.add_argument(
        "--risk_stats",
        type=str,
        required=True,
        help="Path to per_video_risk_stats_model-<vlm>.csv",
    )
    parser.add_argument(
        "--vsb_align",
        type=str,
        required=True,
        help="Path to vsb_category_alignment_model-<vlm>.csv",
    )
    parser.add_argument(
        "--clipscore",
        type=str,
        required=True,
        help="Path to clipscore_per_video_model-<vlm>.csv",
    )
    parser.add_argument(
        "--exp1_results",
        type=str,
        default=None,
        help="(Optional) Path to per-video Exp1 behavior results CSV (must contain columns like global_id, condition, final_label).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output JSON path for Exp2 summary.",
    )
    args = parser.parse_args()

    risk_df = pd.read_csv(args.risk_stats)
    align_df = pd.read_csv(args.vsb_align)
    clip_df = pd.read_csv(args.clipscore)

    summary: Dict[str, Any] = {"dataset_stats": {}, "risk_vs_asr": {}}

    # Dataset-level stats
    datasets = sorted(set(risk_df["dataset"].dropna().unique()) | set(clip_df["dataset"].dropna().unique()))
    for ds in datasets:
        ds_stats: Dict[str, Any] = {}
        sub_risk = risk_df[risk_df["dataset"] == ds]
        if not sub_risk.empty:
            ds_stats["mean_risk_ratio"] = float(sub_risk["risk_ratio"].mean())
        sub_clip = clip_df[clip_df["dataset"] == ds]
        if not sub_clip.empty:
            ds_stats["avg_clipscore_pos"] = float(sub_clip["clipscore_pos"].mean())
            ds_stats["avg_clipscore_neg"] = float(sub_clip["clipscore_neg_mean"].mean())
        if ds == "VSB":
            sub_align = align_df
            if not sub_align.empty:
                ds_stats["keyword_class_align_acc"] = float(sub_align["correct"].mean())
        summary["dataset_stats"][ds] = ds_stats

    # Optional: risk_vs_asr using Exp1 per-video results (if provided and columns exist)
    if args.exp1_results:
        try:
            exp1_df = pd.read_csv(args.exp1_results)
            required_cols = {"global_id", "condition", "final_label"}
            if required_cols.issubset(exp1_df.columns):
                # Example: focus on VH_BQ condition
                cond_name = "VH_BQ"
                sub = exp1_df[exp1_df["condition"] == cond_name]
                merged = sub.merge(risk_df[["global_id", "risk_ratio"]], on="global_id", how="inner")
                if not merged.empty:
                    median_rr = merged["risk_ratio"].median()
                    high = merged[merged["risk_ratio"] >= median_rr]
                    low = merged[merged["risk_ratio"] < median_rr]

                    def asr(df):
                        if df.empty:
                            return 0.0
                        return float((df["final_label"] == "HARMFUL").mean())

                    summary["risk_vs_asr"][cond_name] = {
                        "split_median_risk_ratio": float(median_rr),
                        "high_risk_group_asr": asr(high),
                        "low_risk_group_asr": asr(low),
                        "num_high": int(len(high)),
                        "num_low": int(len(low)),
                    }
                else:
                    print("[WARN] After merging Exp1 and risk stats, no overlapping rows; skip risk_vs_asr.")
            else:
                print(f"[WARN] Exp1 results missing required columns {required_cols}; skip risk_vs_asr.")
        except Exception as e:
            print(f"[WARN] Failed to load or process Exp1 results: {e}")

    with open(args.out_path, "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote Exp2 summary to {args.out_path}")

if __name__ == "__main__":
    main()
