#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
0_40_check_vsb_subset_integrity.py

对 manifest_vsb_subset_seedX.jsonl 做完整性检查：
  - video_path 是否存在
  - 是否能读取一帧
  - Q_h / Q_b 是否为空
输出:
  - manifest_vsb_subset_seedX_integrity_log.jsonl
  - manifest_vsb_subset_seedX_integrity_summary.json
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm


def check_video_file(path: str):
    """
    使用 OpenCV 尝试读取一帧视频，返回 (status, error_msg)
    status in {"ok", "missing_file", "corrupted_video"}
    """
    if not os.path.exists(path):
        return "missing_file", "file does not exist"

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return "corrupted_video", "cv2.VideoCapture cannot open file"

    ok, _ = cap.read()
    cap.release()
    if not ok:
        return "corrupted_video", "cv2 cannot read first frame"

    return "ok", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/vsb_exp0/manifest_vsb_subset_seed0.jsonl",
        help="0_30 输出的 manifest_vsb_subset_seedX.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/vsb_exp0",
        help="完整性检查日志输出目录",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log_path = args.manifest_path.replace(".jsonl", "_integrity_log.jsonl")
    summary_path = args.manifest_path.replace(".jsonl", "_integrity_summary.json")

    num_total = 0
    num_ok = 0
    num_missing_file = 0
    num_corrupted_video = 0
    num_empty_Q_h = 0
    num_empty_Q_b = 0

    print(f"[INFO] Checking manifest: {args.manifest_path}")
    with open(args.manifest_path, "r", encoding="utf-8") as f_in, \
            open(log_path, "w", encoding="utf-8") as f_log:

        for line in tqdm(f_in, desc="checking", ncols=100):
            line = line.strip()
            if not line:
                continue
            num_total += 1
            rec = json.loads(line)

            vid = rec["video_id"]
            vpath = rec["video_path"]

            status, error_msg = check_video_file(vpath)

            error_type = None

            # 检查文本
            qh = (rec.get("Q_h") or "").strip()
            qb = (rec.get("Q_b") or "").strip()

            if not qh:
                num_empty_Q_h += 1
                if error_type is None:
                    error_type = "empty_Q_h"
                else:
                    error_type += "|empty_Q_h"

            if not qb:
                num_empty_Q_b += 1
                if error_type is None:
                    error_type = "empty_Q_b"
                else:
                    error_type += "|empty_Q_b"

            if status == "missing_file":
                num_missing_file += 1
                if error_type is None:
                    error_type = "missing_file"
                else:
                    error_type += "|missing_file"

            elif status == "corrupted_video":
                num_corrupted_video += 1
                if error_type is None:
                    error_type = "corrupted_video"
                else:
                    error_type += "|corrupted_video"

            if status == "ok" and error_type is None:
                num_ok += 1
                log_status = "ok"
            else:
                log_status = "error"

            log_rec = {
                "video_id": vid,
                "video_path": vpath,
                "status": log_status,
                "video_status": status,
                "error_type": error_type,
                "error_msg": error_msg,
            }
            f_log.write(json.dumps(log_rec, ensure_ascii=False) + "\n")

    summary = {
        "manifest_path": args.manifest_path,
        "log_path": log_path,
        "num_total": num_total,
        "num_ok": num_ok,
        "num_missing_file": num_missing_file,
        "num_corrupted_video": num_corrupted_video,
        "num_empty_Q_h": num_empty_Q_h,
        "num_empty_Q_b": num_empty_Q_b,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Integrity log written to {log_path}")
    print(f"[INFO] Integrity summary written to {summary_path}")
    print(f"[INFO] Summary: {summary}")


if __name__ == "__main__":
    main()
