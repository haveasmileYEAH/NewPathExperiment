#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from typing import Any, Dict

def main():
    parser = argparse.ArgumentParser(
        description="Collect unique keywords from per-video keyword JSONL."
    )
    parser.add_argument(
        "--keywords_path",
        type=str,
        required=True,
        help="Path to data/exp2/keywords_model-<vlm>.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output JSONL path for unique keywords with frequencies.",
    )
    args = parser.parse_args()

    counter: Counter[str] = Counter()
    with open(args.keywords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)
            kw_list = obj.get("keywords_list") or []
            for kw in kw_list:
                kw_norm = str(kw).strip().lower()
                if kw_norm:
                    counter[kw_norm] += 1

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for kw, freq in counter.most_common():
            out_obj = {"keyword": kw, "freq": freq}
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Found {len(counter)} unique keywords. Wrote to {args.out_path}")

if __name__ == "__main__":
    main()
