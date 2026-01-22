#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2_31_parse_risk_features_exp2.py

功能：
  - 解析 2_21_run_model_risk_features_exp2.py 生成的 risk_features JSONL 文件，
    将 raw_answer_text 中的 JSON 字符串解析为结构化字段：
      * key_objects
      * key_actions
      * risk_keywords
      * video_intent
  - 额外生成一些统计辅助字段：
      * num_key_objects
      * num_key_actions
      * num_risk_keywords
      * has_risk_keywords (bool)

  - 输出新的 JSONL 文件，便于后续与 Exp1 行为标签 merge 和统计分析。
"""

import argparse
import json
from pathlib import Path
from collections import Counter

from typing import Any, Dict


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_parse_inner_json(raw_text: str) -> Dict[str, Any]:
    """
    尝试从 raw_answer_text 中解析出 JSON 对象。

    当前你的输出形态大致为：
      "{\n  \"key_objects\": [...], ... }"

    但为了鲁棒性，这里仍然做一个简单的：
      - 找到第一个 '{' 和最后一个 '}' 之间的子串
      - 用 json.loads 尝试解析
      - 失败时返回一个默认结构
    """
    default = {
        "key_objects": [],
        "key_actions": [],
        "risk_keywords": [],
        "video_intent": "unknown",
    }

    if not raw_text or not isinstance(raw_text, str):
        return default

    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return default
        sub = raw_text[start : end + 1]
        obj = json.loads(sub)

        # 做一些字段上的兜底
        key_objects = obj.get("key_objects", [])
        key_actions = obj.get("key_actions", [])
        risk_keywords = obj.get("risk_keywords", [])
        video_intent = obj.get("video_intent", "unknown")

        if not isinstance(key_objects, list):
            key_objects = []
        if not isinstance(key_actions, list):
            key_actions = []
        if not isinstance(risk_keywords, list):
            risk_keywords = []
        if not isinstance(video_intent, str):
            video_intent = "unknown"

        return {
            "key_objects": key_objects,
            "key_actions": key_actions,
            "risk_keywords": risk_keywords,
            "video_intent": video_intent,
        }
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk_features_in",
        type=str,
        default="data/vsb_exp2/risk_features_model-qwen2_5_vl_7b.jsonl",
        help="2_21 生成的风险理解结果 JSONL 路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/vsb_exp2/risk_features_parsed_model-qwen2_5_vl_7b.jsonl",
        help="解析后的输出 JSONL 路径",
    )
    args = parser.parse_args()

    in_path = args.risk_features_in
    out_path = args.out

    print(f"[INFO] 读取风险理解结果: {in_path}")
    samples = list(iter_jsonl(in_path))
    total = len(samples)
    print(f"[INFO] 总样本数: {total}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    intent_counter = Counter()
    has_risk_counter = Counter()

    parsed_count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in samples:
            raw = rec.get("raw_answer_text", "")

            parsed_json = safe_parse_inner_json(raw)
            key_objects = parsed_json["key_objects"]
            key_actions = parsed_json["key_actions"]
            risk_keywords = parsed_json["risk_keywords"]
            video_intent = parsed_json["video_intent"]

            num_key_objects = len(key_objects)
            num_key_actions = len(key_actions)
            num_risk_keywords = len(risk_keywords)
            has_risk = num_risk_keywords > 0

            intent_counter[video_intent] += 1
            has_risk_counter["has_risk" if has_risk else "no_risk"] += 1

            new_rec = dict(rec)  # 保留原始字段
            new_rec.update(
                {
                    "key_objects": key_objects,
                    "key_actions": key_actions,
                    "risk_keywords": risk_keywords,
                    "video_intent": video_intent,
                    "num_key_objects": num_key_objects,
                    "num_key_actions": num_key_actions,
                    "num_risk_keywords": num_risk_keywords,
                    "has_risk_keywords": has_risk,
                }
            )

            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            parsed_count += 1

    print(f"[INFO] 解析完成样本数: {parsed_count}")
    print("[INFO] video_intent 分布:")
    for intent, cnt in intent_counter.most_common():
        ratio = cnt / total if total > 0 else 0.0
        print(f"  - {intent:24s}: {cnt:4d} ({ratio:.3f})")

    print("[INFO] risk_keywords 是否为空的分布:")
    for k, cnt in has_risk_counter.items():
        ratio = cnt / total if total > 0 else 0.0
        print(f"  - {k:8s}: {cnt:4d} ({ratio:.3f})")

    print(f"[INFO] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
