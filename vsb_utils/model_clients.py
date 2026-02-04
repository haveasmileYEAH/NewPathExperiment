# common/model_clients.py
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import yaml
from together import Together


# ---------------- 配置结构 ----------------

@dataclass
class VUSEvalConfig:
    # 基础路径
    vsb_manifest_path: str = ""
    vsb_video_root: str = ""

    # 模型名称
    teacher_model_name: str = ""
    student_model_name: str = ""
    judge_model_name: str = ""

    # Teacher 配置
    teacher_max_frames: int = 8
    teacher_temperature: float = 0.0
    teacher_max_tokens: int = 512
    
    # Judge 配置
    judge_temperature: float = 0.0
    judge_max_tokens: int = 256

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VUSEvalConfig":
        import inspect
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        # --- 核心修复：只提取类中定义了的字段 ---
        # 获取构造函数的参数列表
        sig = inspect.signature(cls)
        # 过滤掉 YAML 中多余的字段
        valid_cfg = {
            k: v for k, v in cfg.items() 
            if k in sig.parameters
        }
        
        return cls(**valid_cfg)


# ---------------- Together Client ----------------

_together_client: Optional[Together] = None


def get_together_client() -> Together:
    global _together_client
    if _together_client is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        _together_client = Together(api_key=api_key)
    return _together_client


# ---------------- 视频抽帧为 base64 图片 ----------------

def extract_keyframes_as_base64(
    video_path: str | Path,
    num_frames: int = 4,
    img_size: int = 512,
) -> List[str]:
    """
    从本地 mp4 中均匀抽取 num_frames 帧，编码成 base64 JPEG 字符串。
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    num_frames = min(num_frames, frame_count)
    indices = [
        int(frame_count * (i + 0.5) / num_frames)
        for i in range(num_frames)
    ]

    results: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > img_size:
            scale = img_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        results.append(b64)

    cap.release()
    if not results:
        raise RuntimeError(f"Failed to extract frames from: {video_path}")
    return results


# ---------------- Teacher: LLaMA Vision via Together ----------------

def call_teacher_llama_vision(
    video_path: str | Path,
    prompt: str,
    model_name: str,
    num_frames: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    调用 Together 的 Vision Chat 接口：
    - 将视频抽成 num_frames 张图片（base64）
    - 以多模态消息的形式传给 LLaMA 模型
    """
    client = get_together_client()
    frames_b64 = extract_keyframes_as_base64(video_path, num_frames=num_frames)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            }
        )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if not resp.choices:
        raise RuntimeError("Empty response from Together teacher model.")

    message = resp.choices[0].message
    # message.content 可能是 list 或 str
    if isinstance(message.content, list):
        text_parts = [
            c.get("text", "")
            for c in message.content
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()
    return str(message.content).strip()
