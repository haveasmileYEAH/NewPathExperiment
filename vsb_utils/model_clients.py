from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import yaml
from together import Together

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


# =====================================================================
# 配置结构：从 vsb_eval_config.yaml 读取需要的字段
# =====================================================================

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
        """
        从 YAML 文件中读取配置，只保留 dataclass 中定义的字段，
        避免 YAML 里多余字段导致构造函数报错。
        """
        import inspect

        with Path(path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sig = inspect.signature(cls)
        valid_cfg = {k: v for k, v in cfg.items() if k in sig.parameters}
        return cls(**valid_cfg)


# =====================================================================
# Together Client（给 Teacher LLaMA 用）
# =====================================================================

_together_client: Optional[Together] = None


def get_together_client() -> Together:
    """
    懒加载 Together 客户端，依赖环境变量 TOGETHER_API_KEY。
    """
    global _together_client
    if _together_client is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        _together_client = Together(api_key=api_key)
    return _together_client


# =====================================================================
# 视频抽帧工具
# =====================================================================

def extract_keyframes_as_base64(
    video_path: str | Path,
    num_frames: int = 4,
    img_size: int = 512,
) -> List[str]:
    """
    从本地 mp4 中均匀抽取 num_frames 帧，编码成 base64 JPEG 字符串。
    - 专门给 Together 的 Vision 模型用（image_url:data:image/jpeg;base64,...）。
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
    # 与后面 Qwen 的 PIL 版本保持一致的“均匀抽帧”策略
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


def extract_keyframes_as_pil(
    video_path: str | Path,
    num_frames: int = 4,
    img_size: int = 512,
) -> List[Image.Image]:
    """
    从本地 mp4 中均匀抽取 num_frames 帧，返回缩放后的 PIL.Image 列表。
    - 抽帧位置策略与 extract_keyframes_as_base64 完全一致（均匀抽帧）。
    - 专门给本地 Qwen VL 模型用。
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

    images: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > img_size:
            scale = img_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # BGR → RGB，再转 PIL.Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        images.append(img)

    cap.release()
    if not images:
        raise RuntimeError(f"Failed to extract frames from: {video_path}")
    return images


# =====================================================================
# Teacher: LLaMA Vision via Together
# =====================================================================

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
    - 返回纯文本（通常是 JSON 字符串，交给上层解析）
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


# =====================================================================
# Student: Qwen2.5-VL-7B-Instruct 本地模型
# =====================================================================

_qwen_model = None
_qwen_processor = None
_qwen_model_name: Optional[str] = None


def get_qwen_vl7b_model_and_processor(model_name: str):
    """
    懒加载 Qwen/Qwen2.5-VL-7B-Instruct，本地 GPU 推理。
    - 使用 AutoModelForCausalLM + AutoProcessor
    - 默认放到 cuda:0（如果有 GPU）
    """
    global _qwen_model, _qwen_processor, _qwen_model_name

    if _qwen_model is not None and _qwen_model_name == model_name:
        return _qwen_model, _qwen_processor

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    _qwen_model = model
    _qwen_processor = processor
    _qwen_model_name = model_name

    return model, processor


def call_qwen_vl7b_video(
    video_path: str | Path,
    prompt: str,
    model_name: str,
    num_frames: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    调用本地 Qwen/Qwen2.5-VL-7B-Instruct 做视频理解：
    - 均匀抽取 num_frames 帧（与 Teacher 一致，默认 8 帧）
    - 将多帧 + 文本 prompt 作为多模态对话输入
    - 返回模型生成的文本（期望是 JSON 字符串，由上层去 json.loads 解析）

    抽帧逻辑与 Teacher 使用的 extract_keyframes_as_base64 保持一致，
    只是这里返回 PIL.Image，直接给 Qwen 的 AutoProcessor 使用。
    """
    model, processor = get_qwen_vl7b_model_and_processor(model_name)

    # 抽帧：均匀 num_frames 帧（img_size=512 与 Teacher 一致）
    images = extract_keyframes_as_pil(video_path, num_frames=num_frames, img_size=512)

    # 构造多模态对话格式：若干 image 占位符 + 一段文本 prompt
    # 对于 Qwen2.5-VL，推荐 messages 结构：
    # messages = [{"role": "user", "content": [{"type": "image"}, ..., {"type": "text", "text": "..."}]}]
    content: List[Dict[str, Any]] = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # 使用 chat_template 生成带 <image> 标记的文本串
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 将文本 + 图像一起编码成张量
    # 注意：text 可以是 str 或 list[str]，这里用单条的 str
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # 部分 Qwen 处理器提供 post_process_generation 来抽取 assistant 回复
    final_text: str
    if hasattr(processor, "post_process_generation"):
        out = processor.post_process_generation(
            generated_text,
            output_type="text",
        )
        # 兼容返回 list 或 str 的情况
        if isinstance(out, list):
            final_text = out[0]
        else:
            final_text = out
    else:
        final_text = generated_text

    return final_text.strip()
