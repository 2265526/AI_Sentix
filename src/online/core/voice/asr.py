# -*- coding: utf-8 -*-
"""
core/voice/asr.py —— 本地语音识别（ASR）
=========================================
阶段四实现：使用 faster-whisper（本地离线，无需外部 key）。

- 模型：本地 `~/whisper-small`（faster-whisper small，已从 HF 缓存复制），直接按路径加载，
  跳过 HuggingFace 联网校验（联网校验是加载卡顿的主要来源）；可用 WHISPER_MODEL_PATH 指定其他路径；
- 设备：自动探测——ctranslate2 检测到可用 CUDA 则用 GPU（float16），否则 CPU（int8）；
  可用 ASR_DEVICE=cpu / cuda 强制指定；
- 输入：音频字节（wav / mp3 / ogg / webm 等，PyAV 自动解码）；
- 输出：简体中文文本。

环境依赖：faster-whisper（pip 安装，内部依赖 PyAV，无需系统 ffmpeg）。
"""
import io
import logging
import os
import threading
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# 模型尺寸（settings.whisper_model_size，默认 small：中文准确率与速度均衡；如追求更快可改 "base"）
_MODEL_SIZE = settings.whisper_model_size
# 本地模型目录（~/.cache 的 HF 缓存复制版）：存在则直接加载，跳过 HuggingFace 联网校验
# （实测每次 WhisperModel('small') 都会联网校验缓存，网络慢时加载卡几分钟）
DEFAULT_LOCAL_MODEL = os.path.expanduser("~/whisper-small")
# 推理设备：默认自动探测（有 CUDA 用 GPU/float16，否则 CPU/int8）；
# 可通过环境变量 ASR_DEVICE=cpu 强制 CPU、ASR_DEVICE=cuda 强制 GPU（GPU 加载异常时回退用）
_DEVICE = settings.asr_device or None
_COMPUTE_TYPE = None

_model = None
_model_lock = threading.Lock()


def _resolve_device():
    """返回 (device, compute_type)：有可用 CUDA 用 GPU，否则 CPU int8。"""
    if _DEVICE is not None:
        return _DEVICE, ("float16" if _DEVICE == "cuda" else "int8")
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _resolve_model_source() -> str:
    """模型来源：WHISPER_MODEL_PATH > ~/whisper-small（本地） > 默认模型名。

    本地路径加载完全离线、秒级校验；模型名加载会走 HuggingFace 缓存校验（网络慢时卡顿）。
    """
    path = settings.whisper_model_path.strip()
    if path:
        return path
    if os.path.isdir(DEFAULT_LOCAL_MODEL):
        return DEFAULT_LOCAL_MODEL
    return _MODEL_SIZE


def get_model():
    """懒加载 faster-whisper 模型（线程安全单例，首次调用自动探测 GPU/CPU）。"""
    global _model, _DEVICE, _COMPUTE_TYPE
    if _model is None:
        with _model_lock:
            if _model is None:
                if _DEVICE is None or _COMPUTE_TYPE is None:
                    _DEVICE, _COMPUTE_TYPE = _resolve_device()
                source = _resolve_model_source()
                logger.info(
                    "asr: 加载 Whisper 模型 %s（device=%s, compute_type=%s，首次加载约 30~60s）...",
                    source, _DEVICE, _COMPUTE_TYPE,
                )
                from faster_whisper import WhisperModel

                _model = WhisperModel(source, device=_DEVICE, compute_type=_COMPUTE_TYPE)
                logger.info("asr: 模型加载完成")
    return _model


def transcribe(audio_bytes: bytes, language: Optional[str] = "zh") -> str:
    """
    音频字节 → 文本。

    Args:
        audio_bytes: 原始音频（wav/mp3/ogg/webm 等）
        language: 识别语言（默认中文）

    Returns:
        识别文本（空音频返回空串）
    """
    if not audio_bytes:
        return ""
    model = get_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language,
        beam_size=5,
        # 不启用 VAD 过滤：实测 Silero VAD 会把低声/短句录音整段误判为静音
        # （如 3s 音频被 "VAD filter removed 00:03.011" 全部丢弃，识别 0 字），
        # 客服语音场景录音通常干净，直接交给 whisper 转录更可靠；
        # 无语音的空白录音由上层空文本校验（422）兜底。
    )
    text = "".join(seg.text for seg in segments).strip()
    logger.info("asr: 识别完成（%d 字节音频 → %d 字）", len(audio_bytes), len(text))
    return text
