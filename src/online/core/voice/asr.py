# -*- coding: utf-8 -*-
"""
core/voice/asr.py —— 本地语音识别（ASR）
=========================================
阶段四实现：使用 faster-whisper（本地离线，无需外部 key）。

- 模型：Systran/faster-whisper-small（CPU int8 量化），首次调用自动下载并缓存；
- 进程内单例：模型只加载一次，避免每次请求重载（首载约 30~60s，之后单条识别 1~2s）；
- 输入：音频字节（wav / mp3 / ogg / webm 等，PyAV 自动解码）；
- 输出：简体中文文本。

环境依赖：faster-whisper（pip 安装，内部依赖 PyAV，无需系统 ffmpeg）。
"""
import io
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# 模型尺寸：small 中文准确率与速度均衡；如追求更快可改 "base"
_MODEL_SIZE = "small"
_DEVICE = "cpu"
_COMPUTE_TYPE = "int8"

_model = None
_model_lock = threading.Lock()


def get_model():
    """懒加载 faster-whisper 模型（线程安全单例）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("asr: 加载 Whisper 模型 %s（首次加载约 30~60s）...", _MODEL_SIZE)
                from faster_whisper import WhisperModel

                _model = WhisperModel(_MODEL_SIZE, device=_DEVICE, compute_type=_COMPUTE_TYPE)
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
        vad_filter=True,  # 内置 VAD，跳过静音段，提高短句识别率
    )
    text = "".join(seg.text for seg in segments).strip()
    logger.info("asr: 识别完成（%d 字节音频 → %d 字）", len(audio_bytes), len(text))
    return text
