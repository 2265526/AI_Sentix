# -*- coding: utf-8 -*-
"""
core/voice/tts.py —— 语音合成（TTS）
=====================================
阶段四实现：使用 edge-tts（微软 Edge 在线语音接口，免 key、服务端调用）。

- 服务端联网即可，用户端（浏览器）零依赖，任何浏览器都能播放返回的 mp3；
- 默认中文女声 zh-CN-XiaoxiaoNeural，可换其他音色；
- 输入：中文文本；输出：mp3 音频字节。

环境依赖：edge-tts（pip 安装）。
"""
import asyncio
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 默认中文音色（可在调用时覆盖）
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

# 音色参考（edge-tts 支持的其他中文音色）
VOICES = {
    "晓晓": "zh-CN-XiaoxiaoNeural",   # 女声，温柔
    "云希": "zh-CN-YunxiNeural",      # 男声，阳光
    "云健": "zh-CN-YunjianNeural",    # 男声，沉稳
    "晓伊": "zh-CN-XiaoyiNeural",     # 女声，甜美
}


def _voices_map() -> None:
    pass  # 占位：如需运行时列出全部音色可调 edge_tts.list_voices()


async def _synthesize_async(text: str, voice: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    文本 → mp3 音频字节。

    Args:
        text: 要合成的中文文本
        voice: 音色（默认晓晓）

    Returns:
        mp3 字节；合成失败返回 b""（调用方兜底）
    """
    if not text or not text.strip():
        return b""
    try:
        audio = asyncio.run(_synthesize_async(text.strip(), voice))
        logger.info("tts: 合成完成（%d 字 → %d 字节）", len(text), len(audio))
        return audio
    except Exception as e:  # 网络/接口异常：返回空，由上层降级为文本回复
        logger.error("tts: 合成失败: %s", e)
        return b""
