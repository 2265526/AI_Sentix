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
import re
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# 默认中文音色（可经环境变量 TTS_VOICE 调整，或在调用时覆盖）
DEFAULT_VOICE = settings.tts_voice

# 音色参考（edge-tts 支持的其他中文音色）
VOICES = {
    "晓晓": "zh-CN-XiaoxiaoNeural",   # 女声，温柔
    "云希": "zh-CN-YunxiNeural",      # 男声，阳光
    "云健": "zh-CN-YunjianNeural",    # 男声，沉稳
    "晓伊": "zh-CN-XiaoyiNeural",     # 女声，甜美
}

# emoji / 特殊符号：TTS 会把表情读成文字（如 😊 → "嘴角含笑"），合成前剔除
# 覆盖：Emoji 主区、扩展区、符号区（箭头/几何）、变体选择符、ZWJ 连接符、肤色修饰符、区域指示符等
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"      # Emoji 主区 + 扩展区
    "\U00002600-\U000027BF"      # 杂项符号（含常见表情 ☀★☺ 等）
    "\U000023E9-\U000023FA"      # 媒体控制符号
    "\U00002B50-\U00002B55"      # 星星/圆圈
    "\U0000FE0F"                 # 变体选择符
    "\u200D"                     # 零宽连接符
    "\U0001F3FB-\U0001F3FF"      # 肤色修饰符
    "\U0001F1E6-\U0001F1FF"      # 区域指示符（国旗）
    "]+"
)


def strip_emoji(text: str) -> str:
    """剔除文本中的 emoji 与特殊符号，避免 TTS 把它们读成文字描述。"""
    if not text:
        return text
    return _EMOJI_RE.sub("", text).strip()


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
    clean = strip_emoji(text)  # 剔除 emoji，避免 TTS 读出"嘴角含笑"等表情描述
    if not clean:
        return b""
    try:
        audio = asyncio.run(_synthesize_async(clean.strip(), voice))
        logger.info("tts: 合成完成（%d 字 → %d 字节）", len(clean), len(audio))
        return audio
    except Exception as e:  # 网络/接口异常：返回空，由上层降级为文本回复
        logger.error("tts: 合成失败: %s", e)
        return b""
