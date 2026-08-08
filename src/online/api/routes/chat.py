"""
对话接口

- 默认流式：SSE（text/event-stream），事件为 JSON：
      {"type":"meta","intent":...,"tools_used":[...]}
      {"type":"token","content":"..."}
      {"type":"done"}
- stream=false：返回普通 JSON（ChatResponse）。
"""
import json
import logging
import os
import re
import time
import uuid
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from psycopg2.extensions import connection

from config.settings import settings
from src.common.utils import sse_format
from src.online.api.models import ChatRequest, ChatResponse
from src.online.core.voice import asr, tts
from src.online.db.session import get_db
from src.online.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])

# 会话标识白名单：仅允许字母/数字/连字符，1~64 位（防超长/特殊字符滥用）
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def _normalize_session(session_id: Optional[str]) -> str:
    """
    会话标识归一化：合法则原样返回；为空或非法时服务端生成 UUID。
    兜底保证无状态客户端（如 curl）也能开启带记忆的会话。
    """
    if session_id and _SESSION_ID_RE.match(session_id):
        return session_id
    return str(uuid.uuid4())


@router.post("/text", response_model=ChatResponse, summary="文本对话（支持流式）")
def chat_text(
    req: ChatRequest,
    conn: connection = Depends(get_db),
    response: Response = None,
):
    """Agent 文本对话：记忆读取 → 问题增强 → 意图识别 → 工具执行 → 二次模型回调。"""
    session_id = _normalize_session(req.session_id)
    response.headers["X-Session-Id"] = session_id

    service = ChatService()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    if not req.stream:
        # 非流式：普通 JSON（携带过期信号与增强前后问题）
        result = service.chat(conn, req.message, history, session_id)
        return ChatResponse(
            reply=result["reply"],
            intent=result["intent"],
            tools_used=result["tools_used"],
            context_reset=result["context_reset"],
            original_query=result["original_query"],
            enriched_query=result["enriched_query"],
        )

    # 流式：SSE（meta 事件含 intent/tools_used/original/enriched/context_reset）
    def event_stream():
        for event in service.chat_stream(conn, req.message, history, session_id):
            yield sse_format(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


# 多模态语音链路（音频 → ASR → LLM → TTS → 音频）
@router.post("/audio", summary="语音对话（音频→ASR→LLM→TTS→mp3 音频）")
def chat_audio(
    file: UploadFile = File(..., description="录音文件（wav/mp3/ogg/webm，≤20MB）"),
    history: str = Form("[]", description="多轮历史 JSON 字符串"),
    session_id: str = Form("", description="会话标识（短期记忆，与文本对话共享）"),
    conn: connection = Depends(get_db),
):
    """
    链路：录音（前端 MediaRecorder）→ 本地 ASR（faster-whisper）→
          Agent 文本对话（ChatService）→ TTS（edge-tts）→ 返回 mp3 音频。
    识别文本与客服回复通过响应头 X-Transcript / X-Reply 返回（URL 编码），
    便于前端同时展示文字。首次请求会加载 Whisper 模型。
    """
    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="音频文件为空")
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="音频超过 20MB 限制")

    # 调试：配置 AUDIO_DEBUG_DIR后保存每次上传的原始音频，便于排查识别为空的录音
    debug_dir = settings.audio_debug_dir
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        fname = f"{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000}.webm"
        with open(os.path.join(debug_dir, fname), "wb") as df:
            df.write(audio_bytes)
        logger.info("chat/audio: 已保存调试音频 %s（%d 字节）", fname, len(audio_bytes))

    # 1) ASR：音频 → 文本
    try:
        text = asr.transcribe(audio_bytes)
    except Exception as e:
        logger.error("chat/audio: ASR 失败: %s", e)
        raise HTTPException(status_code=500, detail=f"语音识别失败: {e}")
    if not text:
        raise HTTPException(status_code=422, detail="未能识别到语音内容，请靠近麦克风重试")

    # 会话标识：空/非法 → 服务端生成，随响应头返回（无状态客户端也能用）
    sid = _normalize_session(session_id or None)

    # 2) LLM：文本对话（记忆 + 意图识别 + 工具 + 二次回调，携带同一 session_id）
    try:
        history_list = json.loads(history or "[]")
        if not isinstance(history_list, list):
            history_list = []
    except json.JSONDecodeError:
        history_list = []
    result = ChatService().chat(conn, text, history_list, sid)

    # 3) TTS：回复文本 → mp3 音频
    mp3 = tts.synthesize(result["reply"])
    if not mp3:  # TTS 失败降级：返回 JSON，前端展示文本
        return JSONResponse(
            status_code=200,
            content={
                "transcript": text,
                "reply": result["reply"],
                "intent": result["intent"],
                "tools_used": result["tools_used"],
                "audio": None,
            },
            headers=_audio_headers(sid, result["context_reset"]),
        )

    # 4) 返回音频流（识别文本/回复放响应头，前端可展示）
    return Response(
        content=mp3,
        media_type="audio/mpeg",
        headers={
            "X-Transcript": quote(text),
            "X-Reply": quote(result["reply"]),
            "X-Intent": quote(result["intent"] or ""),
            "Cache-Control": "no-cache",
            **_audio_headers(sid, result["context_reset"]),
        },
    )


def _audio_headers(session_id: str, context_reset: bool) -> dict:
    """语音接口公共响应头：会话标识 + 过期信号（二进制流无法携带 meta 事件，走响应头）。"""
    headers = {"X-Session-Id": session_id}
    if context_reset:
        headers["X-Context-Expired"] = "true"
    return headers
