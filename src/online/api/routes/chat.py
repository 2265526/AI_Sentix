# -*- coding: utf-8 -*-
"""
api/routes/chat.py —— /v1/chat/text 对话接口
=============================================
对应《开发文档》阶段三产出物：
  "Agent 核心服务，支持文本输入、流式输出"

- 默认流式：SSE（text/event-stream），事件为 JSON：
      {"type":"meta","intent":...,"tools_used":[...]}
      {"type":"token","content":"..."}
      {"type":"done"}
- stream=false：返回普通 JSON（ChatResponse）。
"""
import json
import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from psycopg2.extensions import connection

from src.online.api.models import ChatRequest, ChatResponse
from src.online.core.voice import asr, tts
from src.online.db.session import get_db
from src.online.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _sse(event: dict) -> str:
    """SSE 事件序列化。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/text", response_model=ChatResponse, summary="文本对话（支持流式）")
def chat_text(req: ChatRequest, conn: connection = Depends(get_db)):
    """Agent 文本对话：意图识别 → 工具执行（SQL/RAG）→ 二次模型回调。"""
    service = ChatService()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    if not req.stream:
        # 非流式：普通 JSON
        result = service.chat(conn, req.message, history)
        return ChatResponse(
            reply=result["reply"],
            intent=result["intent"],
            tools_used=result["tools_used"],
        )

    # 流式：SSE
    def event_stream():
        for event in service.chat_stream(conn, req.message, history):
            yield _sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# 阶段四：多模态语音链路（音频 → ASR → LLM → TTS → 音频）
# ============================================================
@router.post("/audio", summary="语音对话（音频→ASR→LLM→TTS→mp3 音频）")
def chat_audio(
    file: UploadFile = File(..., description="录音文件（wav/mp3/ogg/webm，≤20MB）"),
    history: str = Form("[]", description="多轮历史 JSON 字符串"),
    conn: connection = Depends(get_db),
):
    """
    阶段四产物：POST /v1/chat/audio。
    链路：录音（前端 MediaRecorder）→ 本地 ASR（faster-whisper）→
          Agent 文本对话（ChatService）→ TTS（edge-tts）→ 返回 mp3 音频。
    识别文本与客服回复通过响应头 X-Transcript / X-Reply 返回（URL 编码），
    便于前端同时展示文字。首次请求会加载 Whisper 模型（约 30~60s），之后秒级。
    """
    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="音频文件为空")
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="音频超过 20MB 限制")

    # 1) ASR：音频 → 文本
    try:
        text = asr.transcribe(audio_bytes)
    except Exception as e:
        logger.error("chat/audio: ASR 失败: %s", e)
        raise HTTPException(status_code=500, detail=f"语音识别失败: {e}")
    if not text:
        raise HTTPException(status_code=422, detail="未能识别到语音内容，请靠近麦克风重试")

    # 2) LLM：文本对话（意图识别 + 工具 + 二次回调）
    try:
        history_list = json.loads(history or "[]")
        if not isinstance(history_list, list):
            history_list = []
    except json.JSONDecodeError:
        history_list = []
    result = ChatService().chat(conn, text, history_list)

    # 3) TTS：回复文本 → mp3 音频
    mp3 = tts.synthesize(result["reply"])
    if not mp3:  # TTS 失败降级：返回 JSON，前端展示文本
        return {
            "transcript": text,
            "reply": result["reply"],
            "intent": result["intent"],
            "tools_used": result["tools_used"],
            "audio": None,
        }

    # 4) 返回音频流（识别文本/回复放响应头，前端可展示）
    return Response(
        content=mp3,
        media_type="audio/mpeg",
        headers={
            "X-Transcript": quote(text),
            "X-Reply": quote(result["reply"]),
            "X-Intent": quote(result["intent"] or ""),
            "Cache-Control": "no-cache",
        },
    )
