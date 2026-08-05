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

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from psycopg2.extensions import connection

from src.online.api.models import ChatRequest, ChatResponse
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
