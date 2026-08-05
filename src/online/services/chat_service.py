# -*- coding: utf-8 -*-
"""
services/chat_service.py —— 文本对话服务编排
=============================================
对应《开发文档》阶段三任务 3 与产出物：
  "将检索结果（SQL 实体或 RAG 片段）和用户问题再次丢给 LLM，生成最终的流式回复"
  "Agent 核心服务，支持文本输入、流式输出"

编排链路（一次对话）：
  1. 意图识别（LLM function calling）→ tool_call
  2. 服务路由（router）执行工具：SQL 检索 或 阶段二 RAG 引擎
  3. 二次模型回调：工具结果 + 用户问题 → LLM 生成最终回复（流式/非流式）

降级策略（文档阶段五完整实现，这里保留基础版）：
  - 意图识别失败：跳过工具，直接 LLM 回答；
  - 二次回调失败：返回预置兜底话术，保证接口不 500。
"""
import logging
from typing import Any, Dict, Iterator, List, Optional

from src.online.core.agent.intent import detect_intent
from src.online.core.agent.router import execute_tool_calls
from src.online.core.llm.client import LLMClient

logger = logging.getLogger(__name__)

# 客服回答系统提示词（与工具返回的参考信息配合）
CUSTOMER_SERVICE_PROMPT = """你是一名专业的电商AI智能客服，负责解答用户在商品、库存、物流、售后、使用说明等方面的问题。
回答规则：
1. 若提供了【工具返回】的参考信息，严格依据参考信息回答；价格、库存、物流时效等数值必须与参考信息一致，不得编造；
2. 参考信息未覆盖的部分，如实说明"该信息暂未收录"，不要猜测；
3. 用户只是寒暄、闲聊时，直接友好回应即可，无需工具信息；
4. 回答使用简体中文，简洁、专业、友好。"""

# 二次模型回调失败时的兜底话术
FALLBACK_REPLY = "抱歉，我这边暂时遇到了点问题，请稍后再试。"


class ChatService:
    """文本对话编排服务（每请求一个实例，连接由 FastAPI 依赖注入）。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    # --------------------------------------------------------
    # 内部：组装二次回调消息
    # --------------------------------------------------------
    def _build_reply_messages(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]],
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        return LLMClient.build_messages(
            system_prompt=CUSTOMER_SERVICE_PROMPT,
            user_query=query,
            history=history,
            tool_results=tool_results,
        )

    # --------------------------------------------------------
    # 非流式对话（测试/内部使用）
    # --------------------------------------------------------
    def chat(
        self,
        conn,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """完整对话一轮，返回 {reply, intent, tools_used}。"""
        # 1) 意图识别
        intent = detect_intent(self.llm, query, history)

        # 2) 执行工具（意图识别出错则跳过工具）
        tool_results: List[Dict[str, Any]] = []
        if intent.has_tool_call:
            try:
                tool_results = execute_tool_calls(conn, intent.tool_calls)
            except Exception as e:  # 工具执行异常不阻断回答
                logger.warning("chat: 工具执行失败 %s: %s", intent.tool_calls, e)

        # 3) 二次模型回调（非流式）
        try:
            messages = self._build_reply_messages(query, history, tool_results)
            reply = self.llm.chat(messages)
        except Exception as e:  # LLM/网络异常统一兜底，保证接口不 500
            logger.error("chat: 二次回调失败: %s", e)
            reply = FALLBACK_REPLY

        return {
            "reply": reply,
            "intent": intent.first_tool_name,
            "tools_used": [tr["name"] for tr in tool_results],
        }

    # --------------------------------------------------------
    # 流式对话（SSE）
    # --------------------------------------------------------
    def chat_stream(
        self,
        conn,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式对话，yield 结构化事件：
            {"type": "meta", "intent": ..., "tools_used": [...]}
            {"type": "token", "content": "..."}
            {"type": "done"}
        """
        intent = detect_intent(self.llm, query, history)

        tool_results: List[Dict[str, Any]] = []
        if intent.has_tool_call:
            try:
                tool_results = execute_tool_calls(conn, intent.tool_calls)
            except Exception as e:
                logger.warning("chat_stream: 工具执行失败 %s: %s", intent.tool_calls, e)

        yield {
            "type": "meta",
            "intent": intent.first_tool_name,
            "tools_used": [tr["name"] for tr in tool_results],
        }

        messages = self._build_reply_messages(query, history, tool_results)
        try:
            for delta in self.llm.chat_stream(messages):
                if delta:
                    yield {"type": "token", "content": delta}
        except Exception as e:  # LLM/网络异常统一兜底，保证 SSE 流不中断
            logger.error("chat_stream: 二次回调失败: %s", e)
            yield {"type": "token", "content": FALLBACK_REPLY}

        yield {"type": "done"}
