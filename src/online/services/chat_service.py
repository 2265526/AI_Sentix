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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from src.online.core.agent.enricher import enrich_query
from src.online.core.agent.intent import detect_intent
from src.online.core.agent.router import execute_tool_calls
from src.online.core.llm.client import LLMClient
from src.online.core.llm.prompt_templates import CUSTOMER_SERVICE_PROMPT, FALLBACK_REPLY
from src.online.core.memory import extractor
from src.online.db.repositories import memory_repo

logger = logging.getLogger(__name__)


@dataclass
class ChatState:
    """
    单次对话的结构化状态（调研方案 P0「structured state」）。

    贯穿 ①记忆读取 → ②增强 → ③意图 → ④工具 → ⑤回写 → ⑥⑦二次回调 全链路，
    让每个环节的产出可观测（trace）、可扩展。
    """

    user_query: str                                   # 用户原始问题
    enhanced_query: str = ""                          # 增强后问题（enricher 输出）
    context_reset: bool = False                       # 会话过期/切换信号
    intent_tag: Optional[str] = None                  # 预分类意图标签（规则分类器）
    intent_tool: Optional[str] = None                 # 实际选择的工具名
    tools_used: List[str] = field(default_factory=list)  # 实际执行的工具
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    degraded: bool = False                            # 结构化检索是否降级命中
    retrieval_hits: int = 0                           # 检索召回条数（RAG/结构化）
    started_at: float = field(default_factory=time.time)  # 请求开始时间（trace）


class ChatService:
    """文本对话编排服务（每请求一个实例，连接由 FastAPI 依赖注入）。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    # --------------------------------------------------------
    # 内部：记忆读取与问题增强（①读 session_context ②enrich_query）
    # --------------------------------------------------------
    def _load_memory(
        self, conn, session_id: str, query: str, history: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """
        读取会话上下文并做问题增强。

        Returns:
            {"enhanced_query", "context_reset": bool, "original"}
            会话过期/首次请求/切换意图清空时 context_reset=True；
            记忆相关异常一律兜底：返回原问题直通（绝不阻断主链路）。
        """
        ctx: Dict[str, Any] = {}
        context_reset = False
        try:
            record = memory_repo.get_session_context(conn, session_id)
            if record is None:
                # 请求带了 session_id 但无活跃上下文（TTL 过期/被清理/首次请求）
                context_reset = True
            else:
                ctx = record["context"] or {}

            result = enrich_query(query, ctx, history)
            if result.switch_intent:
                # 用户明确切换话题：清空服务端上下文，并向前端发过期信号
                try:
                    memory_repo.clear_session_context(conn, session_id)
                except Exception as e:
                    logger.warning("chat: 清空会话上下文失败: %s", e)
                context_reset = True
            return {
                "enhanced_query": result.query,
                "context_reset": context_reset,
                "original": result.original,
            }
        except Exception as e:
            logger.warning("chat: 记忆读取/问题增强失败，原问题直通: %s", e)
            try:
                conn.rollback()  # 恢复事务，避免影响后续工具 SQL
            except Exception:
                pass
            return {"enhanced_query": query, "context_reset": False, "original": query}

    # --------------------------------------------------------
    # 内部：记忆回写（⑤实体抽取回写 + 交互日志）
    # --------------------------------------------------------
    def _save_memory(
        self,
        conn,
        session_id: str,
        query: str,
        enhanced_query: str,
        intent,
        tool_results: List[Dict[str, Any]],
        tool_failed: bool,
    ) -> None:
        """回写会话上下文与交互日志；任何异常仅记日志，不阻断主链路。"""
        try:
            if tool_failed or not tool_results:
                # 工具执行失败/无结果：不更新实体，仅轮次+1、刷新 TTL（保留旧快照）
                old = memory_repo.get_session_context(conn, session_id)
                memory_repo.upsert_session_context(conn, session_id, old["context"] if old else {})
            else:
                extractor.update_session_context(conn, session_id, query, intent, tool_results)
            # 交互流水（P1：画像原料）
            entities = extractor.extract_entities(query, intent, tool_results)
            memory_repo.log_interaction(
                conn,
                user_id=None,
                session_id=session_id,
                query=query,
                enhanced_query=enhanced_query,
                tool_called=intent.first_tool_name if intent.has_tool_call else None,
                result_count=len(tool_results),
                entities=entities,
            )
        except Exception as e:
            logger.warning("chat: 记忆回写失败: %s", e)
            try:
                conn.rollback()  # 恢复事务，避免影响请求内后续 SQL
            except Exception:
                pass

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
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """完整对话一轮，返回 {reply, intent, tools_used, context_reset, original_query, enriched_query}。"""
        state = ChatState(user_query=query)

        # ①② 记忆读取 + 问题增强（session_id 为空则跳过，行为与旧版一致）
        if session_id:
            memory = self._load_memory(conn, session_id, query, history)
            state.enhanced_query = memory["enhanced_query"]
            state.context_reset = memory["context_reset"]
        else:
            state.enhanced_query = query

        # ③④ 意图识别（预分类限制工具集）+ 工具执行
        intent = detect_intent(self.llm, state.enhanced_query, history)
        state.intent_tag = intent.intent
        state.intent_tool = intent.first_tool_name

        tool_results: List[Dict[str, Any]] = []
        tool_failed = False
        if intent.has_tool_call:
            try:
                tool_results = execute_tool_calls(conn, intent.tool_calls)
                state.tool_results = tool_results
                state.tools_used = [tr["name"] for tr in tool_results]
                state.retrieval_hits = sum(len(tr.get("raw") or []) for tr in tool_results)
                state.degraded = any("近似匹配" in (tr.get("result") or "") for tr in tool_results)
            except Exception as e:  # 工具执行异常不阻断回答
                tool_failed = True
                logger.warning("chat: 工具执行失败 %s: %s", intent.tool_calls, e)
                try:
                    conn.rollback()  # 工具 SQL 失败后恢复事务，保证记忆写入/后续可用
                except Exception:
                    pass

        # ⑤ 记忆回写
        if session_id:
            self._save_memory(conn, session_id, query, state.enhanced_query, intent, tool_results, tool_failed)

        # ⑥⑦ 二次模型回调（非流式，传增强后问题）
        try:
            messages = self._build_reply_messages(state.enhanced_query, history, tool_results)
            reply = self.llm.chat(messages)
        except Exception as e:  # LLM/网络异常统一兜底，保证接口不 500
            logger.error("chat: 二次回调失败: %s", e)
            reply = FALLBACK_REPLY

        # trace：请求级可观测日志（意图/工具/召回/降级/耗时）
        logger.info(
            "chat trace: query=%r intent_tag=%s tool=%s tools=%s hits=%d degraded=%s ctx_reset=%s %.2fs",
            state.user_query, state.intent_tag, state.intent_tool, state.tools_used,
            state.retrieval_hits, state.degraded, state.context_reset,
            time.time() - state.started_at,
        )

        return {
            "reply": reply,
            "intent": state.intent_tool,
            "tools_used": state.tools_used,
            "context_reset": state.context_reset,
            "original_query": state.user_query,
            "enriched_query": state.enhanced_query,
        }

    # --------------------------------------------------------
    # 流式对话（SSE）
    # --------------------------------------------------------
    def chat_stream(
        self,
        conn,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式对话，yield 结构化事件：
            {"type": "meta", "intent": ..., "tools_used": [...],
             "original": ..., "enriched": ..., "context_reset": bool}
            {"type": "token", "content": "..."}
            {"type": "done"}
        """
        state = ChatState(user_query=query)

        # ①② 记忆读取 + 问题增强
        if session_id:
            memory = self._load_memory(conn, session_id, query, history)
            state.enhanced_query = memory["enhanced_query"]
            state.context_reset = memory["context_reset"]
        else:
            state.enhanced_query = query

        # ③④ 意图识别 + 工具执行
        intent = detect_intent(self.llm, state.enhanced_query, history)
        state.intent_tag = intent.intent
        state.intent_tool = intent.first_tool_name

        tool_results: List[Dict[str, Any]] = []
        tool_failed = False
        if intent.has_tool_call:
            try:
                tool_results = execute_tool_calls(conn, intent.tool_calls)
                state.tool_results = tool_results
                state.tools_used = [tr["name"] for tr in tool_results]
                state.retrieval_hits = sum(len(tr.get("raw") or []) for tr in tool_results)
                state.degraded = any("近似匹配" in (tr.get("result") or "") for tr in tool_results)
            except Exception as e:
                tool_failed = True
                logger.warning("chat_stream: 工具执行失败 %s: %s", intent.tool_calls, e)
                try:
                    conn.rollback()  # 恢复事务，保证记忆写入/后续可用
                except Exception:
                    pass

        yield {
            "type": "meta",
            "intent": state.intent_tool,
            "tools_used": state.tools_used,
            "original": state.user_query,
            "enriched": state.enhanced_query,
            "context_reset": state.context_reset,
        }

        # ⑤ 记忆回写
        if session_id:
            self._save_memory(conn, session_id, query, state.enhanced_query, intent, tool_results, tool_failed)

        # ⑥⑦ 二次模型回调（流式，传增强后问题）
        messages = self._build_reply_messages(state.enhanced_query, history, tool_results)
        try:
            for delta in self.llm.chat_stream(messages):
                if delta:
                    yield {"type": "token", "content": delta}
        except Exception as e:  # LLM/网络异常统一兜底，保证 SSE 流不中断
            logger.error("chat_stream: 二次回调失败: %s", e)
            yield {"type": "token", "content": FALLBACK_REPLY}

        # trace：请求级可观测日志
        logger.info(
            "chat_stream trace: query=%r intent_tag=%s tool=%s tools=%s hits=%d degraded=%s ctx_reset=%s %.2fs",
            state.user_query, state.intent_tag, state.intent_tool, state.tools_used,
            state.retrieval_hits, state.degraded, state.context_reset,
            time.time() - state.started_at,
        )

        yield {"type": "done"}
