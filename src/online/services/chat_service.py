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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from src.online.core.agent import params
from src.online.core.agent.enricher import enrich_query
from src.online.core.agent.intent import detect_intent
from src.online.core.agent.router import execute_tool_calls
from src.online.core.llm.client import LLMClient
from src.online.core.llm.prompt_templates import CUSTOMER_SERVICE_PROMPT, FALLBACK_REPLY
from src.online.core.memory import extractor
from src.online.core.monitor.monitor import MonitorRequest, MonitorStep, monitor_store
from src.online.db.repositories import memory_repo

logger = logging.getLogger(__name__)


def _ms(t0: float) -> int:
    """耗时（毫秒）。"""
    return int(round((time.time() - t0) * 1000))


def _summarize_tool_results(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成工具返回摘要（name / 命中条数 / 返回文本预览），供监控详情与导出。

    raw 为结构化行（列表）时记命中条数；result 为给 LLM 的格式化文本，
    预览截断 300 字符，避免环形缓冲被大返回撑爆。
    """
    out: List[Dict[str, Any]] = []
    for tr in tool_results or []:
        raw = tr.get("raw")
        hits = len(raw) if isinstance(raw, list) else 0
        result = str(tr.get("result") or "")
        out.append({
            "name": tr.get("name"),
            "hits": hits,
            "preview": result[:300],
        })
    return out


def _accumulate_tokens(prompt: int, completion: int, usage: Dict[str, Any]) -> tuple:
    """把一次 LLM 调用的 usage 累加进 (prompt_tokens, completion_tokens)。"""
    if not usage:
        return prompt, completion
    return (
        prompt + int(usage.get("prompt_tokens") or 0),
        completion + int(usage.get("completion_tokens") or 0),
    )

# SKU 模式（如 G000115 / 21873056212；子串匹配：'G000115 多少钱' 中也能提取）
_SKU_RE = re.compile(r"[A-Za-z]{1,4}\d{3,}|\d{6,}")

# 预分类标签 → 兜底工具（模型未调用工具时的决策兜底）
_FALLBACK_TOOL = {
    "INVENTORY": "get_product_inventory",
    "PRICE": "get_product_price",
    "RECOMMEND": "product_recommendation",
}

# 预分类标签 → 直驱工具（P1-5：预分类命中直接执行，不依赖 LLM function calling 选工具）
_INTENT_TOOL = {
    "FAQ": "get_knowledge_base",
    "RECOMMEND": "product_recommendation",
    "PRICE": "get_product_price",
    "INVENTORY": "get_product_inventory",
}


def _has_product_signal(query: str) -> bool:
    """query 是否含明确的商品检索信号（品牌 / 品类 / SKU / 价格词）。"""
    if not query:
        return False
    if _SKU_RE.search(query):
        return True
    if params.extract_brand(query):
        return True
    low = query.lower()
    if any(w.lower() in low for w in params.CATEGORY_WORDS):
        return True
    return any(w in query for w in ("有货", "现货", "库存", "多少钱", "价格", "推荐", "发货"))


def _fallback_tool_call(intent_tag: Optional[str], query: str) -> Optional[Dict[str, Any]]:
    """
    决策兜底：模型未调用工具时，依据预分类标签 / SKU / 商品词信号强制走一次结构化检索。

    P0-2：FAQ 语义强制走一次知识库检索（与商品兜底对称，避免 LLM 未调用
          get_knowledge_base 时零检索瞎答"未收录"）；RAG 为语义检索，query 用原句不清洗。
    P0-3：query 含 SKU 码（如 G000115）时直接按 sku_code 精确匹配，绕过名称模糊。
    P0-4：product_name 先经 clean_product_name 预清洗（去掉'库存/价格/多少'等
          意图词），保证 L1 名称模糊匹配能命中精确商品。
    返回兜底 tool_call 或 None（纯寒暄/FAQ 语义不兜底商品检索）。
    """
    if intent_tag == "FAQ":
        # 售后/使用说明语义走知识库（RAG 语义检索，原句效果最好）
        return {"name": "get_knowledge_base", "arguments": {"query": query}}
    sku = _SKU_RE.search(query or "")
    if sku:
        # 用户直接报 SKU 码（G000115 / IP15PM256）：走 L0 精确匹配
        tool = _FALLBACK_TOOL.get(intent_tag, "get_product_price")
        return {"name": tool, "arguments": {"sku_code": sku.group(0)}}
    if intent_tag in _FALLBACK_TOOL:
        clean = params.clean_product_name(query) or query
        return {"name": _FALLBACK_TOOL[intent_tag], "arguments": {"product_name": clean}}
    if _has_product_signal(query):
        # 预分类未命中但含商品词（如'休闲，宽松的衬衫'）→ 按导购语义走推荐检索
        clean = params.clean_product_name(query) or query
        return {"name": "product_recommendation", "arguments": {"product_name": clean}}
    return None


def _resolve_tool_calls(intent, query: str) -> List[Dict[str, Any]]:
    """
    P1-5 预分类直驱：预分类命中（实测规则分类器 4/4 稳定）时，工具已确定，
    直接构造对应工具调用——不再依赖 LLM function calling 做路由选择；
    LLM 若返回了 arguments（如品牌/型号/价格）则优先采用，缺失键用兜底默认值补全。
    预分类未命中：沿用 LLM tool_calls；为空再走 _fallback_tool_call。
    """
    if intent.intent in _INTENT_TOOL:
        name = _INTENT_TOOL[intent.intent]
        args: Dict[str, Any] = {}
        if intent.tool_calls:
            args = dict(intent.tool_calls[0].get("arguments") or {})
        if name == "get_knowledge_base":
            # RAG 语义检索：原句作为 query 效果最好
            args.setdefault("query", query)
        else:
            sku = _SKU_RE.search(query or "")
            if sku:
                # SKU 精确优先（用户报码直接命中）
                args.setdefault("sku_code", sku.group(0))
            elif not args.get("product_name"):
                args["product_name"] = params.clean_product_name(query) or query
        return [{"name": name, "arguments": args}]
    # 预分类未命中：LLM 调用优先，空则决策兜底
    calls = list(intent.tool_calls)
    if not calls:
        fallback = _fallback_tool_call(intent.intent, query)
        if fallback:
            return [fallback]
    return calls


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
    ) -> bool:
        """回写会话上下文与交互日志；任何异常仅记日志不阻断主链路。返回是否成功。"""
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
            return True
        except Exception as e:
            logger.warning("chat: 记忆回写失败: %s", e)
            try:
                conn.rollback()  # 恢复事务，避免影响请求内后续 SQL
            except Exception:
                pass
            return False

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
        steps: List[MonitorStep] = []
        fallback_used = False
        llm_ok = True
        prompt_tokens = 0
        completion_tokens = 0
        t_total = time.time()

        # ①② 记忆读取 + 问题增强（session_id 为空则跳过，行为与旧版一致）
        if session_id:
            t0 = time.time()
            memory = self._load_memory(conn, session_id, query, history)
            state.enhanced_query = memory["enhanced_query"]
            state.context_reset = memory["context_reset"]
            steps.append(MonitorStep(
                stage="memory",
                status="ok" if not state.context_reset else "degraded",
                detail=f"读会话上下文 + 问题增强（context_reset={state.context_reset}）",
                ms=_ms(t0),
            ))
        else:
            state.enhanced_query = query

        # ③ 意图识别（预分类限制工具集）
        t0 = time.time()
        intent = detect_intent(self.llm, state.enhanced_query, history)
        state.intent_tag = intent.intent
        state.intent_tool = intent.first_tool_name
        intent_extra: Dict[str, Any] = {}
        if intent.error:
            intent_extra["error"] = intent.error
        if intent.raw_response:
            intent_extra["raw"] = intent.raw_response[:200]
        steps.append(MonitorStep(
            stage="intent",
            status="error" if intent.error else "ok",
            detail=f"预分类={state.intent_tag or '无'} 工具={state.intent_tool or '无'}",
            ms=_ms(t0),
            extra=intent_extra,
        ))
        prompt_tokens, completion_tokens = _accumulate_tokens(
            prompt_tokens, completion_tokens, getattr(self.llm, "get_last_usage", lambda: {})())

        # ④ 工具调用：预分类直驱 → LLM tool_calls → 决策兜底（P1-5 不依赖 LLM 选工具）
        calls = _resolve_tool_calls(intent, state.enhanced_query or query)
        if not intent.has_tool_call:
            fallback_used = True
            if calls:
                logger.info("chat: 模型未调用工具，按预分类/兜底执行 %s（query=%r）", calls[0].get("name"), query)

        tool_results: List[Dict[str, Any]] = []
        tool_failed = False
        if calls:
            t0 = time.time()
            try:
                tool_results = execute_tool_calls(conn, calls)
                state.tool_results = tool_results
                state.tools_used = [tr["name"] for tr in tool_results]
                state.retrieval_hits = sum(len(tr.get("raw") or []) for tr in tool_results)
                state.degraded = any("相关商品" in (tr.get("result") or "") for tr in tool_results)
                steps.append(MonitorStep(
                    stage="tool", status="degraded" if state.degraded else "ok",
                    detail=f"执行 {state.tools_used} 命中{state.retrieval_hits}"
                           f"{'（决策兜底）' if fallback_used else ''}"
                           f"{'（降级）' if state.degraded else ''}",
                    ms=_ms(t0),
                    extra={"hits": state.retrieval_hits, "fallback": fallback_used},
                ))
            except Exception as e:  # 工具执行异常不阻断回答
                tool_failed = True
                logger.warning("chat: 工具执行失败 %s: %s", calls, e)
                steps.append(MonitorStep(stage="tool", status="error",
                                         detail=f"工具执行失败: {e}", ms=_ms(t0)))
                try:
                    conn.rollback()  # 工具 SQL 失败后恢复事务，保证记忆写入/后续可用
                except Exception:
                    pass
        else:
            steps.append(MonitorStep(stage="tool", status="skipped", detail="无工具调用"))

        # ⑤ 记忆回写
        if session_id:
            t0 = time.time()
            save_ok = self._save_memory(conn, session_id, query, state.enhanced_query, intent, tool_results, tool_failed)
            steps.append(MonitorStep(stage="save", status="ok" if save_ok else "error",
                                     detail="记忆回写" + ("" if save_ok else "失败（不影响对话）"), ms=_ms(t0)))
        else:
            steps.append(MonitorStep(stage="save", status="skipped", detail="未启用记忆"))

        # ⑥⑦ 二次模型回调（非流式，传增强后问题）
        t0 = time.time()
        try:
            messages = self._build_reply_messages(state.enhanced_query, history, tool_results)
            reply = self.llm.chat(messages)
        except Exception as e:  # LLM/网络异常统一兜底，保证接口不 500
            logger.error("chat: 二次回调失败: %s", e)
            reply = FALLBACK_REPLY
            llm_ok = False
        steps.append(MonitorStep(stage="reply", status="ok" if llm_ok else "error",
                                 detail=f"二次模型回调 {'成功' if llm_ok else '失败，返回兜底话术'}",
                                 ms=_ms(t0)))
        prompt_tokens, completion_tokens = _accumulate_tokens(
            prompt_tokens, completion_tokens, getattr(self.llm, "get_last_usage", lambda: {})())

        # 监控记录（V2.2.2：请求级时间线；tool 记录"实际执行"的主工具，而非意图阶段选中的工具）
        monitor_store.record(MonitorRequest(
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id or "",
            query=state.user_query,
            enhanced_query=state.enhanced_query,
            intent_tag=state.intent_tag,
            intent_tool=state.intent_tool,
            tool=state.tools_used[0] if state.tools_used else state.intent_tool,
            tools_used=state.tools_used,
            tool_inputs=[{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls],
            tool_results_summary=_summarize_tool_results(tool_results),
            reply=reply[:2000],
            hits=state.retrieval_hits,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            degraded=state.degraded,
            fallback=fallback_used,
            context_reset=state.context_reset,
            llm_ok=llm_ok,
            total_ms=_ms(t_total),
            steps=steps,
        ))

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
        steps: List[MonitorStep] = []
        fallback_used = False
        llm_ok = True
        prompt_tokens = 0
        completion_tokens = 0
        t_total = time.time()

        # ①② 记忆读取 + 问题增强
        if session_id:
            t0 = time.time()
            memory = self._load_memory(conn, session_id, query, history)
            state.enhanced_query = memory["enhanced_query"]
            state.context_reset = memory["context_reset"]
            steps.append(MonitorStep(
                stage="memory",
                status="ok" if not state.context_reset else "degraded",
                detail=f"读会话上下文 + 问题增强（context_reset={state.context_reset}）",
                ms=_ms(t0),
            ))
        else:
            state.enhanced_query = query

        # ③ 意图识别
        t0 = time.time()
        intent = detect_intent(self.llm, state.enhanced_query, history)
        state.intent_tag = intent.intent
        state.intent_tool = intent.first_tool_name
        intent_extra: Dict[str, Any] = {}
        if intent.error:
            intent_extra["error"] = intent.error
        if intent.raw_response:
            intent_extra["raw"] = intent.raw_response[:200]
        steps.append(MonitorStep(
            stage="intent",
            status="error" if intent.error else "ok",
            detail=f"预分类={state.intent_tag or '无'} 工具={state.intent_tool or '无'}",
            ms=_ms(t0),
            extra=intent_extra,
        ))
        prompt_tokens, completion_tokens = _accumulate_tokens(
            prompt_tokens, completion_tokens, getattr(self.llm, "get_last_usage", lambda: {})())

        # ④ 工具调用：预分类直驱 → LLM tool_calls → 决策兜底（P1-5 不依赖 LLM 选工具）
        calls = _resolve_tool_calls(intent, state.enhanced_query or query)
        if not intent.has_tool_call:
            fallback_used = True
            if calls:
                logger.info("chat_stream: 模型未调用工具，按预分类/兜底执行 %s（query=%r）", calls[0].get("name"), query)

        tool_results: List[Dict[str, Any]] = []
        tool_failed = False
        if calls:
            t0 = time.time()
            try:
                tool_results = execute_tool_calls(conn, calls)
                state.tool_results = tool_results
                state.tools_used = [tr["name"] for tr in tool_results]
                state.retrieval_hits = sum(len(tr.get("raw") or []) for tr in tool_results)
                state.degraded = any("相关商品" in (tr.get("result") or "") for tr in tool_results)
                steps.append(MonitorStep(
                    stage="tool", status="degraded" if state.degraded else "ok",
                    detail=f"执行 {state.tools_used} 命中{state.retrieval_hits}"
                           f"{'（决策兜底）' if fallback_used else ''}"
                           f"{'（降级）' if state.degraded else ''}",
                    ms=_ms(t0),
                    extra={"hits": state.retrieval_hits, "fallback": fallback_used},
                ))
            except Exception as e:
                tool_failed = True
                logger.warning("chat_stream: 工具执行失败 %s: %s", calls, e)
                steps.append(MonitorStep(stage="tool", status="error",
                                         detail=f"工具执行失败: {e}", ms=_ms(t0)))
                try:
                    conn.rollback()  # 恢复事务，保证记忆写入/后续可用
                except Exception:
                    pass
        else:
            steps.append(MonitorStep(stage="tool", status="skipped", detail="无工具调用"))

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
            t0 = time.time()
            save_ok = self._save_memory(conn, session_id, query, state.enhanced_query, intent, tool_results, tool_failed)
            steps.append(MonitorStep(stage="save", status="ok" if save_ok else "error",
                                     detail="记忆回写" + ("" if save_ok else "失败（不影响对话）"), ms=_ms(t0)))
        else:
            steps.append(MonitorStep(stage="save", status="skipped", detail="未启用记忆"))

        # ⑥⑦ 二次模型回调（流式，传增强后问题）
        messages = self._build_reply_messages(state.enhanced_query, history, tool_results)
        t0 = time.time()
        reply_parts: List[str] = []
        try:
            for delta in self.llm.chat_stream(messages):
                if delta:
                    reply_parts.append(delta)
                    yield {"type": "token", "content": delta}
        except Exception as e:  # LLM/网络异常统一兜底，保证 SSE 流不中断
            logger.error("chat_stream: 二次回调失败: %s", e)
            llm_ok = False
            reply_parts.append(FALLBACK_REPLY)
            yield {"type": "token", "content": FALLBACK_REPLY}
        reply = "".join(reply_parts)
        steps.append(MonitorStep(stage="reply", status="ok" if llm_ok else "error",
                                 detail=f"二次模型回调 {'成功' if llm_ok else '失败，返回兜底话术'}",
                                 ms=_ms(t0)))
        prompt_tokens, completion_tokens = _accumulate_tokens(
            prompt_tokens, completion_tokens, getattr(self.llm, "get_last_usage", lambda: {})())

        # 监控记录（V2.2.2；tool 记录"实际执行"的主工具，而非意图阶段选中的工具）
        monitor_store.record(MonitorRequest(
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id or "",
            query=state.user_query,
            enhanced_query=state.enhanced_query,
            intent_tag=state.intent_tag,
            intent_tool=state.intent_tool,
            tool=state.tools_used[0] if state.tools_used else state.intent_tool,
            tools_used=state.tools_used,
            tool_inputs=[{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls],
            tool_results_summary=_summarize_tool_results(tool_results),
            reply=reply[:2000],
            hits=state.retrieval_hits,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            degraded=state.degraded,
            fallback=fallback_used,
            context_reset=state.context_reset,
            llm_ok=llm_ok,
            total_ms=_ms(t_total),
            steps=steps,
        ))

        # trace：请求级可观测日志
        logger.info(
            "chat_stream trace: query=%r intent_tag=%s tool=%s tools=%s hits=%d degraded=%s ctx_reset=%s %.2fs",
            state.user_query, state.intent_tag, state.intent_tool, state.tools_used,
            state.retrieval_hits, state.degraded, state.context_reset,
            time.time() - state.started_at,
        )

        yield {"type": "done"}
