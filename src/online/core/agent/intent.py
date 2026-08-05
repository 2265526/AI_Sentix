# -*- coding: utf-8 -*-
"""
core/agent/intent.py —— 意图识别（LLM function calling）
=========================================================
对应《开发文档》阶段三任务 1：
  "在'意图识别'环节，大模型输出 tool_call"

流程：把用户问题 + 工具 schema 交给 LLM，模型返回要调用的工具及参数；
无工具调用 → 视为普通对话，直接由客服回答。

返回结构：
    IntentResult.tool_calls = [{"id", "name", "arguments": dict}, ...]
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.online.core.agent.tools import TOOLS, TOOL_NAMES
from src.online.core.llm.client import INTENT_SYSTEM_PROMPT, LLMClient, LLMError


@dataclass
class IntentResult:
    """意图识别结果。"""

    query: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # 意图识别阶段是否发生了 LLM 错误（由上层决定兜底策略）
    error: Optional[str] = None

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def first_tool_name(self) -> Optional[str]:
        return self.tool_calls[0]["name"] if self.tool_calls else None


def _fill_default_arguments(
    tool_name: str, query: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    工具参数缺失，或模型用了非标准键（如 category / product_type）导致
    检索主键缺失时，注入用户原话作为默认检索词（query / product_name），
    保证检索不落空。已有有效检索主键（query / product_name / sku_code /
    category 等）时不覆盖。
    """
    args = dict(arguments) if isinstance(arguments, dict) else {}
    # 检索主键：这些键任一存在即视为"模型已给出可检索参数"
    main_keys = {
        "query", "question", "text", "keyword",
        "product_name", "product", "name", "title",
        "sku_code", "sku",
        "category", "product_type", "type",
        "preferences", "style", "需求",
    }
    if (set(args) - {"_raw"}) & main_keys:
        return args
    if tool_name in ("get_product_price", "get_product_inventory"):
        args["product_name"] = query
    elif tool_name in ("get_knowledge_base", "product_recommendation"):
        args["query"] = query
    return args


def detect_intent(
    llm: LLMClient,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentResult:
    """
    意图识别：判断用户问题需要调用哪个工具。

    Args:
        llm: LLMClient 实例
        query: 用户问题
        history: 多轮历史（可选）

    Returns:
        IntentResult：tool_calls 为空表示无需工具，直接对话。
    """
    messages = LLMClient.build_messages(INTENT_SYSTEM_PROMPT, query, history)
    try:
        tool_calls = llm.function_call(messages, TOOLS)
    except LLMError as e:
        return IntentResult(query=query, error=str(e))

    # 容错：
    #  - 工具名不在 schema 中（或别名无法归一化）时忽略该调用；
    #  - arguments 非 dict（如未解析的 JSON 字符串）时保留为 {"_raw": ...}；
    #  - 参数为空时注入默认参数（用用户原话）。
    valid: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        if not isinstance(name, str) or name not in TOOL_NAMES:
            continue
        args = tc.get("arguments")
        if not isinstance(args, dict):
            args = {"_raw": args} if args is not None else {}
        args = _fill_default_arguments(name, query, args)
        tc = dict(tc)
        tc["name"] = name
        tc["arguments"] = args
        valid.append(tc)
    return IntentResult(query=query, tool_calls=valid)
