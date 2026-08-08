"""
意图识别（意图预分类 + LLM function calling）

先由规则预分类器 _classify_intent 输出意图标签（FAQ / RECOMMEND / PRICE / INVENTORY），
命中时把 function calling 的工具集限制为该意图对应的原子工具，避免 LLM 在全部工具间误选；
未命中则走全量工具。无工具调用 → 视为普通对话，直接由客服回答。

返回结构：IntentResult.tool_calls = [{"id", "name", "arguments": dict}, ...]
"""
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from src.common.exceptions import LLMError
from src.online.core.agent.tools import TOOLS, TOOL_NAMES
from src.online.core.llm.client import LLMClient
from src.online.core.llm.prompt_templates import INTENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """意图识别结果。"""

    query: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # 意图识别阶段是否发生了 LLM 错误（由上层决定兜底策略）
    error: Optional[str] = None
    # 预分类意图标签（规则分类器输出，供 trace / 链路观测）
    intent: Optional[str] = None
    # LLM function calling 的原始响应摘要（供监控排查：模型为何没调用工具）
    raw_response: str = ""

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def first_tool_name(self) -> Optional[str]:
        return self.tool_calls[0]["name"] if self.tool_calls else None


# 规则意图预分类（命中强信号词时限制工具集，避免 LLM 误调用）
_INTENT_RULES = [
    # 售后 / 使用说明类（最优先：'手机进水怎么办' 必须走知识库而非商品检索）
    ("FAQ", ("怎么办", "坏了", "进水", "售后", "保修", "退换", "退货", "退款", "政策",
             "使用说明", "怎么用", "怎么修", "故障", "维修", "订单", "发票", "运费", "投诉")),
    # 推荐类
    ("RECOMMEND", ("推荐", "适合", "送礼", "送人", "送男", "送女", "送老", "性价比高", "挑几款")),
    # 价格类
    ("PRICE", ("多少钱", "价格", "售价", "贵不贵", "价位", "便宜吗")),
    # 库存 / 物流类
    ("INVENTORY", ("有货", "库存", "现货", "几天", "发货", "仓库", "物流", "配送", "到货")),
]

# 预分类 → 可用工具子集（命中时只给这些工具，选择准确率更高）
_TOOLS_BY_INTENT = {
    "FAQ": [t for t in TOOLS if t["function"]["name"] == "get_knowledge_base"],
    "RECOMMEND": [t for t in TOOLS if t["function"]["name"] == "product_recommendation"],
    "PRICE": [t for t in TOOLS if t["function"]["name"] == "get_product_price"],
    "INVENTORY": [t for t in TOOLS if t["function"]["name"] == "get_product_inventory"],
}


def _classify_intent(query: str) -> Optional[str]:
    """规则意图预分类：命中强信号词返回意图标签，否则 None（不限制工具集）。"""
    for intent, words in _INTENT_RULES:
        if any(w in query for w in words):
            return intent
    return None


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
    # 商品推荐：无论模型是否给了筛选条件，都保证有可检索的商品关键词（用户原话兜底）
    if tool_name == "product_recommendation":
        args.setdefault("product_name", query)
        return args
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
    if tool_name in ("get_product_price", "get_product_inventory", "product_recommendation"):
        # 结构化检索工具：注入用户原话作为商品关键词
        args["product_name"] = query
    elif tool_name in ("get_knowledge_base",):
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
    # 规则预分类：命中强信号词时限制工具集（intent 与 tool selection 分离）
    intent_tag = _classify_intent(query)
    tools = _TOOLS_BY_INTENT.get(intent_tag, TOOLS)
    if tools is not TOOLS:
        logger.info("intent: 预分类 %s → 工具集限制为 %s 个", intent_tag, len(tools))
    try:
        tool_calls = llm.function_call(messages, tools)
    except LLMError as e:
        return IntentResult(query=query, error=str(e), intent=intent_tag)

    # 原始响应摘要（LLMClient 记录；桩 LLM 无此方法时忽略）
    raw_response = ""
    try:
        raw_response = getattr(llm, "get_last_function_call_raw", lambda: "")()
    except Exception:
        pass

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
    return IntentResult(query=query, tool_calls=valid, intent=intent_tag, raw_response=raw_response)
