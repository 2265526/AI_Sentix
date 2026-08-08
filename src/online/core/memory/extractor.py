"""
实体抽取与上下文回写

从「工具原始数据 → 工具参数 → 用户问题正则兜底」三级抽取实体快照，
回写 session_context（防呆：实体全空不覆盖旧快照），并规则拼装 last_answer_summary
（不调用 LLM，避免流式链路阻塞与额外计费）。
"""
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.online.core.agent import params
from src.online.core.agent.enricher import _find_brand
from src.online.db.repositories import memory_repo

# 正则（价格 / SKU 兜底提取）
# 价格数字：'3000以内' / '5千' / '1万' / '3000-5000' / '5000左右'
_PRICE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万|千)?\s*"
    r"(?:以内|以下|左右|到|至|~|－|-|—)?\s*"
    r"(\d+(?:\.\d+)?)?\s*(万|千)?"
)

# 工具返回文本中的 SKU：'SKU G054002：xxx'
_SKU_RE = re.compile(r"SKU\s+([A-Za-z0-9]+)")

# 工具返回文本中的价格：'¥6999.00' / '¥6999'
_PRICE_IN_RAW_RE = re.compile(r"¥\s*([\d.]+)")

# 快照中参与「防呆全空判断」的实体键
_ENTITY_KEYS = (
    "last_category_big", "last_category_small", "last_brand",
    "last_mentioned_sku", "last_price_range", "last_query_keyword",
    "last_answer_summary",
)


def _extract_price_from_query(query: str) -> Optional[List[float]]:
    """从用户问题正则兜底提取价格区间（如 '3000以内' → [None, 3000]、'3000-5000' → [3000,5000]）。"""
    nums: List[float] = []
    for m in _PRICE_RE.finditer(query):
        if not m.group(1):
            continue
        v = float(m.group(1)) * (10000 if m.group(2) == "万" else 1000 if m.group(2) == "千" else 1)
        nums.append(v)
        if m.group(3):
            v2 = float(m.group(3)) * (10000 if m.group(4) == "万" else 1000 if m.group(4) == "千" else 1)
            nums.append(v2)
    if not nums:
        return None
    return [min(nums), max(nums)]


def _build_summary(snapshot: Dict[str, Any]) -> Optional[str]:
    """规则拼装一句话摘要（不调 LLM，P1 阶段实现）。"""
    parts: List[str] = []
    if snapshot.get("last_brand"):
        parts.append(snapshot["last_brand"])
    if snapshot.get("last_category_big"):
        parts.append(snapshot["last_category_big"])
    if snapshot.get("last_category_small"):
        parts.append(snapshot["last_category_small"])
    price_range = snapshot.get("last_price_range")
    price_desc = ""
    if price_range and len(price_range) == 2 and price_range[0] is not None and price_range[1] is not None:
        price_desc = f"，价位 {int(price_range[0])}-{int(price_range[1])}"
    elif price_range and len(price_range) == 2 and price_range[1] is not None:
        price_desc = f"，价位 {int(price_range[1])} 以内"
    if parts or price_desc:
        return f"围绕 {'·'.join(parts)}{price_desc}"
    keyword = snapshot.get("last_query_keyword")
    if keyword:
        return f"查询了 {keyword} 相关信息"
    return None


def extract_entities(
    query: str,
    intent,
    tool_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    三级抽取实体快照（全量覆盖式，字段缺失置 None）。

    Args:
        query: 用户原始问题
        intent: IntentResult（含 tool_calls）
        tool_results: 工具执行结果 [{"name", "result", "raw"}...]

    Returns:
        完整快照 dict（含 confidence）
    """
    snapshot: Dict[str, Any] = {
        "last_category_big": None,
        "last_category_small": None,
        "last_brand": None,
        "last_mentioned_sku": None,
        "last_price_range": None,
        "last_tool": None,
        "last_query_keyword": None,
        "last_answer_summary": None,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": {},
    }
    conf: Dict[str, float] = snapshot["confidence"]

    # 工具名与参数（优先级②：结构化参数最可靠）
    tool_name: Optional[str] = None
    args: Dict[str, Any] = {}
    if intent is not None and getattr(intent, "tool_calls", None):
        first = intent.tool_calls[0]
        tool_name = first.get("name")
        args = first.get("arguments") or {}
    snapshot["last_tool"] = tool_name
    if tool_name:
        conf["tool"] = 1.0

    if args.get("category_big"):
        snapshot["last_category_big"] = str(args["category_big"])
        conf["category_big"] = 1.0
    if args.get("category_small"):
        snapshot["last_category_small"] = str(args["category_small"])
        conf["category_small"] = 1.0
    if args.get("sku_code"):
        snapshot["last_mentioned_sku"] = str(args["sku_code"])
        conf["sku"] = 1.0

    # 价格区间：优先工具参数（数值直接取；模型给的字符串走文本解析，min/max 分别解析防拼接错误）
    min_price = args.get("min_price")
    max_price = args.get("max_price")
    if min_price is not None or max_price is not None:
        lo = hi = None
        if isinstance(min_price, (int, float)):
            lo = float(min_price)
        elif isinstance(min_price, str):
            lo, hi_tmp = params.parse_price_text(min_price)
            if hi_tmp is not None:
                hi = hi_tmp
        if isinstance(max_price, (int, float)):
            hi = float(max_price)
        elif isinstance(max_price, str):
            _, hi_tmp = params.parse_price_text(max_price)
            if hi_tmp is not None:
                hi = hi_tmp
        if lo is not None or hi is not None:
            snapshot["last_price_range"] = [lo, hi]
            conf["price_range"] = 1.0

    # 优先级①：工具返回原始数据补充 SKU / 价格（缺失时才回填）
    for tr in tool_results or []:
        raw = tr.get("raw") or tr.get("result") or ""
        if not isinstance(raw, str):
            raw = str(raw)
        if not snapshot["last_mentioned_sku"]:
            m = _SKU_RE.search(raw)
            if m:
                snapshot["last_mentioned_sku"] = m.group(1)
                conf["sku"] = 1.0
        if not snapshot["last_price_range"]:
            prices = [float(p) for p in _PRICE_IN_RAW_RE.findall(raw)]
            if prices:
                snapshot["last_price_range"] = [min(prices), max(prices)]
                conf["price_range"] = 1.0

    # 优先级③：用户问题正则兜底（品牌词表 + 价格数字）
    brand = _find_brand(query)
    if brand and not snapshot["last_brand"]:
        snapshot["last_brand"] = brand
        conf["brand"] = 0.8
    if not snapshot["last_price_range"]:
        price_from_query = _extract_price_from_query(query)
        if price_from_query:
            snapshot["last_price_range"] = price_from_query
            conf["price_range"] = 0.7

    # 检索核心词（参数 product_name 优先，统一用 params 清洗——含意图词/疑问词的整句不会原样入库）
    keyword = params.clean_product_name(args.get("product_name"))
    if not keyword:
        # 兜底：品牌 / 品类 / 价格信息（纯寒暄不污染记忆）
        if snapshot["last_brand"]:
            keyword = snapshot["last_brand"]
        else:
            low = query.lower()
            has_category = any(w.lower() in low for w in params.CATEGORY_WORDS)
            if has_category or snapshot.get("last_price_range"):
                keyword = params.clean_product_name(query)
    if keyword:
        snapshot["last_query_keyword"] = keyword[:64]
        conf["keyword"] = 0.8 if keyword == snapshot["last_brand"] else 0.9

    # last_answer_summary：规则拼装，不调用 LLM
    snapshot["last_answer_summary"] = _build_summary(snapshot)
    if snapshot["last_answer_summary"]:
        conf["summary"] = 1.0

    return snapshot


def update_session_context(conn, session_id: str, query: str, intent, tool_results: List[Dict[str, Any]]) -> None:
    """
    抽取实体并回写 session_context（防呆：实体全空时保留旧快照，仅轮次+1/刷新 TTL）。
    """
    ctx = extract_entities(query, intent, tool_results)

    # 防呆：所有实体均空（如纯寒暄/无信息问题）→ 不覆盖旧快照
    if not any(ctx.get(k) not in (None, [], "") for k in _ENTITY_KEYS):
        old = memory_repo.get_session_context(conn, session_id)
        ctx = old["context"] if old else {}

    memory_repo.upsert_session_context(conn, session_id, ctx)
