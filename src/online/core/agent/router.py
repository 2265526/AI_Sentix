# -*- coding: utf-8 -*-
"""
core/agent/router.py —— 服务路由（工具调用分发）
=================================================
对应《开发文档》阶段三任务 2：
  (1) get_product_inventory / get_product_price → SQL 检索 → 返回严格参数
  (2) get_knowledge_base / product_recommendation → 调用阶段二 RAG 引擎

分发规则：
  - 结构化工具：product_repo / inventory_repo（精确数值，参数化 SQL）
  - RAG 工具：kb_repo（混合检索 + Rerank）
  - 未知工具：返回错误信息，不影响主流程
"""
import logging
from typing import Any, Dict, Optional

from src.online.db.repositories import inventory_repo, kb_repo, product_repo
from src.online.core.agent.tools import RAG_TOOLS, SQL_TOOLS, TOOL_NAMES

logger = logging.getLogger(__name__)

# 模型可能用别名键填参（实测：product / query / category / product_type 等），做键名归一化
_KEY_ALIASES: Dict[str, tuple] = {
    "sku_code": ("sku_code", "sku"),
    "product_name": ("product_name", "product", "name", "title", "goods"),
    "query": ("query", "question", "text", "keyword", "category", "product_type", "type", "需求"),
    "doc_type": ("doc_type", "type"),
    # V1.2.5 分级过滤参数（结构化商品检索）
    "category_big": ("category_big", "big_category", "大类"),
    "category_small": ("category_small", "small_category", "sub_category", "小类"),
    "category_path": ("category_path", "path", "category", "类目", "类目路径"),
    "min_price": ("min_price", "price_min", "minPrice", "最低价"),
    "max_price": ("max_price", "price_max", "maxPrice", "最高价"),
    "in_stock_only": ("in_stock_only", "inStock", "有货", "仅看有货"),
}

# doc_type 白名单（与 kb_documents.doc_type 数据一致）
_ALLOWED_DOC_TYPES = ("product_manual", "faq", "policy")


def _arg(arguments: Dict[str, Any], key: str) -> Optional[str]:
    """从工具参数中安全取值：支持别名键；缺失/非字符串返回 None。"""
    if not isinstance(arguments, dict):
        return None
    for alias in _KEY_ALIASES.get(key, (key,)):
        val = arguments.get(alias)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _num_arg(arguments: Dict[str, Any], key: str) -> Optional[float]:
    """数值型参数取值（价格区间等）：兼容数字与数字字符串；非法值返回 None。"""
    if not isinstance(arguments, dict):
        return None
    for alias in _KEY_ALIASES.get(key, (key,)):
        val = arguments.get(alias)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and val.strip():
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _bool_arg(arguments: Dict[str, Any], key: str) -> bool:
    """布尔型参数取值（有库存过滤等）：兼容 True/False 与常见真值字符串。"""
    if not isinstance(arguments, dict):
        return False
    for alias in _KEY_ALIASES.get(key, (key,)):
        val = arguments.get(alias)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val > 0
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("1", "true", "yes", "是", "有", "有货"):
                return True
            if v in ("0", "false", "no", "否", "无"):
                return False
    return False


def _doc_type_arg(arguments: Dict[str, Any]) -> Optional[str]:
    """doc_type 参数白名单校验：非法值返回 None（不过滤）。"""
    dt = _arg(arguments, "doc_type")
    return dt if dt in _ALLOWED_DOC_TYPES else None


def execute_tool(conn, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行一次工具调用，返回给 LLM 的格式化结果。

    Args:
        conn: 数据库连接（来自请求级依赖）
        tool_call: {"name": str, "arguments": dict}

    Returns:
        {"name": str, "result": str(给LLM的文本), "raw": dict(结构化数据,可选)}
    """
    name = tool_call.get("name", "")
    arguments = tool_call.get("arguments") or {}

    # ---- 结构化 SQL 检索 ----
    if name == "get_product_inventory":
        rows = inventory_repo.search_inventory(
            conn,
            sku_code=_arg(arguments, "sku_code"),
            product_name=_arg(arguments, "product_name"),
            category_big=_arg(arguments, "category_big"),
            category_small=_arg(arguments, "category_small"),
            category_path=_arg(arguments, "category_path"),
            min_price=_num_arg(arguments, "min_price"),
            max_price=_num_arg(arguments, "max_price"),
            in_stock_only=_bool_arg(arguments, "in_stock_only"),
        )
        return {
            "name": name,
            "result": inventory_repo.format_inventory(rows),
            "raw": rows,
        }

    if name == "get_product_price":
        rows = product_repo.search_products(
            conn,
            sku_code=_arg(arguments, "sku_code"),
            product_name=_arg(arguments, "product_name"),
            category_big=_arg(arguments, "category_big"),
            category_small=_arg(arguments, "category_small"),
            category_path=_arg(arguments, "category_path"),
            min_price=_num_arg(arguments, "min_price"),
            max_price=_num_arg(arguments, "max_price"),
            in_stock_only=_bool_arg(arguments, "in_stock_only"),
        )
        return {
            "name": name,
            "result": product_repo.format_products(rows),
            "raw": rows,
        }

    # ---- RAG 向量检索 ----
    if name == "get_knowledge_base":
        chunks = kb_repo.search_kb(
            conn,
            query=_arg(arguments, "query") or "",
            doc_type=_doc_type_arg(arguments),
        )
        return {
            "name": name,
            "result": kb_repo.format_kb(chunks),
            "raw": chunks,
        }

    if name == "product_recommendation":
        # 商品推荐基于商品说明书知识库（product_manual）
        chunks = kb_repo.search_kb(
            conn,
            query=_arg(arguments, "query") or "",
            doc_type="product_manual",
        )
        return {
            "name": name,
            "result": kb_repo.format_kb(chunks),
            "raw": chunks,
        }

    # ---- 未知工具 ----
    logger.warning("router: 未知工具名 %r（已知: %s）", name, TOOL_NAMES)
    return {
        "name": name,
        "result": f"未知的工具调用：{name}",
        "raw": None,
    }


def execute_tool_calls(
    conn, tool_calls: list
) -> list:
    """顺序执行多个工具调用（当前业务每次一个，循环保留扩展性）。"""
    return [execute_tool(conn, tc) for tc in tool_calls]
