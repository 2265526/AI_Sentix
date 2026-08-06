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
from typing import Any, Dict, List, Optional

import jieba

from src.online.db.repositories import inventory_repo, kb_repo, product_repo
from src.online.core.agent.tools import RAG_TOOLS, SQL_TOOLS, TOOL_NAMES
from src.offline.etl.category_classifier import CATEGORY_TREE

logger = logging.getLogger(__name__)

# 推荐场景修饰词（模型可能把整句塞进 product_name，检索前剔除这些词提取核心实体）
_RECOMMEND_STOP = frozenset(
    "推荐 适合 送 几款 款 个 以内 的 吗 么 什么 有什么 想要 要 好 点 一些 件 条 台 只 个 "
    "帮我 给我 我想 我要 有没有 可以 怎么 这样 那种 类似".split()
)

# 类目词表（大类/中类/小类名，按长度降序），用于从整句中识别核心品类词
_CATEGORY_WORDS: List[str] = sorted(
    {w for big, middles in CATEGORY_TREE.items() for w in [big, *middles.keys(), *sum(middles.values(), [])]},
    key=len,
    reverse=True,
)
_CATEGORY_WORDS_SET = frozenset(_CATEGORY_WORDS)

# 常见口语商品词 → 标准类目小类名（用于识别类目词表外的高频词）
_CATEGORY_SYNONYMS = {
    "跑步鞋": "运动鞋", "跑鞋": "运动鞋", "球鞋": "运动鞋",
    "智能手机": "智能手机", "手机壳": "手机壳", "女装": "女装", "男装": "男装",
    "裙子": "连衣裙", "毛衣": "毛衣", "卫衣": "卫衣", "裤子": "裤装",
    "拖鞋": "拖鞋", "凉鞋": "凉鞋", "外套": "外套", "大衣": "大衣",
}

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
    "max_price": ("max_price", "price_max", "price_limit", "maxPrice", "最高价"),
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


def _core_keywords(text: str) -> List[str]:
    """从（可能是整句的）需求文本中提取核心商品词。

    策略：
      1. 优先匹配类目词表（大类/中类/小类名，最长优先）——如"适合送长辈的智能手表"→"智能手表"；
      2. 否则 jieba 分词 + 剔除修饰词，按长度降序（保持稳定顺序）。
    """
    if not text:
        return []
    # 1) 类目词表最长匹配
    for w in _CATEGORY_WORDS:
        if w in text:
            rest = [t for t in jieba.cut(text) if t.strip() and t != w]
            others = [
                t.strip()
                for t in rest
                if t.strip() not in _RECOMMEND_STOP
                and len(t.strip()) >= 2
                and not t.strip().isdigit()
            ]
            return [w, *dict.fromkeys(others)]
    # 1.5) 口语同义词 → 标准类目词（如"跑步鞋"→"运动鞋"）
    for slang, standard in _CATEGORY_SYNONYMS.items():
        if slang in text:
            return [standard]
    # 2) jieba 分词兜底（保序去重，长度降序稳定排列）
    toks = list(dict.fromkeys(
        t.strip()
        for t in jieba.cut(text)
        if t.strip() and t not in _RECOMMEND_STOP and len(t.strip()) >= 2 and not t.strip().isdigit()
    ))
    return sorted(toks, key=len, reverse=True)


def _brand_words(text: str) -> list:
    """从需求文本中提取非类目实体词（品牌/型号，如 iphone、小米、Mate）。

    与 _core_keywords 互补：_core_keywords 优先类目词（"iphone手机"→["手机"]），
    _brand_words 提取被类目词"掩盖"的品牌词，用于降级时组合过滤（保护品牌不丢失）。
    """
    if not text:
        return []
    toks = list(dict.fromkeys(
        t.strip()
        for t in jieba.cut(text)
        if t.strip() and t not in _RECOMMEND_STOP
        and len(t.strip()) >= 2 and not t.strip().isdigit()
        and t.strip() not in _CATEGORY_WORDS_SET
    ))
    return sorted(toks, key=len, reverse=True)


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
        query = _arg(arguments, "query") or ""
        # 意图层类目自动映射：query 含明确类目词时，作为类目过滤传给 RAG 检索
        # （如"手机有什么问题"→ 限定"手机"类目，避免召回手机壳等无关内容）
        cat_words = [w for w in _core_keywords(query) if w in _CATEGORY_WORDS_SET]
        cat_path = cat_words[0] if cat_words else None
        chunks = kb_repo.search_kb(
            conn,
            query=query,
            doc_type=_doc_type_arg(arguments),
            category_path=f"%{cat_path}%" if cat_path else None,
        )
        return {
            "name": name,
            "result": kb_repo.format_kb(chunks),
            "raw": chunks,
        }

    if name == "product_recommendation":
        # 结构化推荐：按商品关键词 + 类目/价格/库存分级过滤查商品表（精确命中规格信息）
        keyword = _arg(arguments, "product_name") or _arg(arguments, "query") or ""
        rows = product_repo.search_products(
            conn,
            sku_code=_arg(arguments, "sku_code"),
            product_name=keyword,
            category_big=_arg(arguments, "category_big"),
            category_small=_arg(arguments, "category_small"),
            category_path=_arg(arguments, "category_path"),
            min_price=_num_arg(arguments, "min_price"),
            max_price=_num_arg(arguments, "max_price"),
            in_stock_only=_bool_arg(arguments, "in_stock_only"),
        )
        # 降级兜底：模型把整句塞进关键词导致空结果时，提取核心词重试。
        # 关键：优先保护品牌/型号词（如 iphone），用「品牌词 + 类目词」组合过滤，
        # 避免类目词覆盖后把品牌丢弃（"iphone手机"被偷换成泛"手机"）。
        if not rows and keyword:
            _mp = _num_arg(arguments, "min_price")
            _xp = _num_arg(arguments, "max_price")
            _st = _bool_arg(arguments, "in_stock_only")
            cats = [w for w in _core_keywords(keyword) if w in _CATEGORY_WORDS_SET]
            brands = _brand_words(keyword)

            # 1) 品牌/型号词 + 类目过滤组合（最优先，如 iphone + 手机类目）
            if brands:
                for b in brands:
                    for c in cats:
                        rows = product_repo.search_products(
                            conn, product_name=b, category_path=c,
                            min_price=_mp, max_price=_xp, in_stock_only=_st,
                        )
                        if rows:
                            break
                    if rows:
                        break
            # 2) 类目词走类目过滤（小类精确/路径模糊）
            if not rows:
                for kw in cats:
                    rows = product_repo.search_products(
                        conn, category_small=kw,
                        min_price=_mp, max_price=_xp, in_stock_only=_st,
                    )
                    if not rows:
                        rows = product_repo.search_products(
                            conn, category_path=kw,
                            min_price=_mp, max_price=_xp, in_stock_only=_st,
                        )
                    if rows:
                        break
            # 3) 品牌/型号词单独名称匹配
            if not rows:
                for b in brands:
                    rows = product_repo.search_products(
                        conn, product_name=b,
                        min_price=_mp, max_price=_xp, in_stock_only=_st,
                    )
                    if rows:
                        break
            # 4) 其余核心词名称匹配
            if not rows:
                for kw in _core_keywords(keyword):
                    if kw in _CATEGORY_WORDS_SET:
                        continue
                    rows = product_repo.search_products(
                        conn, product_name=kw,
                        min_price=_mp, max_price=_xp, in_stock_only=_st,
                    )
                    if rows:
                        break
        return {
            "name": name,
            "result": product_repo.format_products(rows),
            "raw": rows,
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
