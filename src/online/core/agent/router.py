# -*- coding: utf-8 -*-
"""
core/agent/router.py —— 服务路由（工具调用分发）
=================================================
对应《开发文档》阶段三任务 2：
  (1) get_product_inventory / get_product_price → SQL 检索 → 返回严格参数
  (2) get_knowledge_base / product_recommendation → 调用阶段二 RAG 引擎

V2.2.1 架构优化（检索参数归一化 + 统一降级链）：
  - LLM 输出的 arguments 先经 params.normalize_arguments 归一化
    （清洗意图词 / 文本价格解析），永不直通 SQL；
  - 结构化检索走统一多级降级链 _structured_search：
      L1 归一化名称精确 → L2 品牌 + 类目组合 → L3 类目检索 → L4 型号/核心词 → 空
    覆盖「整句塞 product_name」「只给类目不给关键词」等模型输出缺陷。

分发规则：
  - 结构化工具：product_repo / inventory_repo（精确数值，参数化 SQL）
  - RAG 工具：kb_repo（混合检索 + Rerank）
  - 未知工具：返回错误信息，不影响主流程
"""
import logging
from typing import Any, Dict, List, Optional

import jieba

from src.common.constants import ALLOWED_DOC_TYPES
from src.online.core.agent import params
from src.online.db.repositories import inventory_repo, kb_repo, product_repo
from src.online.core.agent.tools import RAG_TOOLS, SQL_TOOLS, TOOL_NAMES
from src.offline.etl.category_classifier import CATEGORY_TREE

logger = logging.getLogger(__name__)

# 类目词表（大类/中类/小类名，按长度降序），用于从整句中识别核心品类词
_CATEGORY_WORDS: List[str] = sorted(
    {w for big, middles in CATEGORY_TREE.items() for w in [big, *middles.keys(), *sum(middles.values(), [])]},
    key=len,
    reverse=True,
)
_CATEGORY_WORDS_SET = frozenset(_CATEGORY_WORDS)

# 类目词 → 过滤层级映射（大类词→category_big，中类词→category_path，小类词→category_small）
_CATEGORY_LEVEL: Dict[str, tuple] = {}
for _big, _middles in CATEGORY_TREE.items():
    _CATEGORY_LEVEL.setdefault(_big, ("category_big", _big))
    for _mid, _smalls in _middles.items():
        _CATEGORY_LEVEL.setdefault(_mid, ("category_path", f"{_big}/{_mid}"))
        for _s in _smalls:
            _CATEGORY_LEVEL.setdefault(_s, ("category_small", _s))

# 推荐场景修饰词（RAG 类目映射时剔除）
_RECOMMEND_STOP = frozenset(
    "推荐 适合 送 几款 款 个 以内 的 吗 么 什么 有什么 想要 要 好 点 一些 件 条 台 只 "
    "帮我 给我 我想 我要 有没有 可以 怎么 这样 那种 类似".split()
)

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


def _arg(arguments: Dict[str, Any], key: str) -> Optional[str]:
    """从工具参数中安全取值：支持别名键；缺失/非字符串返回 None。"""
    if not isinstance(arguments, dict):
        return None
    for alias in _KEY_ALIASES.get(key, (key,)):
        val = arguments.get(alias)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _doc_type_arg(arguments: Dict[str, Any]) -> Optional[str]:
    """doc_type 参数白名单校验：非法值返回 None（不过滤）。"""
    dt = _arg(arguments, "doc_type")
    return dt if dt in ALLOWED_DOC_TYPES else None


def _core_keywords(text: str) -> List[str]:
    """从（可能是整句的）需求文本中提取核心商品词（供 RAG 类目映射）。

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
    # 2) jieba 分词兜底（保序去重，长度降序稳定排列）
    toks = list(dict.fromkeys(
        t.strip()
        for t in jieba.cut(text)
        if t.strip() and t not in _RECOMMEND_STOP and len(t.strip()) >= 2 and not t.strip().isdigit()
    ))
    return sorted(toks, key=len, reverse=True)


# ============================================================
# 统一结构化检索（多级降级链）
# ============================================================
def _structured_search(conn, tool_name: str, norm: Dict[str, Any]) -> tuple:
    """
    多级降级检索商品（库存联表 / 价格），覆盖模型参数缺陷：

      L0 SKU 精确（最高优先级）；
      L1 归一化商品名 ILIKE + 过滤条件；
      L2 仅品牌词；
      L3 类目（category_small / category_path / category_big，支持只给类目不给关键词）；
      L4 型号串；
      全空返回 ([], 0)。

    Returns:
        (rows, matched_level)：matched_level ∈ {1:精确(L0/L1), 2:品牌, 3:类目, 4:型号, 0:未命中}。
        降级命中的结果由调用方在文本中标注「近似匹配」，避免 LLM 误报为精确结果。

    所有 SQL 均为参数化查询（product_repo / inventory_repo 内部实现）。
    """
    repo = inventory_repo if tool_name == "get_product_inventory" else product_repo
    fn = repo.search_inventory if tool_name == "get_product_inventory" else repo.search_products

    def _call(**overrides) -> List[Dict[str, Any]]:
        return fn(
            conn,
            sku_code=norm.get("sku_code"),
            product_name=overrides.get("product_name"),
            category_big=overrides.get("category_big", norm.get("category_big")),
            category_small=overrides.get("category_small", norm.get("category_small")),
            category_path=overrides.get("category_path", norm.get("category_path")),
            min_price=overrides.get("min_price", norm.get("min_price")),
            max_price=overrides.get("max_price", norm.get("max_price")),
            in_stock_only=overrides.get("in_stock_only", norm.get("in_stock_only", False)),
        )

    name = norm.get("product_name")

    # L0：SKU 精确匹配（最高优先级，不依赖名称清洗）
    if norm.get("sku_code"):
        rows = _call(product_name=None)
        if rows:
            return rows, 1

    # L0.5：纯类目词优先走类目检索（P1-6）。
    # '手机'/'衬衫'/'连衣裙' 等泛化品类词直接 ILIKE 名称会被配件/周边淹没
    # （实测名称含'手机' 674 件几乎全是手机包/钱包，真正手机名不含'手机'），
    # 类目检索能精确落到该品类本体（'手机'→ 手机数码/手机 中类，排除手机配件）。
    if name and name in _CATEGORY_WORDS_SET:
        level_key, level_val = _CATEGORY_LEVEL[name]
        if level_key == "category_path":
            # 段边界：'手机数码/手机/' 不匹配 '手机数码/手机配件/...'（同级前缀混淆）
            level_val = f"{level_val}/"
        rows = _call(product_name=None, **{level_key: level_val})
        if rows:
            return rows, 3

    # L1：归一化名称 + 全部过滤条件
    if name:
        rows = _call(product_name=name)
        if rows:
            return rows, 1

    # L2：品牌词单独（保护品牌不因全称不匹配而丢失）
    brand = params.extract_brand(name or "")
    if brand and brand != name:
        rows = _call(product_name=brand)
        if rows:
            return rows, 2

    # L3：类目检索（名称为空 / 过泛时；category-only 场景的关键路径）
    for key in ("category_small", "category_path", "category_big"):
        if norm.get(key):
            rows = _call(product_name=None, **{key: norm[key]})
            if rows:
                return rows, 3
    # 名称含类目词时按层级过滤（大类→category_big，中类→category_path，小类→category_small）
    if name:
        for w in _CATEGORY_WORDS:
            if w in name:
                level_key, level_val = _CATEGORY_LEVEL[w]
                rows = _call(product_name=None, **{level_key: level_val})
                if rows:
                    return rows, 3
                # 该类目词 0 行时继续尝试名称中的其他类目词（如'智能手表 充电器'）
                continue

    # L4：型号串单独（'苹果iPhone 15' → 'iPhone 15'）
    model = params.extract_model(name or "")
    if model and model != name:
        rows = _call(product_name=model)
        if rows:
            return rows, 4

    return [], 0


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
    # 原始 query 兜底来源：intent 层已注入 product_name/query（用户原话），此处取其一
    raw_query = (
        str(arguments.get("query") or arguments.get("product_name") or "")
        if isinstance(arguments, dict)
        else ""
    )

    # ---- 结构化 SQL 检索（参数先归一化，LLM 原始值不直通 SQL）----
    if name in ("get_product_inventory", "get_product_price", "product_recommendation"):
        norm = params.normalize_arguments(name, arguments, raw_query)
        rows, level = _structured_search(conn, name, norm)
        if name == "get_product_inventory":
            result = inventory_repo.format_inventory(rows)
        else:
            result = product_repo.format_products(rows)
        if rows and level >= 2:
            # 降级命中（品牌/类目/型号近似）：积极引导 LLM 展示相关商品，
            # 避免"未找到完全匹配"等措辞让 LLM 误答"没查到"
            result = "为您找到以下相关商品（可能包含同系列/同品牌商品，按相关度排序）：\n" + result
        return {"name": name, "result": result, "raw": rows}

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
