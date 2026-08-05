# -*- coding: utf-8 -*-
"""
core/agent/tools.py —— function-calling 工具 schema
=====================================================
对应《开发文档》阶段三任务 1（意图识别 Tools 设计）与「一、核心路由策略」：
  - query_inventory / query_price → 结构化数据库检索（SQL）
  - query_knowledge / product_recommendation → RAG 向量检索

工具清单：
  1. get_product_inventory  查商品库存与物流时效（SQL：product_catalog + inventory_logistics）
  2. get_product_price      查商品售价（SQL：product_catalog）
  3. get_knowledge_base     查知识库（RAG：售后政策/使用说明/FAQ）
  4. product_recommendation 商品推荐（RAG：商品说明书/评价语料）

每个工具的参数均为可选字段：模型能填多少填多少，
例如按 SKU 精确查，或按商品名称模糊查。
"""
import re
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# 工具定义（OpenAI function-calling 格式）
# ------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_product_inventory",
            "description": "查询商品库存数量与物流时效（预计送达天数、仓库位置）。"
                           "用户询问'有货吗''库存''几天能到''哪个仓'时使用。"
                           "可按 SKU 精确查询、按商品名称模糊查询，"
                           "也可按大类/小类/类目路径分级过滤，或用价格区间、有货状态进一步筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku_code": {
                        "type": "string",
                        "description": "商品 SKU 编码（如 21873056212），精确匹配",
                    },
                    "product_name": {
                        "type": "string",
                        "description": "商品名称关键词，模糊匹配（如：连衣裙、手机壳）",
                    },
                    "category_big": {
                        "type": "string",
                        "description": "大类过滤（如：服装鞋包）",
                    },
                    "category_small": {
                        "type": "string",
                        "description": "小类过滤（如：衬衫）",
                    },
                    "category_path": {
                        "type": "string",
                        "description": "类目完整路径过滤（如：服装鞋包/女装/衬衫）",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "最低价格过滤（元）",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格过滤（元）",
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "是否仅返回有库存商品",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "查询商品售价。用户询问'多少钱''价格''售价'时使用。"
                           "可按 SKU 精确查询、按商品名称模糊查询，"
                           "也可按大类/小类/类目路径分级过滤，或用价格区间、有货状态进一步筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku_code": {
                        "type": "string",
                        "description": "商品 SKU 编码，精确匹配",
                    },
                    "product_name": {
                        "type": "string",
                        "description": "商品名称关键词，模糊匹配",
                    },
                    "category_big": {
                        "type": "string",
                        "description": "大类过滤（如：服装鞋包）",
                    },
                    "category_small": {
                        "type": "string",
                        "description": "小类过滤（如：衬衫）",
                    },
                    "category_path": {
                        "type": "string",
                        "description": "类目完整路径过滤（如：服装鞋包/女装/衬衫）",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "最低价格过滤（元）",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格过滤（元）",
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "是否仅返回有库存商品",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_base",
            "description": "检索知识库（售后政策、使用说明、常见问题 FAQ）。"
                           "用户询问'退货流程''怎么洗''保修''售后政策'等非结构化知识时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索问题原文（尽量保留用户原话）",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["policy", "faq", "product_manual"],
                        "description": "限定知识类型，不明确时省略",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "product_recommendation",
            "description": "根据用户需求推荐商品。用户请求'推荐''适合我''有什么好的''送人'时使用。"
                           "基于商品说明书知识库做语义检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "推荐需求描述（如：适合200斤男士穿的显瘦外套）",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# 工具名 → 便捷索引
TOOL_NAMES: List[str] = [t["function"]["name"] for t in TOOLS]

# 工具名 → 该工具是否为结构化数据库检索（走 SQL）
SQL_TOOLS = {"get_product_inventory", "get_product_price"}

# 工具名 → 该工具是否为 RAG 知识库检索（走向量库）
RAG_TOOLS = {"get_knowledge_base", "product_recommendation"}

# ------------------------------------------------------------
# 工具名容错映射
# ------------------------------------------------------------
# 实测 DeepSeek deepseek-chat 可能"发明"工具名（如 price_inventory）或以
# 不同写法返回（query_price / knowledge 等），这里做别名归一化。
TOOL_NAME_ALIASES: Dict[str, str] = {
    # 价格
    "get_product_price": "get_product_price",
    "price": "get_product_price",
    "pricing": "get_product_price",
    "query_price": "get_product_price",
    "product_price": "get_product_price",
    # 库存/物流
    "get_product_inventory": "get_product_inventory",
    "inventory": "get_product_inventory",
    "stock": "get_product_inventory",
    "query_inventory": "get_product_inventory",
    "price_inventory": "get_product_inventory",
    "logistics": "get_product_inventory",
    # 知识库
    "get_knowledge_base": "get_knowledge_base",
    "knowledge": "get_knowledge_base",
    "kb": "get_knowledge_base",
    "query_knowledge": "get_knowledge_base",
    "faq": "get_knowledge_base",
    "policy": "get_knowledge_base",
    # 推荐
    "product_recommendation": "product_recommendation",
    "recommendation": "product_recommendation",
    "recommend": "product_recommendation",
    "product_recommend": "product_recommendation",
}

# 文本启发式：模型在 content 里以自然语言提到工具时，用正则提取工具名关键词
_TOOL_NAME_RE = (
    r"get_product_(?:price|inventory|knowledge_base)|product_recommendation|"
    r"price_inventory|query_(?:price|inventory|knowledge)|"
    r"\b(?:price|pricing|inventory|stock|knowledge|kb|faq|policy|recommend(?:ation)?)\b"
)


def normalize_tool_name(name: str) -> Optional[str]:
    """
    把模型返回的工具名归一化为 schema 中的正式工具名。

    匹配顺序：精确 → 别名 → 子串包含。无法识别返回 None。
    """
    name = (name or "").strip()
    if not name:
        return None
    if name in TOOL_NAMES:
        return name
    if name in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[name]
    lower = name.lower()
    for alias, canonical in TOOL_NAME_ALIASES.items():
        if alias in lower or lower in alias:
            return canonical
    return None
