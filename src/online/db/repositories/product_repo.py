# -*- coding: utf-8 -*-
"""
db/repositories/product_repo.py —— 商品数据访问层（product_catalog）
====================================================================
阶段三结构化检索（查价格）用：按 SKU 精确 / 商品名称模糊查询商品。

安全：所有条件参数化绑定（%s），LIKE 模式经参数传入，无字符串拼接。
"""
from typing import Any, Dict, List, Optional

import psycopg2.extras

_MAX_RESULTS = 5


def _like_pattern(keyword: str) -> str:
    """把用户关键词转成 ILIKE 模式（转义 % _ 通配符，防 LIKE 注入）。"""
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_products(
    conn,
    sku_code: Optional[str] = None,
    product_name: Optional[str] = None,
    limit: int = _MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """
    查询商品基础信息（id / sku_code / product_name / category_id / price）。

    Args:
        sku_code: SKU 精确匹配（优先级最高）
        product_name: 名称关键词模糊匹配（sku 未命中或未提供时使用）

    Returns:
        [{id, sku_code, product_name, category_id, price}, ...]
    """
    sql = """
        SELECT id, sku_code, product_name, category_id, price
        FROM product_catalog
        WHERE (
              (%s::text IS NOT NULL AND sku_code = %s)
           OR (%s::text IS NOT NULL AND product_name ILIKE %s ESCAPE '\\')
        )
        ORDER BY id
        LIMIT %s
    """
    name_pat = _like_pattern(product_name) if product_name else None
    params = (sku_code, sku_code, name_pat, name_pat, limit)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "sku_code": r["sku_code"],
            "product_name": r["product_name"],
            "category_id": r["category_id"],
            "price": float(r["price"]) if r["price"] is not None else None,
        }
        for r in rows
    ]


def format_products(products: List[Dict[str, Any]]) -> str:
    """把商品列表格式化为给 LLM 的简洁文本。"""
    if not products:
        return "未找到匹配的商品。"
    lines = []
    for p in products:
        price = f"¥{p['price']:.2f}" if p["price"] is not None else "价格未公布"
        lines.append(
            f"- SKU {p['sku_code']}：{p['product_name']}，{price}"
            f"（category_id={p['category_id']}）"
        )
    return "\n".join(lines)
