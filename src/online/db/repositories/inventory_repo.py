# -*- coding: utf-8 -*-
"""
db/repositories/inventory_repo.py —— 库存与物流数据访问层
==========================================================
阶段三结构化检索（查库存/物流时效）用：
product_catalog + inventory_logistics 联表，按 SKU 精确 / 名称模糊查询。

安全：所有条件参数化绑定，LIKE 模式经参数传入。
"""
from typing import Any, Dict, List, Optional

import psycopg2.extras

_MAX_RESULTS = 5


def _like_pattern(keyword: str) -> str:
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_inventory(
    conn,
    sku_code: Optional[str] = None,
    product_name: Optional[str] = None,
    limit: int = _MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """
    联表查询商品 + 库存 + 物流信息。

    Args:
        sku_code: SKU 精确匹配（优先级最高）
        product_name: 名称关键词模糊匹配

    Returns:
        [{product_id, sku_code, product_name, price, stock_quantity,
          warehouse_location, delivery_estimate_days}, ...]
    """
    sql = """
        SELECT p.id          AS product_id,
               p.sku_code,
               p.product_name,
               p.price,
               i.stock_quantity,
               i.warehouse_location,
               i.delivery_estimate_days
        FROM product_catalog p
        LEFT JOIN inventory_logistics i ON i.product_id = p.id
        WHERE (
              (%s::text IS NOT NULL AND p.sku_code = %s)
           OR (%s::text IS NOT NULL AND p.product_name ILIKE %s ESCAPE '\\')
        )
        ORDER BY p.id
        LIMIT %s
    """
    name_pat = _like_pattern(product_name) if product_name else None
    params = (sku_code, sku_code, name_pat, name_pat, limit)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "product_id": r["product_id"],
            "sku_code": r["sku_code"],
            "product_name": r["product_name"],
            "price": float(r["price"]) if r["price"] is not None else None,
            "stock_quantity": r["stock_quantity"],
            "warehouse_location": r["warehouse_location"],
            "delivery_estimate_days": r["delivery_estimate_days"],
        }
        for r in rows
    ]


def format_inventory(items: List[Dict[str, Any]]) -> str:
    """把库存/物流列表格式化为给 LLM 的简洁文本。"""
    if not items:
        return "未找到匹配的商品库存信息。"
    lines = []
    for it in items:
        stock = (
            f"库存 {it['stock_quantity']} 件"
            if it["stock_quantity"] is not None
            else "库存未知"
        )
        delivery = (
            f"预计 {it['delivery_estimate_days']} 天送达"
            if it["delivery_estimate_days"] is not None
            else "物流时效未知"
        )
        wh = it["warehouse_location"] or "仓库未知"
        price = f"¥{it['price']:.2f}" if it["price"] is not None else "价格未公布"
        lines.append(
            f"- SKU {it['sku_code']}：{it['product_name']}，{price}，"
            f"{stock}，{delivery}（{wh}）"
        )
    return "\n".join(lines)
