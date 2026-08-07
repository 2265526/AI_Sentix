# -*- coding: utf-8 -*-
"""
db/repositories/inventory_repo.py —— 库存与物流数据访问层
==========================================================
阶段三结构化检索（查库存/物流时效）用：
product_catalog + inventory_logistics 联表，按 SKU 精确 / 名称模糊查询。

V1.2.5 起支持「分级过滤」检索（与 product_repo 同规则，参考《数据库设计手册》商品信息表）：
    一级  sku_code        精确匹配
    二级  product_name    名称关键词模糊匹配（ILIKE）
    三级  category_big    大类（中文）精确过滤
    四级  category_small  小类（中文）精确过滤
    五级  category_path   类目完整路径模糊匹配（ILIKE，含前缀）
    六级  min_price / max_price  价格区间过滤
    七级  in_stock_only   仅返回有库存商品
规则：已提供的条件全部生效（AND 逐级收紧），未提供的条件自动跳过；
无任何过滤条件时返回空列表（避免全表扫描）。

安全：所有条件参数化绑定，LIKE 模式经参数传入。
"""
from typing import Any, Dict, List, Optional

import psycopg2.extras

from src.common.utils import like_pattern, relevance_order

_MAX_RESULTS = 5


def _filters(
    sku_code: Optional[str] = None,
    product_name: Optional[str] = None,
    category_big: Optional[str] = None,
    category_small: Optional[str] = None,
    category_path: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = False,
):
    """按「分级过滤规则」动态拼装 WHERE 条件与参数（全部参数化绑定）。"""
    clauses: List[str] = []
    params: List[Any] = []

    if sku_code:
        clauses.append("p.sku_code = %s")
        params.append(sku_code)

    if product_name:
        clauses.append("p.product_name ILIKE %s ESCAPE '\\'")
        params.append(like_pattern(product_name))

    if category_big:
        clauses.append("p.category_big = %s")
        params.append(category_big)

    if category_small:
        clauses.append("p.category_small = %s")
        params.append(category_small)

    if category_path:
        clauses.append("p.category_path ILIKE %s ESCAPE '\\'")
        params.append(like_pattern(category_path))

    if min_price is not None:
        clauses.append("p.price >= %s")
        params.append(min_price)

    if max_price is not None:
        clauses.append("p.price <= %s")
        params.append(max_price)

    if in_stock_only:
        clauses.append("i.stock_quantity > 0")

    return clauses, params


def search_inventory(
    conn,
    sku_code: Optional[str] = None,
    product_name: Optional[str] = None,
    category_big: Optional[str] = None,
    category_small: Optional[str] = None,
    category_path: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = False,
    limit: int = _MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """
    联表查询商品 + 库存 + 物流信息（分级过滤，规则同 product_repo.search_products）。

    Returns:
        [{product_id, sku_code, product_name, price, stock_quantity,
          warehouse_location, delivery_estimate_days}, ...]
    """
    clauses, params = _filters(
        sku_code=sku_code,
        product_name=product_name,
        category_big=category_big,
        category_small=category_small,
        category_path=category_path,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
    )
    if not clauses:
        return []
    # 相关性排序（P1-7）：名称关键词存在时 精确 > 前缀 > 包含 > id
    order_frag, sort_params = relevance_order(product_name)

    sql = f"""
        SELECT p.id          AS product_id,
               p.sku_code,
               p.product_name,
               p.price,
               i.stock_quantity,
               i.warehouse_location,
               i.delivery_estimate_days
        FROM product_catalog p
        LEFT JOIN inventory_logistics i ON i.product_id = p.id
        WHERE {' AND '.join(clauses)}
        ORDER BY {order_frag}
        LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params + sort_params + [limit])
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
