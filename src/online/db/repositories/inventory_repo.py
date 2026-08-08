"""
 库存与物流数据访问层

product_catalog + inventory_logistics 联表，按 SKU 精确 / 名称模糊查询，
支持分级过滤（规则同 product_repo）：SKU 精确 → 名称模糊 → 类目 → 价格区间 → 仅看有库存。
无任何过滤条件时返回空列表（避免全表扫描）；所有条件参数化绑定。
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
    # 相关性排序：名称关键词存在时 精确 > 前缀 > 包含 > id
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
