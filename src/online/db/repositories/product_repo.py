# -*- coding: utf-8 -*-
"""
db/repositories/product_repo.py —— 商品数据访问层（product_catalog）
====================================================================
阶段三结构化检索（查价格）用：按 SKU 精确 / 商品名称模糊查询商品。

V1.2.5 起支持「分级过滤」检索（参考《数据库设计手册》商品信息表字段）：
    一级（最精确）  sku_code        精确匹配
    二级            product_name    名称关键词模糊匹配（ILIKE）
    三级            category_big    大类（中文）精确过滤
    四级            category_small  小类（中文）精确过滤
    五级            category_path   类目完整路径模糊匹配（ILIKE，含前缀）
    六级            min_price / max_price  价格区间过滤
    七级            in_stock_only   仅返回有库存（stock_quantity > 0）商品
规则：已提供的条件全部生效（AND 逐级收紧），未提供的条件自动跳过；
无任何过滤条件时返回空列表（避免全表扫描）。

安全：所有条件参数化绑定（%s），LIKE 模式经参数传入，无字符串拼接。
"""
from typing import Any, Dict, List, Optional

import psycopg2.extras

_MAX_RESULTS = 5


def _like_pattern(keyword: str) -> str:
    """把用户关键词转成 ILIKE 模式（转义 % _ 通配符，防 LIKE 注入）。"""
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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
    """
    按「分级过滤规则」动态拼装 WHERE 条件与参数（全部参数化绑定）。

    Returns: (where_clauses, params)
    """
    clauses: List[str] = []
    params: List[Any] = []

    if sku_code:
        clauses.append("p.sku_code = %s")
        params.append(sku_code)

    if product_name:
        clauses.append("p.product_name ILIKE %s ESCAPE '\\'")
        params.append(_like_pattern(product_name))

    if category_big:
        clauses.append("p.category_big = %s")
        params.append(category_big)

    if category_small:
        clauses.append("p.category_small = %s")
        params.append(category_small)

    if category_path:
        clauses.append("p.category_path ILIKE %s ESCAPE '\\'")
        params.append(_like_pattern(category_path))

    if min_price is not None:
        clauses.append("p.price >= %s")
        params.append(min_price)

    if max_price is not None:
        clauses.append("p.price <= %s")
        params.append(max_price)

    if in_stock_only:
        clauses.append("i.stock_quantity > 0")

    return clauses, params


def search_products(
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
    查询商品基础信息（id / sku_code / product_name / 类目 / price）。

    Args:
        sku_code: SKU 精确匹配（一级过滤，优先级最高）
        product_name: 名称关键词模糊匹配（二级过滤）
        category_big: 大类精确过滤（三级过滤，如：服装鞋包）
        category_small: 小类精确过滤（四级过滤，如：衬衫）
        category_path: 类目完整路径模糊过滤（五级过滤，如：服装鞋包/女装/衬衫）
        min_price / max_price: 价格区间过滤（六级过滤）
        in_stock_only: 仅返回有库存商品（七级过滤，联表 inventory_logistics）
        limit: 返回条数上限

    Returns:
        [{id, sku_code, product_name, category_big, category_small,
          category_path, price}, ...]
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
        # 无任何过滤条件：不返回数据，避免全表扫描
        return []

    sql = f"""
        SELECT p.id, p.sku_code, p.product_name,
               p.category_big, p.category_small, p.category_path, p.price
        FROM product_catalog p
        LEFT JOIN inventory_logistics i ON i.product_id = p.id
        WHERE {' AND '.join(clauses)}
        ORDER BY p.id
        LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params + [limit])
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "sku_code": r["sku_code"],
            "product_name": r["product_name"],
            "category_big": r["category_big"],
            "category_small": r["category_small"],
            "category_path": r["category_path"],
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
        category = p.get("category_path") or (
            "/".join(
                c for c in (p.get("category_big"), p.get("category_small")) if c
            )
        )
        lines.append(
            f"- SKU {p['sku_code']}：{p['product_name']}，{price}"
            + (f"（类目：{category}）" if category else "")
        )
    return "\n".join(lines)
