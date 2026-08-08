"""
通用工具函数
"""
import json
from typing import Any, Dict, Optional


def like_pattern(keyword: str) -> str:
    """
    把用户关键词转成 ILIKE 模糊匹配模式（转义 % _ 通配符，防 LIKE 注入）。

    供商品/库存等结构化检索共用（product_repo / inventory_repo）。
    """
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def like_prefix(keyword: str) -> str:
    """关键词前缀匹配模式（'关键词%'），用于相关性排序：名称以关键词开头的商品优先。"""
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def relevance_order(product_name: Optional[str]) -> tuple:
    """
    生成相关性排序 SQL 片段与绑定参数（供 product_repo / inventory_repo 共用）。

    名称关键词存在时：名称精确命中 > 名称前缀命中 > 名称包含 > 兜底按 id 排序，
    避免降级检索把不相关商品（如手机配件）排在真正命中的商品前面；
    类目/价格等过滤（无名称关键词）时退化为 id 排序。

    Returns: (order_fragment, sort_params)
    """
    if not product_name:
        return "p.id", []
    return (
        """
        CASE WHEN p.product_name = %s THEN 0
             WHEN p.product_name LIKE %s ESCAPE '\\' THEN 1
             WHEN p.product_name ILIKE %s ESCAPE '\\' THEN 2
             ELSE 3 END, p.id
        """,
        [product_name, like_prefix(product_name), like_pattern(product_name)],
    )


def sse_format(event: Dict[str, Any]) -> str:
    """把事件 dict 序列化为一条 SSE 消息（data: ... 空行）。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
