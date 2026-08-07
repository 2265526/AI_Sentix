# -*- coding: utf-8 -*-
"""
src/common/utils.py —— 通用工具函数
====================================
对应《目录结构》中 src/common/utils.py 的职责："辅助函数"。

放置被多个模块复用的纯函数；单一模块专用的逻辑保留在各模块内部。
"""
import json
from typing import Any, Dict


def like_pattern(keyword: str) -> str:
    """
    把用户关键词转成 ILIKE 模糊匹配模式（转义 % _ 通配符，防 LIKE 注入）。

    供商品/库存等结构化检索共用（product_repo / inventory_repo）。
    """
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def sse_format(event: Dict[str, Any]) -> str:
    """把事件 dict 序列化为一条 SSE 消息（data: ... 空行）。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
