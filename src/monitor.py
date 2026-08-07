# -*- coding: utf-8 -*-
"""
api/routes/monitor.py —— 请求监控接口（V2.2.2）
====================================================
供前端「监控」页面查询请求级时间线，定位出错环节。
数据源：core/monitor.py 的内存环形缓冲（最近 200 条，重启清空）。
"""
from fastapi import APIRouter, HTTPException, Query

from src.online.core.monitor import monitor_store

router = APIRouter(prefix="/v1/monitor", tags=["monitor"])


@router.get("/summary", summary="监控概览统计")
def monitor_summary():
    """最近请求的总量 / 错误 / 兜底 / 降级 / LLM 失败 / 平均耗时。"""
    return monitor_store.summary()


@router.get("/requests", summary="最近请求列表（轻量，不含时间线）")
def monitor_requests(
    limit: int = Query(50, ge=1, le=200, description="返回条数"),
    status: str = Query("all", pattern="^(all|error|degraded)$",
                        description="筛选：all 全部 / error 出错 / degraded 降级或兜底"),
):
    return {"items": monitor_store.recent(limit=limit, status=status)}


@router.get("/requests/{request_id}", summary="单请求详情（全链路时间线）")
def monitor_request_detail(request_id: str):
    """返回该请求的完整监控记录（含 steps 时间线）；不存在返回 404。"""
    detail = monitor_store.get(request_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="请求记录不存在或已超出环形缓冲")
    return detail
