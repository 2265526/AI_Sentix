# -*- coding: utf-8 -*-
"""
api/routes/monitor.py —— 请求监控接口（V2.2.2）
====================================================
供前端「监控」页面查询请求级时间线，定位出错环节，并支持导出监控日志。
数据源：core/monitor.py 的内存环形缓冲（最近 200 条，重启清空）。
"""
import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.online.core.monitor.monitor import MonitorRequest, monitor_store

router = APIRouter(prefix="/v1/monitor", tags=["monitor"])

# 导出格式 → (媒体类型, 文件扩展名)
_FORMAT_META = {
    "csv": ("text/csv; charset=utf-8", "csv"),
    "json": ("application/json; charset=utf-8", "json"),
    "txt": ("text/plain; charset=utf-8", "txt"),
}


def _steps_summary(record: dict) -> str:
    """时间线步骤摘要：stage:status:detail 用分号连接。"""
    parts = []
    for s in record.get("steps", []):
        parts.append(f"{s['stage']}:{s['status']}:{s.get('detail', '')}")
    return "; ".join(parts)


def _tool_inputs_text(record: dict) -> str:
    """工具调用参数摘要：name({arguments}) 用分号连接。"""
    parts = []
    for t in record.get("tool_inputs", []):
        args = t.get("arguments") or {}
        parts.append(f"{t.get('name', '')}({json.dumps(args, ensure_ascii=False)})")
    return "; ".join(parts)


def _tool_results_text(record: dict) -> str:
    """工具返回摘要：name xN条 预览前 80 字。"""
    parts = []
    for t in record.get("tool_results_summary", []):
        preview = (t.get("preview") or "").replace("\n", " ")[:80]
        parts.append(f"{t.get('name', '')} x{t.get('hits', 0)}条 {preview}")
    return "; ".join(parts)


def _csv_export(records) -> str:
    """CSV：每条请求一行，含关键字段 + 步骤/工具摘要。utf-8-sig 保证 Excel 打开不乱码。"""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "ID", "时间", "会话ID", "原始问题", "增强后问题", "预分类意图", "意图工具",
        "实际工具", "实际工具列表", "工具参数", "工具返回摘要", "命中", "降级",
        "决策兜底", "会话过期", "LLM成功", "总耗时(ms)", "Prompt Tokens",
        "Completion Tokens", "Total Tokens", "步骤摘要", "回复(截断)",
    ])
    for r in records:
        d = r.to_dict(with_steps=True)
        writer.writerow([
            d["id"], d["ts"], d["session_id"], d["query"], d["enhanced_query"],
            d["intent_tag"] or "", d["intent_tool"] or "", d["tool"] or "",
            ", ".join(d["tools_used"]), _tool_inputs_text(d), _tool_results_text(d),
            d["hits"], "是" if d["degraded"] else "否", "是" if d["fallback"] else "否",
            "是" if d["context_reset"] else "否", "是" if d["llm_ok"] else "否",
            d["total_ms"], d.get("prompt_tokens", 0), d.get("completion_tokens", 0),
            d.get("total_tokens", 0), _steps_summary(d),
            (d.get("reply") or "")[:200].replace("\n", " "),
        ])
    return "\ufeff" + buf.getvalue()  # BOM：Excel 兼容中文


def _json_export(records) -> str:
    """JSON：完整结构化记录（含 steps / tool_inputs / reply）。"""
    data = [r.to_dict(with_steps=True) for r in records]
    return json.dumps(data, ensure_ascii=False, indent=2)


def _txt_export(records) -> str:
    """TXT：人类可读的监控日志（请求块 + 时间线）。"""
    lines = [
        "# 电商AI智能客服 监控日志",
        f"# 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# 记录数: {len(records)}",
        "=" * 72,
    ]
    for r in records:
        d = r.to_dict(with_steps=True)
        status = "正常" if d["llm_ok"] and not d["degraded"] and not d["fallback"] else (
            "降级" if d["degraded"] or d["fallback"] else "错误")
        lines += [
            f"[请求 {d['id']}] {d['ts']}  状态: {status}  耗时: {d['total_ms']}ms",
            f"  会话: {d['session_id'] or '（无）'} | 预分类意图: {d['intent_tag'] or '无'} | "
            f"意图工具: {d['intent_tool'] or '无'} | 实际工具: {d['tool'] or '无'} | 命中: {d['hits']}",
            f"  标记: {'降级 ' if d['degraded'] else ''}{'决策兜底 ' if d['fallback'] else ''}"
            f"{'会话过期 ' if d['context_reset'] else ''}{'LLM失败' if not d['llm_ok'] else 'LLM成功'}",
            f"  Token 消耗: prompt {d.get('prompt_tokens', 0)} / completion {d.get('completion_tokens', 0)} / "
            f"total {d.get('total_tokens', 0)}",
            f"  原始问题: {d['query']}",
        ]
        if d["enhanced_query"] != d["query"]:
            lines.append(f"  增强后问题: {d['enhanced_query']}")
        if d["tool_inputs"]:
            lines.append(f"  工具调用: {_tool_inputs_text(d)}")
        if d["tool_results_summary"]:
            lines.append(f"  工具返回: {_tool_results_text(d)}")
        if d.get("reply"):
            lines.append(f"  回复: {(d['reply'] or '').replace(chr(10), ' / ')[:300]}")
        lines.append("  ---- 时间线 ----")
        for s in d["steps"]:
            line = f"  [{s['stage']}] {s['status']} {s['ms']}ms  {s.get('detail', '')}"
            if s.get("extra", {}).get("error"):
                line += f"  ERROR: {s['extra']['error']}"
            lines.append(line)
        lines.append("=" * 72)
    return "\n".join(lines)


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


@router.get("/export", summary="导出监控日志（CSV/JSON/TXT）")
def monitor_export(
    status: str = Query("all", pattern="^(all|error|degraded)$",
                        description="筛选：all 全部 / error 出错 / degraded 降级或兜底"),
    format: str = Query("csv", pattern="^(csv|json|txt)$", description="导出格式"),
):
    """导出当前环形缓冲内的监控记录（最近 200 条）为附件文件。

    - csv:  每条请求一行，含关键字段与步骤/工具摘要（Excel 可直接打开）；
    - json: 完整结构化记录（含 steps 时间线 / 工具参数 / 回复）；
    - txt:  人类可读的日志风格，按请求分块展示全链路时间线。
    """
    records: list = monitor_store.all_records(status=status)
    media_type, ext = _FORMAT_META[format]
    if format == "csv":
        content = _csv_export(records)
    elif format == "json":
        content = _json_export(records)
    else:
        content = _txt_export(records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monitor_log_{ts}.{ext}"
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
