"""
请求级监控（内存环形缓冲）

把 chat_service 全链路（记忆读取 → 问题增强 → 意图识别 → 工具执行 → 记忆回写 → 二次回调）
的每个阶段记录为结构化时间线，供前端「监控」页面查询定位问题。

设计：
  - MonitorStore 内存环形缓冲（deque maxlen=200，线程安全），零部署零依赖；
  - record() 由 chat_service 在每轮请求结束时调用。
"""
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 环形缓冲容量：保留最近 N 条请求（超出自动丢弃最旧）
BUFFER_SIZE = 200


@dataclass
class MonitorStep:
    """单个阶段的时间线记录。"""

    stage: str                 # memory / enhance / intent / tool / save / reply
    status: str                # ok | degraded | error | skipped
    detail: str = ""           # 人类可读描述（如"降级到品牌层""LLM 调用失败"）
    ms: int = 0                # 阶段耗时（毫秒）
    extra: Dict[str, Any] = field(default_factory=dict)  # 参数摘要（工具名/召回数/错误信息）


@dataclass
class MonitorRequest:
    """一轮对话请求的完整监控记录。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = ""               # 完成时间 ISO
    session_id: str = ""
    query: str = ""            # 原始问题
    enhanced_query: str = ""   # 增强后问题
    intent_tag: Optional[str] = None  # 预分类意图标签
    intent_tool: Optional[str] = None # 意图识别阶段模型选择的工具（可能未实际执行）
    tool: Optional[str] = None        # 实际执行的主工具（兜底/容错后真正跑的）
    tools_used: List[str] = field(default_factory=list)  # 实际成功执行的工具
    tool_inputs: List[Dict[str, Any]] = field(default_factory=list)  # 实际执行的工具调用（含参数）
    tool_results_summary: List[Dict[str, Any]] = field(default_factory=list)  # 工具返回摘要
    reply: str = ""           # 最终回复（截断存储，避免撑爆环形缓冲）
    hits: int = 0              # 结构化检索召回条数
    prompt_tokens: int = 0     # 本轮请求 LLM 输入 token 合计（意图 + 二次回调）
    completion_tokens: int = 0 # 本轮请求 LLM 输出 token 合计
    degraded: bool = False     # 降级命中（品牌/类目/型号）
    fallback: bool = False     # 决策兜底触发（模型未调用工具）
    context_reset: bool = False
    llm_ok: bool = True        # 二次回调是否成功
    total_ms: int = 0
    steps: List[MonitorStep] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """本轮请求 LLM token 总消耗。"""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self, with_steps: bool = True) -> Dict[str, Any]:
        data = {
            "id": self.id, "ts": self.ts, "session_id": self.session_id,
            "query": self.query, "enhanced_query": self.enhanced_query,
            "intent_tag": self.intent_tag, "intent_tool": self.intent_tool,
            "tool": self.tool, "tools_used": self.tools_used,
            "tool_inputs": self.tool_inputs,
            "tool_results_summary": self.tool_results_summary,
            "reply": self.reply,
            "hits": self.hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "degraded": self.degraded, "fallback": self.fallback,
            "context_reset": self.context_reset, "llm_ok": self.llm_ok,
            "total_ms": self.total_ms,
        }
        if with_steps:
            data["steps"] = [s.__dict__ for s in self.steps]
        return data


class MonitorStore:
    """请求监控存储（内存环形缓冲，线程安全）。"""

    def __init__(self, size: int = BUFFER_SIZE):
        self._buf: deque = deque(maxlen=size)
        self._lock = threading.RLock()

    def record(self, req: MonitorRequest) -> None:
        with self._lock:
            self._buf.append(req)

    def summary(self) -> Dict[str, Any]:
        """概览统计：总量 / 错误 / 兜底 / 降级 / LLM 失败 / 平均耗时。"""
        with self._lock:
            items = list(self._buf)
        n = len(items)
        if n == 0:
            return {"total": 0, "errors": 0, "degraded": 0, "fallback": 0,
                    "llm_errors": 0, "tool_calls": 0, "avg_ms": 0}
        return {
            "total": n,
            "errors": sum(1 for r in items if not r.llm_ok or any(s.status == "error" for s in r.steps)),
            "degraded": sum(1 for r in items if r.degraded),
            "fallback": sum(1 for r in items if r.fallback),
            "llm_errors": sum(1 for r in items if not r.llm_ok),
            "tool_calls": sum(len(r.tools_used) for r in items),
            "avg_ms": round(sum(r.total_ms for r in items) / n, 1),
        }

    def recent(self, limit: int = 50, status: str = "all") -> List[Dict[str, Any]]:
        """最近请求列表（不携带 steps，轻量）。status: all | error | degraded。"""
        with self._lock:
            items = list(self._buf)[-limit:]
        return [r.to_dict(with_steps=False) for r in self._filter_status(items, status)]

    def get(self, req_id: str) -> Optional[Dict[str, Any]]:
        """单请求详情（含全链路 steps 时间线）。"""
        with self._lock:
            for r in self._buf:
                if r.id == req_id:
                    return r.to_dict(with_steps=True)
        return None

    def all_records(self, status: str = "all") -> List[MonitorRequest]:
        """导出用：返回全部（或按状态筛选的）监控记录对象（含 steps，按时间正序）。"""
        with self._lock:
            items = list(self._buf)
        return self._filter_status(items, status)

    @staticmethod
    def _filter_status(items: List[MonitorRequest], status: str) -> List[MonitorRequest]:
        """按状态筛选：all 全部 / error 出错 / degraded 降级或兜底。"""
        if status == "error":
            return [r for r in items if not r.llm_ok or any(s.status == "error" for s in r.steps)]
        if status == "degraded":
            return [r for r in items if r.degraded or r.fallback]
        return items


# 全局单例：全进程共享
monitor_store = MonitorStore()
