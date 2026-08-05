# -*- coding: utf-8 -*-
"""阶段三：Agent（tools / intent / router / llm / chat_service）单元测试。
不依赖真实数据库与外部 LLM：LLMClient 与数据库连接均用 mock。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.online.core.agent.intent import IntentResult, detect_intent
from src.online.core.agent.router import execute_tool
from src.online.core.agent.tools import TOOLS, TOOL_NAMES, RAG_TOOLS, SQL_TOOLS
from src.online.core.llm.client import LLMClient, LLMError
from src.online.services.chat_service import ChatService


# ============================================================
# tools schema
# ============================================================
def test_tools_schema_complete():
    """4 个工具齐全，schema 字段完整、工具名唯一。"""
    assert len(TOOLS) == 4
    assert len(TOOL_NAMES) == len(set(TOOL_NAMES))
    for t in TOOLS:
        fn = t["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert isinstance(fn["parameters"]["properties"], dict)


def test_tools_route_classification():
    """结构化工具与 RAG 工具分类正确。"""
    assert SQL_TOOLS == {"get_product_inventory", "get_product_price"}
    assert RAG_TOOLS == {"get_knowledge_base", "product_recommendation"}
    assert not (SQL_TOOLS & RAG_TOOLS)


def test_tools_required_fields():
    """get_knowledge_base 的 query 必填；其余工具参数均可选。"""
    kb = next(t for t in TOOLS if t["function"]["name"] == "get_knowledge_base")
    assert "query" in kb["function"]["parameters"]["required"]
    inv = next(t for t in TOOLS if t["function"]["name"] == "get_product_inventory")
    assert inv["function"]["parameters"]["required"] == []


# ============================================================
# intent（LLM function calling 解析）
# ============================================================
class _FakeLLM:
    """可控的 LLM 桩：模拟 function_call 返回。"""

    def __init__(self, tool_calls=None, error=None):
        self._calls = tool_calls
        self._error = error

    def function_call(self, messages, tools, tool_choice="auto"):
        if self._error:
            raise LLMError(self._error)
        return self._calls

    def chat(self, messages, **kw):
        return "模拟回复"

    def chat_stream(self, messages, **kw):
        yield "模拟"
        yield "回复"


def test_detect_intent_with_tool_call():
    llm = _FakeLLM(
        tool_calls=[
            {
                "id": "call_1",
                "name": "get_product_price",
                "arguments": {"sku_code": "21873056212"},
            }
        ]
    )
    result = detect_intent(llm, "这个多少钱？")
    assert result.has_tool_call
    assert result.first_tool_name == "get_product_price"
    assert result.tool_calls[0]["arguments"]["sku_code"] == "21873056212"


def test_detect_intent_no_tool_call():
    result = detect_intent(_FakeLLM(tool_calls=[]), "你好")
    assert not result.has_tool_call
    assert result.tool_calls == []


def test_detect_intent_filters_unknown_tool():
    llm = _FakeLLM(tool_calls=[{"id": "x", "name": "not_a_tool", "arguments": {}}])
    result = detect_intent(llm, "测试")
    assert result.tool_calls == []  # 非法工具被过滤


def test_detect_intent_llm_error():
    result = detect_intent(_FakeLLM(error="timeout"), "测试")
    assert result.error
    assert not result.has_tool_call


def test_detect_intent_bad_arguments_json():
    """arguments 非法 JSON 时保留 _raw，不崩溃。"""
    llm = _FakeLLM(
        tool_calls=[{"id": "x", "name": "get_product_price", "arguments": "not-json"}]
    )
    result = detect_intent(llm, "价格")
    assert result.tool_calls[0]["arguments"].get("_raw") == "not-json"


# ============================================================
# LLMClient.content 工具调用解析（DeepSeek 不稳定格式兼容）
# ============================================================
def test_parse_content_call_name_arguments():
    calls = LLMClient._parse_content_calls(
        '{"name": "get_product_price", "arguments": {"sku_code": "123"}}'
    )
    assert calls[0]["name"] == "get_product_price"
    assert calls[0]["arguments"] == {"sku_code": "123"}


def test_parse_content_call_tool_params():
    """DeepSeek 实测格式：{"tool": ..., "params": ...}"""
    calls = LLMClient._parse_content_calls(
        '{"tool": "price_inventory", "params": {"product": "iPhone"}}'
    )
    # 发明名 price_inventory 归一化到 get_product_inventory；params 键归一化
    assert calls[0]["name"] == "get_product_inventory"
    assert calls[0]["arguments"] == {"product": "iPhone"}


def test_parse_content_call_unknown_tool_dropped():
    """无法识别的工具名被丢弃。"""
    calls = LLMClient._parse_content_calls('{"tool": "hack_xyz", "params": {}}')
    assert calls == []


def test_parse_content_text_mentions_tool():
    """自然语言文本中提取工具名。"""
    calls = LLMClient._parse_content_calls("需要查询价格，调用 get_product_price。")
    assert calls[0]["name"] == "get_product_price"
    assert calls[0]["arguments"] == {}  # 参数由 intent 层注入默认值


def test_parse_content_non_json_no_tool():
    assert LLMClient._parse_content_calls("您好，请问有什么可以帮您？") == []


def test_detect_intent_fills_default_arguments():
    """文本模式（无参数）时注入用户原话作为默认参数。"""
    llm = _FakeLLM(
        tool_calls=[{"id": "x", "name": "get_product_price", "arguments": {}}]
    )
    result = detect_intent(llm, "iPhone 15 Pro Max 多少钱")
    assert result.tool_calls[0]["arguments"]["product_name"] == "iPhone 15 Pro Max 多少钱"


def test_detect_intent_raw_only_fills_default_arguments():
    """arguments 仅含 _raw（JSON 解析失败）时同样注入默认参数，避免检索落空。"""
    llm = _FakeLLM(
        tool_calls=[{"id": "x", "name": "get_product_price", "arguments": {"_raw": "bad"}}]
    )
    result = detect_intent(llm, "手机壳多少钱")
    assert result.tool_calls[0]["arguments"]["product_name"] == "手机壳多少钱"


# ============================================================
# LLMClient 空响应防御
# ============================================================
class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeCompletionsResp:
    def __init__(self, choices):
        self.choices = choices


class _FakeCompletions:
    def __init__(self, resp):
        self._resp = resp

    def create(self, **kw):
        return self._resp


def test_function_call_empty_choices_returns_empty():
    """API 返回空 choices 时视为无工具调用，不越界。"""
    llm = LLMClient(api_key="test")
    llm._client.chat.completions = _FakeCompletions(_FakeCompletionsResp([]))
    assert llm.function_call([{"role": "user", "content": "hi"}], TOOLS) == []


def test_chat_empty_choices_raises_llm_error():
    """chat 空 choices 抛 LLMError（由上层兜底），不 IndexError。"""
    llm = LLMClient(api_key="test")
    llm._client.chat.completions = _FakeCompletions(_FakeCompletionsResp([]))
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        assert False, "应当抛 LLMError"
    except LLMError:
        pass


# ============================================================
# chat_service 流式异常兜底
# ============================================================
class _BrokenLLM(_FakeLLM):
    def __init__(self):
        super().__init__(tool_calls=[])

    def chat_stream(self, messages, **kw):
        raise LLMError("模拟流式故障")
        yield  # pragma: no cover


def test_chat_service_stream_fallback_on_error():
    """流式二次回调失败时下发兜底话术，事件流不中断。"""
    svc = ChatService(llm=_BrokenLLM())
    events = list(svc.chat_stream(_FakeConn([]), "测试"))
    assert events[0]["type"] == "meta"
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "抱歉" in tokens
    assert events[-1]["type"] == "done"


# ============================================================
# llm.build_messages
# ============================================================
def test_build_messages_with_tool_results():
    msgs = LLMClient.build_messages(
        system_prompt="SYS",
        user_query="多少钱？",
        history=[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好"}],
        tool_results=[{"name": "get_product_price", "result": "- SKU 1：商品A，¥100"}],
    )
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "你好"
    assert "参考信息" in msgs[-1]["content"]
    assert "¥100" in msgs[-1]["content"]
    assert msgs[-1]["role"] == "user"


def test_build_messages_history_truncated():
    hist = [{"role": "user", "content": f"消息{i}"} for i in range(15)]
    msgs = LLMClient.build_messages("SYS", "q", history=hist)
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) <= 11  # 历史最多 10 条 + 本次 query


# ============================================================
# router
# ============================================================
class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    """提供 RealDictCursor 风格的假连接。"""

    def __init__(self, rows):
        self._rows = rows

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self._rows)


def test_router_unknown_tool():
    out = execute_tool(_FakeConn([]), {"name": "hack_tool", "arguments": {}})
    assert "未知的工具调用" in out["result"]


def test_router_sql_tool_inventory():
    rows = [
        {
            "product_id": 1, "sku_code": "21873056212", "product_name": "天丝牛仔上衣",
            "price": 868.0, "stock_quantity": 5, "warehouse_location": "上海仓",
            "delivery_estimate_days": 2,
        }
    ]
    out = execute_tool(
        _FakeConn(rows),
        {"name": "get_product_inventory", "arguments": {"sku_code": "21873056212"}},
    )
    assert out["name"] == "get_product_inventory"
    assert "库存 5 件" in out["result"]
    assert "2 天送达" in out["result"]
    assert out["raw"][0]["stock_quantity"] == 5


def test_router_sql_tool_price():
    rows = [
        {
            "id": 1, "sku_code": "21873056212", "product_name": "天丝牛仔上衣",
            "category_id": 10, "price": 868.0,
        }
    ]
    out = execute_tool(
        _FakeConn(rows),
        {"name": "get_product_price", "arguments": {"sku_code": "21873056212"}},
    )
    assert "¥868.00" in out["result"]


# ============================================================
# chat_service 编排（全 mock，不连库/不调外部 LLM）
# ============================================================
def test_chat_service_direct_reply():
    """无工具调用：直接 LLM 回答。"""
    svc = ChatService(llm=_FakeLLM(tool_calls=[]))
    result = svc.chat(_FakeConn([]), "你好呀")
    assert result["reply"] == "模拟回复"
    assert result["intent"] is None
    assert result["tools_used"] == []


def test_chat_service_tool_then_reply():
    """有工具调用：先执行工具，再二次回调。"""
    svc = ChatService(
        llm=_FakeLLM(
            tool_calls=[
                {"id": "c", "name": "get_product_price", "arguments": {"sku_code": "1"}}
            ]
        )
    )
    result = svc.chat(_FakeConn([]), "多少钱？")
    assert result["intent"] == "get_product_price"
    assert result["tools_used"] == ["get_product_price"]


def test_chat_service_stream_events():
    svc = ChatService(llm=_FakeLLM(tool_calls=[]))
    events = list(svc.chat_stream(_FakeConn([]), "你好"))
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "模拟回复"
