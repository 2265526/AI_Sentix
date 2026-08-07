# -*- coding: utf-8 -*-
"""
core/llm/client.py —— LLM 统一调用（DeepSeek）
==============================================
对应《开发文档》阶段三：
  - 意图识别：function calling 输出 tool_call
  - 二次模型回调：把检索结果与用户问题再次交给 LLM，生成（流式）回复

技术选型（文档未细化的部分）：
  - 使用 openai SDK 对接 DeepSeek（OpenAI 兼容 API），密钥、base_url、模型名等
    配置统一在 config/settings.py（DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL /
    DEEPSEEK_MODEL，默认 deepseek-chat），本模块不再直接读取环境变量；
  - 提供三类能力：
        chat()            非流式补全（测试/内部用）
        chat_stream()     流式补全（SSE 逐字下发，供 /v1/chat/text）
        function_call()   工具调用（意图识别），返回解析后的 tool_calls
  - 超时：统一取 settings.deepseek_timeout，避免请求挂死；
    调用失败抛 LLMError（定义于 src/common/exceptions.py），由上层 chat_service 做兜底。
"""
import json
import re
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

from config.settings import settings
from src.common.exceptions import LLMError
from src.online.core.agent.tools import _TOOL_NAME_RE, normalize_tool_name


class LLMClient:
    """DeepSeek 统一客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        api_key = api_key or settings.deepseek_api_key
        if not api_key:
            raise LLMError("未配置 DEEPSEEK_API_KEY，请在 .env 中设置")
        self.model = model or settings.deepseek_model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or settings.deepseek_base_url,
            timeout=timeout or settings.deepseek_timeout,
        )

    # --------------------------------------------------------
    # 非流式补全
    # --------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
    ) -> str:
        """非流式补全，返回完整回复文本。"""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            if not resp.choices:
                raise LLMError("LLM 响应为空（choices 为空）")
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            raise LLMError(f"LLM 非流式调用失败: {e}") from e

    # --------------------------------------------------------
    # 流式补全
    # --------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """流式补全，逐段 yield 文本增量（不含 tool_calls 内容）。"""
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            raise LLMError(f"LLM 流式调用失败: {e}") from e

    # --------------------------------------------------------
    # Function calling（意图识别）
    # --------------------------------------------------------
    def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        调用带工具 schema 的补全，解析模型返回的工具调用。

        兼容两种响应格式：
          1. 标准 OpenAI tool_calls（message.tool_calls）；
          2. DeepSeek deepseek-chat 实测会把工具调用序列化为 JSON 文本
             放在 message.content 中（如 {"name": "...", "arguments": {...}}）——
             自动解析并归一化为同一结构。

        Returns:
            [{"id": str, "name": str, "arguments": dict}, ...]
            arguments 已从 JSON 解析为 dict；解析失败时保留 {"_raw": ...}。
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )
        except Exception as e:
            raise LLMError(f"LLM 意图识别失败: {e}") from e

        if not resp.choices:
            return []  # 空响应视为"无工具调用"
        message = resp.choices[0].message
        parsed: List[Dict[str, Any]] = []

        # 格式 1：标准 tool_calls
        if message.tool_calls:
            for tc in message.tool_calls:
                fn = tc.function
                name = normalize_tool_name(fn.name or "")
                if name is None:
                    continue
                parsed.append(
                    self._parse_call(tc.id, name, fn.arguments or "")
                )
            return parsed

        # 格式 2：content 里是工具调用 JSON（DeepSeek 兼容格式）
        content = (message.content or "").strip()
        if content:
            parsed = self._parse_content_calls(content)
        return parsed

    # --------------------------------------------------------
    # 工具调用解析辅助
    # --------------------------------------------------------
    @staticmethod
    def _parse_call(call_id: str, name: str, arguments_raw: str) -> Dict[str, Any]:
        """把原始 arguments 字符串解析为 dict（失败保留 _raw）。"""
        try:
            args = json.loads(arguments_raw or "{}")
        except json.JSONDecodeError:
            args = {"_raw": arguments_raw}
        if not isinstance(args, dict):
            args = {"_raw": arguments_raw}
        return {"id": call_id, "name": name, "arguments": args}

    @staticmethod
    def _parse_content_calls(content: str) -> List[Dict[str, Any]]:
        """解析 content 中的工具调用。

        兼容（实测 DeepSeek 响应不稳定）：
          - JSON 对象：{"name"/"tool"/"function": 工具名, "arguments"/"args"/"params": 参数}
          - JSON 数组：[{...}, ...]
          - 自然语言文本："需要查询价格，调用 get_product_price" → 正则提取工具名
        工具名经 normalize_tool_name 归一化（容忍模型发明的别名），
        无法识别为任何工具时丢弃该候选。
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return LLMClient._parse_text_calls(content)

        candidates: List[Any] = data if isinstance(data, list) else [data]
        calls: List[Dict[str, Any]] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            raw_name = c.get("name") or c.get("tool") or c.get("function") or c.get("tool_name")
            name = normalize_tool_name(raw_name if isinstance(raw_name, str) else "")
            if name is None:
                continue
            raw_args = c.get("arguments", c.get("args", c.get("params", c.get("parameters", "{}"))))
            raw_args = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
            calls.append(LLMClient._parse_call(f"call_{len(calls)}", name, raw_args))
        return calls

    @staticmethod
    def _parse_text_calls(content: str) -> List[Dict[str, Any]]:
        """从自然语言文本中提取工具调用（模型未输出 JSON 时的兜底）。"""
        matches = re.findall(_TOOL_NAME_RE, content)
        calls: List[Dict[str, Any]] = []
        seen = set()
        for m in matches:
            name = normalize_tool_name(m)
            if name is None or name in seen:
                continue
            seen.add(name)
            calls.append({"id": f"call_{len(calls)}", "name": name, "arguments": {}})
        return calls

    # --------------------------------------------------------
    # 消息构建辅助
    # --------------------------------------------------------
    @staticmethod
    def build_messages(
        system_prompt: str,
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """
        组装一次对话的完整消息列表。

        Args:
            system_prompt: 系统提示词
            user_query: 本次用户问题
            history: 历史消息 [{"role": "user"|"assistant", "content": ...}]
            tool_results: 工具执行结果 [{"name", "result": str}...]；
                          有则拼接进 user 消息（以"参考信息"形式提供给模型）

        Returns:
            OpenAI 消息列表
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        # 历史消息（限制最近 10 条，防止上下文膨胀）
        if history:
            messages.extend(history[-10:])

        if tool_results:
            refs = "\n\n".join(
                f"【工具 {r['name']} 返回】\n{r['result']}" for r in tool_results
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"{user_query}\n\n以下是查询到的参考信息，请据此回答：\n{refs}",
                }
            )
        else:
            messages.append({"role": "user", "content": user_query})
        return messages
