"""
自定义异常
"""


class LLMError(Exception):
    """LLM 调用异常（超时/网络/API 错误）。"""


class ConfigError(RuntimeError):
    """配置缺失或非法（如未配置 DATABASE_URL）。"""
