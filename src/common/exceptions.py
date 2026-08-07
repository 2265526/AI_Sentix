# -*- coding: utf-8 -*-
"""
src/common/exceptions.py —— 自定义异常
=======================================
对应《目录结构》中 src/common/exceptions.py 的职责："自定义异常"。

各业务模块统一从这里导入/抛出自定义异常；
为避免破坏既有 import 路径，原定义处（如 core/llm/client.py）保留重新导出。
"""


class LLMError(Exception):
    """LLM 调用异常（超时/网络/API 错误）。"""


# 预留：后续业务异常（如配置缺失、检索异常）在此扩展
class ConfigError(RuntimeError):
    """配置缺失或非法（如未配置 DATABASE_URL）。"""
