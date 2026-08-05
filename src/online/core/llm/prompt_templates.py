# -*- coding: utf-8 -*-
"""
 Prompt 模板
============================================
RAG 问答模板：把检索到的上下文 + 用户问题拼成完整 Prompt，
Agent 二次模型回调使用；也可单独用于 /rag/search 的调试返回。

模板设计要点：
  - 知识片段可能为空 → 明确要求模型基于自身能力兜底并提示"未收录"；
  - 片段有明确来源标签（doc_type/SKU 等）→ 要求模型只依据给定片段回答，
    不编造片段外的信息，避免幻觉；
  - 上下文与问题以分隔符隔开，方便模型定位。
"""
from typing import Optional

# 系统提示词：电商客服角色 + RAG 使用约束
RAG_SYSTEM_PROMPT = """你是一名专业的电商AI智能客服，负责解答用户在商品、库存、物流、售后、使用说明等方面的问题。
回答规则：
1. 优先依据下方提供的"知识片段"回答，做到准确、简洁、友好；
2. 若知识片段不足以回答用户问题，如实说明"该信息暂未收录"，不要编造；
3. 涉及价格、库存、物流时效等具体数据时，只引用片段中明确出现的内容；
4. 全程使用简体中文回答。"""

# 用户提示词模板：{context} 为 build_context 组装的知识片段，{question} 为用户问题
RAG_USER_TEMPLATE = """【知识片段】
{context}

【用户问题】
{question}

请根据知识片段回答用户问题。若片段与问题无关或信息不足，请直接告知未收录相关信息。"""

# 无知识片段时的兜底提示词（片段为空走 LLM 泛化能力）
RAG_FALLBACK_TEMPLATE = """用户问题：{question}
（本次未检索到相关知识片段，请基于自身能力判断：能回答则简要回答，无法确认则回复"抱歉，我还没收录相关信息"。）"""


def build_rag_prompt(
    question: str,
    context: Optional[str] = None,
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> list:
    """
    组装 RAG 问答 Prompt（OpenAI 消息格式，供deepseek 客户端直接调用）。

    Args:
        question: 用户问题
        context: build_context() 的产物；为空或 None 时使用兜底模板
        system_prompt: 系统提示词（默认 RAG_SYSTEM_PROMPT）

    Returns:
        [{"role": "system", ...}, {"role": "user", ...}] 消息列表
    """
    context = (context or "").strip()
    if context:
        user_msg = RAG_USER_TEMPLATE.format(context=context, question=question)
    else:
        user_msg = RAG_FALLBACK_TEMPLATE.format(question=question)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
