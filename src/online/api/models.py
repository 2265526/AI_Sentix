# -*- coding: utf-8 -*-
"""
Pydantic 请求/响应模型
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RAGSearchRequest(BaseModel):
    """/rag/search 请求体。"""

    query: str = Field(..., min_length=1, max_length=512, description="用户查询文本")
    top_k: int = Field(5, ge=1, le=20, description="返回的最相关知识片段数（Top-K）")
    threshold: float = Field(
        0.4, ge=0.0, le=1.0, description="相关性阈值，最终得分 <= 该值的结果被丢弃"
    )
    doc_type: Optional[str] = Field(
        None, description="按文档类型过滤（product_manual / faq / policy），不传表示全部"
    )
    category_big: Optional[str] = Field(
        None, description="类目过滤：大类（如：手机数码），精确匹配 meta_data.category_big"
    )
    category_small: Optional[str] = Field(
        None, description="类目过滤：小类（如：智能手机），精确匹配 meta_data.category_small"
    )
    category_path: Optional[str] = Field(
        None, description="类目过滤：类目路径（如：手机数码/手机/%），LIKE 模糊匹配 meta_data.category_path"
    )


class RAGChunk(BaseModel):
    """一条检索结果（知识分块）。"""

    chunk_id: int = Field(..., description="kb_chunks.id")
    doc_id: int = Field(..., description="所属原始文档 kb_documents.id")
    chunk_index: int = Field(..., description="分块序号")
    chunk_text: str = Field(..., description="分块文本内容")
    doc_type: Optional[str] = Field(None, description="文档类型")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    score: float = Field(..., description="最终综合得分（Rerank 后）")
    vector_score: Optional[float] = Field(None, description="向量相似度 [0,1]，可能为空")
    bm25_score: Optional[float] = Field(None, description="BM25 归一化分数 [0,1]，可能为空")


class RAGSearchResponse(BaseModel):
    """/rag/search 响应体。"""

    query: str = Field(..., description="回显查询文本")
    total: int = Field(..., description="通过阈值后返回的结果数")
    threshold: float = Field(..., description="实际使用的相关性阈值")
    results: List[RAGChunk] = Field(default_factory=list, description="Top-K 知识片段（按得分降序）")
    degraded: bool = Field(
        False, description="是否降级（embedding 服务不可用时仅 BM25 召回）"
    )


# ============================================================
# 阶段三：Agent 对话
# ============================================================
class ChatMessage(BaseModel):
    """历史对话消息。"""

    role: str = Field(..., pattern="^(user|assistant)$", description="消息角色")
    content: str = Field(..., min_length=1, max_length=4000, description="消息内容")


class ChatRequest(BaseModel):
    """/v1/chat/text 请求体。"""

    message: str = Field(..., min_length=1, max_length=2000, description="用户输入")
    history: List[ChatMessage] = Field(
        default_factory=list, max_length=20, description="多轮历史（最近 10 条生效）"
    )
    stream: bool = Field(True, description="是否流式返回（SSE）")
    session_id: Optional[str] = Field(
        None, max_length=64,
        description="会话标识（短期记忆载体，白名单 ^[A-Za-z0-9-]{1,64}$；空则服务端生成）",
    )


class ChatResponse(BaseModel):
    """/v1/chat/text 非流式响应体。"""

    reply: str = Field(..., description="客服回复")
    intent: Optional[str] = Field(None, description="识别到的意图工具名（无则 None）")
    tools_used: List[str] = Field(default_factory=list, description="实际执行的工具列表")
    context_reset: bool = Field(
        False, description="会话过期/首次访问信号：为 True 时前端应清空本地历史"
    )
    original_query: Optional[str] = Field(None, description="原始用户问题（问题增强前）")
    enriched_query: Optional[str] = Field(None, description="增强后问题（无增强时等于原始问题）")
