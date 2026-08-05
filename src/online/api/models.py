# -*- coding: utf-8 -*-
"""
Pydantic 请求/响应模型
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# 允许的文档类型（与 kb_documents.doc_type 数据一致，None 表示不过滤）
ALLOWED_DOC_TYPES = ("product_manual", "faq", "policy")


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


class ChatResponse(BaseModel):
    """/v1/chat/text 非流式响应体。"""

    reply: str = Field(..., description="客服回复")
    intent: Optional[str] = Field(None, description="识别到的意图工具名（无则 None）")
    tools_used: List[str] = Field(default_factory=list, description="实际执行的工具列表")
