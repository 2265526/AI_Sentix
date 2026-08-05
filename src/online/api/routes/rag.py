# -*- coding: utf-8 -*-
"""
检索接口
  支持输入 Query 返回最相关的 Top-5 知识片段"

调用链：POST /rag/search
  RAGSearchRequest
    → HybridRetriever.search()   （BM25 + 向量双路召回）
    → Reranker.rerank()          （综合得分、阈值 0.4、Top-5）
    → RAGSearchResponse
"""
import logging

from fastapi import APIRouter, Depends
from psycopg2.extensions import connection

from src.online.api.dependencies import get_rag_engine
from src.online.api.models import (
    RAGChunk,
    RAGSearchRequest,
    RAGSearchResponse,
)
from src.online.core.rag.reranker import Reranker
from src.online.core.rag.retriever import HybridRetriever
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])
@router.post("/refresh", summary="重建 BM25 索引（离线 ETL 入库后调用）")
def refresh_bm25(engine: dict = Depends(get_rag_engine)):
    """
    重建进程内的 BM25 关键词索引。
    离线 ETL写入新知识后，
    在线服务的 BM25 索引不会自动更新，需调用本接口（或重启服务）刷新，
    否则新知识在 BM25 路检索不到。
    """
    retriever: HybridRetriever = engine["retriever"]
    retriever.bm25_retriever.refresh()
    stats = retriever.stats()
    logger.info("rag/refresh BM25 索引已重建: %s", stats)
    return {"status": "ok", **stats}
@router.post("/search", response_model=RAGSearchResponse, summary="RAG 混合检索 Top-K")
def search(
    req: RAGSearchRequest,
    engine: dict = Depends(get_rag_engine),
):
    """
    混合检索（BM25 + 语义向量双路召回）→ Rerank（阈值过滤 + Top-K）。
    """
    retriever: HybridRetriever = engine["retriever"]
    # 按请求参数构造 Reranker（threshold / top_k 随请求生效）
    reranker: Reranker = Reranker(threshold=req.threshold, top_k=req.top_k)
    # 1) 双路召回（每路召回 candidate_k 条作为 Rerank 候选池）
    candidate_k = max(req.top_k * 6, 30)  # 候选池按 top_k 放大，保证 Rerank 有足够素材
    hits = retriever.search(req.query, candidate_k=candidate_k, doc_type=req.doc_type)
    # 2) 重排序：综合得分、阈值过滤、降序取 Top-K
    top = reranker.rerank(hits, query=req.query)
    results = [
        RAGChunk(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            chunk_index=h.chunk_index,
            chunk_text=h.chunk_text,
            doc_type=h.doc_type,
            meta_data=h.meta_data,
            score=round(h.score, 4),
            vector_score=round(h.vector_score, 4) if h.vector_score is not None else None,
            bm25_score=round(h.bm25_score, 4) if h.bm25_score is not None else None,
        )
        for h in top
    ]
    # 降级标记：向量路为空说明 embedding 服务不可用，本次仅 BM25 召回
    degraded = all(h.vector_score is None for h in hits) if hits else True
    logger.info(
        "rag/search query=%r doc_type=%s candidates=%d passed=%d degraded=%s",
        req.query, req.doc_type, len(hits), len(top), degraded,
    )
    return RAGSearchResponse(
        query=req.query,
        total=len(results),
        threshold=req.threshold,
        results=results,
        degraded=degraded,
    )
