# -*- coding: utf-8 -*-
"""
db/repositories/kb_repo.py —— 知识库数据访问层
===============================================
阶段三 RAG 检索（get_knowledge_base / product_recommendation）的封装：
复用阶段二混合检索引擎（HybridRetriever + Reranker），与 /rag/search 同链路。
"""
from typing import Any, Dict, List, Optional

from src.online.core.rag.reranker import Reranker
from src.online.core.rag.retriever import HybridRetriever


def search_kb(
    conn,
    query: str,
    top_k: int = 5,
    threshold: float = 0.4,
    doc_type: Optional[str] = None,
    category_big: Optional[str] = None,
    category_small: Optional[str] = None,
    category_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    知识库混合检索（BM25 + 向量双路召回 → Rerank 阈值过滤 → Top-K）。

    Args:
        category_big / category_small / category_path: 类目级过滤
            （meta_data 类目字段，如"手机"类目排除"手机壳"等无关内容）

    Returns:
        [{chunk_id, doc_id, chunk_index, chunk_text, doc_type, meta_data,
          score, vector_score, bm25_score}, ...]
    """
    retriever = HybridRetriever(conn)
    hits = retriever.search(
        query, candidate_k=max(top_k * 6, 30), doc_type=doc_type,
        category_big=category_big, category_small=category_small,
        category_path=category_path,
    )
    top = Reranker(threshold=threshold, top_k=top_k).rerank(hits, query=query)
    return [
        {
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "chunk_index": h.chunk_index,
            "chunk_text": h.chunk_text,
            "doc_type": h.doc_type,
            "meta_data": h.meta_data,
            "score": round(h.score, 4),
            "vector_score": round(h.vector_score, 4) if h.vector_score is not None else None,
            "bm25_score": round(h.bm25_score, 4) if h.bm25_score is not None else None,
        }
        for h in top
    ]


def format_kb(chunks: List[Dict[str, Any]], max_chars: int = 1500) -> str:
    """
    把知识分块格式化为给 LLM 的参考文本（含来源标签，超长截断）。
    """
    if not chunks:
        return "知识库未检索到相关内容（该问题暂未收录，建议咨询人工客服或换个问法）。"
    parts = []
    total = 0
    for i, c in enumerate(chunks):
        source = c.get("doc_type") or "unknown"
        block = (
            f"[{i + 1}]（来源：{source}，相关度 {c['score']:.3f}）\n"
            f"{c['chunk_text']}"
        )
        if total + len(block) > max_chars:
            block = block[: max_chars - total]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n\n".join(parts)
