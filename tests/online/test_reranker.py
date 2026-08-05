# -*- coding: utf-8 -*-
"""阶段二：reranker 单元测试（综合得分 / 阈值过滤 / Top-K / 单路降级）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.online.core.rag.retriever import ChunkHit
from src.online.core.rag.reranker import Reranker


def make_hit(
    chunk_id,
    vector_score=None,
    bm25_score=None,
    chunk_text="测试文本",
    doc_type="faq",
):
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=1,
        chunk_index=chunk_id,
        chunk_text=chunk_text,
        doc_type=doc_type,
        vector_score=vector_score,
        bm25_score=bm25_score,
    )


def test_double_route_weighted_fusion():
    """双路分数齐全时按权重融合；BM25 先按当批最大分归一化。

    A: v=0.8, b=10 → 0.7*0.8 + 0.3*1.0 = 0.86
    B: v=0.6, b=5  → 0.7*0.6 + 0.3*0.5 = 0.57
    """
    hits = [
        make_hit(1, vector_score=0.8, bm25_score=10.0),
        make_hit(2, vector_score=0.6, bm25_score=5.0),
    ]
    out = Reranker().rerank(hits)
    by_id = {h.chunk_id: h for h in out}
    assert abs(by_id[1].score - 0.86) < 1e-9
    assert abs(by_id[2].score - 0.57) < 1e-9
    assert out[0].chunk_id == 1  # 得分高者在前


def test_threshold_filter_drops_low():
    """最终得分 <= 0.4 的被丢弃；> 0.4 的保留。

    单路向量不打折（0.4 阈值语义即余弦相似度阈值）：0.3 被丢、0.6 保留；
    双路 0.7*0.7+0.3*1.0=0.79 保留且排最前。
    """
    hits = [
        make_hit(1, vector_score=0.3),              # 单路 0.3 → 丢弃
        make_hit(2, vector_score=0.6),              # 单路 0.6 → 保留
        make_hit(3, vector_score=0.7, bm25_score=0.5),  # 双路 0.79 → 保留
    ]
    out = Reranker(threshold=0.4, top_k=5).rerank(hits)
    ids = [h.chunk_id for h in out]
    assert ids == [3, 2]
    assert all(h.score > 0.4 for h in out)


def test_top_k_limits_and_order():
    """通过阈值后按得分降序，只保留 top_k。"""
    hits = [make_hit(i, vector_score=0.5 + i * 0.1) for i in range(1, 7)]
    out = Reranker(threshold=0.4, top_k=5).rerank(hits)
    assert len(out) == 5
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)


def test_single_route_fallback():
    """只有 BM25 分（向量路未召回/降级模式）时用归一化分 × 0.7，仅最强命中过阈值。"""
    hits = [
        make_hit(1, bm25_score=10.0),
        make_hit(2, bm25_score=5.0),
    ]
    out = Reranker().rerank(hits)
    # 归一化后 1.0×0.7=0.7 保留；0.5×0.7=0.35 <= 0.4 被阈值过滤
    assert len(out) == 1
    assert abs(out[0].score - 0.7) < 1e-9
    assert out[0].chunk_id == 1


def test_empty_hits():
    """空候选直接返回空列表。"""
    assert Reranker().rerank([]) == []


def test_score_fn_override():
    """注入外部重排函数时代替加权融合。"""
    hits = [make_hit(1, vector_score=0.9), make_hit(2, vector_score=0.1)]
    out = Reranker(score_fn=lambda q, t: 0.5).rerank(hits)
    assert all(abs(h.score - 0.5) < 1e-9 for h in out)
