# -*- coding: utf-8 -*-
"""阶段二：HybridRetriever 双路合并去重 与 tokenize 单元测试（不依赖数据库）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.online.core.rag.retriever import (
    ChunkHit,
    HybridRetriever,
    STOPWORDS,
    tokenize,
)


def make_vec_hit(chunk_id, doc_id=1, idx=0, score=0.8):
    return ChunkHit(
        chunk_id=chunk_id, doc_id=doc_id, chunk_index=idx,
        chunk_text="向量命中文本", doc_type="faq",
        meta_data={}, vector_score=score,
    )


def make_bm25_hit(chunk_id, doc_id=1, idx=0, score=10.0):
    return ChunkHit(
        chunk_id=chunk_id, doc_id=doc_id, chunk_index=idx,
        chunk_text="关键词命中文本", doc_type="faq",
        meta_data={}, bm25_score=score,
    )


class _FakeVector:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=30, doc_type=None):
        return self._hits


class _FakeBM25:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=30, doc_type=None):
        return self._hits


def test_merge_dedup_shared_hit():
    """同一 (doc_id, chunk_index) 两路都命中：只保留一条，两路分数都合并。"""
    vec = [make_vec_hit(chunk_id=100, doc_id=1, idx=0)]
    bm = [make_bm25_hit(chunk_id=100, doc_id=1, idx=0)]
    hr = HybridRetriever(None, _FakeVector(vec), _FakeBM25(bm))
    out = hr.search("测试")
    assert len(out) == 1
    assert out[0].vector_score == 0.8
    assert out[0].bm25_score == 10.0


def test_merge_keeps_distinct_hits():
    """两路各自命中不同 chunk：两条都保留，分数各归其位。"""
    vec = [make_vec_hit(chunk_id=1, doc_id=1, idx=0)]
    bm = [make_bm25_hit(chunk_id=2, doc_id=2, idx=3)]
    hr = HybridRetriever(None, _FakeVector(vec), _FakeBM25(bm))
    out = hr.search("测试")
    assert len(out) == 2
    by_id = {h.chunk_id: h for h in out}
    assert by_id[1].vector_score == 0.8 and by_id[1].bm25_score is None
    assert by_id[2].bm25_score == 10.0 and by_id[2].vector_score is None


def test_merge_preserves_vector_score_when_only_bm25_hits():
    """同 key 仅 BM25 命中时，向量分保持 None（交由 Reranker 单路降级处理）。"""
    bm = [make_bm25_hit(chunk_id=9, doc_id=9, idx=0)]
    hr = HybridRetriever(None, _FakeVector([]), _FakeBM25(bm))
    out = hr.search("测试")
    assert len(out) == 1
    assert out[0].vector_score is None
    assert out[0].bm25_score == 10.0


def test_empty_query_returns_empty():
    hr = HybridRetriever(None, _FakeVector([]), _FakeBM25([]))
    assert hr.search("") == []
    assert hr.search("   ") == []


def test_tokenize_keeps_words_and_drops_stopwords():
    toks = tokenize("连衣裙怎么洗不容易变形")
    assert "连衣裙" in toks
    assert "洗" in toks
    assert "怎么" not in toks          # 疑问词被停用词过滤
    assert all(t not in STOPWORDS for t in toks)


def test_tokenize_english_mixed():
    toks = tokenize("iPhone 15 Pro Max 256G 多少钱")
    assert "iPhone" in toks
    assert "256G" in toks
