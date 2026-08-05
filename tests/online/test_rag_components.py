# -*- coding: utf-8 -*-
"""阶段二：context_builder / prompt_templates 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.online.core.llm.prompt_templates import build_rag_prompt
from src.online.core.rag.context_builder import build_context
from src.online.core.rag.retriever import ChunkHit


def make_hit(chunk_id, text="内容文本", doc_type="faq", meta=None, score=0.6):
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=10,
        chunk_index=chunk_id,
        chunk_text=text,
        doc_type=doc_type,
        meta_data=meta or {},
        score=score,
    )


def test_build_context_numbers_and_source():
    """上下文包含编号、来源标签（doc_type/SKU）与得分。"""
    hits = [
        make_hit(1, text="七天无理由退货", meta={"sku": "IP15PM256"}),
        make_hit(2, text="清洗注意事项", doc_type="product_manual"),
    ]
    ctx = build_context(hits)
    assert "[片段1]" in ctx and "[片段2]" in ctx
    assert "[faq]" in ctx and "[product_manual]" in ctx
    assert "SKU=IP15PM256" in ctx
    assert "相关度=0.600" in ctx


def test_build_context_empty():
    assert build_context([]) == ""


def test_build_context_truncation():
    """超过 max_chars 时截断后续片段。"""
    hits = [make_hit(i, text="长" * 200) for i in range(1, 6)]
    ctx = build_context(hits, max_chars=300)
    assert len(ctx) <= 300 + 200  # 允许最后一个片段整体跳过/部分计入的边界
    assert "[片段1]" in ctx


def test_build_rag_prompt_with_context():
    msgs = build_rag_prompt("怎么退货？", context="【片段1】七天无理由退货")
    assert msgs[0]["role"] == "system"
    assert "知识片段" in msgs[1]["content"]
    assert "怎么退货？" in msgs[1]["content"]
    assert "七天无理由退货" in msgs[1]["content"]


def test_build_rag_prompt_fallback():
    """无上下文时走兜底模板。"""
    msgs = build_rag_prompt("今天天气怎么样？", context=None)
    assert "未检索到相关知识片段" in msgs[1]["content"]
