# -*- coding: utf-8 -*-
"""
上下文组装

职责：
  - 把 Reranker 输出的 Top-K 知识分块拼装成带编号、带来源的上下文文本；
  - 控制上下文总长度（按字符数截断，避免超出 LLM 上下文窗口）；
  - 与 prompt_templates.py 的 RAG 模板配合，生成最终 Prompt。
"""
from typing import List

from src.common.constants import MAX_CONTEXT_CHARS
from .retriever import ChunkHit

# 上下文总长度上限（字符，统一见 src/common/constants.py）。按中文场景估算
# ~2 个字符/token，默认上限 ~6k token 的片段量，防止拼接超窗；可按模型调整。


def _format_hit(hit: ChunkHit, idx: int) -> str:
    """单条知识分块 → 编号片段文本（含来源标签与得分）。"""
    source = hit.doc_type or "unknown"
    # meta_data 防御：ETL 写入为 JSONB 对象，但历史/异常数据可能非 dict
    meta = hit.meta_data if isinstance(hit.meta_data, dict) else {}
    # 补充可读来源信息（有则展示）
    extra = []
    if meta.get("sku"):
        extra.append(f"SKU={meta['sku']}")
    if meta.get("product_id"):
        extra.append(f"product_id={meta['product_id']}")
    if meta.get("chapter"):
        extra.append(f"章节={meta['chapter']}")
    source_desc = f"[{source}]" + (f"({'/'.join(extra)})" if extra else "")
    return f"[片段{idx + 1}] {source_desc} 相关度={hit.score:.3f}\n{hit.chunk_text}"


def build_context(
    hits: List[ChunkHit], max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    把 Top-K 分块组装成上下文文本。

    Args:
        hits: Reranker.rerank() 的输出（已按得分降序）
        max_chars: 上下文总字符上限

    Returns:
        拼接后的上下文字符串（无结果时返回空串）
    """
    if not hits:
        return ""

    parts: List[str] = []
    total = 0
    for i, hit in enumerate(hits):
        block = _format_hit(hit, i)
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
