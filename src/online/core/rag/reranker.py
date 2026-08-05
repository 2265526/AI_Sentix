# -*- coding: utf-8 -*-
"""
重排序服务（综合得分 + 相关性阈值 + Top-K）
===================================================================

实现说明：
  - 综合得分 = 加权融合两路召回分数：
        final_score = w_v * norm(vector_score) + w_b * norm(bm25_score)
    其中 norm() 把各路分数归一化到 [0, 1]：
        - 向量相似度（1 - 余弦距离）天然在 [0, 1]，直接使用；
        - BM25 原始分无上界，按当批最大分归一化（max 归一化）。
  - 权重默认 w_v=0.7 / w_b=0.3（语义为主、关键词为辅），可通过构造参数调整。
  - 若某候选只有单路被召回（另一路分数为 None），则只用有分的那一路；
    两路都未命中（理论不会出现）得分为 0。
  - 阈值过滤：final_score <= threshold（默认 0.4）直接丢弃，> threshold 保留；
    保留结果按 final_score 降序，取 top_k（默认 5）。
  - 预留 cross-encoder 重排接口（可选）：
    若部署了 bge-reranker 等模型，可注入 score_fn 对候选重新打分后套用同一阈值逻辑。
"""
from typing import Callable, List, Optional

from .retriever import ChunkHit

DEFAULT_THRESHOLD = 0.4
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3


class Reranker:
    """混合检索结果的重排序器。"""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        single_route_discount: float = 1.0,
        bm25_only_discount: float = 0.7,
        score_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Args:
            threshold: 相关性阈值，最终得分 <= 该值的分块被丢弃（默认 0.4）
            top_k: 通过阈值后保留的最多结果数（默认 5）
            vector_weight / bm25_weight: 向量分与 BM25 分的融合权重（和为 1）
            single_route_discount: 仅向量路被召回时的降权系数（默认 1.0，不打折）。
                      0.4 阈值语义上是"余弦相似度阈值"，向量相似度（1-余弦距离）
                      应直接用它衡量；打折会让语义命中（如 0.47）跌破 0.4 被误杀。
            bm25_only_discount: 仅 BM25 路被召回（embedding 服务不可用、
                      降级模式）时的降权系数（默认 0.7）。BM25 分数按当批
                      max 归一化，最高分恒为 1.0——若不额外降权，降级模式下
                      几乎所有候选都能过 0.4 阈值，过滤形同虚设；
                      0.7 折扣使仅最强命中（0.7 > 0.4）保留、次优命中被过滤。
            score_fn: 可选的重打分函数 f(query, chunk_text) -> float；
                      提供时替代加权融合，作为"最终得分"（如未来接 cross-encoder）
        """
        self.threshold = threshold
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.single_route_discount = single_route_discount
        self.bm25_only_discount = bm25_only_discount
        self.score_fn = score_fn

    # --------------------------------------------------------
    def _normalize_bm25(self, hits: List[ChunkHit]) -> None:
        """把当批 BM25 原始分按最大值归一化到 [0,1]。"""
        max_score = max((h.bm25_score or 0.0) for h in hits)
        if max_score <= 0:
            return
        for h in hits:
            if h.bm25_score is not None:
                h.bm25_score = h.bm25_score / max_score
    def _compute_final_score(self, query: str, hit: ChunkHit) -> float:
        """计算单条候选的最终综合得分。

        融合规则：
          - 双路都被召回：final = w_v * norm(vector) + w_b * norm(bm25)；
          - 仅向量路召回：final = norm(vector) × single_route_discount（默认 1.0，
            直接以余弦相似度衡量——0.4 阈值语义即"余弦相似度阈值"）；
          - 仅 BM25 路召回（降级模式）：final = norm(bm25) × bm25_only_discount
            （BM25 按当批 max 归一化最高恒为 1.0，需额外降权，否则降级模式下
            阈值过滤形同虚设——只有最强关键词命中能过 0.4）；
          - 两路都无分：0。
        """
        if self.score_fn is not None:
            try:
                return float(self.score_fn(query, hit.chunk_text))
            except Exception:
                # 外部重排器异常时退回加权融合，保证主流程可用
                pass

        if hit.vector_score is not None and hit.bm25_score is not None:
            return (
                self.vector_weight * hit.vector_score
                + self.bm25_weight * hit.bm25_score
            )
        if hit.vector_score is not None:
            return hit.vector_score * self.single_route_discount
        if hit.bm25_score is not None:
            return hit.bm25_score * self.bm25_only_discount
        return 0.0
    # --------------------------------------------------------
    def rerank(self, hits: List[ChunkHit], query: str = "") -> List[ChunkHit]:
        """
        对混合召回结果重排序：
          1. BM25 分数 max 归一化
          2. 计算每条最终综合得分
          3. 丢弃最终得分 <= threshold 的结果
          4. 按最终得分降序，取 top_k

        Args:
            hits: HybridRetriever.search() 的合并去重结果
            query: 原始查询文本（供外部 score_fn 使用）

        Returns:
            按最终得分降序的 Top-K 结果
        """
        if not hits:
            return []

        self._normalize_bm25(hits)

        for h in hits:
            h.score = self._compute_final_score(query, h)

        passed = [h for h in hits if h.score > self.threshold]
        passed.sort(key=lambda h: h.score, reverse=True)
        return passed[: self.top_k]
