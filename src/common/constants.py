"""
业务常量
"""
from typing import Tuple

# 文档类型（None 表示不过滤）
ALLOWED_DOC_TYPES: Tuple[str, ...] = ("product_manual", "faq", "policy")

# RAG 重排序默认参数
DEFAULT_THRESHOLD: float = 0.4        # 相关性阈值：最终得分 <= 该值的分块被丢弃
DEFAULT_TOP_K: int = 5                # 通过阈值后保留的最多结果数
DEFAULT_VECTOR_WEIGHT: float = 0.7    # 融合权重：向量相似度
DEFAULT_BM25_WEIGHT: float = 0.3      # 融合权重：BM25 归一化分（与向量权重和为 1）
SINGLE_ROUTE_DISCOUNT: float = 1.0    # 仅向量路召回时的降权系数（1.0 = 不打折）
BM25_ONLY_DISCOUNT: float = 0.7       # 仅 BM25 路召回（embedding 降级）时的降权系数

# 上下文组装上限（字符）
MAX_CONTEXT_CHARS: int = 12000        # RAG 上下文总长度上限，防止拼接超窗
