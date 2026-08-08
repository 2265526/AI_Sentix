"""
— 混合检索引擎（BM25 关键词 + 语义向量 双路召回）
"""
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jieba
import psycopg2.extras
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config.settings import settings

# 召回参数默认值
VECTOR_TOP_K = settings.vector_top_k   # 向量路每路召回数（Rerank 前的候选池）
BM25_TOP_K = settings.bm25_top_k       # BM25 路每路召回数
MAX_QUERY_LEN = settings.max_query_len  # 查询文本截断长度（与 /rag/search 接口 query 上限一致）

# token 过滤：jieba 分词结果直接作为 token（中文词整体保留，不再二次拆字；
# 英文数字词由 jieba 自行切分）。
_TOKEN_RE = re.compile(r"^[a-zA-Z0-9]+$")

# 中文停用词（高频虚词 + 通用疑问词）。
# IDF 接近 0，参与 BM25 打分只会抬升所有候选的分数、稀释区分度。
STOPWORDS = frozenset(
    """
    的 了 吗 呢 吧 啊 呀 嘛 哦 嗯 是 在 和 与 及 或 就 都 而 于 之 其 这 那
    你 我 他 她 它 我们 你们 他们 她们 它们 咱们
    怎么 怎样 如何 什么样 为什么 什么 哪 哪些 哪个 谁 多少 多久 哪里 哪儿
    请问 可以 能 会 要 有 没有 不 也 还 很 更 最 太 只 但 但是 因为 所以 如果
    一下 一个 一种 一样 是否 是不是 请问 您好 你好 谢谢 了
    """.split()
)


# 复用 embedding 客户端（避免每次请求新建连接开销；配置统一取自 settings）
_EMBEDDING_CLIENT = OpenAI(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
    timeout=settings.embedding_timeout,
)


def get_embedding(text: str) -> Optional[List[float]]:
    """
    调用本地 Ollama embedding 接口生成向量。
    失败或超时时返回 None（调用方据此降级为纯 BM25 召回，保证服务可用）。
    """
    if not text or not text.strip():
        return None
    try:
        resp = _EMBEDDING_CLIENT.embeddings.create(
            input=text[:MAX_QUERY_LEN], model=settings.embedding_model
        )
        return resp.data[0].embedding
    except Exception:
        return None


def tokenize(text: str) -> List[str]:
    """
    使用 jieba 分词，词作为整体 token。
    注意：不能对分词结果再按单字拆分——否则「连衣裙」会拆成「连」「衣」「裙」，
    导致单字在无关文本中大面积误命中，BM25 噪音剧增。
    """
    if not text:
        return []
    tokens: List[str] = []
    for seg in jieba.cut(text):
        seg = seg.strip()
        if not seg:
            continue
        # 纯标点/符号、停用词过滤掉；中英文实词保留
        if seg in STOPWORDS:
            continue
        if _TOKEN_RE.match(seg) or any("\u4e00" <= ch <= "\u9fff" for ch in seg):
            tokens.append(seg)
    return tokens


# 统一的召回结果结构
@dataclass
class ChunkHit:
    """一条召回的知识分块结果（双路分数 + 融合分数）。"""
    chunk_id: int
    doc_id: int
    chunk_index: int
    chunk_text: str
    doc_type: Optional[str] = None
    meta_data: Dict[str, Any] = field(default_factory=dict)
    vector_score: Optional[float] = None   # 向量相似度 [0,1]；未召回该路为 None
    bm25_score: Optional[float] = None     # BM25 原始分（非 [0,1]）
    score: float = 0.0                     # Rerank 后的最终综合得分

    @property
    def key(self) -> tuple:
        """去重键：同一文档分块在两路召回中只保留一条。"""
        return (self.doc_id, self.chunk_index)


# 语义向量召回
class VectorRetriever:
    """基于 pgvector 余弦距离的语义召回。"""

    def __init__(self, conn):
        self._conn = conn

    def search(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K,
        doc_type: Optional[str] = None,
        category_big: Optional[str] = None,
        category_small: Optional[str] = None,
        category_path: Optional[str] = None,
    ) -> List[ChunkHit]:
        vector = get_embedding(query)
        if vector is None:
            # embedding 服务不可用：向量路降级为空，由混合层走 BM25
            return []

        sql = """
            SELECT c.id, c.doc_id, c.chunk_index, c.chunk_text,
                   d.doc_type, c.meta_data,
                   1 - (c.chunk_vector <=> %s::vector) AS vector_score
            FROM kb_chunks c
            JOIN kb_documents d ON d.id = c.doc_id
            WHERE (%s::text IS NULL OR d.doc_type = %s)
        """
        params: List[Any] = [vector, doc_type, doc_type]

        # 类目级过滤（meta_data 冗余类目字段；无类目字段的 chunk（如 faq/policy）保留，
        # 有类目的按条件匹配——"手机"类目过滤排除其他类目商品，不误伤售后/政策知识）
        if category_big:
            sql += " AND (c.meta_data->>'category_big' IS NULL OR c.meta_data->>'category_big' = %s)"
            params.append(category_big)
        if category_small:
            sql += " AND (c.meta_data->>'category_small' IS NULL OR c.meta_data->>'category_small' = %s)"
            params.append(category_small)
        if category_path:
            sql += " AND (c.meta_data->>'category_path' IS NULL OR c.meta_data->>'category_path' LIKE %s)"
            params.append(category_path)

        sql += " ORDER BY c.chunk_vector <=> %s::vector LIMIT %s"
        params += [vector, top_k]

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        hits: List[ChunkHit] = []
        for r in rows:
            hits.append(
                ChunkHit(
                    chunk_id=r["id"],
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    chunk_text=r["chunk_text"],
                    doc_type=r["doc_type"],
                    meta_data=r["meta_data"] or {},
                    vector_score=float(r["vector_score"]),
                )
            )
        return hits


# 关键词 BM25 召回
class BM25Retriever:
    """
    jieba 分词 + BM25Okapi 关键词召回。
    BM25 索引（语料 + 分词结果）在进程内缓存，避免每个请求重建。
    """

    def __init__(self, conn):
        self._conn = conn

    def _load_corpus(self) -> List[Dict[str, Any]]:
        """从 kb_chunks 全量加载 (id, doc_id, chunk_index, chunk_text, doc_type, meta_data)。"""
        sql = """
            SELECT c.id, c.doc_id, c.chunk_index, c.chunk_text,
                   d.doc_type, c.meta_data
            FROM kb_chunks c
            JOIN kb_documents d ON d.id = c.doc_id
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            {
                "chunk_id": r["id"],
                "doc_id": r["doc_id"],
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "doc_type": r["doc_type"],
                "meta_data": r["meta_data"] or {},
            }
            for r in rows
        ]

    def _build_index(self) -> None:
        """构建并缓存 BM25 索引（进程内单例，加锁防并发重复构建）。

        缓存结构：_BM25_CACHE["index"] = (BM25Okapi, corpus) 整体原子赋值，
        读取方一次性取出，避免并发首建/refresh 时拿到不匹配的 (bm25, corpus) 对。
        """
        if _BM25_CACHE["index"] is not None:
            return
        with _BM25_LOCK:
            if _BM25_CACHE["index"] is not None:  # 双重检查
                return
            t0 = time.time()
            corpus = self._load_corpus()
            tokenized = [tokenize(c["chunk_text"]) for c in corpus]
            _BM25_CACHE["index"] = (BM25Okapi(tokenized), corpus)
            _BM25_CACHE["corpus_size"] = len(corpus)
            _BM25_CACHE["build_ms"] = int((time.time() - t0) * 1000)

    def refresh(self) -> None:
        """强制重建 BM25 索引（知识库数据更新后调用，否则新数据检索不到）。"""
        with _BM25_LOCK:
            _BM25_CACHE["index"] = None
            self._build_index()

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        doc_type: Optional[str] = None,
        category_big: Optional[str] = None,
        category_small: Optional[str] = None,
        category_path: Optional[str] = None,
    ) -> List[ChunkHit]:
        # 与 refresh() 并发时可能读到刚置 None 的缓存：重试一次构建后再读
        for _ in range(2):
            self._build_index()
            entry = _BM25_CACHE["index"]
            if entry is not None:
                break
        else:
            return []
        bm25: BM25Okapi
        corpus: List[Dict[str, Any]]
        bm25, corpus = entry
        if not corpus:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        def _cat_ok(meta: Dict[str, Any]) -> bool:
            """类目过滤：无类目字段的 chunk（如 faq/policy）保留，有类目的按条件匹配。"""
            if category_big and meta.get("category_big") and meta.get("category_big") != category_big:
                return False
            if category_small and meta.get("category_small") and meta.get("category_small") != category_small:
                return False
            if category_path and meta.get("category_path") and category_path not in meta["category_path"]:
                return False
            return True

        scores = bm25.get_scores(q_tokens)
        # 取分数 Top-N 的候选（含 doc_type / 类目过滤）
        ranked = sorted(
            (
                (scores[i], i)
                for i in range(len(corpus))
                if (doc_type is None or corpus[i]["doc_type"] == doc_type)
                and _cat_ok(corpus[i]["meta_data"])
            ),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        hits: List[ChunkHit] = []
        for raw_score, idx in ranked:
            if raw_score <= 0:
                continue
            c = corpus[idx]
            hits.append(
                ChunkHit(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    chunk_index=c["chunk_index"],
                    chunk_text=c["chunk_text"],
                    doc_type=c["doc_type"],
                    meta_data=c["meta_data"],
                    bm25_score=float(raw_score),
                )
            )
        return hits


# BM25 索引缓存（单进程单例；多 worker 部署时每进程各持一份）。
# index 为 (BM25Okapi, corpus) 整体原子赋值，配合 _BM25_LOCK 保证并发安全。
_BM25_CACHE: Dict[str, Any] = {
    "index": None,
    "corpus_size": 0,
    "build_ms": 0,
}
_BM25_LOCK = threading.RLock()


# 混合检索（双路召回 + 归一化融合）
class HybridRetriever:
    """
    双路召回入口：向量路 + BM25 路并行召回，合并去重后交给 Reranker。
    融合打分（加权、阈值过滤、Top-K）在 reranker.py 的 Reranker 中完成。
    """

    def __init__(
        self,
        conn,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        self._conn = conn
        self.vector_retriever = vector_retriever or VectorRetriever(conn)
        self.bm25_retriever = bm25_retriever or BM25Retriever(conn)

    def search(
        self,
        query: str,
        candidate_k: int = VECTOR_TOP_K,
        doc_type: Optional[str] = None,
        category_big: Optional[str] = None,
        category_small: Optional[str] = None,
        category_path: Optional[str] = None,
    ) -> List[ChunkHit]:
        """
        双路召回并合并（支持类目级过滤）。

        Args:
            query: 用户查询文本
            candidate_k: 每路召回候选数（Rerank 的输入池大小）
            doc_type: 按文档类型过滤（product_manual / faq / policy / None=全部）
            category_big / category_small / category_path:
                类目级过滤（meta_data 类目字段，如"手机"类目排除"手机壳"等无关内容）

        Returns:
            合并去重后的 ChunkHit 列表（score 字段暂为 0，由 Reranker 填充）
        """
        if not query or not query.strip():
            return []

        vec_hits = self.vector_retriever.search(
            query, top_k=candidate_k, doc_type=doc_type,
            category_big=category_big, category_small=category_small,
            category_path=category_path,
        )
        bm25_hits = self.bm25_retriever.search(
            query, top_k=candidate_k, doc_type=doc_type,
            category_big=category_big, category_small=category_small,
            category_path=category_path,
        )

        # 按 (doc_id, chunk_index) 合并去重，两路分数都保留
        merged: Dict[tuple, ChunkHit] = {}
        for hit in vec_hits:
            merged[hit.key] = hit
        for hit in bm25_hits:
            if hit.key in merged:
                merged[hit.key].bm25_score = hit.bm25_score
            else:
                merged[hit.key] = hit
        return list(merged.values())

    def stats(self) -> Dict[str, Any]:
        """BM25 索引构建信息（供调试/日志）。"""
        self.bm25_retriever._build_index()
        return {
            "bm25_corpus_size": _BM25_CACHE["corpus_size"],
            "bm25_build_ms": _BM25_CACHE["build_ms"],
        }
