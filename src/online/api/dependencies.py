"""
FastAPI 依赖：数据库连接、RAG 引擎实例。
"""
from fastapi import Depends
from psycopg2.extensions import connection

from src.online.core.rag.retriever import HybridRetriever
from src.online.db.session import get_db


def get_rag_engine(conn: connection = Depends(get_db)):
    """
    RAG 检索引擎：混合检索器（BM25 + 向量双路召回）。
    连接来自请求级依赖；BM25 索引在 retriever 内部按进程缓存，不随请求重建。
    """
    return {"retriever": HybridRetriever(conn)}
