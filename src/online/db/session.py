# -*- coding: utf-8 -*-
"""
db/session.py —— 数据库会话管理
================================
阶段二 RAG 引擎的数据库连接层：
  - 从 .env 读取 DATABASE_URL（postgresql://...），未配置直接报错（不内置明文密码）；
  - 使用 psycopg2 连接池（SimpleConnectionPool），避免每个请求新建连接的
    开销与 max_connections 风险；
  - register_vector() 全局注册 pgvector 适配器（池内所有连接自动支持 VECTOR 类型）；
  - 提供 FastAPI 依赖 get_db()：按请求借出连接、请求结束归还连接池。

用法（FastAPI 依赖注入）：
    @router.post("/search")
    def search(req: RAGSearchRequest, conn=Depends(get_db)):
        ...
"""
import threading

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector

from config.settings import settings
from src.common.exceptions import ConfigError

DATABASE_URL = settings.database_url
if not DATABASE_URL:
    raise ConfigError(
        "未配置 DATABASE_URL 环境变量（请检查 .env）："
        "postgresql://user:password@host:port/dbname"
    )

# pgvector 类型适配器：pgvector 0.5.x 的 register_vector 需按连接注册，
# 在借出连接时调用（幂等；池内连接对象复用，注册一次持续有效）
_POOL_SIZE = settings.db_pool_size
_pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=_POOL_SIZE, dsn=DATABASE_URL)

# 信号量：限制同时借出的连接数并支持获取超时（池满时拿不到即报错，
# 避免 psycopg2 SimpleConnectionPool.getconn 在池满时无限阻塞挂死）
_semaphore = threading.BoundedSemaphore(_POOL_SIZE)
_POOL_TIMEOUT = settings.db_pool_timeout


def _register(conn) -> None:
    """为连接注册 pgvector 适配器（幂等）。"""
    register_vector(conn)


def _acquire():
    """从池中借出连接：信号量限流 + 超时保护。"""
    if not _semaphore.acquire(timeout=_POOL_TIMEOUT):
        raise RuntimeError("数据库连接池繁忙，请稍后重试")
    try:
        conn = _pool.getconn()
    except Exception:
        _semaphore.release()
        raise
    return conn


def get_connection():
    """从连接池借出一个连接（用完必须归还；register 失败时归还避免泄漏）。"""
    conn = _acquire()
    try:
        _register(conn)
        return conn
    except Exception:
        _pool.putconn(conn)
        _semaphore.release()
        raise


def get_db():
    """
    FastAPI 依赖：从连接池借出连接，请求结束归还。
    使用示例见模块 docstring。
    """
    conn = _acquire()
    try:
        _register(conn)
        yield conn
    finally:
        _pool.putconn(conn)
        _semaphore.release()
