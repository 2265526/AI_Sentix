"""
记忆数据访问层

短期/长期记忆：
  - session_context       会话上下文快照（短期记忆，30 分钟过期，turn_count 超限自动清空）
  - user_interaction_log  交互日志（预留，P0 起写入，供长期画像统计）

写入统一 conn.commit()，
读取用 RealDictCursor。
"""
import json
from typing import Any, Dict, Optional

import psycopg2.extras

# turn_count 超过该值后，下一轮 upsert 触发上下文清空（context 归空、计数归 1）
MAX_TURNS = 10

# 会话有效期：expires_at = last_active_at + 30 分钟
SESSION_TTL = "30 minutes"


def get_session_context(conn, session_id: str) -> Optional[Dict[str, Any]]:
    """
    读取未过期会话的上下文快照。

    Returns:
        {"session_id", "context": dict, "turn_count": int}；无记录或已过期返回 None。
    """
    sql = """
        SELECT context, turn_count
        FROM session_context
        WHERE session_id = %s AND expires_at > NOW()
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (session_id,))
            row = cur.fetchone()
    except Exception:
        # 关键：SELECT 失败必须回滚，否则 PostgreSQL 事务进入 aborted 状态，
        # 同一连接后续所有 SQL（含工具查询）都会报 "current transaction is aborted"
        conn.rollback()
        raise
    if not row:
        return None
    return {
        "session_id": session_id,
        "context": row["context"],
        "turn_count": row["turn_count"],
    }


def upsert_session_context(conn, session_id: str, context: Dict[str, Any]) -> None:
    """
    写入会话上下文（幂等 upsert）：
      - 新会话：turn_count 置 1；
      - 已有会话：context 全量覆盖、turn_count+1、刷新过期时间；
      - turn_count 超过 MAX_TURNS 时重置：context 清空、turn_count 归 1。
    """
    sql = """
        INSERT INTO session_context (session_id, context, turn_count, expires_at, last_active_at)
        VALUES (%s, %s, 1, NOW() + %s::interval, NOW())
        ON CONFLICT (session_id) DO UPDATE SET
            context = CASE WHEN session_context.turn_count >= %s
                           THEN '{}'::jsonb ELSE EXCLUDED.context END,
            turn_count = CASE WHEN session_context.turn_count >= %s
                              THEN 1 ELSE session_context.turn_count + 1 END,
            expires_at = NOW() + %s::interval,
            last_active_at = NOW()
    """
    cur = conn.cursor()
    try:
        cur.execute(
            sql,
            (
                session_id,
                json.dumps(context, ensure_ascii=False),
                SESSION_TTL,
                MAX_TURNS,
                MAX_TURNS,
                SESSION_TTL,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def clear_session_context(conn, session_id: str) -> None:
    """删除指定会话的上下文（切换意图时清空，开启新记忆周期）。"""
    sql = "DELETE FROM session_context WHERE session_id = %s"
    cur = conn.cursor()
    try:
        cur.execute(sql, (session_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def log_interaction(
    conn,
    user_id: Optional[str],
    session_id: Optional[str],
    query: str,
    enhanced_query: Optional[str],
    tool_called: Optional[str],
    result_count: Optional[int],
    entities: Dict[str, Any],
) -> None:
    """写入一条交互日志（user_interaction_log，预留表 P0 起写入）。"""
    sql = """
        INSERT INTO user_interaction_log
            (user_id, session_id, query, enhanced_query, tool_called, result_count, entities)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cur = conn.cursor()
    try:
        cur.execute(
            sql,
            (
                user_id,
                session_id,
                query,
                enhanced_query,
                tool_called,
                result_count,
                json.dumps(entities, ensure_ascii=False),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def delete_expired(conn) -> int:
    """手动清理过期会话（等价于 pg_cron 每小时任务），返回删除条数。"""
    sql = "DELETE FROM session_context WHERE expires_at < NOW()"
    cur = conn.cursor()
    try:
        cur.execute(sql)
        deleted = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
    return deleted or 0
