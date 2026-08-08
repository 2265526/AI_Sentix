"""
全局统一配置
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    """读取环境变量。"""
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    """全部可配置项。"""

    # ---- LLM（DeepSeek）----
    deepseek_api_key: str          # DEEPSEEK_API_KEY
    deepseek_base_url: str         # DEEPSEEK_BASE_URL
    deepseek_model: str            # DEEPSEEK_MODEL
    deepseek_timeout: float        # DEEPSEEK_TIMEOUT（秒，连接超时）

    # ---- Embedding（本地 Ollama qwen3-embedding）----
    embedding_base_url: str        # EMBEDDING_BASE_URL
    embedding_api_key: str         # EMBEDDING_API_KEY
    embedding_model: str           # EMBEDDING_MODEL
    embedding_dim: int             # EMBEDDING_DIM（向量维度）
    embedding_timeout: float       # EMBEDDING_TIMEOUT（秒）,网络请求超时控制参数

    # ---- 数据库（PostgreSQL + pgvector）----
    database_url: str              # DATABASE_URL
    db_pool_size: int              # DB_POOL_SIZE（连接池上限）
    db_pool_timeout: float         # DB_POOL_TIMEOUT（连接超时，秒）

    # ---- 语音 ASR / TTS ----
    asr_device: str                # ASR_DEVICE（空=自动探测）
    whisper_model_path: str        # WHISPER_MODEL_PATH（本地模型目录，空=自动）
    whisper_model_size: str        # WHISPER_MODEL_SIZE（默认 small）
    tts_voice: str                 # TTS_VOICE（默认 zh-CN-XiaoxiaoNeural）

    # ---- 应用 ----
    audio_debug_dir: str           # AUDIO_DEBUG_DIR（调试音频保存目录，空=不保存）

    # ---- RAG 检索参数 ----
    vector_top_k: int              # VECTOR_TOP_K（向量路每路召回数）
    bm25_top_k: int                # BM25_TOP_K（BM25 路每路召回数）
    max_query_len: int             # MAX_QUERY_LEN（查询文本截断长度）

    # ---- 离线 ETL ----
    etl_chunk_size: int            # ETL_CHUNK_SIZE（知识文档分块大小）
    etl_chunk_overlap: int         # ETL_CHUNK_OVERLAP（分块重叠）

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置（未配置项使用默认值）。"""
        return cls(
            # LLM
            deepseek_api_key=_get("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_timeout=float(_get("DEEPSEEK_TIMEOUT", "60")),
            # Embedding
            embedding_base_url=_get("EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
            embedding_api_key=_get("EMBEDDING_API_KEY", "ollama"),
            embedding_model=_get("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            embedding_dim=int(_get("EMBEDDING_DIM", "1024")),
            embedding_timeout=float(_get("EMBEDDING_TIMEOUT", "5")),
            # 数据库
            database_url=_get("DATABASE_URL", ""),
            db_pool_size=int(_get("DB_POOL_SIZE", "10")),
            db_pool_timeout=float(_get("DB_POOL_TIMEOUT", "5")),
            # 语音
            asr_device=_get("ASR_DEVICE", ""),
            whisper_model_path=_get("WHISPER_MODEL_PATH", ""),
            whisper_model_size=_get("WHISPER_MODEL_SIZE", "small"),
            tts_voice=_get("TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
            # 应用
            audio_debug_dir=_get("AUDIO_DEBUG_DIR", ""),
            # RAG
            vector_top_k=int(_get("VECTOR_TOP_K", "30")),
            bm25_top_k=int(_get("BM25_TOP_K", "30")),
            max_query_len=int(_get("MAX_QUERY_LEN", "512")),
            # ETL
            etl_chunk_size=int(_get("ETL_CHUNK_SIZE", "500")),
            etl_chunk_overlap=int(_get("ETL_CHUNK_OVERLAP", "50")),
        )


# 全局共享同一份配置
settings = Settings.from_env()
