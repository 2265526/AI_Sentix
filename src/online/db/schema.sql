-- ============================================================
-- 电商AI智能客服系统 —— 数据库 Schema（与《数据库设计手册 V1.0》对应）
-- ============================================================
-- 数据库：PostgreSQL（含 pgvector 扩展）
-- 说明：本脚本为全新初始化脚本。执行会删除并重建以下 4 张表（数据会丢失）：
--       product_catalog / inventory_logistics / kb_documents / kb_chunks
-- 用法：psql "$DATABASE_URL" -f schema.sql
-- ============================================================

-- pgvector 扩展（VECTOR(1024) 类型依赖）
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. 结构化业务表（精准检索）
-- ============================================================

-- 2.0 分级类目表（大类 → 中类 → 小类 三级树）
DROP TABLE IF EXISTS category CASCADE;
CREATE TABLE category (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,          -- 类目名称（中文）
    parent_id   BIGINT,                         -- 父类目（NULL = 一级大类）
    level       INT          NOT NULL,          -- 层级：1 大类 / 2 中类 / 3 小类
    path        VARCHAR(200),                   -- 完整路径（如：服装鞋包/女装/衬衫）
    external_id INT,                            -- 外部平台类目 ID（Shopee，可空）
    created_at  TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT category_parent_fkey
        FOREIGN KEY (parent_id) REFERENCES category (id) ON DELETE CASCADE
);

-- 2.1 商品基础表
DROP TABLE IF EXISTS product_catalog CASCADE;
CREATE TABLE product_catalog (
    id              BIGSERIAL PRIMARY KEY,
    sku_code        VARCHAR(50)  NOT NULL,
    product_name    VARCHAR(255) NOT NULL,
    category_big    VARCHAR(50),                -- 大类（中文，如：服装鞋包）
    category_small  VARCHAR(50),                -- 小类（中文，如：衬衫）
    category_path   VARCHAR(200),               -- 完整类目路径（如：服装鞋包/女装/衬衫）
    price           DECIMAL(15, 2),
    raw_description TEXT,
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- sku_code 唯一约束（手册：UNIQUE NOT NULL，商品SKU唯一标识）
CREATE UNIQUE INDEX product_catalog_sku_code_key ON product_catalog (sku_code);

-- 2.2 库存与物流信息表（与商品一对一，product_id 唯一）
DROP TABLE IF EXISTS inventory_logistics CASCADE;
CREATE TABLE inventory_logistics (
    id                     BIGSERIAL PRIMARY KEY,
    product_id             BIGINT        NOT NULL,
    stock_quantity         INT           DEFAULT 0,
    warehouse_location     VARCHAR(100),
    delivery_estimate_days INT,
    created_at             TIMESTAMP     DEFAULT NOW(),
    CONSTRAINT inventory_logistics_product_id_fkey
        FOREIGN KEY (product_id) REFERENCES product_catalog (id) ON DELETE CASCADE
);

-- 唯一约束：确保每个商品只有一条库存记录（强制一对一）
CREATE UNIQUE INDEX unique_product_id ON inventory_logistics (product_id);

-- ============================================================
-- 3. RAG 知识库向量表（非结构化检索）
-- ============================================================

-- 3.1 知识库原始文档表
DROP TABLE IF EXISTS kb_documents CASCADE;
CREATE TABLE kb_documents (
    id          BIGSERIAL PRIMARY KEY,
    doc_type    VARCHAR(50) NOT NULL,
    source_url  TEXT,
    raw_content TEXT        NOT NULL,
    created_at  TIMESTAMP   DEFAULT NOW()
);

-- 3.2 知识库向量分块表（核心表，1024 维向量）
DROP TABLE IF EXISTS kb_chunks CASCADE;
CREATE TABLE kb_chunks (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       BIGINT       NOT NULL,
    chunk_index  INT          NOT NULL,
    chunk_text   TEXT         NOT NULL,
    chunk_vector VECTOR(1024),
    meta_data    JSONB,
    created_at   TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT kb_chunks_doc_id_fkey
        FOREIGN KEY (doc_id) REFERENCES kb_documents (id) ON DELETE CASCADE
);

-- ============================================================
-- 4. 索引策略（性能优化）
-- ============================================================

-- 类目树查询（按父类目/名称）
CREATE INDEX idx_category_parent ON category (parent_id);
CREATE INDEX idx_category_name ON category (name);
-- 加速按类目路径查询类目树（类目扩充后 643 节点）
CREATE INDEX idx_category_path ON category (path);

-- 加速按 SKU 检索商品（手册：idx_product_sku，B-Tree）
CREATE INDEX idx_product_sku ON product_catalog (sku_code);

-- 加速按类目分级过滤商品（手册：idx_product_category_*，B-Tree）
CREATE INDEX idx_product_category_big   ON product_catalog (category_big);
CREATE INDEX idx_product_category_small ON product_catalog (category_small);
CREATE INDEX idx_product_category_path  ON product_catalog (category_path);

-- 加速商品关联库存查询（手册：idx_inventory_product，B-Tree）
CREATE INDEX idx_inventory_product ON inventory_logistics (product_id);

-- 加速按文档 ID 查询分块（手册：idx_chunks_doc_id，B-Tree）
CREATE INDEX idx_chunks_doc_id ON kb_chunks (doc_id);

-- 加速按分块序号排序（手册：idx_chunks_index，B-Tree）
CREATE INDEX idx_chunks_index ON kb_chunks (chunk_index);

-- 向量相似度搜索（手册：idx_chunks_vector_hnsw，HNSW 余弦距离，vector_cosine_ops）
CREATE INDEX idx_chunks_vector_hnsw ON kb_chunks
    USING hnsw (chunk_vector vector_cosine_ops);
-- 加速按类目过滤知识分块（V2.1.0：meta_data 冗余类目字段的表达式索引）
CREATE INDEX idx_chunks_category_big ON kb_chunks ((meta_data->>'category_big'));
CREATE INDEX idx_chunks_category_small ON kb_chunks ((meta_data->>'category_small'));
CREATE INDEX idx_chunks_category_path ON kb_chunks ((meta_data->>'category_path'));

-- ============================================================
-- 字段注释（与手册「注释说明」列一致）
-- ============================================================
COMMENT ON COLUMN product_catalog.id              IS '主键ID（自增）';
COMMENT ON COLUMN product_catalog.sku_code        IS '商品SKU编码（唯一标识，如 IP15PM256）';
COMMENT ON COLUMN product_catalog.product_name    IS '商品名称';
COMMENT ON COLUMN product_catalog.price           IS '商品售价（单位：元，保留两位小数）';
COMMENT ON COLUMN product_catalog.raw_description IS '原始商品描述文本（用于后续提取关键词或生成向量）';
COMMENT ON COLUMN product_catalog.created_at      IS '记录创建时间';

COMMENT ON COLUMN inventory_logistics.product_id             IS '关联商品ID（外键指向 product_catalog.id，级联删除）';
COMMENT ON COLUMN inventory_logistics.stock_quantity         IS '当前库存数量';
COMMENT ON COLUMN inventory_logistics.warehouse_location     IS '仓库位置（如：上海仓、广州仓）';
COMMENT ON COLUMN inventory_logistics.delivery_estimate_days IS '预计送达天数（如 2 代表两天送达）';
COMMENT ON COLUMN inventory_logistics.created_at             IS '记录创建时间';

COMMENT ON COLUMN kb_documents.doc_type    IS '文档类型标签（return_policy 售后政策 / product_manual 说明书 / faq 常见问题）';
COMMENT ON COLUMN kb_documents.source_url  IS '文档来源URL（用于追溯版权或更新源）';
COMMENT ON COLUMN kb_documents.raw_content IS '原始全文内容（未经切分的完整文本）';
COMMENT ON COLUMN kb_documents.created_at  IS '记录创建时间';

COMMENT ON COLUMN kb_chunks.doc_id       IS '关联原始文档ID（外键指向 kb_documents.id，级联删除）';
COMMENT ON COLUMN kb_chunks.chunk_index  IS '分块序号（从 0 开始，用于还原顺序或展示上下文）';
COMMENT ON COLUMN kb_chunks.chunk_text   IS '分块后的纯文本内容（用于向量化及展示给大模型）';
COMMENT ON COLUMN kb_chunks.chunk_vector IS '文本向量（1024维），支持余弦距离（<=>）、内积（<#>）、L2距离（<->）等向量运算';
COMMENT ON COLUMN kb_chunks.meta_data    IS '元数据标签，用于 Rerank 阶段过滤，标准格式见《数据库设计手册》5. 数据字典';
COMMENT ON COLUMN kb_chunks.created_at   IS '记录创建时间';

-- ============================================================
-- V2.2.0 短期/长期记忆（会话上下文 + 用户画像 + 交互日志）
-- ============================================================

-- 5.1 会话上下文表（短期记忆：会话维度 JSONB 快照，30 分钟过期）
DROP TABLE IF EXISTS session_context CASCADE;
CREATE TABLE session_context (
    session_id     VARCHAR(64) PRIMARY KEY,
    user_id        VARCHAR(64),
    context        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    turn_count     INT         NOT NULL DEFAULT 0,
    expires_at     TIMESTAMP   NOT NULL,
    last_active_at TIMESTAMP   DEFAULT NOW(),
    created_at     TIMESTAMP   DEFAULT NOW()
);

-- 过期清理 / 按用户查询 / 上下文 GIN 索引
CREATE INDEX idx_session_context_expires ON session_context (expires_at);
CREATE INDEX idx_session_context_user ON session_context (user_id, last_active_at);
CREATE INDEX idx_session_context_ctx ON session_context USING GIN (context);

-- 5.2 用户长期画像表（预留：长期记忆，后续由画像任务回填）
DROP TABLE IF EXISTS user_long_term_profile CASCADE;
CREATE TABLE user_long_term_profile (
    user_id              VARCHAR(64) PRIMARY KEY,
    preferred_brands     TEXT[],
    preferred_categories TEXT[],
    price_sensitivity    VARCHAR(20) CHECK (price_sensitivity IN ('low','medium','high')),
    price_stats          JSONB,
    frequent_skus        TEXT[],
    summary              TEXT,
    profile_embedding    VECTOR(1024),
    total_interactions   INT         NOT NULL DEFAULT 0,
    last_active_at       TIMESTAMP,
    created_at           TIMESTAMP   DEFAULT NOW(),
    updated_at           TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX idx_profile_brands ON user_long_term_profile USING GIN (preferred_brands);
CREATE INDEX idx_profile_categories ON user_long_term_profile USING GIN (preferred_categories);
CREATE INDEX idx_profile_embedding ON user_long_term_profile
    USING hnsw (profile_embedding vector_cosine_ops);
CREATE INDEX idx_profile_last_active ON user_long_term_profile (last_active_at);

-- 5.3 用户交互日志表（预留：P0 起写入，供画像统计）
DROP TABLE IF EXISTS user_interaction_log CASCADE;
CREATE TABLE user_interaction_log (
    id             BIGSERIAL PRIMARY KEY,
    user_id        VARCHAR(64),
    session_id     VARCHAR(64),
    query          TEXT,
    enhanced_query TEXT,
    tool_called    VARCHAR(50),
    result_count   INT,
    entities       JSONB,
    created_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_interaction_user_time ON user_interaction_log (user_id, created_at DESC);
CREATE INDEX idx_interaction_tool ON user_interaction_log (tool_called);

-- 字段注释（与《数据库设计手册》「注释说明」列一致）
COMMENT ON COLUMN session_context.session_id     IS '会话ID（VARCHAR(64)，主键）';
COMMENT ON COLUMN session_context.user_id        IS '用户ID（可空，未登录场景为空；预留不设外键）';
COMMENT ON COLUMN session_context.context        IS '上下文快照（JSONB）';
COMMENT ON COLUMN session_context.turn_count     IS '累计对话轮数，超过10触发自动清空';
COMMENT ON COLUMN session_context.expires_at     IS '过期时间=last_active_at+30分钟';
COMMENT ON COLUMN session_context.last_active_at IS '最后活跃时间（每次对话自动刷新）';
COMMENT ON COLUMN session_context.created_at     IS '记录创建时间';

COMMENT ON COLUMN user_long_term_profile.user_id              IS '用户ID（主键；预留，不设外键松耦合）';
COMMENT ON COLUMN user_long_term_profile.preferred_brands     IS '偏好品牌（TEXT[]，如 {iPhone,华为}）';
COMMENT ON COLUMN user_long_term_profile.preferred_categories IS '偏好类目（TEXT[]）';
COMMENT ON COLUMN user_long_term_profile.price_sensitivity    IS '价格敏感度（low/medium/high）';
COMMENT ON COLUMN user_long_term_profile.price_stats          IS '价格统计信息（JSONB，如 {avg,min,max,samples}）';
COMMENT ON COLUMN user_long_term_profile.frequent_skus        IS '高频商品SKU（TEXT[]）';
COMMENT ON COLUMN user_long_term_profile.summary              IS '用户画像摘要（文本，LLM 生成）';
COMMENT ON COLUMN user_long_term_profile.profile_embedding    IS '画像向量（1024维，须与kb_chunks同模型）';
COMMENT ON COLUMN user_long_term_profile.total_interactions   IS '累计交互次数';
COMMENT ON COLUMN user_long_term_profile.last_active_at       IS '最后活跃时间';
COMMENT ON COLUMN user_long_term_profile.created_at           IS '记录创建时间';
COMMENT ON COLUMN user_long_term_profile.updated_at           IS '记录更新时间（聚合/抽取时刷新）';

COMMENT ON COLUMN user_interaction_log.id             IS '主键ID（自增）';
COMMENT ON COLUMN user_interaction_log.user_id        IS '用户ID（未登录时与 session_id 相同）';
COMMENT ON COLUMN user_interaction_log.session_id     IS '会话ID（关联 session_context.session_id）';
COMMENT ON COLUMN user_interaction_log.query          IS '用户原始问题';
COMMENT ON COLUMN user_interaction_log.enhanced_query IS '增强后的问题（无增强时为原句）';
COMMENT ON COLUMN user_interaction_log.tool_called    IS '调用的工具名（无工具调用为 NULL）';
COMMENT ON COLUMN user_interaction_log.result_count   IS '工具返回结果条数';
COMMENT ON COLUMN user_interaction_log.entities       IS '抽取的实体快照（JSONB）';
COMMENT ON COLUMN user_interaction_log.created_at     IS '记录创建时间';

-- 过期会话自动清理（pg_cron，每小时执行一次；pg_cron 为 PostgreSQL 官方扩展，未安装时可改用
-- memory_repo.delete_expired() 手动清理）
CREATE EXTENSION IF NOT EXISTS pg_cron;
SELECT cron.schedule('cleanup_session_context','0 * * * *', $$DELETE FROM session_context WHERE expires_at < NOW()$$);
