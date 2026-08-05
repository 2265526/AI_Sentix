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

-- 2.1 商品基础表
DROP TABLE IF EXISTS product_catalog CASCADE;
CREATE TABLE product_catalog (
    id              BIGSERIAL PRIMARY KEY,
    sku_code        VARCHAR(50)  NOT NULL,
    product_name    VARCHAR(255) NOT NULL,
    category_id     INT,
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

-- 加速按 SKU 检索商品（手册：idx_product_sku，B-Tree）
CREATE INDEX idx_product_sku ON product_catalog (sku_code);

-- 加速商品关联库存查询（手册：idx_inventory_product，B-Tree）
CREATE INDEX idx_inventory_product ON inventory_logistics (product_id);

-- 加速按文档 ID 查询分块（手册：idx_chunks_doc_id，B-Tree）
CREATE INDEX idx_chunks_doc_id ON kb_chunks (doc_id);

-- 加速按分块序号排序（手册：idx_chunks_index，B-Tree）
CREATE INDEX idx_chunks_index ON kb_chunks (chunk_index);

-- 向量相似度搜索（手册：idx_chunks_vector_hnsw，HNSW 余弦距离，vector_cosine_ops）
CREATE INDEX idx_chunks_vector_hnsw ON kb_chunks
    USING hnsw (chunk_vector vector_cosine_ops);

-- ============================================================
-- 字段注释（与手册「注释说明」列一致）
-- ============================================================
COMMENT ON COLUMN product_catalog.id              IS '主键ID（自增）';
COMMENT ON COLUMN product_catalog.sku_code        IS '商品SKU编码（唯一标识，如 IP15PM256）';
COMMENT ON COLUMN product_catalog.product_name    IS '商品名称';
COMMENT ON COLUMN product_catalog.category_id     IS '商品类目ID（关联外部类目系统）';
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
