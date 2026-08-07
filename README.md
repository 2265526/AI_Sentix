# AI_Sentix
Al_Sentix-—电商系统AI智能客服（含语音链路）

为电商平台提供智能化客服，减少人工投入，支持文字/语音问询、语音对话，精准意图识别、工具调用、知识库召回、流式输出

技术实现：自研RAG框架（chunking+hybrid retrieval + rerank)+DeepSeek API集成+ASR（语音识别)→ LLM→ TTS（语 音合成）完整多模态链路+错误fallback机制+FastAPI服务化。

## 🧰 技术栈

| 层级 | 技术 |
|---|---|
| 后端框架 | Python 3.12 · **FastAPI** · uvicorn |
| 数据库 | **PostgreSQL**（关系 + 向量同库）· **pgvector** 扩展（HNSW 索引，1024 维） |
| 数据库驱动 | psycopg2（连接池 `SimpleConnectionPool`） |
| Embedding | 本地 **Ollama** 的 `qwen3-embedding:0.6b`（1024 维，OpenAI 兼容接口） |
| 大模型 | **DeepSeek**（`deepseek-chat`，openai SDK 兼容调用，function calling / 流式） |
| 关键词检索 | **jieba** 分词 + **rank_bm25**（BM25Okapi） |
| 语义检索 | pgvector 余弦距离（`<=>`）+ HNSW 索引 |
| 文本处理 | langchain `RecursiveCharacterTextSplitter`（分块）· pypdf（PDF）· python-docx（DOCX） |
| 前端 | **React 18** · **Vite 5** · **Ant Design 5** · axios（SSE 用 fetch ReadableStream） |
| 语音识别（ASR） | **faster-whisper**（本地离线small 模型，PyAV 解码 wav/mp3/ogg/webm） |
| 语音合成（TTS） | **edge-tts**（微软在线接口，服务端调用；中文音色晓晓/云希/云健等） |

| 自研RAG 框架（召回 → 融合 → Rerank → 上下文组装），使用 langchain 的文本分块工具；向量检索基于 pgvector 原生 SQL。


## ✨ 版本更新

## 当前最新版本为 `V2.1.0`。

<details>
<summary>🏗️ V1.x.x 系列（点击展开 / 收起）</summary>

<!-- ================= V1.0.0 ================= -->
<details>
<summary>🏷️ V1.0.0 — 基础版本（点击展开）</summary>

#### ✨ 功能特性
##### 1. RAG 知识库检索（混合检索 + 重排序）
- **双路召回**：BM25 关键词检索（jieba 分词）+ 语义向量检索（pgvector 余弦距离，HNSW 索引）
- **重排序**：双路分数加权融合，相关性阈值 0.4 过滤，Top-5 输出

##### 2. Agent 智能对话（意图识别 + 工具路由 + 流式回复）
- **意图识别**：DeepSeek function calling 判断用户意图，自动容忍模型输出的参数键名波动
- **4 个工具**：
  | 工具 | 用途 | 检索方式 |
  |---|---|---|
  | `get_product_inventory` | 查库存/物流时效 | SQL（精确） |
  | `get_product_price` | 查商品价格 | SQL（精确） |
  | `get_knowledge_base` | 查售后/使用说明/FAQ | RAG（语义） |
  | `product_recommendation` | 商品推荐 | RAG（语义） |
- **二次模型回调**：工具结果 + 用户问题重新组装消息 → DeepSeek 流式生成最终回复（SSE）

##### 3. 管理员功能（Web 页面，无需登录）
- **知识库文档管理**：上传 TXT / PDF，自动分块 + 向量化入库，上传后立即可被检索
- **商品数据同步**：上传 CSV，增量同步商品/库存，并**同步为每个商品生成知识库向量

##### 4. Web 验证台（React + Ant Design）
- 客服聊天页（SSE 流式展示）、RAG 检索验证页（Top-K/阈值/类型过滤）、管理员页

</details>

<!-- ================= V1.1.0 ================= -->
<details>
<summary>🏷️ V1.1.0 — 功能修复与增强（点击展开）</summary>

#### 🔧 更新内容
- **商品 CSV 导入同步生成知识库向量**：结构化入库的同时为每个商品生成 `product_manual` 知识文档（商品名 + 描述 → 分块 → 向量化），上传后立即可被 RAG 检索
- **修复「未收录」误判**：意图识别工具参数键名波动（`query` / `category` / `product_type` 等）导致检索词缺失 → router 参数别名扩展 + intent 层自动注入用户原话兜底，检索不再落空
- **修复前端聊天流式 token 重复与换行折叠**：移除 React StrictMode 对 setState updater 的双调用、消息渲染保留换行（`white-space: pre-wrap`）
- **管理接口增强**：CORS 支持、知识库上传后自动重建 BM25 索引、`.gitignore` 密钥防护

</details>

<!-- ================= V1.2.0 ================= -->
<details>
<summary>🏷️ V1.2.0 — 分级类目体系（点击展开）</summary>

#### 🔧 更新内容
- **新增分级类目体系**：14 大类 × 103 中类 × 526 小类三级类目，新增 `category` 表
- **入库自动分类**：商品导入时按中文关键词自动归类；商品知识库向量同步标注类目
- **商品表新增类目字段**：`category_big` / `category_small` / `category_path`；`kb_chunks.meta_data` 同步类目字段

</details>

<!-- ================= V1.2.5 ================= -->
<details>
<summary>🏷️ V1.2.5 — 检索分级过滤（点击展开）</summary>

#### 🔧 更新内容
- **删除外部平台类目 ID 属性列**：`product_catalog.category_id` 列移除（含导入、检索、建表 SQL 全部读写逻辑）
- **meta_data 调整**：`kb_chunks.meta_data` 取消 `category_id`、`product_ids` 字段，类目过滤改由 `category_big` / `category_small` / `category_path` 承担
- **产品信息检索分级过滤**：按商品信息表字段逐级收紧（SKU → 名称 → 大类 → 小类 → 类目路径 → 价格区间 → 库存状态），条件参数化防注入
- **新增类目过滤索引**：`idx_product_category_big` / `idx_product_category_small` / `idx_product_category_path`

</details>

</details>

<!-- ================= V2.x.x 系列 ================= -->
<details open>
<summary>🏗️ V2.x.x 系列（点击展开 / 收起）</summary>

<!-- ================= V2.0.0 ================= -->
<details>
<summary>🏷️ V2.0.0 — 多模态语音链路（点击展开）</summary>

#### 🔧 更新内容
- **语音链路**：音频 → ASR → LLM → TTS 全链路
- **ASR**：本地 `faster-whisper`（small，单例缓存）
- **TTS**：`edge-tts`（微软在线接口，服务端调用，中文音色晓晓/云希等）
- **新接口**：`POST /v1/chat/audio`（上传录音 → 返回 mp3 音频流，识别文本/回复经响应头返回）
- **前端**：聊天页新增「语音提问」按钮（录音 → 识别 → 播放回复语音）

</details>

<!-- ================= V2.0.1 ================= -->
<details>
<summary>🏷️ V2.0.1 — 语音体验优化与加载修复（点击展开）</summary>

#### 🔧 更新内容
- **TTS 剔除 emoji**：合成前清除表情符号，不再读出「嘴角含笑」等表情描述
- **录音音量条**：语音提问时实时显示麦克风音量（微信式竖条），结束录音自动消失
- **前端优化**：调整了导航区和聊天框的位置及大小、调整消息字体大小，调整麦克风图标的位置
- **Whisper 加载优化**：模型改本地路径 `~/whisper-small` 加载，跳过 HuggingFace 联网校验（加载由数分钟降至秒级）；

</details>

<!-- ================= V2.1.0 ================= -->
<details>
<summary>🏷️ V2.1.0 — 检索精准度与稳定性优化（点击展开）</summary>

#### 🔧 更新内容
- **商品检索返回总数**：`search_products` 用 `COUNT(*) OVER()` 返回匹配总数，`format_products` 展示「共找到 N 件（展示前 M 件）」，避免 LLM 把分页结果误认为全部
- **推荐 LIMIT 5→10**：单次推荐最多展示 10 件
- **推荐降级保护品牌词**：新增 `_brand_words` 提取品牌/型号词（如 iphone），降级时按「品牌词 + 类目词」组合过滤，不再被类目词覆盖丢弃品牌
- **RAG 类目级过滤**：Vector / BM25 / Hybrid 检索器支持 `category_big` / `category_small` / `category_path` 过滤；`/rag/search` 新增类目参数；意图层（get_knowledge_base）自动提取查询类目词过滤；faq/policy 等无类目分块不受影响
- **类目过滤索引**：`kb_chunks` 新增 `meta_data` 类目表达式索引

</details>

<!-- ================= V2.2.0 ================= -->
<details>
<summary>🏷️ V2.2.0 — 短期记忆与长期记忆（点击展开）</summary>
(长期记忆预留，视后续具体业务环境开发)

#### 🔧 更新内容
- **新增记忆仓储**：会话上下文读写、超轮次自动清空（turn_count > 10）、TTL 过期处理
- **新增问题增强**：4 条规则（切换检测 / 指代消解 / 品牌品类补全 / 价格补全），输出 `enhanced_query` 再进入意图识别
- **数据表与索引**：三张记忆表 + GIN / HNSW 索引

</details>

</details>

---

