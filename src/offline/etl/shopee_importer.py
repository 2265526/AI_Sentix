import os
import re
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List
import json
import time

load_dotenv()

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATABASE_URL = os.getenv("DATABASE_URL")
DATA_FILE = "/home/cai/t_data/AI_Sentix/src/offline/data/shopee-products.csv"  # 你的数据集文件名
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIM = 1024

# ★ 新增：翻译相关配置
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "deepseek-chat")  # 用于翻译的对话模型
TRANSLATE_CACHE_FILE = os.getenv("TRANSLATE_CACHE_FILE", "translation_cache.json")  # 翻译缓存，避免重复调用
# 当文本中中文字符占比 >= 该阈值时，认为已是中文，跳过翻译
CHINESE_RATIO_THRESHOLD = 0.5
# 批量翻译：每批多少条文本合并成一次 API 调用
BATCH_TRANSLATE_SIZE = int(os.getenv("BATCH_TRANSLATE_SIZE", "20"))

# ============================================================
# 初始化
# ============================================================
# 用于非 embedding 的客户端（这里也复用它来做翻译）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 专门用于 embedding 的客户端（从环境变量读取本地 Ollama 地址）
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
embedding_client = OpenAI(api_key="ollama", base_url=EMBEDDING_BASE_URL)  # Ollama 不需要真实 key

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def get_embedding(text: str) -> List[float]:
    if not text or len(text.strip()) < 5:
        return [0.0] * EMBEDDING_DIM
    try:
        text = text[:8000]
        resp = embedding_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        print(f"  ⚠️ 向量生成失败: {e}")
        return [0.0] * EMBEDDING_DIM


# ============================================================
# ★ 新增：批量翻译模块（入库前把非中文文本转换为中文）
#   - 先收集所有待翻译文本（title + description），去重
#   - 按批次合并成一次 API 调用，用 JSON 结构化返回
#   - 结果写入缓存，入库时直接查缓存
# ============================================================
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
# 用于判断“有意义字符”的正则（字母 + 中文），用来估算中文占比
_MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")

# 内存 + 磁盘翻译缓存：相同文本只翻译一次
_translation_cache = {}


def _load_translation_cache():
    """加载磁盘上的翻译缓存（如果存在）。"""
    global _translation_cache
    if os.path.exists(TRANSLATE_CACHE_FILE):
        try:
            with open(TRANSLATE_CACHE_FILE, "r", encoding="utf-8") as f:
                _translation_cache = json.load(f)
            print(f"  已加载翻译缓存: {len(_translation_cache)} 条")
        except Exception as e:
            print(f"  ⚠️ 翻译缓存加载失败，将重新构建: {e}")
            _translation_cache = {}


def _save_translation_cache():
    """把翻译缓存写回磁盘。"""
    try:
        with open(TRANSLATE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_translation_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️ 翻译缓存保存失败: {e}")


def is_chinese_enough(text: str) -> bool:
    """判断文本是否已经基本是中文（中文字符占有意义字符的比例 >= 阈值）。"""
    if not text:
        return True
    meaningful = _MEANINGFUL_CHAR_RE.findall(text)
    if not meaningful:
        # 纯数字/符号（如尺寸、编号），无需翻译
        return True
    chinese = _CHINESE_CHAR_RE.findall(text)
    return (len(chinese) / len(meaningful)) >= CHINESE_RATIO_THRESHOLD


def translate_to_chinese(text: str) -> str:
    """查缓存获取译文；缓存未命中则原样返回（保证不阻塞流程）。
    正常流程下，所有待翻译文本都已通过 batch_translate_all 预先翻译好并写入缓存。"""
    if text is None:
        return text
    text = str(text)
    if not text.strip():
        return text
    if is_chinese_enough(text):
        return text
    return _translation_cache.get(text, text)


def _parse_batch_translation_response(content: str, batch_texts: list) -> dict:
    """解析模型返回的批量翻译结果，返回 {原文: 译文} 字典。
    优先尝试解析 JSON；解析失败则降级为按行匹配。"""
    result = {}
    content = (content or "").strip()

    # 尝试 1：直接解析 JSON 数组
    try:
        items = json.loads(content)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    src = item.get("source") or item.get("原文") or item.get("id") or ""
                    tgt = item.get("translation") or item.get("译文") or item.get("target") or ""
                    if src and tgt:
                        result[src] = tgt
            if result:
                return result
    except Exception:
        pass

    # 尝试 2：从 ```json ... ``` 代码块中提取
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        src = item.get("source") or item.get("原文") or item.get("id") or ""
                        tgt = item.get("translation") or item.get("译文") or item.get("target") or ""
                        if src and tgt:
                            result[src] = tgt
                if result:
                    return result
        except Exception:
            pass

    # 尝试 3：按行简单匹配（兜底，不保证 100% 准确）
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) >= len(batch_texts):
        for i, src in enumerate(batch_texts):
            if i < len(lines):
                result[src] = lines[i]

    return result


def _batch_translate_one(texts: list) -> dict:
    """对一批文本执行一次批量翻译，返回 {原文: 译文}。"""
    if not texts:
        return {}

    # 构造请求体：每条带 id（用序号）和 source
    items = [{"id": i, "source": t[:2000]} for i, t in enumerate(texts)]
    user_content = json.dumps(items, ensure_ascii=False, indent=2)

    system_prompt = (
        "你是专业的电商商品文本翻译助手。用户会给你一个 JSON 数组，"
        "每个元素包含 id 和 source 两个字段，source 是需要翻译的电商商品文本（商品名或商品描述）。\n"
        "请将每条 source 翻译成简体中文，保留其中的数字、尺寸、规格、单位、型号、品牌名与表情符号。\n"
        "输出必须是一个 JSON 数组，每个元素包含 id、source、translation 三个字段，"
        "id 和 source 必须与输入完全一致，translation 是翻译后的中文文本。\n"
        "不要添加任何解释、说明或 Markdown 格式，只输出纯 JSON。"
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=TRANSLATE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                stream=False,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_batch_translation_response(content, texts)

            # 用 id 反查原文，建立 原文->译文 映射
            # 但 parsed 里的 key 可能是 id 或 source，需要统一处理
            final = {}
            # 先按 source 匹配
            for src in texts:
                if src in parsed:
                    final[src] = parsed[src]
            # 再按 id 匹配
            if len(final) < len(texts):
                try:
                    raw_items = json.loads(content)
                    if isinstance(raw_items, list):
                        id_to_source = {str(i): t for i, t in enumerate(texts)}
                        for item in raw_items:
                            if isinstance(item, dict):
                                item_id = str(item.get("id", ""))
                                tgt = item.get("translation") or item.get("译文") or ""
                                if item_id in id_to_source and tgt:
                                    final[id_to_source[item_id]] = tgt
                except Exception:
                    pass

            if final:
                return final
            print(f"  ⚠️ 批量翻译返回解析为空(第{attempt + 1}次)，重试中...")
        except Exception as e:
            print(f"  ⚠️ 批量翻译失败(第{attempt + 1}次): {e}")
        time.sleep(1.5 * (attempt + 1))

    return {}


def batch_translate_all(texts: list) -> int:
    """批量翻译所有待翻译文本，结果写入缓存。返回实际新翻译的条数。"""
    # 过滤：非空、非中文、未在缓存中
    pending = []
    seen = set()
    for t in texts:
        if t is None:
            continue
        t = str(t)
        if not t.strip():
            continue
        if is_chinese_enough(t):
            continue
        if t in _translation_cache:
            continue
        if t in seen:
            continue
        seen.add(t)
        pending.append(t)

    total = len(pending)
    if total == 0:
        print("  无需翻译（全部已是中文或已缓存）")
        return 0

    print(f"  待翻译文本: {total} 条，每批 {BATCH_TRANSLATE_SIZE} 条，共 { (total + BATCH_TRANSLATE_SIZE - 1) // BATCH_TRANSLATE_SIZE } 批")

    new_count = 0
    for i in range(0, total, BATCH_TRANSLATE_SIZE):
        batch = pending[i:i + BATCH_TRANSLATE_SIZE]
        batch_num = i // BATCH_TRANSLATE_SIZE + 1
        print(f"  翻译第 {batch_num} 批 ({len(batch)} 条)...", end=" ", flush=True)

        result = _batch_translate_one(batch)
        success = 0
        for src, tgt in result.items():
            if src and tgt and src in seen:
                _translation_cache[src] = tgt
                success += 1
                new_count += 1

        print(f"成功 {success}/{len(batch)}")

        # 每 5 批落盘一次缓存，防止意外丢失
        if batch_num % 5 == 0:
            _save_translation_cache()

    _save_translation_cache()
    print(f"  ✅ 批量翻译完成，新增 {new_count} 条译文，缓存总计 {len(_translation_cache)} 条")
    return new_count


# ============================================================
# 主导入逻辑
# ============================================================
def import_data():
    print("=" * 60)
    print("🚀 导入 Shopee 数据 (使用 DeepSeek Embedding)")
    print("=" * 60)

    # ★ 加载翻译缓存
    _load_translation_cache()

    # 1. 读取 CSV
    print(f"\n[1/5] 读取数据: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  共 {len(df)} 条商品记录")

    # 重要：处理列名中有空格的情况（'Product Description' 有空格）
    # 将列名中的空格替换为下划线，方便后续使用
    df.columns = df.columns.str.replace(' ', '_')
    print(f"  列名已规范化: {list(df.columns)}")

    # ★ 批量翻译：先收集所有 title 和 description，一次性批量翻译后写入缓存
    print("\n[1.5/5] 批量翻译商品名与描述...")
    all_texts = []
    for _, row in df.iterrows():
        title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
        desc = str(row.get('Product_Description', '')) if pd.notna(row.get('Product_Description')) else ''
        if title:
            all_texts.append(title)
        if desc:
            all_texts.append(desc)
    batch_translate_all(all_texts)

    # 2. 连接数据库
    print("\n[2/5] 连接数据库...")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()

    # 3. 导入商品和库存（翻译已在步骤 1.5 批量完成，这里直接查缓存）
    print("\n[3/5] 导入商品和库存数据...")
    product_count = 0
    for idx, row in df.iterrows():
        try:
            sku = str(row.get('id', f"SKU_{idx:06d}"))[:50]
            name = str(row.get('title', '未知商品'))
            price = float(row.get('final_price', 0.0)) if pd.notna(row.get('final_price')) else 0.0
            description = str(row.get('Product_Description', '')) if pd.notna(row.get('Product_Description')) else ''
            category_id = int(row.get('category_id', 0)) if pd.notna(row.get('category_id')) else 0
            stock = int(row.get('stock', 0)) if pd.notna(row.get('stock')) else 0

            # ★ 关键改动：入库前把非中文的商品名与描述翻译成中文
            name = translate_to_chinese(name)[:255]
            description = translate_to_chinese(description)

            # 插入商品表
            product_sql = """
                INSERT INTO product_catalog 
                (sku_code, product_name, category_id, price, raw_description)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (sku_code) DO UPDATE SET
                    product_name = EXCLUDED.product_name,
                    category_id = EXCLUDED.category_id,
                    price = EXCLUDED.price,
                    raw_description = EXCLUDED.raw_description
                RETURNING id
            """
            cur.execute(product_sql, (sku, name, category_id, price, description))
            product_id = cur.fetchone()[0]

            # 插入库存表
            stock_sql = """
                INSERT INTO inventory_logistics 
                (product_id, stock_quantity, warehouse_location, delivery_estimate_days)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (product_id) DO UPDATE SET
                    stock_quantity = EXCLUDED.stock_quantity
            """
            cur.execute(stock_sql, (
                product_id,
                stock,
                "默认仓",
                (idx % 5) + 1
            ))

            product_count += 1
            if product_count % 100 == 0:
                print(f"  已导入 {product_count} 件商品")
                conn.commit()
        except Exception as e:
            print(f"  ⚠️ 第 {idx} 条数据导入失败: {e}")
            continue

    conn.commit()
    print(f"  ✅ 成功导入 {product_count} 件商品")

    # 4. 构建知识库（只处理有描述的商品）
    #    注意：raw_description 已经是翻译后的中文，这里的向量也基于中文文本生成
    print("\n[4/5] 构建 RAG 知识库（生成向量）...")
    cur.execute("""
        SELECT id, sku_code, product_name, raw_description 
        FROM product_catalog 
        WHERE raw_description IS NOT NULL AND raw_description != ''
    """)
    products = cur.fetchall()

    kb_count = 0
    for prod_id, sku, name, desc in products:
        doc_content = f"商品名称：{name}\n商品描述：{desc}\nSKU：{sku}"

        # 插入原始文档
        doc_sql = """
            INSERT INTO kb_documents (doc_type, source_url, raw_content)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        cur.execute(doc_sql, ("product_manual", f"product_{prod_id}", doc_content))
        doc_id = cur.fetchone()[0]

        # 切分并生成向量
        chunks = text_splitter.split_text(doc_content)
        for chunk_idx, chunk_text in enumerate(chunks):
            vector = get_embedding(chunk_text)
            meta_data = {
                "product_id": prod_id,
                "sku": sku,
                "doc_type": "product_manual",
                "embedding_model": "deepseek-embedding"
            }
            chunk_sql = """
                INSERT INTO kb_chunks 
                (doc_id, chunk_index, chunk_text, chunk_vector, meta_data)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(chunk_sql, (
                doc_id,
                chunk_idx,
                chunk_text,
                vector,
                json.dumps(meta_data)
            ))
            kb_count += 1
            # time.sleep(0.08)  # 控制频率

        if kb_count % 20 == 0:
            print(f"  已生成 {kb_count} 个向量块")
            conn.commit()

    conn.commit()
    print(f"  ✅ 生成 {kb_count} 个向量分块")

    # 5. 统计
    print("\n[5/5] 📊 导入完成，数据统计：")
    cur.execute("SELECT COUNT(*) FROM product_catalog")
    print(f"  product_catalog: {cur.fetchone()[0]} 条")
    cur.execute("SELECT COUNT(*) FROM inventory_logistics")
    print(f"  inventory_logistics: {cur.fetchone()[0]} 条")
    cur.execute("SELECT COUNT(*) FROM kb_documents")
    print(f"  kb_documents: {cur.fetchone()[0]} 条")
    cur.execute("SELECT COUNT(*) FROM kb_chunks")
    print(f"  kb_chunks: {cur.fetchone()[0]} 条")

    cur.close()
    conn.close()
    print("\n✅ 全部导入完成！")


if __name__ == "__main__":
    start = time.time()
    import_data()
    print(f"\n⏱️ 总耗时: {time.time() - start:.2f} 秒")
