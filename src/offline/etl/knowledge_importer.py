# -*- coding: utf-8 -*-
"""
knowledge_importer.py —— 外部知识文档（政策/FAQ 数据）导入 RAG 知识库
====================================================================
配置与风格参考 shopee_importer.py：
  - 从 .env 读取数据库连接与 embedding 配置
  - 文档切分（langchain RecursiveCharacterTextSplitter）+ 向量化
  - 写入 pgvector 表：kb_documents（原始文档）/ kb_chunks（向量分块）
  - 支持幂等重导：同 (doc_type, source_url) 的旧数据先删除再写入

用法：
  # 导入目录下所有 .docx / .txt / .md（默认 doc_type=policy）
  python knowledge_importer.py --dir  /home/cai/t_data/AI_Sentix/knowledge_docs

  # 导入单个文件，并指定文档类型
  python knowledge_importer.py --file  /path/to/电商常见问题FAQ.docx --doc-type faq

  # 多个文件
  python knowledge_importer.py --file a.docx --file b.md --doc-type policy

依赖（缺失会给出安装提示）：
  pip install python-docx psycopg2-binary pgvector openai langchain-text-splitters python-dotenv

.env 参考配置（均可选，脚本有默认值，默认指向本地 Ollama）：
  EMBEDDING_BASE_URL=http://localhost:11434/v1
  EMBEDDING_API_KEY=ollama
  EMBEDDING_MODEL=qwen3-embedding:0.6b
  EMBEDDING_DIM=1024
  DATABASE_URL=postgresql://postgres:236591@localhost:5432/postgres
"""

import os
import sys
import json
import time
import argparse
import glob
from typing import List, Optional, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置（与 shopee_importer.py 同风格）
# ============================================================
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ 未配置 DATABASE_URL，请在 .env 中设置")
    sys.exit(1)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_EXT = (".docx", ".txt", ".md")

# ============================================================
# 初始化
# ============================================================
client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def get_embedding(text: str) -> List[float]:
    """调用 embedding API；失败时返回全零向量（与 shopee_importer.py 一致）"""
    if not text or len(text.strip()) < 5:
        return [0.0] * EMBEDDING_DIM
    try:
        text = text[:8000]
        resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        print(f"  ⚠️ 向量生成失败: {e}")
        return [0.0] * EMBEDDING_DIM


# ============================================================
# 文档解析
# ============================================================
def read_docx(file_path: str) -> List[Tuple[Optional[str], str]]:
    """读取 .docx，按 Heading 样式拆分为 [(章节标题, 正文), ...]"""
    try:
        from docx import Document
    except ImportError:
        print("❌ 缺少 python-docx，请执行: pip install python-docx")
        sys.exit(1)

    doc = Document(file_path)
    sections: List[Tuple[Optional[str], List[str]]] = []
    cur_title: Optional[str] = None
    cur_lines: List[str] = []

    def flush():
        if cur_lines:
            sections.append((cur_title, "\n".join(cur_lines)))
            cur_lines.clear()

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style_name = p.style.name if p.style else ""
        if style_name.startswith("Heading"):
            flush()
            cur_title = text
        else:
            cur_lines.append(text)
    flush()

    if not sections:
        print("  ⚠️ 文档为空，跳过")
        return []
    return [(title, content) for title, content in sections]


def read_text_file(file_path: str) -> List[Tuple[Optional[str], str]]:
    """读取 .txt / .md；md 按 # 标题拆分，txt 整个文件为一个章节"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if file_path.endswith(".md"):
        sections = []
        cur_title: Optional[str] = None
        cur_lines = []
        for line in content.splitlines():
            s = line.strip()
            if s.startswith("#"):
                if cur_lines:
                    sections.append((cur_title, "\n".join(cur_lines)))
                    cur_lines = []
                cur_title = s.lstrip("#").strip()
            elif s:
                cur_lines.append(line)
        if cur_lines:
            sections.append((cur_title, "\n".join(cur_lines)))
        return sections or [(None, content)]
    return [(None, content)]


def parse_document(file_path: str) -> List[Tuple[Optional[str], str]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return read_docx(file_path)
    if ext in (".txt", ".md"):
        return read_text_file(file_path)
    print(f"  ⚠️ 不支持的格式: {ext}（支持 {SUPPORTED_EXT}）")
    return []


# ============================================================
# 导入逻辑
# ============================================================
def import_document(cur, file_path: str, doc_type: str) -> Tuple[int, int]:
    """
    导入单个文件。返回 (文档数, 向量块数)。
    幂等：按 (doc_type, source_url) 先删除旧数据（ON DELETE CASCADE 自动清 chunks）再写入。
    """
    print(f"\n📄 处理文件: {file_path}")
    sections = parse_document(file_path)
    if not sections:
        return 0, 0

    file_key = os.path.abspath(file_path)
    docs_count = 0
    chunks_count = 0

    for sec_idx, (title, content) in enumerate(sections):
        if not content.strip():
            continue

        # 章节唯一标识：绝对路径 + 章节序号（标题仅作元数据，不参与幂等键）
        source_url = f"{file_key}#{sec_idx}"
        chapter_title = title or f"第{sec_idx + 1}节"

        # 1. 幂等清理：删除同源旧文档（级联删除旧 chunks）
        cur.execute(
            "DELETE FROM kb_documents WHERE doc_type = %s AND source_url = %s",
            (doc_type, source_url),
        )

        # 2. 插入原始文档
        doc_sql = """
            INSERT INTO kb_documents (doc_type, source_url, raw_content)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        cur.execute(doc_sql, (doc_type, source_url, content))
        doc_id = cur.fetchone()[0]
        docs_count += 1

        # 3. 切分并生成向量
        chunks = text_splitter.split_text(content)
        for chunk_idx, chunk_text in enumerate(chunks):
            vector = get_embedding(chunk_text)
            meta_data = {
                "source_file": os.path.basename(file_path),
                "chapter": chapter_title,
                "doc_type": doc_type,
                "embedding_model": EMBEDDING_MODEL,
            }
            chunk_sql = """
                INSERT INTO kb_chunks (doc_id, chunk_index, chunk_text, chunk_vector, meta_data)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(chunk_sql, (doc_id, chunk_idx, chunk_text, vector, json.dumps(meta_data)))
            chunks_count += 1
            time.sleep(0.05)  # 控制频率，避免本地服务过载

        if docs_count % 5 == 0:
            print(f"  已处理 {docs_count} 个章节 / {chunks_count} 个向量块")
            cur.connection.commit()

    print(f"  ✅ 文件完成: {docs_count} 个章节, {chunks_count} 个向量块")
    return docs_count, chunks_count


def main():
    parser = argparse.ArgumentParser(description="导入外部知识文档（政策/FAQ）到 RAG 知识库")
    parser.add_argument("--dir", type=str, help="目录：导入其中所有支持的文档")
    parser.add_argument("--file", type=str, action="append", help="单个文件（可多次指定）")
    parser.add_argument("--doc-type", type=str, default="policy",
                        help="文档类型，写入 kb_documents.doc_type（默认 policy，FAQ 建议传 faq）")
    args = parser.parse_args()

    # 收集待导入文件
    files = list(args.file or [])
    if args.dir:
        for ext in SUPPORTED_EXT:
            files.extend(glob.glob(os.path.join(args.dir, "**", f"*{ext}"), recursive=True))
    files = list(dict.fromkeys(files))  # 去重且保持顺序

    if not files:
        parser.print_help()
        print("\n❌ 请通过 --dir 或 --file 指定要导入的文档")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 导入知识文档 (embedding: {EMBEDDING_MODEL}, 维度: {EMBEDDING_DIM})")
    print(f"   文档类型: {args.doc_type}，共 {len(files)} 个文件")
    print("=" * 60)

    # 连接数据库
    print("\n[连接数据库] ...")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()

    total_docs = 0
    total_chunks = 0
    failed = 0
    for f in files:
        try:
            d, c = import_document(cur, f, args.doc_type)
            total_docs += d
            total_chunks += c
            conn.commit()
        except Exception as e:
            failed += 1
            print(f"  ❌ 导入失败 {f}: {e}")
            conn.rollback()
            continue

    # 统计
    print("\n" + "=" * 60)
    print("📊 导入完成，数据统计：")
    cur.execute("SELECT doc_type, COUNT(*) FROM kb_documents GROUP BY doc_type ORDER BY doc_type")
    for dt, cnt in cur.fetchall():
        print(f"  kb_documents[{dt}]: {cnt} 条")
    cur.execute("SELECT COUNT(*) FROM kb_chunks")
    print(f"  kb_chunks: {cur.fetchone()[0]} 条")
    print(f"  本次新增: {total_docs} 个文档 / {total_chunks} 个向量块，失败 {failed} 个文件")

    cur.close()
    conn.close()
    print("✅ 全部完成！")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\n⏱️ 总耗时: {time.time() - start:.2f} 秒")
