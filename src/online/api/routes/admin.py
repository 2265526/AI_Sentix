"""
管理员接口

1. POST /admin/kb/upload     知识库文档上传（TXT / PDF）：解析 → 分块 → 向量化 → 入库（幂等），并重建在线 BM25 索引
2. POST /admin/products/import  商品数据同步（CSV 导入）：product_catalog / inventory_logistics 增量入库（ON CONFLICT 更新）
3. 类目管理：
     - GET  /admin/categories          类目树（大类/中类/小类 三级）
     - GET  /admin/categories/search   类目搜索（名称/路径模糊匹配）
     - POST /admin/categories          新增类目（层级 + 父级校验，自动拼 path）
"""
import hashlib
import io
import json
import logging
import os
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from psycopg2.extensions import connection
from pydantic import BaseModel, Field

from src.common.constants import ALLOWED_DOC_TYPES
from src.online.db.session import get_db
from src.offline.etl.category_classifier import classify_by_keywords
from src.offline.etl.knowledge_importer import (
    get_embedding,
    read_text_file,
    text_splitter,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# 支持的文档类型
ALLOWED_TEXT_EXT = (".txt", ".md")
ALLOWED_PDF_EXT = (".pdf")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


# 类目管理（category 表，三级树）
class CategoryCreate(BaseModel):
    """新增类目请求体。level: 1=大类 2=中类 3=小类。"""

    name: str = Field(..., min_length=1, max_length=50, description="类目名称")
    level: int = Field(..., ge=1, le=3, description="层级：1 大类 / 2 中类 / 3 小类")
    parent_id: Optional[int] = Field(None, description="父类目 id（大类不填；中类填大类；小类填中类）")


def _category_rows(cur) -> List[tuple]:
    """读取全部类目（扁平行）。"""
    cur.execute(
        "SELECT id, name, parent_id, level, path FROM category ORDER BY level, id"
    )
    return cur.fetchall()


def _build_tree(rows: List[tuple]) -> list:
    """扁平行 → 嵌套树（level 1 → 2 → 3）。"""
    nodes = {
        r[0]: {"id": r[0], "name": r[1], "parent_id": r[2], "level": r[3], "path": r[4], "children": []}
        for r in rows
    }
    roots = []
    for r in rows:
        node = nodes[r[0]]
        if r[2] is None:
            roots.append(node)
        elif r[2] in nodes:
            nodes[r[2]]["children"].append(node)
    return roots


def _validate_category(cur, name: str, level: int, parent_id: Optional[int]) -> str:
    """
    校验新增类目并计算完整路径 path。

    Raises:
        ValueError: 层级/父级不匹配、父类目不存在、同级重名。
    """
    name = name.strip()
    if not name:
        raise ValueError("类目名称不能为空")
    if len(name) > 50:
        raise ValueError("类目名称最长 50 字")

    if level == 1:
        if parent_id is not None:
            raise ValueError("大类不能指定父类目")
        return name

    if parent_id is None:
        raise ValueError("中类/小类必须指定父类目")
    cur.execute("SELECT name, level, path FROM category WHERE id = %s", (parent_id,))
    parent = cur.fetchone()
    if parent is None:
        raise ValueError("父类目不存在")
    if level == 2 and parent[1] != 1:
        raise ValueError("中类的父级必须是大类（level=1）")
    if level == 3 and parent[1] != 2:
        raise ValueError("小类的父级必须是中类（level=2）")

    # 同级下名称唯一（IS NOT DISTINCT FROM 兼容 parent_id 为 NULL）
    cur.execute(
        "SELECT 1 FROM category WHERE name = %s AND parent_id IS NOT DISTINCT FROM %s",
        (name, parent_id),
    )
    if cur.fetchone():
        raise ValueError(f"类目「{name}」在所选父级下已存在")

    return f"{parent[2]}/{name}"


@router.get("/categories", summary="类目树（大类/中类/小类三级）")
def list_categories(conn: connection = Depends(get_db)):
    """返回完整类目树（含层级与完整路径），供前端树展示与级联下拉。"""
    cur = conn.cursor()
    try:
        rows = _category_rows(cur)
    finally:
        cur.close()
    return {"total": len(rows), "tree": _build_tree(rows)}


@router.get("/categories/search", summary="类目搜索（名称/路径模糊匹配）")
def search_categories(
    q: str = Query(..., min_length=1, max_length=50, description="搜索关键词（支持相似/包含匹配）"),
    conn: connection = Depends(get_db),
):
    """按类目名称或完整路径模糊搜索（ILIKE），返回命中的类目及其层级/路径。"""
    pattern = f"%{q}%"
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, name, parent_id, level, path
            FROM category
            WHERE name ILIKE %s OR path ILIKE %s
            ORDER BY level, id
            LIMIT 200
            """,
            (pattern, pattern),
        )
        items = [
            {"id": r[0], "name": r[1], "parent_id": r[2], "level": r[3], "path": r[4]}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
    return {"query": q, "total": len(items), "items": items}


@router.post("/categories", summary="新增类目（大类/中类/小类）")
def create_category(
    body: CategoryCreate,
    conn: connection = Depends(get_db),
):
    """新增类目：中类必须选所属大类，小类必须选所属中类；自动拼接完整 path。"""
    cur = conn.cursor()
    try:
        path = _validate_category(cur, body.name, body.level, body.parent_id)
        cur.execute(
            "INSERT INTO category (name, parent_id, level, path) VALUES (%s, %s, %s, %s) RETURNING id, created_at",
            (body.name.strip(), body.parent_id, body.level, path),
        )
        new_id, created_at = cur.fetchone()
        conn.commit()
    except ValueError as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()

    logger.info("admin/categories: create id=%s name=%s level=%s path=%s", new_id, body.name, body.level, path)
    return {"status": "ok", "id": new_id, "name": body.name.strip(), "level": body.level, "path": path, "created_at": created_at}


# 知识库文档上传
def _extract_text(filename: str, content: bytes) -> str:
    """按扩展名提取纯文本：TXT/MD 直接解码，PDF 用 pypdf 逐页提取。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ALLOWED_TEXT_EXT:
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return content.decode(enc)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore")
    if ext in ALLOWED_PDF_EXT:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    raise ValueError(f"不支持的文件类型 {ext}，仅支持 {'/'.join(ALLOWED_TEXT_EXT + ALLOWED_PDF_EXT)}")


def _import_text_to_kb(
    cur,
    doc_type: str,
    source_url: str,
    raw_content: str,
    meta_extra: dict = None,
) -> Tuple[int, int]:
    """把整段文本分块、向量化并入库（幂等：同 source_url 先删旧数据）。

    Args:
        meta_extra: 附加到 meta_data 的字段（如 product_skus、类目字段）

    Returns: (文档数, 分块数)
    """
    # 幂等清理：删除同源旧文档（ON DELETE CASCADE 自动清 chunks）
    cur.execute(
        "DELETE FROM kb_documents WHERE doc_type = %s AND source_url = %s",
        (doc_type, source_url),
    )

    cur.execute(
        "INSERT INTO kb_documents (doc_type, source_url, raw_content) VALUES (%s, %s, %s) RETURNING id",
        (doc_type, source_url, raw_content),
    )
    doc_id = cur.fetchone()[0]

    # 切分 + 向量化 + 入库
    chunks = text_splitter.split_text(raw_content)
    for chunk_idx, chunk_text in enumerate(chunks):
        vector = get_embedding(chunk_text)
        meta_data = {"doc_type": doc_type, **(meta_extra or {})}
        cur.execute(
            "INSERT INTO kb_chunks (doc_id, chunk_index, chunk_text, chunk_vector, meta_data)"
            " VALUES (%s, %s, %s, %s, %s)",
            (doc_id, chunk_idx, chunk_text, vector, json.dumps(meta_data)),
        )
    return 1, len(chunks)


@router.post("/kb/upload", summary="知识库文档上传（TXT/PDF，自动分块向量化入库）")
def upload_kb_document(
    file: UploadFile = File(..., description="TXT / PDF 文件（≤20MB）"),
    doc_type: str = Form("policy", description="文档类型：policy / faq / product_manual"),
    conn: connection = Depends(get_db),
):
    """上传知识文档，自动分块 + 向量化写入知识库。"""
    if doc_type not in ALLOWED_DOC_TYPES:
        raise ValueError(f"doc_type 必须是 {'/'.join(ALLOWED_DOC_TYPES)} 之一")

    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("文件超过 20MB 限制")

    raw_content = _extract_text(file.filename or "upload.txt", content)
    if not raw_content.strip():
        raise ValueError("未能从文件中提取到文本内容")

    # 文档唯一标识：上传文件名 + 内容哈希（同名文件重新上传即覆盖）
    source_url = f"upload:{file.filename}:{hashlib.md5(raw_content.encode()).hexdigest()[:8]}"

    cur = conn.cursor()
    try:
        doc_count, chunk_count = _import_text_to_kb(cur, doc_type, source_url, raw_content)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()

    # 重建在线进程的 BM25 索引（新知识立即可被关键词召回）
    from src.online.core.rag.retriever import HybridRetriever

    HybridRetriever(conn).bm25_retriever.refresh()

    logger.info(
        "admin/kb/upload: file=%s doc_type=%s docs=%d chunks=%d",
        file.filename, doc_type, doc_count, chunk_count,
    )
    return {
        "status": "ok",
        "filename": file.filename,
        "doc_type": doc_type,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
    }


# 商品数据同步（CSV 导入）
def _parse_csv(content: bytes) -> List[dict]:
    """
    解析 CSV（自动探测常见编码），返回规范化后的行字典。
    使用标准库 csv 而非 pandas：避免 .pylib 旧版 numpy 与系统 scipy
    二进制冲突导致进程崩溃。
    """
    import csv

    text = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("无法识别 CSV 编码")

    def _num(value, default):
        try:
            s = str(value).strip().replace(",", "")
            return float(s) if s else default
        except (TypeError, ValueError):
            return default

    rows: List[dict] = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        sku = str(r.get("id") or "").strip()
        if not sku:
            continue
        rows.append(
            {
                "sku_code": sku[:50],
                "product_name": str(r.get("title") or "未知商品")[:255],
                "price": _num(r.get("final_price") or r.get("initial_price"), 0.0),
                "raw_description": str(r.get("Product Description") or "")[:20000],
                "stock_quantity": int(_num(r.get("stock"), 0)),
            }
        )
    return rows


@router.post("/products/import", summary="商品数据同步（CSV 导入，同时生成商品知识库向量）")
def import_products_csv(
    file: UploadFile = File(..., description="CSV 文件（列：id/title/final_price/stock/Product Description）"),
    conn: connection = Depends(get_db),
):
    """
    CSV 导入商品与库存数据（ON CONFLICT 增量更新），
    并同步为每个商品生成知识库文档（doc_type=product_manual）：
    商品名 + 描述 → 分块 → embedding 向量化 → kb_documents / kb_chunks（幂等覆盖）。
    """
    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("文件超过 20MB 限制")

    rows = _parse_csv(content)
    if not rows:
        raise ValueError("CSV 中没有可导入的数据（缺少 id/title 列）")

    cur = conn.cursor()
    imported = failed = 0
    kb_docs = kb_chunks = 0
    errors: List[str] = []
    try:
        for row in rows:
            try:
                # 自动分类：按商品名关键词判断大类/小类
                category_big, category_small, category_path = classify_by_keywords(
                    row["product_name"]
                )
                cur.execute(
                    """
                    INSERT INTO product_catalog (sku_code, product_name, category_big, category_small, category_path, price, raw_description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sku_code) DO UPDATE SET
                        product_name = EXCLUDED.product_name,
                        category_big = EXCLUDED.category_big,
                        category_small = EXCLUDED.category_small,
                        category_path = EXCLUDED.category_path,
                        price = EXCLUDED.price,
                        raw_description = EXCLUDED.raw_description
                    RETURNING id
                    """,
                    (row["sku_code"], row["product_name"], category_big, category_small, category_path, row["price"], row["raw_description"]),
                )
                product_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO inventory_logistics (product_id, stock_quantity, warehouse_location, delivery_estimate_days)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (product_id) DO UPDATE SET
                        stock_quantity = EXCLUDED.stock_quantity
                    """,
                    (product_id, row["stock_quantity"], "默认仓", 3),
                )

                # 同步生成商品知识库文档（商品名 + 描述）
                desc = (row["raw_description"] or "").strip()
                kb_content = row["product_name"] + (f"\n\n{desc}" if desc else "")
                if kb_content.strip():
                    d, c = _import_text_to_kb(
                        cur,
                        doc_type="product_manual",
                        source_url=f"csv_product:{row['sku_code']}",
                        raw_content=kb_content,
                        meta_extra={
                            "category_big": category_big,
                            "category_small": category_small,
                            "category_path": category_path,
                            "product_skus": [row["sku_code"]],
                            "applicable_audience": "通用",
                        },
                    )
                    kb_docs += d
                    kb_chunks += c
                imported += 1
            except Exception as e:  # 单行失败不阻断整体
                failed += 1
                if len(errors) < 5:
                    errors.append(f"SKU {row['sku_code']}: {e}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()

    # 重建在线进程的 BM25 索引（新商品知识文档立即可被关键词召回）
    if kb_docs > 0:
        from src.online.core.rag.retriever import HybridRetriever

        HybridRetriever(conn).bm25_retriever.refresh()

    logger.info(
        "admin/products/import: rows=%d imported=%d failed=%d kb_docs=%d kb_chunks=%d",
        len(rows), imported, failed, kb_docs, kb_chunks,
    )
    return {
        "status": "ok",
        "filename": file.filename,
        "total": len(rows),
        "imported": imported,
        "failed": failed,
        "errors": errors,
        "kb_docs_created": kb_docs,
        "kb_chunks_created": kb_chunks,
    }
