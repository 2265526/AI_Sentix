# -*- coding: utf-8 -*-
"""
api/main.py —— FastAPI 应用入口
=================================
阶段二：RAG 检索服务；阶段三：Agent 对话；阶段四：管理员接口（前端验证用）。

启动方式：
    uvicorn src.online.api.main:app --host 0.0.0.0 --port 8000
Swagger 文档：http://localhost:8000/docs
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.logger import setup_logging
from src.online.api.routes import admin, chat, rag

setup_logging()

app = FastAPI(
    title="电商AI智能客服",
    description="阶段二：混合检索与重排序；阶段三：Agent 意图识别与工具路由；管理接口：知识库上传 / 商品同步",
    version="0.4.0",
)

# CORS：允许前端（Vite dev server / 静态构建）跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/health", tags=["health"], summary="健康检查")
def health():
    return {"status": "ok", "service": "ecommerce-ai-cs"}
