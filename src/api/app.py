"""FastAPI 应用工厂

create_app() 装配应用：注册路由 + 配置 LangSmith 追踪（可选）。
get_app() 返回模块级单例，供 uvicorn 直接引用。

LangSmith 集成：零代码改动，仅通过环境变量启用。
设置 LANGSMITH_API_KEY + LANGSMITH_TRACING=true 后，
LangChain/LangGraph 自动上报 trace 到 LangSmith 项目。
"""

import os

from fastapi import FastAPI

from src.api.routes import router
from src.config import settings
from src.utils.logger import logger

_app: FastAPI | None = None


def _configure_langsmith() -> None:
    """根据配置启用 LangSmith 追踪（仅设置环境变量，LangChain 自动读取）。"""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info(f"[api] LangSmith 追踪已启用，项目={settings.langsmith_project}")
    elif settings.langsmith_tracing:
        logger.warning("[api] LANGSMITH_TRACING=true 但未配置 LANGSMITH_API_KEY，跳过")


def create_app() -> FastAPI:
    """构建 FastAPI 应用实例。"""
    _configure_langsmith()

    application = FastAPI(
        title="RAGForge",
        description="Agentic RAG — LangGraph 状态机驱动的自适应检索增强生成",
        version="0.1.0",
    )
    application.include_router(router)
    logger.info("[api] FastAPI 应用构建完成（/health /chat /chat/stream）")
    return application


def get_app() -> FastAPI:
    """返回模块级 FastAPI 单例（供 uvicorn `--factory` 或直接引用）。"""
    global _app
    if _app is None:
        _app = create_app()
    return _app
