"""FastAPI 应用工厂

create_app() 装配应用：注册路由 + CORS 中间件 + LangSmith 追踪（可选）。
lifespan 在启动时构建 Agent 管道（重型：解析文档 + 建索引 + 加载模型），
构建失败直接终止启动进程（fail-fast），避免首个请求超时或反复重建。
get_app() 返回模块级单例，供 uvicorn 直接引用。

LangSmith 集成：零代码改动，仅通过环境变量启用。
设置 LANGSMITH_API_KEY + LANGSMITH_TRACING=true 后，
LangChain/LangGraph 自动上报 trace 到 LangSmith 项目。
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import router
from src.config import settings
from src.utils.logger import logger

_app: FastAPI | None = None


def _configure_langsmith() -> None:
    """根据配置启用 LangSmith 追踪（仅设置环境变量，LangChain 自动读取）。"""
    if settings.langsmith_tracing and settings.langsmith_key_value():
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_key_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info(f"[api] LangSmith 追踪已启用，项目={settings.langsmith_project}")
    elif settings.langsmith_tracing:
        logger.warning("[api] LANGSMITH_TRACING=true 但未配置 LANGSMITH_API_KEY，跳过")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """启动时构建 Agent 管道并校验关键配置（fail-fast），关闭时无持久资源需清理。"""
    if not settings.deepseek_key_value():
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置：请在 .env 填入有效 key 后启动服务"
        )
    logger.info("[api] 正在构建 Agent 管道（解析文档 → 建索引 → 加载模型）...")
    # 延迟导入，避免模块加载即触发重型管道构建
    from src.pipeline import build_agent_pipeline

    application.state.graph = build_agent_pipeline()
    application.state.ready = True
    logger.info("[api] Agent 管道构建完成，服务就绪")
    yield
    logger.info("[api] 服务关闭")


def create_app() -> FastAPI:
    """构建 FastAPI 应用实例。"""
    _configure_langsmith()

    application = FastAPI(
        title="RAGForge",
        description="Agentic RAG — LangGraph 状态机驱动的自适应检索增强生成",
        version=__version__,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
