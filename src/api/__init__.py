"""FastAPI 应用模块

提供 HTTP API + SSE 流式接口，封装 Agent 状态机管道。
入口：create_app() 构建应用实例，供 uvicorn 启动。
"""

from src.api.app import create_app, get_app

__all__ = ["create_app", "get_app"]
