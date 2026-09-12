"""CLI 与 Server 启动模式

提供交互式命令行和 FastAPI + SSE 服务两种运行模式。
从 main.py 迁出，使 main.py 保持精简。
"""

import argparse

from src.config import settings
from src.pipeline import ask_agent, build_agent_pipeline
from src.utils.logger import logger


def run_cli() -> None:
    """交互式命令行模式：逐问题执行 Agent 状态机。"""
    logger.info("=" * 60)
    logger.info("  RAGForge — Agentic RAG（LangGraph 状态机）[CLI 模式]")
    logger.info("  输入问题开始对话，输入 quit 退出")
    logger.info("=" * 60)

    graph = build_agent_pipeline()
    logger.info("准备就绪！试试问：你好 / Python 装饰器怎么用？ / 什么是过拟合？")

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        text, state = ask_agent(graph, question)
        logger.info(f"助手：{text}")
        # 决策轨迹（调试可见）
        for m in state.get("messages", [])[-5:]:
            logger.debug(f"  · {m}")
        print()

    logger.info("再见！")


def run_server(host: str | None = None, port: int | None = None) -> None:
    """启动 FastAPI + SSE 服务模式。

    Args:
        host: 监听地址（默认读 settings.api_host）
        port: 监听端口（默认读 settings.api_port）
    """
    import uvicorn

    from src.api import create_app

    # 用 is not None 判断：显式传入的端口 0 等合法值不被默认值静默覆盖
    host = host if host is not None else settings.api_host
    port = port if port is not None else settings.api_port

    logger.info("=" * 60)
    logger.info("  RAGForge — Agentic RAG [API 服务模式]")
    logger.info(f"  监听 http://{host}:{port}")
    logger.info(f"  文档 http://{host}:{port}/docs")
    logger.info("  按 Ctrl+C 停止")
    logger.info("=" * 60)

    app = create_app()
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """程序入口：解析命令行参数选择模式。

    - 无参数 / --serve：启动 API 服务
    - --cli：交互式命令行
    - --host/--port：自定义服务地址端口
    """
    parser = argparse.ArgumentParser(
        description="RAGForge — Agentic RAG 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python -m src.main              # 启动 API 服务\n"
               "  python -m src.main --cli        # 交互式 CLI\n"
               "  python -m src.main --port 9000  # 自定义端口",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cli", action="store_true", help="交互式命令行模式")
    mode.add_argument("--serve", action="store_true", help="启动 API 服务模式（默认）")
    parser.add_argument("--host", default=None, help=f"API 监听地址（默认 {settings.api_host}）")
    parser.add_argument("--port", type=int, default=None, help=f"API 监听端口（默认 {settings.api_port}）")

    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
