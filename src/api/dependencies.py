"""FastAPI 依赖注入：Agent graph 单例 + Metrics 单例

启动时惰性初始化 Agent graph（构建成本高：解析文档+构建索引+加载 embedding），
后续请求复用同一实例。线程安全。
"""

import threading

from src.utils.logger import logger
from src.utils.metrics import MetricsCollector

# 模块级单例（惰性初始化）
_graph_lock = threading.Lock()
_graph = None


def get_graph():
    """获取编译后的 LangGraph Agent 单例（惰性初始化）。

    首次调用时构建（解析文档→切分→嵌入→组装状态机），后续复用。
    FastAPI Depends 注入此函数。

    Returns:
        编译后的 LangGraph 可执行图
    """
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:  # double-check
                # 延迟导入，避免 API 模块加载即触发重型管道构建
                from src.pipeline import build_agent_pipeline

                logger.info("[api] 首次请求，正在构建 Agent 管道...")
                _graph = build_agent_pipeline()
                logger.info("[api] Agent 管道构建完成，复用单例")
    return _graph


def get_metrics() -> MetricsCollector:
    """获取 MetricsCollector 单例（与全局单例相同）。"""
    return MetricsCollector()
