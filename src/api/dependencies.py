"""FastAPI 依赖注入：Agent graph（由 lifespan 启动时构建，存于 app.state）

启动预热（lifespan）代替惰性构建：重型管道（解析文档 + 建索引 + 加载模型）
在服务启动阶段完成，构建失败直接终止启动，而非在首个请求内超时或反复重试。
"""

from fastapi import HTTPException, Request


def get_graph(request: Request):
    """获取编译后的 LangGraph Agent（lifespan 启动时已构建）。

    FastAPI Depends 注入此函数。

    Returns:
        编译后的 LangGraph 可执行图

    Raises:
        HTTPException: 503，Agent 管道尚未构建完成
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪，Agent 管道仍在构建中")
    return graph
