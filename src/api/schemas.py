"""API 请求/响应 Pydantic 模型

定义 HTTP 接口的输入输出 schema，供 routes.py 使用。
所有模型字段类型严格，便于 OpenAPI 文档自动生成。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求。"""

    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(None, description="会话 ID（可选，用于追踪）")


class SourceDoc(BaseModel):
    """检索到的来源文档。"""

    content: str = Field(..., description="文档片段内容")
    source: str = Field("unknown", description="来源文件名")
    score: float = Field(0.0, description="相关性分数")


class ResponseMetrics(BaseModel):
    """单次响应的性能指标。"""

    retrieval_latency_ms: float = Field(0.0, description="检索延迟(ms)")
    e2e_latency_ms: float = Field(0.0, description="端到端延迟(ms)")
    token_usage: dict = Field(default_factory=dict, description="token 消耗明细")


class ChatResponse(BaseModel):
    """对话响应（同步端点）。"""

    answer: str = Field(..., description="生成的回答")
    sources: list[SourceDoc] = Field(default_factory=list, description="检索到的来源文档")
    agent_trace: list[str] = Field(default_factory=list, description="Agent 决策轨迹")
    metrics: ResponseMetrics = Field(default_factory=ResponseMetrics, description="性能指标")
    session_id: str | None = Field(None, description="会话 ID（回传）")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field("ok", description="服务状态")
    version: str = Field("0.1.0", description="服务版本")
