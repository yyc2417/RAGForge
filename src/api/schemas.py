"""API 请求/响应 Pydantic 模型

定义 HTTP 接口的输入输出 schema，供 routes.py 使用。
所有模型字段类型严格，便于 OpenAPI 文档自动生成。
"""

from pydantic import BaseModel, Field, field_validator

from src import __version__


class ChatRequest(BaseModel):
    """对话请求。"""

    query: str = Field(..., min_length=1, max_length=4000, description="用户问题（1~4000 字符）")
    session_id: str | None = Field(None, description="会话 ID（可选，用于追踪）")

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        """去除首尾空白；纯空白输入直接拒绝（避免空查询触发无效检索）。"""
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空白字符")
        return value


class Clarification(BaseModel):
    """clarify 终止分支的响应：本轮只返回澄清问题。"""

    question: str = Field(..., description="向用户提出的澄清问题")
    ambiguity_type: str | None = Field(None, description="歧义类型")


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

    answer: str = Field("", description="生成的回答；clarify 时为空")
    clarification: Clarification | None = Field(None, description="需要澄清时的问题")
    generation_failed: bool = Field(
        False, description="回答是否为生成失败后的降级文案"
    )
    generation_error: str | None = Field(None, description="生成失败原因（截断）")
    sources: list[SourceDoc] = Field(default_factory=list, description="检索到的来源文档")
    agent_trace: list[str] = Field(default_factory=list, description="Agent 决策轨迹")
    metrics: ResponseMetrics = Field(default_factory=ResponseMetrics, description="性能指标")
    session_id: str | None = Field(None, description="会话 ID（回传）")
    request_id: str | None = Field(None, description="请求链路 ID")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field("ok", description="服务状态：ok=就绪，starting=构建中")
    version: str = Field(__version__, description="服务版本")
    components: dict = Field(default_factory=dict, description="分组件状态")
