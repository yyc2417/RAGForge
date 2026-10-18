"""会话数据模型

会话以 JSON 文件落盘；消息保留回答、来源、trace、指标与反馈字段。
"""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

FeedbackRating = Literal["up", "down"]
MessageType = Literal["answer", "clarify", "error"]


def utc_now() -> str:
    """返回 ISO8601 UTC 时间字符串（秒级）。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    """生成 UUID4 字符串。"""
    return str(uuid4())


class SessionMessage(BaseModel):
    """一条会话消息；assistant 消息可携带来源、trace 与反馈。"""

    id: str = Field(default_factory=new_id)
    role: Literal["user", "assistant"]
    content: str
    created_at: str = Field(default_factory=utc_now)
    message_type: MessageType = "answer"
    original_query: str | None = None
    resolved_query: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    feedback: FeedbackRating | None = None
    archived: bool = False


class Session(BaseModel):
    """一个完整会话。"""

    session_id: str
    title: str = "新会话"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionMeta(BaseModel):
    """历史会话列表项。"""

    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
