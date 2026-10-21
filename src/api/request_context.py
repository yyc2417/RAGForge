"""请求上下文：request_id 的 ContextVar。"""

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """获取当前请求 ID；非 HTTP 上下文返回 '-'。"""
    return request_id_var.get()
