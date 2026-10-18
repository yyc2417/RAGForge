"""API 横切能力：request_id、可选 API key、进程内限流。"""

import secrets
import time
import uuid
from collections import defaultdict, deque
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.request_context import request_id_var
from src.config import settings
from src.utils.logger import logger

_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/透传 X-Request-ID 并记录入口/出口日志。"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                f"[api] {request.method} {request.url.path} "
                f"status={response.status_code} e2e={elapsed:.0f}ms request_id={request_id}"
            )
            return response
        finally:
            request_id_var.reset(token)


class AccessControlMiddleware(BaseHTTPMiddleware):
    """可选 X-API-Key 鉴权 + 每客户端每分钟滑动窗口限流。"""

    _hits: dict[str, deque[float]] = defaultdict(deque)
    _lock = Lock()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        expected = settings.api_auth_token_value()
        if expected:
            provided = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(provided, expected):
                return JSONResponse(status_code=401, content={"detail": "未授权：X-API-Key 无效"})

        limit = settings.api_rate_limit_per_minute
        if limit > 0:
            host = request.client.host if request.client else "unknown"
            now = time.monotonic()
            with self._lock:
                hits = self._hits[host]
                while hits and now - hits[0] > 60:
                    hits.popleft()
                if len(hits) >= limit:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"请求过于频繁：每分钟最多 {limit} 次"},
                    )
                hits.append(now)

        return await call_next(request)
