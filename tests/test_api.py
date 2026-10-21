"""API 横切能力测试：request_id / 鉴权 / 限流 / 超时 / 健康检查"""

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.middleware import AccessControlMiddleware, RequestContextMiddleware
from src.api.routes import router
from src.config import settings


class FakeGraph:
    def __init__(self, delay: float = 0.0, clarification: str | None = None) -> None:
        self.delay = delay
        self.clarification = clarification

    async def ainvoke(self, state, config=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        return {
            "answer": "" if self.clarification else "测试回答",
            "clarification_question": self.clarification,
            "ambiguity_type": "missing_slot" if self.clarification else None,
            "documents": [],
            "retrieval_scores": [],
            "messages": ["analyze: ok"],
            "is_faithful": True,
            "iteration_count": 1,
        }


def _make_client(graph: FakeGraph) -> TestClient:
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    app.state.graph = graph
    app.state.ready = True
    return TestClient(app)


def test_health_components_and_request_id():
    client = _make_client(FakeGraph())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["graph"] == "ready"


def test_chat_response_includes_request_id():
    client = _make_client(FakeGraph())
    response = client.post("/chat", json={"query": "你好"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "测试回答"
    assert body["request_id"]
    assert response.headers.get("X-Request-ID") == body["request_id"]


def test_chat_clarification_response():
    client = _make_client(FakeGraph(clarification="你指的是哪个对象？"))
    response = client.post("/chat", json={"query": "它怎么用？"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ""
    assert body["clarification"]["question"] == "你指的是哪个对象？"


def test_api_auth(monkeypatch):
    monkeypatch.setattr(settings, "api_auth_token", SecretStr("secret"))
    client = _make_client(FakeGraph())
    assert client.post("/chat", json={"query": "hi"}).status_code == 401
    ok = client.post("/chat", json={"query": "hi"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    # health 豁免
    assert client.get("/health").status_code == 200


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "api_auth_token", SecretStr(""))
    monkeypatch.setattr(settings, "api_rate_limit_per_minute", 1)
    AccessControlMiddleware._hits.clear()
    client = _make_client(FakeGraph())
    assert client.post("/chat", json={"query": "hi"}).status_code == 200
    assert client.post("/chat", json={"query": "hi"}).status_code == 429


def test_chat_timeout(monkeypatch):
    monkeypatch.setattr(settings, "api_request_timeout", 0.01)
    client = _make_client(FakeGraph(delay=0.05))
    response = client.post("/chat", json={"query": "hi"})
    assert response.status_code == 504
