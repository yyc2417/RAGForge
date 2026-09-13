"""测试用 Fake 组件（无网络，供单元测试复用）

- FakeLLM：按 schema 名称脚本化返回结果的 LLMClient stub
- FakeRetriever：固定返回文档的检索器 stub
"""

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from src.generation.schemas import (
    AnalyzeResult,
    EvaluateResult,
    ReformulateResult,
    VerifyResult,
)


class FakeLLM:
    """按 schema 名称脚本化返回结果的 LLM stub。

    script 值可以是 Pydantic 实例（恒定返回）或 factory 函数（按 kwargs 动态返回，
    用于实现「第一次 False 之后 True」之类的有状态脚本）。
    """

    def __init__(self, script: dict[str, Any] | None = None) -> None:
        self.script: dict[str, Any] = dict(script or {})
        self.structured_calls: list[str] = []
        self.structured_kwargs: list[dict[str, Any]] = []
        self.generate_calls: list[tuple[str, int]] = []

    def invoke_structured(self, prompt, schema, **kwargs):
        self.structured_calls.append(schema.__name__)
        self.structured_kwargs.append(kwargs)
        factory = self.script.get(schema.__name__)
        if factory is None:
            raise AssertionError(f"FakeLLM 未配置 {schema.__name__} 的返回脚本")
        return factory(**kwargs) if callable(factory) else factory

    def generate(
        self, query: str, context: list[Document], scores: list[float] | None = None
    ) -> str:
        self.generate_calls.append((query, len(context)))
        return "测试回答"


class FakeRetriever:
    """固定返回 n 篇高分文档的检索器 stub，接口与 VectorStore/BM25Retriever 一致。"""

    def search_with_scores(self, query: str, k: int | None = None) -> tuple[list[Document], list[float]]:
        n = k if k is not None else 3
        docs = [
            Document(
                page_content=f"内容 {i}（{query}）",
                metadata={"source": "fake.md", "chunk_index": i},
            )
            for i in range(n)
        ]
        return docs, [0.9] * n

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return self.search_with_scores(query, k)[0]


def default_script() -> dict[str, Any]:
    """默认脚本：factual 查询 → 检索充足 → 生成 → 验证通过。"""
    return {
        "AnalyzeResult": AnalyzeResult(query_type="factual", needs_retrieval=True),
        "EvaluateResult": EvaluateResult(failure_mode="sufficient", suggested_action="proceed"),
        "ReformulateResult": lambda **kw: ReformulateResult(
            reformulated_query="改写查询", rewrite_strategy="specify"
        ),
        "VerifyResult": VerifyResult(is_faithful=True),
    }


def make_script(**overrides: Any) -> dict[str, Any]:
    """在默认脚本基础上覆盖指定 schema 的返回。"""
    script = default_script()
    script.update(overrides)
    return script


def scripted_verify(results: list[bool]) -> Callable[..., VerifyResult]:
    """有状态 verify 脚本：按调用次数依次返回 results，超出后返回最后一个。"""
    calls = {"n": 0}

    def _factory(**kw) -> VerifyResult:
        idx = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return VerifyResult(is_faithful=results[idx])

    return _factory
