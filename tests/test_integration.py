"""集成测试：真实调用 DeepSeek API——默认跳过

运行方式（会产生真实 API 费用，每次运行约数千 tokens）：
    RUN_INTEGRATION=1 uv run pytest tests/test_integration.py -v
"""

import os

import pytest

from src.agent import build_agent_graph_from_pipeline, initial_state
from src.agent.nodes import analyze_node
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager

requires_llm = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="集成测试需真实 DeepSeek API：设 RUN_INTEGRATION=1 启用",
)


@requires_llm
def test_analyze_real_llm():
    """analyze 节点：真实 LLM 的意图分类与结构化解析。"""
    out = analyze_node(
        {"query": "什么是过拟合？"}, llm=LLMClient(), prompts=PromptManager()
    )
    assert out["needs_retrieval"] is True
    assert out["query_type"] in ("factual", "reasoning", "complex")


@requires_llm
def test_chitchat_skips_retrieval(vector_store, bm25):
    """寒暄查询：strategy=none，跳过检索与 verify 直接结束。"""
    graph = build_agent_graph_from_pipeline(vector_store, bm25, LLMClient())
    state = graph.invoke(initial_state("你好"), config={"recursion_limit": 50})
    assert state.get("retrieval_strategy") == "none"
    assert state.get("answer"), "寒暄也应有回答"


@requires_llm
def test_factual_e2e(vector_store, bm25):
    """事实查询端到端：产出回答，检索次数在上限内。"""
    graph = build_agent_graph_from_pipeline(vector_store, bm25, LLMClient())
    state = graph.invoke(
        initial_state("Python 装饰器怎么用？"), config={"recursion_limit": 50}
    )
    assert state.get("answer")
    assert state.get("iteration_count", 0) <= 3


@requires_llm
def test_max_iterations_bounded(vector_store, bm25):
    """知识库外的生僻查询：状态机在预算内强制收敛。"""
    graph = build_agent_graph_from_pipeline(vector_store, bm25, LLMClient())
    state = graph.invoke(
        initial_state("量子纠缠的贝尔不等式推导", max_iterations=2),
        config={"recursion_limit": 50},
    )
    assert state.get("answer")
    assert state.get("iteration_count", 0) <= 2
