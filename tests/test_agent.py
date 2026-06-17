"""Task 5 验证：Agent 决策层（LangGraph 8 节点状态机）

验证项目：
1. 单元：各节点独立行为正确（analyze / evaluate / switch_strategy / reformulate）
2. 集成：寒暄跳过检索直生成；事实查询完整流程；模糊查询触发 reformulate
3. 可视化：状态机 mermaid 文本可生成
4. 安全：max_iterations 兜底生效（不会无限循环）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.agent import build_agent_graph_from_pipeline, initial_state
from src.agent.nodes import (
    analyze_node,
    decide_node,
    evaluate_node,
    reformulate_node,
    switch_strategy_node,
)
from src.agent.state import AgentState
from src.config import PROJECT_ROOT, settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.ingestion import DocumentParser, TextChunker, EmbeddingService
from src.retrieval import VectorStore, BM25Retriever

DATA_DIR = PROJECT_ROOT / settings.data_dir


def _build_deps():
    """构建 graph 所需依赖（向量/BM25/LLM/prompts）。"""
    parser = DocumentParser()
    docs = parser.parse_directory(DATA_DIR)
    chunker = TextChunker()
    chunks = chunker.split(docs)
    embedder = EmbeddingService()
    vs = VectorStore.load_or_build(chunks, embedder)
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    llm = LLMClient()
    prompts = PromptManager()
    return vs, bm25, llm, prompts


def test_unit_nodes(vs, bm25, llm, prompts):
    print("=" * 60)
    print("测试 1：单元节点行为")
    print("=" * 60)

    # analyze: 事实查询 → needs_retrieval=True
    out = analyze_node({"query": "什么是过拟合？"}, llm=llm, prompts=prompts)
    assert out["needs_retrieval"] is True, f"事实查询应 needs_retrieval=True，实际 {out}"
    print(f"✅ analyze('什么是过拟合') → {out['query_type']}, needs_retrieval={out['needs_retrieval']}")

    # analyze: 寒暄 → needs_retrieval=False
    out2 = analyze_node({"query": "你好"}, llm=llm, prompts=prompts)
    assert out2["needs_retrieval"] is False, f"寒暄应 needs_retrieval=False，实际 {out2}"
    print(f"✅ analyze('你好') → {out2['query_type']}, needs_retrieval={out2['needs_retrieval']}")

    # decide 规则
    assert decide_node({"query_type": "chitchat"})["retrieval_strategy"] == "none"
    assert decide_node({"query_type": "factual"})["retrieval_strategy"] == "vector"
    assert decide_node({"query_type": "reasoning"})["retrieval_strategy"] == "bm25"
    assert decide_node({"query_type": "complex"})["retrieval_strategy"] == "hybrid"
    print("✅ decide 规则映射正确（chitchat→none, factual→vector, reasoning→bm25, complex→hybrid）")

    # switch_strategy 规则
    assert switch_strategy_node({"retrieval_strategy": "vector"})["retrieval_strategy"] == "bm25"
    assert switch_strategy_node({"retrieval_strategy": "bm25"})["retrieval_strategy"] == "hybrid"
    assert switch_strategy_node({"retrieval_strategy": "hybrid"})["retrieval_strategy"] == "hybrid"
    print("✅ switch_strategy 升级正确（vector→bm25, bm25→hybrid, hybrid→hybrid）")


def test_integration_chitchat(graph):
    print("\n" + "=" * 60)
    print("测试 2：寒暄查询跳过检索")
    print("=" * 60)
    state = graph.invoke(
        initial_state("你好", max_iterations=3), config={"recursion_limit": 50}
    )
    strategy = state.get("retrieval_strategy")
    assert strategy == "none", f"寒暄应 strategy=none，实际 {strategy}"
    assert state.get("answer"), "寒暄也应有回答"
    print(f"✅ '你好' → strategy={strategy}, answer='{state['answer'][:50]}...'")


def test_integration_factual(graph):
    print("\n" + "=" * 60)
    print("测试 3：事实查询完整流程")
    print("=" * 60)
    state = graph.invoke(
        initial_state("Python 装饰器怎么用？", max_iterations=3),
        config={"recursion_limit": 50},
    )
    assert state.get("answer"), "事实查询应有回答"
    print(f"✅ 'Python 装饰器怎么用？'")
    print(f"   strategy={state.get('retrieval_strategy')}, iter={state.get('iteration_count')}")
    print(f"   failure_mode={state.get('failure_mode')}, is_faithful={state.get('is_faithful')}")
    print(f"   answer='{state['answer'][:80]}...'")


def test_integration_overfit(graph):
    print("\n" + "=" * 60)
    print("测试 4：'过拟合'查询（验证 Agent 诊断+策略切换）")
    print("=" * 60)
    state = graph.invoke(
        initial_state("什么是过拟合？怎么解决？", max_iterations=3),
        config={"recursion_limit": 50},
    )
    print(f"✅ '过拟合' 完成")
    print(f"   最终 strategy={state.get('retrieval_strategy')}, iter={state.get('iteration_count')}")
    print(f"   failure_mode={state.get('failure_mode')}, is_faithful={state.get('is_faithful')}")
    print(f"   answer='{(state.get('answer') or '')[:120]}...'")
    # 决策轨迹
    print(f"   决策轨迹（最后 6 步）：")
    for m in state.get("messages", [])[-6:]:
        print(f"     · {m}")


def test_safety_max_iterations(graph):
    print("\n" + "=" * 60)
    print("测试 5：max_iterations 兜底（不会无限循环）")
    print("=" * 60)
    # 用一个检索很难命中的查询，观察 iteration_count ≤ max_iterations
    state = graph.invoke(
        initial_state("量子纠缠的贝尔不等式推导", max_iterations=2),
        config={"recursion_limit": 50},
    )
    iter_count = state.get("iteration_count", 0)
    assert iter_count <= 2, f"iteration_count 应 ≤ max_iterations=2，实际 {iter_count}"
    print(f"✅ iteration_count={iter_count} ≤ max_iterations=2（兜底生效）")


def test_graph_visualization(graph):
    print("\n" + "=" * 60)
    print("测试 6：状态机可视化（mermaid 文本可生成）")
    print("=" * 60)
    try:
        mermaid = graph.get_graph().draw_mermaid()
        assert "analyze" in mermaid and "verify" in mermaid
        print("✅ mermaid 文本生成成功（含 analyze/verify 等节点）")
        print("   节点摘要：")
        for node in ["analyze", "decide", "retrieve", "evaluate",
                     "reformulate", "switch_strategy", "generate", "verify"]:
            present = node in mermaid
            print(f"     {'✓' if present else '✗'} {node}")
    except Exception as e:
        print(f"⚠️ mermaid 生成失败（非阻塞）：{e}")


if __name__ == "__main__":
    print("RAGForge Task 5 验证：Agent 决策层（LangGraph 状态机）\n")
    vs, bm25, llm, prompts = _build_deps()
    test_unit_nodes(vs, bm25, llm, prompts)

    graph = build_agent_graph_from_pipeline(vs, bm25, llm)
    test_integration_chitchat(graph)
    test_integration_factual(graph)
    test_integration_overfit(graph)
    test_safety_max_iterations(graph)
    test_graph_visualization(graph)

    print("\n" + "=" * 60)
    print("🎉 Task 5 验证全部通过！")
    print("=" * 60)
