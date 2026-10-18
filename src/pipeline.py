"""Agent 管道构建与执行

封装完整的 Agent 状态机管道构建（含运行时知识库）和执行逻辑。
- build_agent_runtime()：返回 graph + KnowledgeBaseService，供 UI/API 入库后重建索引；
- build_agent_pipeline()：向后兼容，只返回 graph。
"""

import time
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from src.agent import build_agent_graph_from_pipeline, initial_state
from src.generation.llm_client import LLMClient
from src.knowledge import KnowledgeBaseService
from src.utils.logger import logger
from src.utils.metrics import MetricsCollector, reset_current_collector, set_current_collector


@dataclass
class AgentRuntime:
    """图 + 知识库服务；二者共享同一批 retriever 对象。"""

    graph: CompiledStateGraph
    knowledge_base: KnowledgeBaseService


def build_agent_runtime() -> AgentRuntime:
    """构建完整运行时：知识库索引 + 7 主链路 + clarify 分支。"""
    kb = KnowledgeBaseService()
    llm = LLMClient()
    graph = build_agent_graph_from_pipeline(
        kb.vector_store,
        kb.bm25,
        llm,
        kb.hybrid,
        kb.reranker,
    )
    logger.info(
        f"[pipeline] Agent 运行时就绪（固定 hybrid，chunks={kb.chunks_count}）"
    )
    return AgentRuntime(graph=graph, knowledge_base=kb)


def build_agent_pipeline() -> CompiledStateGraph:
    """向后兼容入口：只返回编译后的 graph。"""
    return build_agent_runtime().graph


def ask_agent(graph, question: str, max_iterations: int = 3) -> tuple[str, dict]:
    """对单个问题执行 Agent 状态机，返回 (回答, 完整最终状态)。"""
    metrics = MetricsCollector()
    context_token = set_current_collector(metrics)
    try:
        t0 = time.perf_counter()
        final_state = graph.invoke(
            initial_state(question, max_iterations=max_iterations),
            config={"recursion_limit": 50},
        )
        e2e = (time.perf_counter() - t0) * 1000
    finally:
        reset_current_collector(context_token)

    metrics.record_e2e_latency(e2e)
    answer_text = final_state.get("answer", "")
    logger.info(f"[agent] 完成 | e2e={e2e:.0f}ms | iter={final_state.get('iteration_count', 0)}")
    return answer_text, final_state
