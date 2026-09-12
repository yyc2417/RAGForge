"""Agent 管道构建与执行

封装完整的 Agent 状态机管道构建（含混合检索）和执行逻辑。
从 main.py 迁出，使 main.py 保持精简。
"""

import time

from langgraph.graph.state import CompiledStateGraph

from src.agent import build_agent_graph_from_pipeline, initial_state
from src.config import PROJECT_ROOT, settings
from src.generation.llm_client import LLMClient
from src.ingestion import DocumentParser, TextChunker, EmbeddingService
from src.retrieval import VectorStore, BM25Retriever, HybridRetriever, Reranker
from src.utils.logger import logger
from src.utils.metrics import MetricsCollector, reset_current_collector, set_current_collector

DATA_DIR = PROJECT_ROOT / settings.data_dir


def build_agent_pipeline() -> CompiledStateGraph:
    """构建完整 Agent 状态机管道：ingestion → retrieval(vector+bm25+hybrid) → agent graph。"""
    parser = DocumentParser()
    docs = parser.parse_directory(DATA_DIR)
    chunker = TextChunker()
    chunks = chunker.split(docs)
    embedder = EmbeddingService()
    vector_store = VectorStore.load_or_build(chunks, embedder)

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    # 混合检索（RRF 融合）+ 重排序（优雅降级：模型不可用自动跳过）
    hybrid = HybridRetriever(vector_store, bm25)
    reranker = Reranker()

    llm = LLMClient()
    graph = build_agent_graph_from_pipeline(vector_store, bm25, llm, hybrid, reranker)
    logger.info("[pipeline] Agent 状态机管道就绪（含混合检索）")
    return graph


def ask_agent(graph, question: str, max_iterations: int = 3) -> tuple[str, dict]:
    """对单个问题执行 Agent 状态机，返回 (回答, 完整最终状态)。"""
    # 独立收集器注入当前上下文：节点内 token/延迟记录都落到本实例
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
