"""执行检索节点：按 retrieval_strategy 调用对应检索器

输入状态：query / reformulated_query, retrieval_strategy
输出状态：documents, retrieval_scores, messages

Task 5：支持 vector / bm25 两种策略。
Task 6：新增 hybrid 策略（HybridRetriever + Reranker），通过 hybrid_retriever/reranker 注入。
"""

import time

from src.agent.state import AgentState
from src.retrieval import VectorStore, BM25Retriever
from src.utils.metrics import get_current_collector
from src.utils.logger import logger


def retrieve_node(
    state: AgentState,
    *,
    vector_store: VectorStore,
    bm25: BM25Retriever,
    hybrid_retriever=None,
    reranker=None,
) -> dict:
    """按策略执行检索，返回文档与分数。

    Args:
        state: 全局状态（读 query/reformulated_query/retrieval_strategy）
        vector_store: 向量检索器
        bm25: BM25 检索器
        hybrid_retriever: 混合检索器（Task 6 注入，可选）
        reranker: 重排序器（Task 6 注入，可选）

    Returns:
        部分状态更新：documents, retrieval_scores, messages
    """
    # 优先使用改写后的查询
    query = state.get("reformulated_query") or state["query"]
    strategy = state.get("retrieval_strategy", "vector")

    t0 = time.perf_counter()

    if strategy == "vector":
        docs, scores = vector_store.search_with_scores(query)
    elif strategy == "bm25":
        docs, scores = bm25.search_with_scores(query)
    elif strategy == "hybrid":
        if hybrid_retriever is None:
            # Task 5 阶段 hybrid 未注入时，退化为向量+BM25 简单拼接去重
            logger.warning("[retrieve] hybrid_retriever 未注入，退化为基础检索")
            v_docs, v_scores = vector_store.search_with_scores(query)
            b_docs, b_scores = bm25.search_with_scores(query)
            docs, scores = _merge_dedupe(v_docs, v_scores, b_docs, b_scores)
        else:
            docs, scores = hybrid_retriever.search_with_scores(query)
            # Task 6：若 reranker 可用则重排序
            if reranker is not None and getattr(reranker, "is_available", lambda: False)():
                # rerank 会重排顺序但返回同一批 Document 对象，
                # 按对象身份回映射原 RRF 分数（置 0 会误导下游展示与评估）
                score_by_doc = {id(doc): s for doc, s in zip(docs, scores)}
                docs = reranker.rerank(query, docs)
                scores = [score_by_doc.get(id(doc), 0.0) for doc in docs]
    else:
        # strategy == none 时不应进入此节点，兜底处理
        docs, scores = [], []

    latency_ms = (time.perf_counter() - t0) * 1000
    get_current_collector().record_retrieval_latency(latency_ms)

    # 迭代计数唯一递增点：每轮真实检索 +1（含首次）。
    # 终止性：任何循环路径（switch/reformulate/幻觉重试）必经本节点，
    # 计数单调递增，达 max_iterations 后路由层强制 generate/END
    iteration_count = state.get("iteration_count", 0) + 1

    msg = (
        f"retrieve: strategy={strategy}, query='{query}', "
        f"hits={len(docs)}, latency={latency_ms:.1f}ms, iter={iteration_count}"
    )
    logger.info(f"[retrieve] {msg}")
    return {
        "documents": docs,
        "retrieval_scores": scores,
        "iteration_count": iteration_count,
        "messages": [msg],
    }


def _merge_dedupe(v_docs, v_scores, b_docs, b_scores):
    """简单合并去重（fallback，非 RRF）。Task 6 用真正的 HybridRetriever 替代。"""
    seen = set()
    docs, scores = [], []
    for doc, score in zip(v_docs, v_scores):
        key = doc.page_content[:50]
        if key not in seen:
            seen.add(key)
            docs.append(doc)
            scores.append(score)
    for doc, score in zip(b_docs, b_scores):
        key = doc.page_content[:50]
        if key not in seen:
            seen.add(key)
            docs.append(doc)
            scores.append(score)
    return docs, scores
