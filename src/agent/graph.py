"""LangGraph 状态机组装：8 节点 + 条件边

状态机拓扑：
  START → analyze → decide
  decide ─(strategy=none)──→ generate ──→ END（chitchat 跳过 verify）
  decide ─(strategy≠none)──→ retrieve → evaluate
  evaluate ──(proceed)──→ generate → verify
  evaluate ──(switch_strategy & 未达上限)──→ switch_strategy → retrieve（循环）
  evaluate ──(switch_strategy & 已是 hybrid)──→ generate（切换无效，短路）
  evaluate ──(reformulate)──→ reformulate ─(改写有效)──→ retrieve（循环）
  reformulate ─(改写无效)──→ generate（跳过注定重复检索的无效循环）
  verify ──(忠实/未验证)──→ END
  verify ──(幻觉 & 未达上限)──→ reformulate（循环）

安全机制：
- iteration_count 由 retrieve 节点唯一递增（每轮真实检索 +1），
  达 max_iterations 后 evaluate/verify 条件边强制走向 generate/END
- verify_failures 由 verify 节点唯一递增（幻觉重试 +1），为「改写无效短路」
  后不经过 retrieve 的 verify→reformulate→generate 循环提供独立终止保证
- 调用方需传 recursion_limit（第二道防线）
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    analyze_node,
    decide_node,
    evaluate_node,
    generate_node,
    reformulate_node,
    retrieve_node,
    switch_strategy_node,
    verify_node,
)
from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.retrieval import BM25Retriever, VectorStore
from src.utils.logger import logger


# ── 路由函数 ──────────────────────────────────────────────────────
def route_after_decide(state: AgentState) -> str:
    """decide 后：strategy=none 跳过检索直接生成，否则检索。"""
    return "generate" if state.get("retrieval_strategy") == "none" else "retrieve"


def route_after_generate(state: AgentState) -> str:
    """generate 后：strategy=none（chitchat）跳过 verify 直接结束。

    chitchat 无检索文档，verify 会因为“无文档支撑”误判为幻觉，
    因此 strategy=none 时直接结束，不走 verify。
    """
    if state.get("retrieval_strategy") == "none":
        logger.info("[route] strategy=none，跳过 verify 直接结束")
        return "end"
    return "verify"


def route_after_evaluate(state: AgentState) -> str:
    """evaluate 后：根据诊断结果三路分支（达上限强制生成）。"""
    action = state.get("suggested_action", "proceed")
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if action == "proceed":
        return "generate"
    if iter_count >= max_iter:
        # 达到检索次数上限，强制进入生成（防死循环）
        logger.warning(f"[route] iter={iter_count}≥{max_iter}，强制 generate")
        return "generate"
    if action == "switch_strategy":
        if state.get("retrieval_strategy") == "hybrid":
            # 已是最强策略，切换无效（相同查询+相同策略=相同结果），
            # 直接生成，避免 switch→retrieve→evaluate 空转
            logger.warning("[route] 已是 hybrid 策略，switch 无效，强制 generate")
            return "generate"
        return "switch_strategy"
    return "reformulate"  # action == "reformulate"


def route_after_reformulate(state: AgentState) -> str:
    """reformulate 后：改写有效则重检索，无效（LLM 失败或与原查询相同）直接生成。"""
    if state.get("rewrite_effective", False):
        return "retrieve"
    logger.warning("[route] 改写无效（LLM 失败或改写结果与原查询相同），直接 generate")
    return "generate"


def route_after_verify(state: AgentState) -> str:
    """verify 后：忠实（或未验证）则结束，确认幻觉则改写重检索（达上限强制结束）。

    is_faithful=None（LLM 校验失败）视为通过：不因验证器故障误触发重检索循环。
    双预算强制终止：verify_failures（幻觉重试上限，覆盖改写无效短路后
    不经过 retrieve 的循环）+ iteration_count（检索次数上限，覆盖
    verify→reformulate→retrieve 对检索预算的消耗）。
    """
    if state.get("is_faithful") is not False:
        return "end"
    max_iter = state.get("max_iterations", 3)
    verify_failures = state.get("verify_failures", 0)
    iter_count = state.get("iteration_count", 0)
    if verify_failures >= max_iter:
        logger.warning(f"[route] verify 后幻觉重试 {verify_failures}≥{max_iter}，强制 END")
        return "end"
    if iter_count >= max_iter:
        logger.warning(f"[route] verify 后检索次数 {iter_count}≥{max_iter}，强制 END")
        return "end"
    return "reformulate"


def build_agent_graph(
    vector_store: VectorStore,
    bm25_retriever: BM25Retriever,
    llm_client: LLMClient,
    prompt_manager: PromptManager,
    hybrid_retriever=None,
    reranker=None,
):
    """组装 LangGraph 状态机。

    Args:
        vector_store: 向量检索器
        bm25_retriever: BM25 检索器
        llm_client: LLM 客户端
        prompt_manager: Prompt 管理器
        hybrid_retriever: 混合检索器（Task 6 可选）
        reranker: 重排序器（Task 6 可选）

    Returns:
        编译后的 LangGraph 可执行图
    """
    graph = StateGraph(AgentState)

    # ── 添加节点（partial 注入依赖）──
    graph.add_node("analyze", partial(analyze_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("decide", decide_node)
    graph.add_node(
        "retrieve",
        partial(
            retrieve_node,
            vector_store=vector_store,
            bm25=bm25_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        ),
    )
    graph.add_node("evaluate", partial(evaluate_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("reformulate", partial(reformulate_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("switch_strategy", switch_strategy_node)
    graph.add_node("generate", partial(generate_node, llm=llm_client))
    graph.add_node("verify", partial(verify_node, llm=llm_client, prompts=prompt_manager))

    # ── 边定义 ──
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "decide")

    # 条件边 1：decide 后是否检索
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "evaluate")

    # 条件边 2：evaluate 后三路分支
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "generate": "generate",
            "reformulate": "reformulate",
            "switch_strategy": "switch_strategy",
        },
    )
    # 循环边：改写/切策略后重新检索（改写无效时短路直达 generate）
    graph.add_conditional_edges(
        "reformulate",
        route_after_reformulate,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("switch_strategy", "retrieve")

    # 生成 → 条件边：chitchat 跳过 verify，其他走 verify
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"verify": "verify", "end": END},
    )

    # 条件边 3：verify 后是否结束
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"end": END, "reformulate": "reformulate"},
    )

    compiled = graph.compile()
    logger.info("[graph] LangGraph 状态机构建完成（8 节点）")
    return compiled


def build_agent_graph_from_pipeline(
    vector_store: VectorStore,
    bm25_retriever: BM25Retriever,
    llm_client: LLMClient,
    hybrid_retriever=None,
    reranker=None,
):
    """便捷封装：内部构造默认 PromptManager 后组装图。"""
    return build_agent_graph(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        llm_client=llm_client,
        prompt_manager=PromptManager(),
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )
