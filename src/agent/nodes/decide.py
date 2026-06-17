"""检索策略决策节点：纯规则，根据 query_type 选检索策略

输入状态：query_type, needs_retrieval
输出状态：retrieval_strategy, messages

决策规则：
- chitchat 或 needs_retrieval=False → none（跳过检索直接生成）
- factual → vector（向量相似度适合事实查询）
- reasoning → bm25（关键词检索适合精确术语推理）
- complex → hybrid（混合检索覆盖多角度）
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def decide_node(state: AgentState) -> dict:
    """根据查询类型规则化决策检索策略（无 LLM 调用）。

    Args:
        state: 全局状态（读 query_type, needs_retrieval）

    Returns:
        部分状态更新：retrieval_strategy, messages
    """
    query_type = state.get("query_type", "factual")
    needs_retrieval = state.get("needs_retrieval", True)

    if not needs_retrieval or query_type == "chitchat":
        strategy = "none"
    elif query_type == "factual":
        strategy = "vector"
    elif query_type == "reasoning":
        strategy = "bm25"
    else:  # complex 或未知
        strategy = "hybrid"

    msg = f"decide: query_type={query_type} → strategy={strategy}"
    logger.info(f"[decide] {msg}")
    return {
        "retrieval_strategy": strategy,
        "messages": [msg],
    }
