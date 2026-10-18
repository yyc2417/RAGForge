"""检索策略决策节点：纯规则，固定 hybrid / none

输入状态：query_type, needs_retrieval
输出状态：retrieval_strategy, messages

决策规则：
- chitchat 或 needs_retrieval=False → none（跳过检索直接生成）
- 其余查询 → hybrid（向量 + BM25 + RRF；固定策略，不再逐级升级）
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def decide_node(state: AgentState) -> dict:
    """根据是否需要检索规则化输出 none / hybrid（无 LLM 调用）。

    Args:
        state: 全局状态（读 query_type, needs_retrieval）

    Returns:
        部分状态更新：retrieval_strategy, messages
    """
    query_type = state.get("query_type", "factual")
    needs_retrieval = state.get("needs_retrieval", True)

    if not needs_retrieval or query_type == "chitchat":
        strategy = "none"
    else:
        strategy = "hybrid"

    msg = f"decide: query_type={query_type} → strategy={strategy}"
    logger.info(f"[decide] {msg}")
    return {
        "retrieval_strategy": strategy,
        "messages": [msg],
    }
