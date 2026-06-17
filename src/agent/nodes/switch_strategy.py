"""检索策略切换节点：纯规则，升级检索策略以提升召回

输入状态：retrieval_strategy, previous_strategy
输出状态：retrieval_strategy, previous_strategy, messages

切换规则（单向升级，不回退）：
- vector → hybrid（向量召回不足时引入 BM25 关键词补充）
- bm25 → hybrid（BM25 召回不足时引入向量语义补充）
- hybrid → hybrid（已是最强策略，保持不变；由 max_iterations 兜底终止）
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def switch_strategy_node(state: AgentState) -> dict:
    """规则化升级检索策略（无 LLM 调用）。

    Args:
        state: 全局状态（读 retrieval_strategy）

    Returns:
        部分状态更新：retrieval_strategy(新), previous_strategy, messages
    """
    current = state.get("retrieval_strategy", "vector")
    previous = current  # 记录切换前的策略

    # 升级映射：任何非 hybrid 策略 → hybrid
    if current == "vector":
        new_strategy = "bm25"  # 先试 BM25（轻量），下一轮再 hybrid
    elif current == "bm25":
        new_strategy = "hybrid"
    else:  # hybrid 已是最强，保持
        new_strategy = "hybrid"

    msg = f"switch_strategy: {previous} → {new_strategy}"
    logger.info(f"[switch_strategy] {msg}")
    # 递增迭代计数，防止 evaluate→switch_strategy→retrieve 循环中计数不增长导致无限循环
    iteration_count = state.get("iteration_count", 0) + 1
    return {
        "retrieval_strategy": new_strategy,
        "previous_strategy": previous,
        "iteration_count": iteration_count,
        "messages": [msg],
    }
