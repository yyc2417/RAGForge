"""检索策略切换节点：纯规则，升级检索策略以提升召回

输入状态：retrieval_strategy
输出状态：retrieval_strategy, messages

切换规则（单向升级，不回退）：
- vector → bm25（先试轻量的 BM25 关键词补充）
- bm25 → hybrid（引入向量语义 + RRF 融合）
- hybrid → hybrid（已是最强策略，保持不变；路由层会在此时强制 generate，
  不会进入本节点的空转循环）
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def switch_strategy_node(state: AgentState) -> dict:
    """规则化升级检索策略（无 LLM 调用）。

    Args:
        state: 全局状态（读 retrieval_strategy）

    Returns:
        部分状态更新：retrieval_strategy(新), messages
    """
    current = state.get("retrieval_strategy", "vector")

    # 升级映射：vector → bm25 → hybrid（单向，不回退）
    if current == "vector":
        new_strategy = "bm25"  # 先试 BM25（轻量），下一轮再 hybrid
    else:  # bm25 或 hybrid
        new_strategy = "hybrid"

    msg = f"switch_strategy: {current} → {new_strategy}"
    logger.info(f"[switch_strategy] {msg}")
    return {
        "retrieval_strategy": new_strategy,
        "messages": [msg],
    }
