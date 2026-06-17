"""Agent 状态定义

AgentState 是 LangGraph 状态机的全局状态容器，贯穿 8 个节点。
每个节点读取所需字段、返回需要更新的字段（部分状态更新）。

关键字段说明：
- failure_mode + suggested_action：诊断式评估的核心，驱动策略切换/改写
- iteration_count / max_iterations：防死循环（默认最多 3 轮重检索）
- messages：决策轨迹日志（Annotated[list, add] 自动追加）
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document

# ── 字面量类型（与各节点输出严格对应）────────────────────────────
QueryType = Literal["factual", "reasoning", "chitchat", "complex"]
RetrievalStrategy = Literal["vector", "bm25", "hybrid", "none"]
FailureMode = Literal["low_recall", "irrelevant", "sufficient"]
SuggestedAction = Literal["reformulate", "switch_strategy", "proceed"]
RewriteStrategy = Literal["specify", "generalize", "synonym_replace"]


class AgentState(TypedDict, total=False):
    """LangGraph 全局状态。total=False 允许节点只更新部分字段。"""

    # ── 输入 ──
    query: str

    # ── analyze 节点输出 ──
    query_type: QueryType
    needs_retrieval: bool

    # ── decide 节点输出 ──
    retrieval_strategy: RetrievalStrategy

    # ── retrieve 节点输出 ──
    documents: list[Document]
    retrieval_scores: list[float]

    # ── evaluate 节点输出（诊断式评估）──
    failure_mode: FailureMode
    suggested_action: SuggestedAction
    diagnosis_reason: str

    # ── reformulate 节点输出 ──
    reformulated_query: str
    rewrite_strategy: RewriteStrategy

    # ── generate 节点输出 ──
    answer: str

    # ── verify 节点输出 ──
    is_faithful: bool
    verification_reason: str

    # ── 策略切换辅助 ──
    previous_strategy: RetrievalStrategy

    # ── 流程控制 ──
    iteration_count: int
    max_iterations: int
    messages: Annotated[list[str], add]


def initial_state(query: str, max_iterations: int = 3) -> AgentState:
    """构造一个初始 AgentState（含默认流程控制字段）。

    Args:
        query: 用户原始查询
        max_iterations: 最大重检索轮数（默认 3）

    Returns:
        初始化后的 AgentState
    """
    return {
        "query": query,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "messages": [],
    }
