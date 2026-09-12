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
RewriteStrategy = Literal["specify", "generalize", "synonym_replace"]  # 仅供 reformulate 节点日志使用


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
    # 改写是否有效（False = LLM 失败或改写结果与原查询相同）：
    # False 时路由直接进入 generate，跳过注定重复检索的无效循环
    rewrite_effective: bool

    # ── generate 节点输出 ──
    answer: str

    # ── verify 节点输出 ──
    # is_faithful: True/False 为判定结果；None 表示 LLM 校验失败未验证
    # （路由层将 None 视为通过，评估层单独归类，避免向「忠实」偏置）
    is_faithful: bool | None
    verification_reason: str

    # ── 流程控制 ──
    # 已执行的检索次数（含首次）：retrieve 节点每执行一次 +1（唯一递增点）。
    # max_iterations=3 表示最多 3 次检索（首次 + 2 轮重检索）
    iteration_count: int
    # 幻觉重试次数：verify 判定不忠实时 +1（唯一递增点）。
    # 改写无效短路后 verify→reformulate→generate 循环不经过 retrieve，
    # 必须由独立计数器保证终止
    verify_failures: int
    max_iterations: int
    messages: Annotated[list[str], add]


def initial_state(query: str, max_iterations: int = 3) -> AgentState:
    """构造一个初始 AgentState（含默认流程控制字段）。

    Args:
        query: 用户原始查询
        max_iterations: 最大检索次数上限（默认 3，含首次检索）

    Returns:
        初始化后的 AgentState
    """
    return {
        "query": query,
        "iteration_count": 0,
        "verify_failures": 0,
        "max_iterations": max_iterations,
        "messages": [],
    }
