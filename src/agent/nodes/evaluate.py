"""检索质量评估节点（诊断式）：输出 failure_mode + suggested_action

输入状态：query, documents, retrieval_scores, retrieval_strategy
输出状态：failure_mode, suggested_action, diagnosis_reason, messages

固定 hybrid 后的诊断逻辑：
1. 规则层：空结果硬规则（召回严重不足 → reformulate）
2. LLM 层：语义判断 insufficient / irrelevant / sufficient
固定 hybrid 的 RRF 融合分不可与绝对阈值比较，且不进入 LLM prompt；
因此不再保留 vector 专属的 0.3 分数阈值规则。

动作只有两种：proceed 直接生成；reformulate 改写查询后重试。
"""

from langchain_core.documents import Document

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import EvaluateResult
from src.utils.logger import logger


def evaluate_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts: PromptManager,
) -> dict:
    """诊断检索质量，决定后续动作（proceed / reformulate）。

    Args:
        state: 全局状态（读 query, documents, retrieval_scores, retrieval_strategy）
        llm: LLM 客户端
        prompts: Prompt 管理器

    Returns:
        部分状态更新：failure_mode, suggested_action, diagnosis_reason, messages
    """
    query = state["query"]
    documents = state.get("documents", [])
    scores = state.get("retrieval_scores", [])
    strategy = state.get("retrieval_strategy", "hybrid")

    # ── 规则层（硬规则）：检索为空，召回严重不足 → 改写查询 ──
    if not documents:
        failure_mode, action = "low_recall", "reformulate"
        reason = "检索返回 0 条文档，召回严重不足"
        msg = f"evaluate: failure_mode={failure_mode}, action={action}（空结果→改写查询）"
        logger.info(f"[evaluate] {msg}")
        return {
            "failure_mode": failure_mode,
            "suggested_action": action,
            "diagnosis_reason": reason,
            "messages": [msg],
        }

    # ── LLM 层：语义诊断（不注入分数，RRF 分量纲不可比）──
    retrieved_docs = _format_docs(documents)
    try:
        result: EvaluateResult = llm.invoke_structured(
            prompts.EVALUATE_PROMPT, EvaluateResult,
            query=query, retrieved_docs=retrieved_docs,
        )
        failure_mode, action = result.failure_mode, result.suggested_action
        reason = result.reason
    except Exception as e:  # noqa: BLE001 - LLM 失败时保守放行，避免误触发循环
        logger.warning(f"[evaluate] LLM 诊断失败，默认 sufficient：{e}")
        failure_mode, action = "sufficient", "proceed"
        reason = "LLM 诊断失败，默认 sufficient（规则兜底）"

    msg = (
        f"evaluate: strategy={strategy}, failure_mode={failure_mode}, "
        f"action={action}, scores={scores[:3]}"
    )
    logger.info(f"[evaluate] {msg}")
    return {
        "failure_mode": failure_mode,
        "suggested_action": action,
        "diagnosis_reason": reason,
        "messages": [msg],
    }


def _format_docs(documents: list[Document]) -> str:
    """把检索结果格式化为 prompt 文本（不含分数：分数已按策略分治，见模块 docstring）。"""
    lines = []
    for i, doc in enumerate(documents):
        snippet = doc.page_content[:150].replace("\n", " ")
        lines.append(f"[{i + 1}] {snippet}...")
    return "\n".join(lines)
