"""检索质量评估节点（诊断式）：输出 failure_mode + suggested_action

输入状态：query, documents, retrieval_scores, retrieval_strategy
输出状态：failure_mode, suggested_action, diagnosis_reason, messages

诊断逻辑融合两路信号：
1. 规则层：绝对分数阈值（仅 vector 策略）+ 空结果硬规则
2. LLM 层：语义判断 irrelevant vs sufficient
两者结合给出最终 failure_mode + suggested_action。

分数语义按策略分治（详见 ADR-006）：
- vector：cosine relevance score ∈ [0,1]，绝对阈值 SCORE_LOW_RECALL_THRESHOLD 有意义
- bm25：分数无上界；hybrid：RRF 融合分上限约 2/(rrf_k+1) ≈ 0.033
  这两类策略的绝对分数不可与 0.3 阈值比较，只依赖空结果硬规则 + LLM 语义诊断，
  且分数不进入任何 LLM prompt（避免 rerank 置零/量纲差异误导评估者）。
"""

from langchain_core.documents import Document

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import EvaluateResult
from src.utils.logger import logger

# 规则阈值：仅用于 vector 策略的 cosine relevance score（0~1）
SCORE_LOW_RECALL_THRESHOLD = 0.3
# 无有效分数时的占位值（仅用于日志展示）
SCORE_NO_HITS = -1.0


def evaluate_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts: PromptManager,
) -> dict:
    """诊断检索质量，决定后续动作（proceed / reformulate / switch_strategy）。

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
    strategy = state.get("retrieval_strategy", "vector")

    # ── 规则层（硬规则）：检索为空，任何策略都判召回不足 ──
    if not documents:
        failure_mode, action = "low_recall", "switch_strategy"
        reason = "检索返回 0 条文档，召回严重不足"
        msg = f"evaluate: failure_mode={failure_mode}, action={action}（空结果→切策略）"
        logger.info(f"[evaluate] {msg}")
        return {
            "failure_mode": failure_mode,
            "suggested_action": action,
            "diagnosis_reason": reason,
            "messages": [msg],
        }

    # ── 规则层（分数规则）：仅对 vector 策略生效 ──
    valid_scores = [s for s in scores if s > 0]
    avg_score = (
        sum(valid_scores) / len(valid_scores) if valid_scores else SCORE_NO_HITS
    )
    score_rule_applicable = strategy == "vector" and avg_score >= 0

    # ── LLM 层：语义诊断（不注入分数，避免量纲误导）──
    retrieved_docs = _format_docs(documents)
    try:
        result: EvaluateResult = llm.invoke_structured(
            prompts.EVALUATE_PROMPT, EvaluateResult,
            query=query, retrieved_docs=retrieved_docs,
        )
        llm_failure = result.failure_mode
        llm_action = result.suggested_action
        reason = result.reason
    except Exception as e:  # noqa: BLE001 - LLM 失败时用规则层兜底
        logger.warning(f"[evaluate] LLM 诊断失败，回退规则层：{e}")
        if score_rule_applicable and avg_score < SCORE_LOW_RECALL_THRESHOLD:
            llm_failure, llm_action = "low_recall", "switch_strategy"
            reason = f"平均相关性分数 {avg_score:.3f} < {SCORE_LOW_RECALL_THRESHOLD}（规则兜底）"
        else:
            llm_failure, llm_action = "sufficient", "proceed"
            reason = "LLM 诊断失败，默认 sufficient（规则兜底）"

    # ── 融合：vector 策略的分数硬证据优先 ──
    if (
        score_rule_applicable
        and avg_score < SCORE_LOW_RECALL_THRESHOLD
        and llm_failure != "irrelevant"
    ):
        failure_mode, action = "low_recall", "switch_strategy"
        reason = f"规则层触发：平均分数 {avg_score:.3f} 过低 → 切换策略 | LLM: {reason}"
    else:
        failure_mode, action = llm_failure, llm_action

    msg = (
        f"evaluate: strategy={strategy}, failure_mode={failure_mode}, "
        f"action={action}, avg_score={avg_score:.3f}"
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
