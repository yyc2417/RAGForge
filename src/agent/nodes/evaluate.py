"""检索质量评估节点（诊断式）：输出 failure_mode + suggested_action

输入状态：query, documents, retrieval_scores
输出状态：failure_mode, suggested_action, diagnosis_reason, messages

诊断逻辑融合两路信号：
1. 规则层：检索分数阈值（向量 relevance score / BM25 分数）判断 low_recall
2. LLM 层：语义判断 irrelevant vs sufficient
两者结合给出最终 failure_mode + suggested_action。
"""

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import EvaluateResult
from src.utils.logger import logger

# 规则阈值：向量 relevance score 低于此值视为召回不足
SCORE_LOW_RECALL_THRESHOLD = 0.3
# 文档为空时直接判为 low_recall
SCORE_NO_HITS = -1.0


def evaluate_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts: PromptManager,
) -> dict:
    """诊断检索质量，决定后续动作（proceed / reformulate / switch_strategy）。

    Args:
        state: 全局状态（读 query, documents, retrieval_scores）
        llm: LLM 客户端
        prompts: Prompt 管理器

    Returns:
        部分状态更新：failure_mode, suggested_action, diagnosis_reason, messages
    """
    query = state["query"]
    documents = state.get("documents", [])
    scores = state.get("retrieval_scores", [])

    # ── 规则层：先看召回是否充足 ──
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

    # 平均相关性分数（过滤掉 rerank 后置零的 0.0，避免误判）
    valid_scores = [s for s in scores if s > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else SCORE_NO_HITS

    # ── LLM 层：语义诊断 ──
    retrieved_docs = _format_docs(documents, scores)
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
        # 规则兜底：分数普遍低 → low_recall+switch，否则 sufficient+proceed
        if avg_score >= 0 and avg_score < SCORE_LOW_RECALL_THRESHOLD:
            llm_failure, llm_action = "low_recall", "switch_strategy"
            reason = f"平均相关性分数 {avg_score:.3f} < {SCORE_LOW_RECALL_THRESHOLD}（规则兜底）"
        else:
            llm_failure, llm_action = "sufficient", "proceed"
            reason = "LLM 诊断失败，默认 sufficient（规则兜底）"

    # ── 融合：规则层的 low_recall 信号优先（分数硬证据）──
    if avg_score >= 0 and avg_score < SCORE_LOW_RECALL_THRESHOLD and llm_failure != "irrelevant":
        failure_mode, action = "low_recall", "switch_strategy"
        reason = f"规则层触发：平均分数 {avg_score:.3f} 过低 → 切换策略 | LLM: {reason}"
    else:
        failure_mode, action = llm_failure, llm_action

    msg = f"evaluate: failure_mode={failure_mode}, action={action}, avg_score={avg_score:.3f}"
    logger.info(f"[evaluate] {msg}")
    return {
        "failure_mode": failure_mode,
        "suggested_action": action,
        "diagnosis_reason": reason,
        "messages": [msg],
    }


def _format_docs(documents, scores) -> str:
    """把检索结果格式化为 prompt 文本。"""
    lines = []
    for i, doc in enumerate(documents):
        score = scores[i] if i < len(scores) else 0.0
        snippet = doc.page_content[:150].replace("\n", " ")
        lines.append(f"[{i+1}] (score={score:.3f}) {snippet}...")
    return "\n".join(lines)
