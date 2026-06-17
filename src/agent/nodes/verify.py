"""一致性校验节点：LLM 检验回答是否忠实于检索文档（幻觉检测）

输入状态：query, answer, documents
输出状态：is_faithful, verification_reason, messages
"""

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import VerifyResult
from src.utils.logger import logger


def verify_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts: PromptManager,
) -> dict:
    """检测回答中是否存在幻觉（无文档支撑的编造）。

    Args:
        state: 全局状态（读 query, answer, documents）
        llm: LLM 客户端
        prompts: Prompt 管理器

    Returns:
        部分状态更新：is_faithful, verification_reason, messages
    """
    query = state["query"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "（无检索文档）"

    try:
        result: VerifyResult = llm.invoke_structured(
            prompts.VERIFY_PROMPT, VerifyResult,
            query=query, answer=answer, context=context,
        )
        is_faithful = result.is_faithful
        reason = result.reason
    except Exception as e:  # noqa: BLE001 - LLM 失败时默认通过（避免误触发重检索循环）
        logger.warning(f"[verify] LLM 校验失败，默认 is_faithful=True：{e}")
        is_faithful = True
        reason = "校验失败，默认通过"

    # 幻觉检测不通过时递增迭代计数，确保 max_iterations 防线生效
    # （防止 verify→reformulate 回退循环中计数不增长导致无限循环）
    if not is_faithful:
        iteration_count = state.get("iteration_count", 0) + 1
    else:
        iteration_count = state.get("iteration_count", 0)

    msg = f"verify: is_faithful={is_faithful} | iter={iteration_count} | {reason}"
    logger.info(f"[verify] {msg}")
    return {
        "is_faithful": is_faithful,
        "verification_reason": reason,
        "iteration_count": iteration_count,
        "messages": [msg],
    }
