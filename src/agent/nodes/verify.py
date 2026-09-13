"""一致性校验节点：LLM 检验回答是否忠实于检索文档（幻觉检测）

输入状态：query, answer, documents
输出状态：is_faithful, verification_reason, messages
"""

from src.agent.state import AgentState
from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import VerifyResult
from src.utils.logger import logger
from src.utils.token_budget import fit_to_budget


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
    documents = fit_to_budget(
        documents, settings.context_token_budget, scores=state.get("retrieval_scores")
    )
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "（无检索文档）"

    try:
        result: VerifyResult = llm.invoke_structured(
            prompts.VERIFY_PROMPT, VerifyResult,
            query=query, answer=answer, context=context,
        )
        is_faithful = result.is_faithful
        reason = result.reason
    except Exception as e:  # noqa: BLE001 - LLM 失败时标记为未验证，不做真伪判定
        # 默认 True 会向评估偏置（幻觉被美化），None 表示「未验证」，
        # 路由层将 None 视为通过（避免误触发重检索循环），评估层单独归类
        logger.warning(f"[verify] LLM 校验失败，标记为未验证：{e}")
        is_faithful = None
        reason = "校验失败，未验证"

    # 幻觉重试计数唯一递增点：改写无效短路后 verify→reformulate→generate
    # 循环不经过 retrieve，需要独立计数器保证终止（与检索计数解耦）
    verify_failures = state.get("verify_failures", 0)
    if is_faithful is False:
        verify_failures += 1

    msg = f"verify: is_faithful={is_faithful} | verify_failures={verify_failures} | {reason}"
    logger.info(f"[verify] {msg}")
    return {
        "is_faithful": is_faithful,
        "verification_reason": reason,
        "verify_failures": verify_failures,
        "messages": [msg],
    }
