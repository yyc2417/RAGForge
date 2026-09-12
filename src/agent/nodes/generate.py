"""答案生成节点：调用 LLMClient 基于 context 生成答案

输入状态：query, documents
输出状态：answer, messages
"""

import time

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.utils.logger import logger

# LLM 生成彻底失败时的降级文案（本节点是全链路唯一无前置兜底的 LLM 调用，
# 不能让异常炸穿整个图作废此前所有轮次的检索与诊断成果）
_FALLBACK_ANSWER = "抱歉，回答生成过程中出现问题，请稍后重试或换个问法。"


def generate_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts=None,
) -> dict:
    """基于检索上下文生成回答（LLM 失败时返回降级文案）。

    Args:
        state: 全局状态（读 query, documents）
        llm: LLM 客户端
        prompts: Prompt 管理器（generate 复用 LLMClient 内置 QA_PROMPT，此参数保留以对齐节点签名）

    Returns:
        部分状态更新：answer, messages
    """
    query = state["query"]
    documents = state.get("documents", [])

    t0 = time.perf_counter()
    try:
        answer_text = llm.generate(query, documents)
    except Exception as e:  # noqa: BLE001 - 降级兜底，保证状态机总能产出答案
        logger.warning(f"[generate] LLM 生成失败，返回降级文案：{e}")
        answer_text = _FALLBACK_ANSWER
    latency_ms = (time.perf_counter() - t0) * 1000

    # 端到端延迟在最终汇总处记录，这里只记录生成耗时（不入指标，避免与 e2e 重复）
    msg = f"generate: answer_len={len(answer_text)}, latency={latency_ms:.0f}ms"
    logger.info(f"[generate] {msg}")
    return {
        "answer": answer_text,
        "messages": [msg],
    }
