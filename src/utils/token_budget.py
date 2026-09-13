"""上下文 token 预算：检索文档拼入 prompt 前的预算裁剪

此前 llm_client 与 verify 节点把检索文档全量拼进 prompt，没有 token 上限——
RETRIEVAL_K / reranker_top_n 调大或语料变大时会逼近模型窗口上限，
单次调用的 token 费用也无上界。本模块提供：

- count_tokens(text)：cl100k_base 口径的 token 计数（与 chunker.py 编码一致）
- fit_to_budget(docs, budget, scores)：按检索分数从高到低装入预算，
  装不下的丢弃并记日志，保底至少保留 1 篇
"""

import tiktoken
from langchain_core.documents import Document

from src.utils.logger import logger

# 与 chunker.py 的 from_tiktoken_encoder(encoding_name=...) 保持同一编码
_ENCODING_NAME = "cl100k_base"
_encoding: tiktoken.Encoding | None = None


def count_tokens(text: str) -> int:
    """统计文本 token 数（cl100k_base 口径，编码器懒加载并复用）。"""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return len(_encoding.encode(text))


def fit_to_budget(
    docs: list[Document],
    budget: int,
    scores: list[float] | None = None,
) -> list[Document]:
    """把文档列表裁剪到 token 预算内，返回可安全拼入 prompt 的文档。

    规则：
    - 总 token 不超预算时原样返回；
    - 超预算时按检索分数从高到低贪心装入（scores 缺省或与 docs 长度
      不齐时按原序装入——检索结果本身按分数降序，等价丢尾部低分文档）；
    - 保底至少保留最高分（或第一篇）文档，即使单篇就超预算，
      保证下游 prompt 上下文不为空；
    - 返回结果保持原检索顺序。

    Args:
        docs: 检索到的文档列表
        budget: token 预算上限（非正数视为不设限，防御性兜底）
        scores: 与 docs 平行的检索分数（可选）

    Returns:
        裁剪后的文档列表（保持原顺序）
    """
    if not docs or budget <= 0:
        return docs

    tokens = [count_tokens(doc.page_content) for doc in docs]
    if sum(tokens) <= budget:
        return docs

    # 装入顺序：有平行分数列表时按分数降序，否则保持原序
    order = list(range(len(docs)))
    if scores is not None and len(scores) == len(docs):
        order.sort(key=lambda i: scores[i], reverse=True)

    kept: set[int] = set()
    used = 0
    for i in order:
        # not kept：保底第一篇（分数最高者）无条件保留
        if used + tokens[i] <= budget or not kept:
            kept.add(i)
            used += tokens[i]

    logger.warning(
        f"[token_budget] 检索上下文超预算（{sum(tokens)} > {budget} tokens），"
        f"丢弃 {len(docs) - len(kept)}/{len(docs)} 篇文档"
    )
    return [docs[i] for i in sorted(kept)]
