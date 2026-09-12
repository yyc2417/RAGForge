"""LLM 客户端封装：ChatOpenAI（DeepSeek 兼容）+ tenacity 重试

提供：
- generate(query, context)：基于上下文生成回答（同步，带重试）
- generate_stream(query, context)：流式生成（供 Task 7 SSE）
- invoke_with_prompt(prompt, **kwargs)：自定义 prompt 调用（供 Agent 节点）
- invoke_structured(prompt, schema, **kwargs)：结构化输出（供 Task 5 with_structured_output）
"""

from collections.abc import Iterator
import json
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.generation.prompts import PromptManager
from src.utils.logger import logger
from src.utils.metrics import get_current_collector


class LLMClient:
    """DeepSeek API 封装（OpenAI 兼容格式）。"""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            temperature=0.3,
            # 超时上限：底层 SDK 默认 600s，一次挂起即可阻塞整个状态机
            timeout=settings.llm_timeout,
            # 内置重试关闭，重试统一交给下方 tenacity 管理（避免双重重试叠加）
            max_retries=0,
        )
        # 结构化输出专用实例：分类/抽取任务不需要创造性，temperature=0
        # 显著降低 JSON 解析失败率与枚举值漂移
        self._llm_structured = ChatOpenAI(
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            temperature=0,
            timeout=settings.llm_timeout,
            max_retries=0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        # 只对瞬时故障重试；401/400 等永久性错误重试无意义且掩盖根因
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
        ),
        reraise=True,
    )
    def generate(self, query: str, context: list[Document]) -> str:
        """基于检索上下文生成回答（带重试）。

        Args:
            query: 用户问题
            context: 检索到的文档列表

        Returns:
            生成的回答文本
        """
        context_text = self._format_context(context)
        chain = PromptManager.QA_PROMPT | self._llm
        resp = chain.invoke({"context": context_text, "input": query})
        self._record_usage(resp)
        return resp.content if hasattr(resp, "content") else str(resp)

    def generate_stream(self, query: str, context: list[Document]) -> Iterator[str]:
        """流式生成回答（逐 token yield，供 Task 7 SSE）。

        Args:
            query: 用户问题
            context: 检索到的文档列表

        Yields:
            回答文本的每个 token 片段
        """
        context_text = self._format_context(context)
        chain = PromptManager.QA_PROMPT | self._llm
        for chunk in chain.stream({"context": context_text, "input": query}):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content

    def invoke_with_prompt(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        """使用自定义 prompt 调用 LLM（供 Agent 节点使用）。

        Args:
            prompt: ChatPromptTemplate 实例
            **kwargs: prompt 变量

        Returns:
            LLM 回复文本
        """
        chain = prompt | self._llm
        resp = chain.invoke(kwargs)
        self._record_usage(resp)
        return resp.content if hasattr(resp, "content") else str(resp)

    def invoke_structured(self, prompt: ChatPromptTemplate, schema: type, **kwargs):
        """结构化输出：通过 prompt 引导 + JSON 解析返回符合 schema 的对象。

        DeepSeek 不支持 response_format=json_schema（会返回 400），因此
        不使用 with_structured_output，改为在 prompt 末尾追加一条独立的
        human 消息要求纯 JSON 输出，再用 Pydantic 解析校验。
        解析失败时携带纠错提示重试 1 次（temperature=0 下重试主要覆盖
        偶发的输出截断/格式漂移）。

        供 Task 5 Agent 节点使用（analyze/evaluate/reformulate/verify）。

        Args:
            prompt: ChatPromptTemplate 实例
            schema: Pydantic 模型类（如 AnalyzeResult）
            **kwargs: prompt 变量

        Returns:
            schema 实例

        Raises:
            Exception: 重试后仍解析失败时抛出最后一次的解析异常
        """
        # 在消息末尾显式追加 JSON 输出要求（独立 human 消息，不依赖
        # 模板对象的运算符拼接行为）
        field_hints = ", ".join(
            f'"{name}"' for name in schema.model_fields  # type: ignore[attr-defined]
        )
        json_suffix = (
            f"请仅输出一个合法 JSON 对象（不要 markdown 代码块、不要多余文字），"
            f"包含字段：{field_hints}。"
        )
        retry_suffix = (
            "上一次输出无法解析为合法 JSON。"
            "请严格只输出一个合法 JSON 对象，不要任何解释文字。"
        )
        messages = [*prompt.messages, ("human", json_suffix)]
        last_error: Exception | None = None

        for attempt in range(2):
            if attempt > 0:
                messages = [*messages, ("human", retry_suffix)]
            chain = ChatPromptTemplate.from_messages(messages) | self._llm_structured
            resp = chain.invoke(kwargs)
            content = resp.content if hasattr(resp, "content") else str(resp)
            self._record_usage(resp)
            try:
                data = json.loads(self._extract_json(content))
                return schema.model_validate(data)
            except Exception as e:  # noqa: BLE001 - 记录后重试一次
                last_error = e
                logger.warning(
                    f"[llm] 结构化输出解析失败（第 {attempt + 1} 次）：{e} "
                    f"| 原始回复: {content[:200]}"
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_json(text: str) -> str:
        """从可能含 markdown 代码块或前后多余文字的文本中提取 JSON 对象。

        依次尝试：代码围栏整体内容（非贪婪截断会破坏嵌套 JSON，因此取
        整段并校验）→ 最外层花括号块 → 原文。返回第一个能通过 json.loads
        校验的候选；全部失败时返回原文交由上层报错。
        """
        candidates: list[str] = []
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            candidates.append(m.group(1).strip())
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            candidates.append(m.group(0))
        candidates.append(text.strip())
        for candidate in candidates:
            try:
                json.loads(candidate)
                return candidate
            except Exception:  # noqa: BLE001 - 尝试下一个候选
                continue
        return text.strip()

    @property
    def llm(self) -> ChatOpenAI:
        """暴露底层 ChatOpenAI 实例（少数需要直接调用的场景）。"""
        return self._llm

    @staticmethod
    def _format_context(context: list[Document]) -> str:
        """将 Document 列表拼接为上下文文本。"""
        if not context:
            return "（未检索到相关文档）"
        return "\n\n".join(doc.page_content for doc in context)

    @staticmethod
    def _record_usage(resp) -> None:
        """从响应中提取 token 使用量并记录到当前上下文的指标收集器。

        API 场景为请求级实例（经 ContextVar 注入），eval/CLI 为进程级默认实例。
        """
        try:
            usage = getattr(resp, "usage_metadata", None) or getattr(resp, "response_metadata", {}).get("token_usage")
            if usage:
                pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                get_current_collector().record_token_usage(int(pt), int(ct))
        except Exception as e:  # noqa: BLE001 - 指标记录失败不应影响主流程
            logger.debug(f"[llm] token 记录失败（可忽略）：{e}")
