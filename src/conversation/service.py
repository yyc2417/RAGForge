"""会话服务：多轮查询改写 + 消息落盘

职责边界：
- 本模块只负责“图外”的会话历史与查询改写；
- 不修改 LangGraph 节点，也不持有图状态；
- 图仍是无状态的，每个 turn 用 standalone query 调用一次。
"""

from typing import Any

from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import CondenseResult
from src.utils.logger import logger
from src.utils.token_budget import count_tokens

from .models import Session, SessionMessage, new_id
from .store import ConversationStore


class ConversationService:
    """多轮会话服务。"""

    def __init__(
        self,
        store: ConversationStore | None = None,
        llm: LLMClient | None = None,
        prompts: PromptManager | None = None,
    ) -> None:
        self.store = store or ConversationStore()
        self._llm = llm
        self._prompts = prompts or PromptManager()

    # ── 会话与消息 ─────────────────────────────────────────────────
    def start_session(self, title: str = "新会话") -> Session:
        return self.store.create_session(title)

    def add_user_message(self, session_id: str, content: str) -> SessionMessage:
        message = SessionMessage(id=new_id(), role="user", content=content)
        self.store.append_message(session_id, message)
        return message

    def record_assistant_state(
        self,
        session_id: str,
        final_state: dict[str, Any],
        *,
        original_query: str | None = None,
        resolved_query: str | None = None,
        latency_ms: float | None = None,
        total_tokens: int | None = None,
        model: str | None = None,
    ) -> SessionMessage:
        """把图最终状态写入 assistant 消息（回答 / 澄清）。"""
        clarification = str(final_state.get("clarification_question") or "").strip()
        message_type = "clarify" if clarification else "answer"
        content = clarification or str(final_state.get("answer") or "").strip() or "（无回答）"
        metrics: dict[str, Any] = {
            "iteration_count": int(final_state.get("iteration_count") or 0),
            "verify_failures": int(final_state.get("verify_failures") or 0),
            "is_faithful": final_state.get("is_faithful"),
            "model": model or settings.deepseek_model,
        }
        if latency_ms is not None:
            metrics["latency_ms"] = round(float(latency_ms), 1)
        if total_tokens is not None:
            metrics["total_tokens"] = int(total_tokens)
        message = SessionMessage(
            id=new_id(),
            role="assistant",
            content=content,
            message_type=message_type,
            original_query=original_query,
            resolved_query=resolved_query,
            sources=self.extract_sources(final_state),
            trace=list(final_state.get("messages") or []),
            metrics=metrics,
        )
        self.store.append_message(session_id, message)
        return message

    def record_feedback(
        self,
        session_id: str,
        message_id: str,
        feedback: str,
        *,
        archived: bool | None = None,
    ) -> bool:
        """记录赞/踩；可同时更新 archived 标记。"""
        if feedback not in ("up", "down"):
            return False
        return self.store.update_feedback(
            session_id, message_id, feedback, archived=archived
        ) is not None

    # ── 多轮查询改写 ───────────────────────────────────────────────
    def prepare_query(self, session_id: str, query: str) -> tuple[str, str | None]:
        """有历史时把追问改写成独立问题；失败或无历史时回退原问题。

        Returns:
            (standalone_query, reason)
        """
        session = self.store.load_session(session_id)
        history = self._format_history(session) if session else ""
        if not history:
            return query, None

        try:
            result: CondenseResult = self._ensure_llm().invoke_structured(
                self._prompts.CONDENSE_PROMPT,
                CondenseResult,
                history=history,
                query=query,
            )
            standalone = result.standalone_query.strip()
            if not standalone:
                return query, None
            if standalone != query:
                logger.info(f"[conversation] 追问改写：{query!r} → {standalone!r}")
            return standalone, result.reason
        except Exception as e:  # noqa: BLE001 - 改写失败不阻塞主流程
            logger.warning(f"[conversation] 多轮改写失败，回退原问题：{e}")
            return query, None

    def _format_history(self, session: Session) -> str:
        """取最近 N 轮，从新到旧装入 token 预算。"""
        if not session.messages:
            return ""
        window = settings.conversation_history_turns * 2
        messages = session.messages[-window:]

        lines: list[str] = []
        used = 0
        for message in reversed(messages):
            text = message.content.strip()
            if not text:
                continue
            if message.role == "user":
                role = "用户"
            elif message.message_type == "clarify":
                role = "助手（澄清）"
            else:
                role = "助手"
            line = f"{role}：{text}"
            cost = count_tokens(line)
            if lines and used + cost > settings.conversation_token_budget:
                break
            lines.append(line)
            used += cost
        return "\n".join(reversed(lines))

    # ── 依赖与来源提取 ─────────────────────────────────────────────
    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    @staticmethod
    def extract_sources(final_state: dict[str, Any], max_content: int = 500) -> list[dict[str, Any]]:
        """把最终状态中的 documents/scores 转成可落盘的来源条目。"""
        docs = final_state.get("documents", []) or []
        scores = final_state.get("retrieval_scores") or []
        rows: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if not hasattr(doc, "page_content"):
                continue
            meta = getattr(doc, "metadata", {}) or {}
            score = scores[i] if i < len(scores) else 0.0
            rows.append({
                "content": doc.page_content[:max_content],
                "source": meta.get("source", "unknown"),
                "score": round(float(score), 6) if score is not None else 0.0,
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index"),
                "source_type": meta.get("source_type", "document"),
            })
        return rows
