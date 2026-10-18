"""Streamlit 聊天式界面

功能：
- 新建会话 / 历史会话切换
- 多轮提问（图外 condense）
- clarify 消息展示
- 来源面板 + 执行轨迹 expander
- 每条 assistant 消息的赞 / 踩反馈

启动：
    uv run --extra ui streamlit run src/ui/streamlit_app.py

说明：
- 图仍是无状态的；会话历史由 ConversationStore 落盘。
- 反馈点击只更新会话文件并 rerun，不重新执行状态机。
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 路径引导：streamlit run 只把脚本目录加入 sys.path，需补项目根
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src.agent.state import initial_state  # noqa: E402
from src.config import settings  # noqa: E402
from src.conversation import ConversationService, ConversationStore, SessionMessage  # noqa: E402
from src.generation.llm_client import LLMClient  # noqa: E402
from src.pipeline import build_agent_runtime  # noqa: E402
from src.ui.event_view import LEVEL_ANSWER, LEVEL_FULL, EventView  # noqa: E402
from src.ui.html_view import CSS  # noqa: E402

# ── 页面配置 ───────────────────────────────────────────────────────
st.set_page_config(page_title="RAGForge", page_icon="⚒️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("RAGForge — Agentic RAG")
st.caption(
    "固定 hybrid 检索；诊断不达标时改写查询重试。"
    "需要澄清时会先追问，答案可展开来源与执行轨迹。"
)


@dataclass
class Runtime:
    graph: Any
    service: ConversationService
    knowledge_base: Any


@st.cache_resource(show_spinner="正在构建 Agent 管道（加载 embedding 模型与向量库）...")
def _get_runtime() -> Runtime:
    agent_runtime = build_agent_runtime()
    service = ConversationService(
        store=ConversationStore(),
        llm=LLMClient(),
        prompts=None,
    )
    return Runtime(
        graph=agent_runtime.graph,
        service=service,
        knowledge_base=agent_runtime.knowledge_base,
    )


if not settings.deepseek_key_value():
    st.error("DEEPSEEK_API_KEY 未配置：请在项目根目录 .env 中填入有效 key 后刷新页面。")
    st.stop()

try:
    runtime = _get_runtime()
except Exception as e:  # noqa: BLE001 - 管道构建失败给可读错误
    st.error(f"Agent 管道构建失败：{type(e).__name__}: {e}")
    st.stop()

service = runtime.service
store = service.store
kb = runtime.knowledge_base


# ── 会话侧栏 ───────────────────────────────────────────────────────
def _current_session_id() -> str | None:
    return st.session_state.get("current_session_id")


def _new_session() -> None:
    st.session_state.current_session_id = None
    st.session_state.pop("pending_query", None)


with st.sidebar:
    st.header("会话")
    if st.button("＋ 新建会话", use_container_width=True):
        _new_session()
        st.rerun()

    sessions = store.list_sessions()
    if not sessions:
        st.caption("暂无历史会话")
    for meta in sessions:
        active = meta.session_id == _current_session_id()
        label = ("● " if active else "") + meta.title
        if st.button(label, key=f"sess-{meta.session_id}", use_container_width=True):
            st.session_state.current_session_id = meta.session_id
            st.rerun()

    st.divider()
    st.header("知识库")
    uploaded = st.file_uploader(
        "上传文件",
        type=["md", "txt", "pdf", "html", "htm", "docx", "csv", "jsonl"],
        accept_multiple_files=True,
        key="kb-uploader",
    )
    if uploaded and st.button("导入上传文件", use_container_width=True):
        with st.spinner("入库并重建索引…"):
            result = kb.ingest_files([(f.name, f.getvalue()) for f in uploaded])
        if result.failed:
            st.error(result.summary + "\n" + "; ".join(f"{n}: {e}" for n, e in result.failed))
        else:
            st.success(result.summary)
        st.rerun()

    folder_path = st.text_input(
        "导入文件夹（项目 data/ 下）",
        key="kb-folder",
        placeholder="例如 data/library/folders/faq",
    )
    if st.button("导入文件夹", use_container_width=True):
        if not folder_path.strip():
            st.warning("请填写文件夹路径")
        else:
            with st.spinner("导入并重建索引…"):
                result = kb.ingest_folder(folder_path.strip())
            if result.failed:
                st.error(result.summary + "\n" + "; ".join(f"{n}: {e}" for n, e in result.failed))
            else:
                st.success(result.summary)
            st.rerun()

    sources = kb.list_sources()
    with st.expander(f"来源列表（{len(sources)}）"):
        for src in sources:
            cols = st.columns([4, 1])
            cols[0].caption(
                f"{src['name']} · {src['scope']} · {src['file_type']} · {src['size']}B"
            )
            if src["scope"] == "library":
                if cols[1].button("移除", key=f"rm-{abs(hash(src['source']))}"):
                    kb.remove_source(src["source"])
                    st.rerun()

    st.divider()
    st.header("本次运行参数")
    st.caption(
        f"LLM `{settings.deepseek_model}` · embedding `{settings.embedding_model}`\n\n"
        f"chunk `{settings.chunk_size}`/`{settings.chunk_overlap}` tokens · "
        f"检索 `top_k={settings.retrieval_k}` · RRF `k={settings.rrf_k}`\n\n"
        f"clarify {'开启' if settings.clarify_enabled else '关闭'}"
    )
    max_iterations = st.slider(
        "最大检索轮次 max_iterations",
        min_value=1,
        max_value=3,
        value=3,
        help="检索次数上限（防死循环预算）。调到 1 可观察自愈被截断。",
    )


# ── 消息渲染 ───────────────────────────────────────────────────────
def _basename(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1] or "unknown"


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"来源（{len(sources)}）"):
        for i, src in enumerate(sources, start=1):
            source = _basename(str(src.get("source", "unknown")))
            page = src.get("page_number")
            chunk = src.get("chunk_index")
            extra = []
            if page is not None:
                extra.append(f"p.{page}")
            if chunk is not None:
                extra.append(f"chunk {chunk}")
            extra.append(f"score {float(src.get('score', 0.0)):.4f}")
            st.markdown(f"**[{i}] {source}** · " + " · ".join(extra))
            st.caption(str(src.get("content", ""))[:300])


def _render_trace(trace: list[str]) -> None:
    if not trace:
        return
    with st.expander("查看状态机细节"):
        st.code("\n".join(trace), language="text")


def _render_feedback(session_id: str, message: SessionMessage) -> None:
    cols = st.columns([1, 1, 6])
    up_disabled = message.feedback == "up"
    down_disabled = message.feedback == "down"
    if cols[0].button("👍 有用", key=f"up-{message.id}", disabled=up_disabled):
        service.record_feedback(session_id, message.id, "up")
        st.toast("已记录：有用")
        st.rerun()
    if cols[1].button("👎 没用", key=f"down-{message.id}", disabled=down_disabled):
        service.record_feedback(session_id, message.id, "down")
        st.toast("已记录：没用")
        st.rerun()
    if message.feedback:
        label = "有用" if message.feedback == "up" else "没用"
        cols[2].caption(f"已标记：{label}")


def _render_message(session_id: str, message: SessionMessage) -> None:
    with st.chat_message(message.role):
        if message.role == "assistant" and message.message_type == "clarify":
            st.warning("需要澄清")
        st.markdown(message.content)
        if message.role != "assistant":
            return
        if message.resolved_query:
            st.caption(f"检索改写：{message.resolved_query}")
        _render_sources(message.sources)
        _render_trace(message.trace)
        _render_feedback(session_id, message)


session_id = _current_session_id()
session = store.load_session(session_id) if session_id else None
if session is not None:
    for message in session.messages:
        _render_message(session.session_id, message)


# ── 新消息 ─────────────────────────────────────────────────────────
def _run_turn(session_id: str, query: str) -> None:
    """执行一轮：condense → 图 → 落盘 → rerun。"""
    standalone, _reason = service.prepare_query(session_id, query)
    service.add_user_message(session_id, query)

    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        view = EventView()

        async def _consume() -> None:
            async for event in runtime.graph.astream_events(
                initial_state(standalone, max_iterations=max_iterations),
                config={"recursion_limit": 50},
                version="v2",
            ):
                level = view.consume(event)
                if level in (LEVEL_ANSWER, LEVEL_FULL):
                    placeholder.markdown(view.answer or "_（处理中…）_")

        with st.spinner("状态机执行中…"):
            try:
                asyncio.run(_consume())
            except Exception as e:  # noqa: BLE001 - UI 层给出可读错误
                st.error(f"执行失败：{type(e).__name__}: {e}")
                service.record_assistant_state(
                    session_id,
                    {"answer": f"执行失败：{type(e).__name__}: {e}"},
                    resolved_query=standalone,
                )
                st.rerun()

    final_state = view.final_state
    if not final_state:
        final_state = {
            "answer": view.answer,
            "clarification_question": view.diagnostic.get("clarification_question"),
            "ambiguity_type": view.diagnostic.get("ambiguity_type"),
            "messages": [],
        }
    service.record_assistant_state(
        session_id,
        final_state,
        resolved_query=standalone if standalone != query else None,
    )
    st.rerun()


prompt = st.chat_input("输入问题…")
if prompt:
    if session is None:
        session = service.start_session()
        st.session_state.current_session_id = session.session_id
    _run_turn(session.session_id, str(prompt))
elif session is None:
    st.info("点击「新建会话」或直接在下方输入问题开始。")
