"""Streamlit 可视化演示界面

进程内直调 LangGraph（不依赖 FastAPI），实时展示 8 节点状态机的
决策与自愈过程：节点三态流转、循环计数、诊断字段、流式答案、来源与收尾指标。

启动：
    uv run --extra ui streamlit run src/ui/streamlit_app.py
"""

import asyncio

import streamlit as st

from src.agent.state import initial_state
from src.config import settings
from src.pipeline import build_agent_pipeline
from src.ui.event_view import DONE, NODES, PENDING, RUNNING, EventView

# ── 页面配置 ───────────────────────────────────────────────────────
st.set_page_config(page_title="RAGForge", page_icon=" forge", layout="wide")
st.title("RAGForge — Agentic RAG 状态机可视化")

# 触发自愈循环的演示问题（评估日志证实会触发 vector → bm25 切换）
_PRESETS = {
    "寒暄（跳过检索）": "你好",
    "普通事实题": "Python 的 GIL 是什么？",
    "触发自愈（策略切换）": "什么是过拟合？怎么解决？",
}

# 自愈动作节点的高亮标记
_SELF_HEAL = {"reformulate", "switch_strategy"}


@st.cache_resource(show_spinner="正在构建 Agent 管道（加载 embedding 模型与向量库）...")
def _get_graph():
    """缓存编译后的 graph：构建成本高（模型加载 + 索引），进程内只构建一次。"""
    return build_agent_pipeline()


# ── 启动前置校验：无 key 直接友好提示，不让 Streamlit 抛红屏 ────────
if not settings.deepseek_key_value():
    st.error("DEEPSEEK_API_KEY 未配置：请在项目根目录 .env 中填入有效 key 后刷新页面。")
    st.stop()

try:
    graph = _get_graph()
except Exception as e:  # noqa: BLE001 - 管道构建失败给可读错误而非红屏
    st.error(f"Agent 管道构建失败：{type(e).__name__}: {e}")
    st.stop()


# ── 事件消费：asyncio 包 astream_events，边收边刷占位符 ────────────
def _run_query_and_render(query: str) -> EventView:
    """执行一次查询，实时渲染事件视图，返回最终 EventView。"""
    view = EventView()
    status = st.empty()

    # 各区域的占位符（先建后刷，保证页面自上而下的稳定布局）
    nodes_ph = st.empty()
    diag_ph = st.empty()
    heal_ph = st.empty()
    answer_ph = st.empty()
    sources_ph = st.empty()
    metrics_ph = st.empty()

    def _render() -> None:
        _render_nodes(nodes_ph, view)
        _render_diagnostic(diag_ph, view)
        _render_self_heal(heal_ph, view)
        _render_answer(answer_ph, view)
        _render_sources(sources_ph, view)
        _render_metrics(metrics_ph, view)

    async def _consume() -> None:
        async for event in graph.astream_events(
            initial_state(query),
            config={"recursion_limit": 50},
            version="v2",
        ):
            view.consume(event)
            _render()

    asyncio.run(_consume())
    status.empty()
    return view


def _render_nodes(ph, view: EventView) -> None:
    """8 节点流水线：三态卡片 + 循环计数（自愈最直观的证据）。"""
    cols = st.columns(len(NODES))
    for col, name in zip(cols, NODES):
        state = view.node_status[name]
        runs = view.node_runs[name]
        if state == RUNNING:
            icon, color = "⏳", "orange"
        elif state == DONE:
            icon, color = "✅", "green"
        else:
            icon, color = "·", "gray"
        label = name if runs <= 1 else f"{name} ×{runs}"
        col.markdown(
            f":{color}[**{icon} {label}**]",
            unsafe_allow_html=True,
        )


def _render_diagnostic(ph, view: EventView) -> None:
    """诊断面板：诊断式评估的核心字段逐项实时刷新。"""
    diag = view.diagnostic
    if not diag:
        ph.info("等待 analyze / evaluate 节点的诊断输出...")
        return
    lines = []
    for field, value in diag.items():
        if value is None or value == "":
            continue
        lines.append(f"- `{field}`: **{value}**")
    ph.markdown("\n".join(lines))


def _render_self_heal(ph, view: EventView) -> None:
    """自愈高亮：reformulate / switch_strategy 被执行时醒目提示。"""
    triggered = [n for n in ("reformulate", "switch_strategy") if view.node_runs[n] > 0]
    if not triggered:
        ph.empty()
        return
    actions = "、".join(triggered)
    ph.warning(f"🔁 触发自愈：{actions} —— 系统在自我修正，注意上方节点的循环计数。")


def _render_answer(ph, view: EventView) -> None:
    ph.subheader("回答")
    text = view.answer
    ph.markdown(text if text else "_（生成中...）_")


def _render_sources(ph, view: EventView) -> None:
    if not view.sources:
        ph.empty()
        return
    ph.subheader("来源文档")
    ph.table(view.sources)


def _render_metrics(ph, view: EventView) -> None:
    summary = view.summary()
    if not view.final_state:
        ph.empty()
        return
    faithful = summary["is_faithful"]
    faithful_text = {True: "✅ 忠实", False: "❌ 检出幻觉", None: "⚠️ 未验证"}.get(
        faithful, str(faithful)
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("端到端状态", "完成")
    m2.metric("检索轮次", summary["iteration_count"])
    m3.metric("幻觉重试", summary["verify_failures"])
    m4.metric("幻觉检测", faithful_text)
    if summary["verification_reason"]:
        ph.caption(f"校验说明：{summary['verification_reason']}")


# ── 输入区 ─────────────────────────────────────────────────────────
query = st.text_input(
    "输入问题",
    value="",
    placeholder="试试下面的问题，或输入任何想问的",
)

preset_cols = st.columns(len(_PRESETS))
clicked = None
for col, (label, preset_query) in zip(preset_cols, _PRESETS.items()):
    if col.button(label):
        clicked = preset_query

if clicked:
    query = clicked

if query:
    st.divider()
    with st.spinner("状态机执行中..."):
        view = _run_query_and_render(str(query))
    if view.error:
        st.error(f"执行过程中出现错误：{view.error}")
else:
    st.info("输入问题或点击上方示例按钮，观察 8 节点状态机的决策与自愈过程。")
