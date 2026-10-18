"""Streamlit 可视化演示界面

进程内直调 LangGraph（不依赖 FastAPI），实时展示 7 节点主链路的决策与自愈过程：
流程轨道 → 分组诊断 → 轮次时间线 → 流式回答 → 折叠来源 → 运行摘要。

启动：
    uv run --extra ui streamlit run src/ui/streamlit_app.py

呈现层两条硬约束（历史版本因违反它们导致流程条重复渲染 2665 次、区块标题消失）：
1. 动态区块各占一个 st.empty()，每轮只写入**一个**元素（多次写入会互相顶掉）；
2. st.columns 等容器化组件不得写在 placeholder 之外（会往页面追加元素而非原地更新）。
因此动态内容统一由 html_view 装配成单个 HTML 片段写入，区块标题在流式循环之外一次性写入。
"""

import asyncio
import sys
from pathlib import Path

# 路径引导：streamlit run 只把脚本所在目录（src/ui）加入 sys.path，项目根不在其中，
# src 布局下 import src.* 会报 ModuleNotFoundError。tests 靠 pytest 的 pythonpath 配置解决，
# Streamlit 没有对应机制，故在入口先补上项目根，其后的导入因此需标注 noqa: E402。
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src.agent.state import initial_state  # noqa: E402
from src.config import settings  # noqa: E402
from src.pipeline import build_agent_pipeline  # noqa: E402
from src.ui import html_view  # noqa: E402
from src.ui.event_view import LEVEL_ANSWER, LEVEL_FULL, EventView  # noqa: E402

# ── 页面配置 ───────────────────────────────────────────────────────
st.set_page_config(page_title="RAGForge", page_icon="⚒️", layout="wide")

# 预设问题：标签只标注实测路径（探针脚本跑出的真实轨迹）。
# 实际走哪条分支由运行时的诊断结论决定，同一问题多次运行可能不经过相同节点。
_PRESETS = {
    "寒暄（不检索）": "你好",
    "混合检索单轮": "Python 的 GIL 是什么？",
    "改写后命中": "Docker 多阶段构建怎么写？",
    "语料外拒答": "K8s 怎么做滚动升级？",
}

_PRESET_NOTES = {
    "寒暄（不检索）": "analyze 判 chitchat → decide 给 strategy=none，retrieve / verify 均不执行",
    "混合检索单轮": "典型 factual 路径，固定 hybrid 一轮检索即判 sufficient",
    "改写后命中": "低召回时 evaluate 建议 reformulate，改写查询后重新走 hybrid",
    "语料外拒答": "实测 irrelevant → 两轮改写后跑满预算，最终诚实拒答（不编造）",
}

st.markdown(html_view.CSS, unsafe_allow_html=True)
st.title("RAGForge — Agentic RAG 状态机可视化")
st.caption(
    "提问后观察状态机如何自主决策：固定 hybrid 检索，诊断质量不达标时改写查询重新检索，"
    "生成答案后做幻觉检测。节点重复执行（徽章显示次数）即自愈的证据。"
)


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


# ── 侧边栏：读图说明 + 字段释义 + 运行参数（把「这是什么」就地答掉）──
def _render_sidebar() -> int:
    """渲染侧边栏说明与参数，返回本次运行的 max_iterations。"""
    with st.sidebar:
        st.header("读图说明")
        st.markdown(
            "**流程轨道**：7 个节点的状态卡片，灰色待执行、琥珀色执行中、绿色已完成、"
            "紫色为自愈动作（reformulate）。卡片徽章只在被执行两次及以上时显示次数。"
        )
        st.markdown("**轮次时间线**：每执行一次检索就记一行，能看到「这轮用什么查询、"
                    "什么策略、命中几篇、诊断结论是什么、下一步做什么」——"
                    "自愈循环的因果链在这里完整呈现。")
        st.markdown("**检索来源**：按轮次分组折叠，点开读 chunk 全文。固定 hybrid 的 RRF 分数"
                    "只能在本轮内比较排序，故条形按本轮最高归一。")

        st.header("示例问题演示什么")
        for label, note in _PRESET_NOTES.items():
            st.markdown(f"- **{label}** — {note}")

        st.header("7 个节点职责")
        st.markdown(
            "\n".join(
                f"- **`{name}`** {cn} — {hint}"
                for name, (cn, hint) in html_view.NODE_META.items()
            )
        )

        st.header("诊断字段含义")
        for title, subtitle, fields in html_view.DIAGNOSIS_GROUPS:
            st.markdown(f"**{title}** · {subtitle}")
            st.markdown(
                "\n".join(
                    f"- `{key}` {label} — {hint}" for key, label, hint in fields
                )
            )

        st.header("本次运行参数")
        st.caption(
            f"LLM `{settings.deepseek_model}` · embedding `{settings.embedding_model}`\n\n"
            f"chunk `{settings.chunk_size}`/`{settings.chunk_overlap}` tokens · "
            f"检索 `top_k={settings.retrieval_k}` · RRF `k={settings.rrf_k}`\n\n"
            f"上下文预算 `{settings.context_token_budget}` tokens · "
            f"reranker `{settings.reranker_model}`（模型缺失时自动降级为纯 RRF）"
        )
        return st.slider(
            "最大检索轮次 max_iterations",
            min_value=1,
            max_value=3,
            value=3,
            help="检索次数上限（防死循环的第一道预算）。调小到 1 可观察自愈被强制截断的行为。",
        )


max_iterations = _render_sidebar()


# ── 输入区 ─────────────────────────────────────────────────────────
query = st.text_input(
    "输入问题",
    value="",
    placeholder="试试下面的问题，或输入任何想问的（语料外问题可验证拒答行为）",
)

preset_cols = st.columns(len(_PRESETS))
clicked = None
for col, (label, preset_query) in zip(preset_cols, _PRESETS.items(), strict=False):
    if col.button(label, use_container_width=True):
        clicked = preset_query

st.caption(
    "按钮标签是实测路径；LLM 判定存在不确定性，同一问题多次运行可能走不同分支，"
    "实际路径以「轮次时间线」为准。"
)

if clicked:
    query = clicked


# ── 事件消费与渲染 ─────────────────────────────────────────────────
def _set(ph, html: str) -> None:
    """把一个 HTML 片段写入占位符（单次写入 = 原地替换）。"""
    ph.markdown(html_view.wrap(html), unsafe_allow_html=True)


def _render_query(query: str) -> EventView:
    """执行一次查询：先搭好区块骨架，再按事件流逐块刷新。"""
    view = EventView()

    # 骨架：标题写入页面主体（一次性），动态内容各占一个 placeholder
    def _section(num: str, title: str, hint: str):
        st.markdown(html_view.wrap(html_view.render_section_title(num, title, hint)),
                    unsafe_allow_html=True)
        return st.empty()

    track_ph = _section("①", "执行流程", "7 节点主链路实时流转；徽章显示次数即重复执行")
    diag_ph = _section("②", "决策诊断", "各节点结论按归属分组，hover 查看字段含义")
    rounds_ph = _section("③", "轮次时间线", "每轮检索的查询、策略、命中与诊断——自愈因果在此")
    heal_ph = st.empty()
    answer_ph = _section("④", "回答", "generate 节点逐 token 流式输出")
    sources_ph = _section("⑤", "检索来源", "按轮次分组，点开阅读 chunk 全文")
    summary_ph = _section("⑥", "运行摘要", "端到端结果，hover 查看统计口径")

    def _render_answer() -> None:
        text = view.answer
        answer_ph.markdown(text if text else "_（生成中…）_")

    def _render_heal() -> None:
        """自愈提示：列出实际触发的动作与内容（方向指引依赖具体文案，不写「上方/下方」）。"""
        actions = [
            f"{name}（{detail}）" for rnd in view.rounds for name, detail in rnd.actions
        ]
        if not actions:
            heal_ph.empty()
            return
        heal_ph.warning("🔁 触发自愈：" + "；".join(actions) + " —— 系统在自我修正。")

    def _render_full() -> None:
        _set(track_ph, html_view.render_track(view))
        _set(diag_ph, html_view.render_diagnosis(view))
        _set(rounds_ph, html_view.render_rounds(view))
        _set(sources_ph, html_view.render_sources(view))
        _set(summary_ph, html_view.render_summary(view))
        _render_heal()

    async def _consume() -> None:
        async for event in graph.astream_events(
            initial_state(query, max_iterations=max_iterations),
            config={"recursion_limit": 50},
            version="v2",
        ):
            # 按变更级别重画：token 只刷答案，节点事件才重画全页
            level = view.consume(event)
            if level == LEVEL_ANSWER:
                _render_answer()
            elif level == LEVEL_FULL:
                _render_full()
                _render_answer()

    with st.spinner("状态机执行中…"):
        asyncio.run(_consume())
    if view.error:
        st.error(f"执行过程中出现错误：{view.error}")
    return view


if query:
    _render_query(str(query))
else:
    st.info("输入问题或点击上方示例按钮，观察状态机的决策与自愈过程。")
