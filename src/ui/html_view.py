"""界面呈现装配：EventView → HTML 片段（纯逻辑，可离线单测）

每个区块返回**单个** HTML 字符串，界面侧一次 st.markdown 写入 placeholder。
这是 Streamlit 的硬约束：placeholder 只有「一次写入」才是原地替换——多次写入
会互相顶掉（历史版本里「回答」「来源文档」两个 subheader 就是这样消失的），
而在 placeholder 之外调用 st.columns 会往页面追加元素（历史版本因此把节点
流程条重复渲染了 2665 次，页面高 25 万像素）。

配色刻意使用中性 rgba 与语义色，不假设亮/暗主题。所有动态文本一律 escape，
chunk 正文含 markdown 与尖括号，直插 HTML 会破坏结构。
"""

import re
from html import escape
from typing import Any

from src.ui.event_view import DONE, NODES, PENDING, RUNNING, EventView, Round

# ── 节点元信息：中文职责名 + 悬浮释义（与 graph.py 的节点一一对应）────
NODE_META: dict[str, tuple[str, str]] = {
    "analyze": ("意图分析", "LLM 判断查询类型、是否需要检索、是否需要澄清"),
    "clarify": ("澄清追问", "需要澄清时终止本轮，向用户提一个问题"),
    "decide": ("检索决策", "规则路由：chitchat / 不需要检索→none，其余→hybrid"),
    "retrieve": ("执行检索", "固定 hybrid：向量 + BM25 + RRF；检索次数的唯一递增点"),
    "evaluate": ("质量诊断", "LLM 诊断式评估：区分召回不足与结果无关，给出 proceed / reformulate"),
    "reformulate": ("查询改写", "按 failure_mode 选择改写策略（specify / synonym_replace / generalize）"),
    "generate": ("生成答案", "基于检索上下文生成回答；LLM 失败时降级兜底"),
    "verify": ("幻觉检测", "LLM 判断答案是否忠实于检索上下文"),
}

# 自愈动作节点（流程条上独立配色，便于一眼看出系统自我修正过）
_SELF_HEAL_NODES = {"reformulate"}

# ── 诊断字段：中文标签 + 释义 + 值映射 ─────────────────────────────
# 分组顺序即阅读顺序；iteration_count / verify_failures 归运行摘要，此处不重复展示
DIAGNOSIS_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "意图与策略",
        "analyze / decide 的结论",
        (
            ("query_type", "查询意图", "LLM 判定的问题类型"),
            ("needs_retrieval", "需要检索", "是否进入检索分支"),
            ("needs_clarification", "需要澄清", "是否需要先向用户确认"),
            ("ambiguity_type", "歧义类型", "缺槽位 / 多解释 / 约束冲突"),
            ("clarification_question", "澄清问题", "本轮向用户提出的问题"),
            ("retrieval_strategy", "检索策略", "本轮实际使用的检索器"),
        ),
    ),
    (
        "检索质量诊断",
        "evaluate 的结论——自愈决策的依据",
        (
            ("failure_mode", "诊断结论", "失败模式；sufficient 表示信息充足"),
            ("suggested_action", "建议动作", "诊断结论推导出的下一步"),
            ("diagnosis_reason", "诊断依据", "LLM 给出的判定理由"),
        ),
    ),
    (
        "改写与校验",
        "reformulate / verify 的产物",
        (
            ("reformulated_query", "改写后查询", "自愈时重新检索使用的查询"),
            ("rewrite_effective", "改写有效", "False 时路由短路，跳过注定重复的检索"),
            ("is_faithful", "忠实于上下文", "verify 结论；未验证表示校验降级"),
            ("verification_reason", "校验说明", "幻觉判定的理由"),
        ),
    ),
)

_VALUE_LABELS: dict[str, dict[Any, str]] = {
    "query_type": {
        "factual": "事实查询", "reasoning": "推理比较",
        "chitchat": "寒暄", "complex": "复杂多跳",
    },
    "retrieval_strategy": {
        "hybrid": "混合 RRF", "none": "不检索",
    },
    "ambiguity_type": {
        "missing_slot": "缺槽位",
        "multiple_interpretations": "多解释",
        "conflicting_constraints": "约束冲突",
    },
    "failure_mode": {
        "low_recall": "召回不足", "irrelevant": "结果无关", "sufficient": "信息充足",
    },
    "suggested_action": {
        "reformulate": "改写查询", "proceed": "直接生成",
    },
}

# 诊断结论/动作的语义色（与流程条共用一套色板）
_MODE_CLASS = {
    "low_recall": "is-warn", "irrelevant": "is-warn", "sufficient": "is-ok",
    "reformulate": "is-heal", "proceed": "is-ok",
}


# ── 区块渲染 ───────────────────────────────────────────────────────
def wrap(html: str) -> str:
    """统一外层容器：色彩变量挂在 :root，字号行高由 .rf-shell 提供。"""
    return f'<div class="rf-shell">{html}</div>'


def render_section_title(num: str, title: str, hint: str) -> str:
    """区块标题：在流式循环之外一次性写入，占用独立元素。"""
    return (
        f'<div class="rf-sec"><span class="rf-sec-num">{escape(num)}</span>'
        f'<span class="rf-sec-title">{escape(title)}</span>'
        f'<span class="rf-sec-hint">{escape(hint)}</span></div>'
    )


def render_track(view: EventView) -> str:
    """① 流程轨道：7 节点状态卡片，带轮次徽章与自愈高亮。"""
    cells = []
    for name in NODES:
        cn, help_text = NODE_META[name]
        status = view.node_status[name]
        runs = view.node_runs[name]
        cls = {PENDING: "is-pending", RUNNING: "is-running", DONE: "is-done"}[status]
        if name in _SELF_HEAL_NODES and runs > 0:
            cls = "is-heal"
        badge = f'<span class="rf-badge">{runs} 次</span>' if runs > 1 else ""
        cells.append(
            f'<div class="rf-node {cls}" title="{escape(help_text, quote=True)}">'
            f'<div class="rf-node-head"><span class="rf-dot"></span>{badge}</div>'
            f'<div class="rf-node-cn">{escape(cn)}</div>'
            f'<div class="rf-node-en">{escape(name)}</div>'
            f"</div>"
        )
    legend = (
        '<div class="rf-legend">'
        '<span><i class="rf-dot is-pending"></i>待执行</span>'
        '<span><i class="rf-dot is-running"></i>执行中</span>'
        '<span><i class="rf-dot is-done"></i>已完成</span>'
        '<span><i class="rf-dot is-heal"></i>自愈动作</span>'
        "</div>"
    )
    return f'<div class="rf-track">{"".join(cells)}</div>{legend}'


def render_diagnosis(view: EventView) -> str:
    """② 决策诊断：按节点归属分组的键值卡片，无值字段不渲染。"""
    blocks = []
    for title, subtitle, fields in DIAGNOSIS_GROUPS:
        rows = []
        for key, label, help_text in fields:
            if key not in view.diagnostic:
                continue
            rows.append(f'<div class="rf-kv" title="{escape(help_text, quote=True)}">'
                        f"<span>{escape(label)}</span>"
                        f"{_fmt_value(key, view.diagnostic[key])}</div>")
        if not rows:
            continue
        blocks.append(
            f'<div class="rf-group"><div class="rf-group-title">{escape(title)}'
            f'<em>{escape(subtitle)}</em></div>{"".join(rows)}</div>'
        )
    if not blocks:
        return '<div class="rf-empty">等待 analyze / evaluate 节点产出诊断结论…</div>'
    return f'<div class="rf-groups">{"".join(blocks)}</div>'


def render_rounds(view: EventView) -> str:
    """③ 轮次时间线：每轮检索一行，呈现自愈因果链（本界面的核心叙事）。"""
    if not view.rounds:
        return '<div class="rf-empty">本次未触发检索（寒暄或改写无效短路），无轮次记录。</div>'
    head = (
        '<div class="rf-round-head"><span>轮次</span><span>检索查询</span>'
        "<span>策略</span><span>命中 / 最高分</span><span>诊断结论</span><span>下一步动作</span></div>"
    )
    body = []
    for rnd in view.rounds:
        heal_cls = " is-healed" if rnd.healed else ""
        mode_chip = _chip(rnd.failure_mode, "failure_mode") if rnd.failure_mode else "—"
        action_cell = _render_round_actions(rnd)
        reason = (
            f'<div class="rf-reason">{escape(rnd.diagnosis_reason)}</div>'
            if rnd.diagnosis_reason
            else ""
        )
        body.append(
            f'<div class="rf-round{heal_cls}">'
            f'<div class="rf-round-row">'
            f'<span class="rf-idx">第 {rnd.index} 轮</span>'
            f'<span class="rf-query" title="{escape(rnd.query, quote=True)}">{escape(rnd.query)}</span>'
            f'<span>{_chip(rnd.strategy, "retrieval_strategy")}</span>'
            f'<span class="rf-score">{rnd.hits} 篇 · {rnd.top_score:.4f}'
            f'<em>{escape(rnd.score_semantics)}</em></span>'
            f'<span>{mode_chip}</span>'
            f'<span>{action_cell}</span>'
            f"</div>{reason}</div>"
        )
    return f'<div class="rf-rounds">{head}{"".join(body)}</div>{_budget_note(view)}'


def render_sources(view: EventView) -> str:
    """④ 检索来源：按轮次分组的折叠条目，替代原先占屏近九成的 st.table。"""
    rounds_with_docs = [rnd for rnd in view.rounds if rnd.sources]
    if not rounds_with_docs:
        return '<div class="rf-empty">本次没有检索到任何文档。</div>'
    blocks = []
    for rnd in rounds_with_docs:
        label = f'第 {rnd.index} 轮 · {rnd.strategy} · {len(rnd.sources)} 篇'
        top = max((s["score"] for s in rnd.sources), default=0.0) or 1.0
        items = []
        for i, src in enumerate(rnd.sources, start=1):
            ratio = max(0.0, min(1.0, src["score"] / top))
            name = _basename(str(src.get("source", "unknown")))
            items.append(
                '<details class="rf-src">'
                f'<summary><span class="rf-src-idx">{i}</span>'
                f'<span class="rf-src-name" title="{escape(str(src.get("source", "")), quote=True)}">'
                f"{escape(name)}</span>"
                f'<span class="rf-bar"><i style="width:{ratio * 100:.0f}%"></i></span>'
                f'<span class="rf-src-score">{src["score"]:.4f}</span></summary>'
                f'<div class="rf-src-body">{_render_body(str(src.get("content", "")))}</div>'
                "</details>"
            )
        blocks.append(
            f'<div class="rf-src-group"><div class="rf-src-group-title">{escape(label)}</div>'
            f'{"".join(items)}</div>'
        )
    return "".join(blocks)


def render_summary(view: EventView) -> str:
    """⑤ 运行摘要：4 张语义色卡片，每项带含义说明。"""
    summary = view.summary()
    faithful = summary["is_faithful"]
    if faithful is True:
        verify_cls, verify_text = "is-ok", "忠实"
    elif faithful is False:
        verify_cls, verify_text = "is-bad", "检出幻觉"
    else:
        verify_cls, verify_text = "is-warn", "未验证"
    heal_names = {name for rnd in view.rounds for name, _ in rnd.actions}
    heal_text = "、".join(sorted(heal_names)) if heal_names else "未触发"
    if summary["error"]:
        end_card = _metric("端到端状态", "异常中止", "is-bad", str(summary["error"]))
    elif summary.get("generation_failed"):
        end_card = _metric(
            "端到端状态",
            "生成降级",
            "is-warn",
            str(summary.get("generation_error") or "LLM 生成失败，返回兜底文案"),
        )
    else:
        end_card = _metric("端到端状态", "完成", "is-ok", "状态机走到 END，未触发递归上限")
    cards = (
        end_card,
        _metric("检索轮次", str(summary["iteration_count"]), "is-plain",
                "retrieve 节点执行次数；上限由 max_iterations 控制"),
        _metric("自愈动作", heal_text, "is-heal" if heal_names else "is-plain",
                "reformulate 的实际触发情况", text=True),
        _metric("幻觉检测", verify_text, verify_cls, "verify 节点对答案忠实性的判定", text=True),
    )
    reason = summary["verification_reason"]
    footnote = f'<div class="rf-note">校验说明：{escape(str(reason))}</div>' if reason else ""
    return f'<div class="rf-metrics">{"".join(cards)}</div>{footnote}'


# ── 片段辅助 ───────────────────────────────────────────────────────
def _fmt_value(key: str, value: Any) -> str:
    """字段值 → 中文语义 + 原始值（保留原始值便于与日志/代码对照）。"""
    if isinstance(value, bool):
        return (
            f'<span class="rf-val is-plain">{"是" if value else "否"}'
            f"<em>{str(value).lower()}</em></span>"
        )
    if value is None:
        return '<span class="rf-val is-warn">未验证</span>'
    label = _VALUE_LABELS.get(key, {}).get(value)
    cls = _MODE_CLASS.get(str(value), "is-plain")
    text = f"{escape(str(label))}<em>{escape(str(value))}</em>" if label else escape(str(value))
    return f'<span class="rf-val {cls}">{text}</span>'


def _chip(value: Any, key: str) -> str:
    """枚举值 → 语义色小标签。"""
    label = _VALUE_LABELS.get(key, {}).get(value, str(value))
    cls = _MODE_CLASS.get(str(value), "is-plain")
    return f'<span class="rf-chip {cls}">{escape(str(label))}</span>'


def _budget_note(view: EventView) -> str:
    """预算截断说明：轮次跑满且最后一轮仍未达标时，循环是被强制停掉的。

    没有这行提示，界面只能看出「检索了 3 轮」，看不出「为什么不再重试」。
    """
    summary = view.summary()
    limit = summary.get("max_iterations")
    rounds = view.rounds
    if not limit or len(rounds) < int(limit):
        return ""
    last = rounds[-1]
    if last.actions or last.failure_mode in (None, "sufficient"):
        return ""
    return (
        f'<div class="rf-note">⚠ 检索轮次已达 max_iterations={int(limit)} 上限，'
        f'路由强制生成答案（第 {last.index} 轮诊断为 {escape(str(last.failure_mode))}）。'
        f'双预算防死循环在此生效：循环被截断，答案质量随之受限。</div>'
    )


def _render_round_actions(rnd: Round) -> str:
    """轮次的下一步动作单元格。"""
    if not rnd.actions:
        suggested = rnd.suggested_action or ""
        if suggested == "proceed":
            return '<span class="rf-chip is-ok">进入生成</span>'
        if not suggested:
            return "—"
        return f'<span class="rf-muted">建议 {_chip(suggested, "suggested_action")}</span>'
    parts = []
    for name, detail in rnd.actions:
        parts.append(f'{_chip(name, "suggested_action")}<em class="rf-detail">{escape(detail)}</em>')
    return "<br>".join(parts)


def _basename(path: str) -> str:
    """只留文件名：绝对路径在窄列里会折成两行，噪音大于信息。"""
    normalized = path.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1] if normalized else "unknown"


def _render_body(text: str) -> str:
    """chunk 正文 → 受控 HTML 段落。

    Streamlit 会把注入的 HTML 再走一遍 markdown 解析：正文里行首的 `1. ` 被识成
    <ol>、空行被拆成 <p>，叠加 white-space: pre-wrap 后实测每个列表前多出约 58px
    无信息间距。这里先把每段压成单行（段内换行用 <br>），markdown 就看不到行首
    列表标记，段间距改由 CSS 精确控制。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return "".join(
        f'<p class="rf-para">{escape(paragraph).replace(chr(10), "<br>")}</p>'
        for paragraph in paragraphs
    ) or '<p class="rf-para rf-muted">（空文档段）</p>'


def _metric(label: str, value: str, cls: str, help_text: str, *, text: bool = False) -> str:
    """摘要卡片；text=True 用小字号长文本样式（避免枚举名被断词）。"""
    value_cls = "rf-metric-value is-text" if text else "rf-metric-value"
    return (
        f'<div class="rf-metric {cls}" title="{escape(help_text, quote=True)}">'
        f'<span class="rf-metric-label">{escape(label)}</span>'
        f'<span class="{value_cls}">{escape(value)}</span></div>'
    )


# ── 样式表（界面侧一次性注入）───────────────────────────────────────
CSS = """
<style>
:root {
  --rf-line: rgba(128,128,128,.26);
  --rf-fill: rgba(128,128,128,.07);
  --rf-muted: rgba(140,140,140,.9);
  --rf-pending: rgba(140,140,140,.55);
  --rf-run: #f0a020;
  --rf-ok: #2fbf71;
  --rf-warn: #e8a33d;
  --rf-bad: #e5484d;
  --rf-heal: #b45cf0;
  --rf-mono: ui-monospace, "Cascadia Mono", "Segoe UI Mono", Consolas, monospace;
}
.rf-shell { font-size: .9rem; line-height: 1.5; container-type: inline-size; }
.rf-sec { display: flex; align-items: baseline; gap: 9px; margin: 22px 0 8px;
  padding-bottom: 5px; border-bottom: 1px solid var(--rf-line); }
.rf-sec-num { font-family: var(--rf-mono); color: var(--rf-ok); font-weight: 700; }
.rf-sec-title { font-size: 1.02rem; font-weight: 700; }
.rf-sec-hint { font-size: .78rem; color: var(--rf-muted); }
/* 列数按容器实宽而非视口：侧边栏展开时主内容区可窄至 400px，@media 会误判 */
.rf-shell .rf-track { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; }
.rf-node { border: 1px solid var(--rf-line); border-radius: 8px; padding: 8px 10px;
  background: var(--rf-fill); transition: border-color .2s; }
.rf-node-head { display: flex; align-items: center; justify-content: space-between; }
.rf-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
.is-pending > .rf-dot, i.is-pending { background: var(--rf-pending); }
.is-running > .rf-dot, i.is-running { background: var(--rf-run); animation: rf-pulse 1s ease-in-out infinite; }
.is-done > .rf-dot, i.is-done { background: var(--rf-ok); }
.is-heal > .rf-dot, i.is-heal { background: var(--rf-heal); }
@keyframes rf-pulse { 0%,100% { opacity: .35; transform: scale(.85);} 50% { opacity: 1; transform: scale(1.15);} }
.rf-node-cn { font-weight: 600; margin-top: 4px; }
.rf-node-en { font-family: var(--rf-mono); font-size: .72rem; color: var(--rf-muted);
  overflow-wrap: break-word; }
.rf-node.is-pending .rf-node-cn { color: var(--rf-pending); }
.rf-node.is-running { border-color: var(--rf-run); }
.rf-node.is-done { border-color: rgba(47,191,113,.55); }
.rf-node.is-heal { border-color: var(--rf-heal); background: rgba(180,92,240,.10); }
.rf-badge { font-family: var(--rf-mono); font-size: .7rem; padding: 1px 6px; border-radius: 999px;
  background: rgba(180,92,240,.18); color: var(--rf-heal); }
.rf-legend { display: flex; gap: 16px; margin-top: 8px; font-size: .76rem; color: var(--rf-muted); }
.rf-legend span { display: flex; align-items: center; gap: 5px; }
.rf-legend i { width: 8px; height: 8px; }

.rf-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
  gap: 10px; align-items: start; }
.rf-group { border: 1px solid var(--rf-line); border-radius: 8px; padding: 10px 12px; background: var(--rf-fill); }
.rf-group-title { font-weight: 600; margin-bottom: 6px; }
.rf-group-title em { display: block; font-style: normal; font-size: .74rem; color: var(--rf-muted); font-weight: 400; }
.rf-kv { display: flex; gap: 8px; align-items: baseline; padding: 3px 0; border-top: 1px dashed rgba(128,128,128,.18); }
.rf-kv > span:first-child { color: var(--rf-muted); flex: none; min-width: 5.5em; }
.rf-val { font-weight: 600; }
.rf-val em, .rf-chip + em { font-style: normal; font-family: var(--rf-mono); font-size: .72rem;
  color: var(--rf-muted); font-weight: 400; margin-left: 5px; }
.rf-val.is-ok, .rf-chip.is-ok { color: var(--rf-ok); }
.rf-val.is-warn, .rf-chip.is-warn { color: var(--rf-warn); }
.rf-val.is-bad, .rf-chip.is-bad { color: var(--rf-bad); }
.rf-val.is-heal, .rf-chip.is-heal { color: var(--rf-heal); }
.rf-chip { font-weight: 600; }

.rf-rounds { border: 1px solid var(--rf-line); border-radius: 8px; overflow: hidden; }
.rf-round-head, .rf-round-row { display: grid; gap: 10px; align-items: baseline;
  grid-template-columns: repeat(2, minmax(0,1fr)); }
.rf-round-head { background: var(--rf-fill); padding: 7px 12px; font-size: .76rem; color: var(--rf-muted); }
.rf-round { border-top: 1px solid var(--rf-line); border-left: 3px solid transparent;
  padding: 9px 12px; }
/* 左框常驻（未自愈时为透明白）：否则自愈行比相邻行多占 3px，同一表内列起点错开 */
.rf-round.is-healed { border-left-color: var(--rf-heal); background: rgba(180,92,240,.06); }
.rf-round-row > span { min-width: 0; }
.rf-idx { font-family: var(--rf-mono); font-size: .78rem; color: var(--rf-muted); }
.rf-query { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden; }
.rf-score em, .rf-detail { display: block; font-style: normal; font-size: .72rem; color: var(--rf-muted); }
.rf-reason { margin-top: 5px; font-size: .8rem; color: var(--rf-muted); }
.rf-muted { color: var(--rf-muted); }

.rf-src-group-title { font-size: .78rem; color: var(--rf-muted); margin: 8px 0 4px; }
.rf-src { border: 1px solid var(--rf-line); border-radius: 7px; margin-bottom: 5px; background: var(--rf-fill); }
.rf-src summary { display: grid; gap: 9px; align-items: center; cursor: pointer;
  padding: 7px 11px; list-style: none; grid-template-columns: 1.6em minmax(0,1fr) 7em 4.5em; }
.rf-src summary::-webkit-details-marker { display: none; }
.rf-src-idx { font-family: var(--rf-mono); color: var(--rf-muted); }
.rf-src-name { font-family: var(--rf-mono); font-size: .82rem; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.rf-src-score { font-family: var(--rf-mono); font-size: .8rem; text-align: right; }
.rf-bar { display: block; height: 5px; border-radius: 3px; background: rgba(128,128,128,.22); overflow: hidden; }
.rf-bar i { display: block; height: 100%; background: var(--rf-ok); }
.rf-src-body { padding: 9px 12px; border-top: 1px solid var(--rf-line);
  font-size: .82rem; overflow-wrap: break-word; }
.rf-para { margin: 0 0 .45em; }
.rf-para:last-child { margin-bottom: 0; }
.rf-src[open] { border-color: rgba(47,191,113,.45); }

.rf-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 10px; }
.rf-metric { border: 1px solid var(--rf-line); border-left-width: 3px; border-radius: 8px;
  padding: 9px 12px; background: var(--rf-fill); }
.rf-metric-label { display: block; font-size: .76rem; color: var(--rf-muted); }
.rf-metric-value { display: block; font-size: 1.35rem; font-weight: 700; margin-top: 2px;
  overflow-wrap: break-word; }
.rf-metric-value.is-text { font-size: .95rem; font-weight: 600; line-height: 1.35; }
.rf-metric.is-ok { border-left-color: var(--rf-ok); }
.rf-metric.is-warn { border-left-color: var(--rf-warn); }
.rf-metric.is-bad { border-left-color: var(--rf-bad); }
.rf-metric.is-heal { border-left-color: var(--rf-heal); }
.rf-metric.is-plain { border-left-color: var(--rf-line); }
.rf-metric.is-ok .rf-metric-value { color: var(--rf-ok); }
.rf-metric.is-bad .rf-metric-value { color: var(--rf-bad); }
.rf-metric.is-heal .rf-metric-value { color: var(--rf-heal); }
.rf-note { margin-top: 8px; font-size: .8rem; color: var(--rf-muted); }
.rf-empty { color: var(--rf-muted); font-size: .85rem; padding: 6px 0; }
@container (min-width: 620px) {
  .rf-shell .rf-track { grid-template-columns: repeat(4, minmax(0,1fr)); }
}
@container (min-width: 900px) {
  .rf-round-head, .rf-round-row {
    grid-template-columns: 4.5em minmax(0,1.6fr) 6.5em minmax(0,1.1fr) 6em minmax(0,1.2fr);
  }
}
</style>
"""
