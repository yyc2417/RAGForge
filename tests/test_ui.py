"""Streamlit UI 单元测试（离线）：EventView 事件蒸馏 + html_view 呈现装配

重点回归：
1. 节点三态与循环计数（自愈可视化的数据来源）
2. token 流按节点过滤（结构化节点的 JSON 碎片不得混入答案流——实测确认的泄漏）
3. 诊断字段提取 / 最终状态捕获 / 来源表格
4. 逐轮归集：第二轮 evaluate 得覆盖第一轮的诊断结论（自愈因果链必须保留）
5. HTML 区块：单元素写入、转义、枚举值中文化
"""

from types import SimpleNamespace

from src.ui import html_view
from src.ui.event_view import (
    DONE,
    LEVEL_ANSWER,
    LEVEL_FULL,
    PENDING,
    RUNNING,
    EventView,
)

# ── 合成事件工厂 ───────────────────────────────────────────────────


def chain_start(name: str, input_state: dict | None = None) -> dict:
    event = {"event": "on_chain_start", "name": name}
    if input_state is not None:
        event["data"] = {"input": input_state}
    return event


def chain_end(name: str, output: dict | None = None) -> dict:
    return {"event": "on_chain_end", "name": name, "data": {"output": output or {}}}


def token_event(node: str, text: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": SimpleNamespace(content=text)},
    }


def doc(source: str, content: str, score: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        page_content=content, metadata={"source": source, "chunk_index": idx}
    )


# ── 节点三态与循环计数 ────────────────────────────────────────────


def test_all_nodes_pending_initially():
    view = EventView()
    assert all(view.node_status[n] == PENDING for n in view.node_status)


def test_node_transitions_start_to_done():
    view = EventView()
    view.consume(chain_start("analyze"))
    assert view.node_status["analyze"] == RUNNING
    view.consume(chain_end("analyze", {"query_type": "factual"}))
    assert view.node_status["analyze"] == DONE
    assert view.node_runs["analyze"] == 1


def test_loop_counting_and_self_heal_flag():
    """自愈序列：retrieve → evaluate(reformulate) → reformulate → retrieve ×2。"""
    view = EventView()
    view.consume(chain_start("analyze"))
    view.consume(chain_end("analyze", {"query_type": "complex"}))
    view.consume(chain_start("decide"))
    view.consume(chain_end("decide", {"retrieval_strategy": "hybrid"}))

    view.consume(chain_start("retrieve"))
    view.consume(chain_end("retrieve", {"documents": [], "retrieval_scores": []}))
    view.consume(chain_start("evaluate"))
    view.consume(chain_end("evaluate", {"failure_mode": "low_recall", "suggested_action": "reformulate"}))
    view.consume(chain_start("reformulate"))
    view.consume(chain_end("reformulate", {"reformulated_query": "改写查询", "rewrite_effective": True}))
    # 第二轮检索
    view.consume(chain_start("retrieve"))
    view.consume(chain_end("retrieve", {"documents": [], "retrieval_scores": []}))

    assert view.node_runs["retrieve"] == 2
    assert view.node_status["retrieve"] == DONE
    assert view.self_heal_triggered is True
    assert view.total_llm_rounds == 2


def test_internal_chain_names_ignored():
    """LangGraph 内部链路（根图/模型/Runnable）不得点亮业务节点。"""
    view = EventView()
    for name in ("LangGraph", "ChatOpenAI", "RunnableSequence", "CaseReducer"):
        view.consume(chain_start(name))
        view.consume(chain_end(name, {"answer": "x"}))
    assert all(runs == 0 for runs in view.node_runs.values())
    assert all(status == PENDING for status in view.node_status.values())


# ── token 过滤（泄漏回归）─────────────────────────────────────────


def test_tokens_only_from_generate_node():
    """结构化节点（analyze）的 JSON 碎片必须被排除，答案流只含 generate 输出。"""
    view = EventView()
    view.consume(token_event("analyze", '{"query_type":"chitchat"'))
    view.consume(token_event("evaluate", ',"failure_mode":"sufficient"}'))
    view.consume(token_event("generate", "你好"))
    view.consume(token_event("generate", "！有什么可以帮你？"))
    assert view.answer == "你好！有什么可以帮你？"


def test_empty_token_chunks_skipped():
    view = EventView()
    view.consume(token_event("generate", ""))
    assert view.answer == ""


# ── 诊断字段与最终状态 ────────────────────────────────────────────


def test_diagnostic_fields_extracted():
    view = EventView()
    view.consume(chain_end("analyze", {"query_type": "complex", "needs_retrieval": True}))
    view.consume(chain_end("decide", {"retrieval_strategy": "hybrid"}))
    view.consume(chain_end(
        "evaluate",
        {"failure_mode": "low_recall", "suggested_action": "reformulate",
         "diagnosis_reason": "关键信息缺失"},
    ))
    assert view.diagnostic["query_type"] == "complex"
    assert view.diagnostic["retrieval_strategy"] == "hybrid"
    assert view.diagnostic["failure_mode"] == "low_recall"
    assert view.diagnostic["diagnosis_reason"] == "关键信息缺失"


def test_final_state_captured_by_answer_key():
    """最终状态按「output 含 answer 键」捕获，不依赖硬编码链路名。"""
    view = EventView()
    # generate 节点输出含 answer，但只算节点输出（DONE + 诊断），不进 final_state
    view.consume(chain_start("generate"))
    view.consume(chain_end("generate", {"answer": "部分", "messages": ["x"]}))
    assert view.node_status["generate"] == DONE
    assert view.final_state == {}
    # 根图最终输出（字段更全）捕获为 final_state
    view.consume(chain_end("LangGraph", {"answer": "完整", "is_faithful": True, "iteration_count": 2}))
    assert view.final_state["answer"] == "完整"
    summary = view.summary()
    assert summary["is_faithful"] is True
    assert summary["iteration_count"] == 2


def test_answer_falls_back_to_final_state():
    """无流式 token 时（如 mock 场景），答案回退到最终状态。"""
    view = EventView()
    view.consume(chain_end("LangGraph", {"answer": "最终答案"}))
    assert view.answer == "最终答案"


# ── 来源表格 ──────────────────────────────────────────────────────


def test_sources_extracted_from_retrieve_output():
    view = EventView()
    view.consume(chain_end("retrieve", {
        "documents": [
            doc("a.md", "内容 A", 0.9, 0),
            doc("b.md", "内容 B", 0.5, 1),
        ],
        "retrieval_scores": [0.9, 0.5],
    }))
    assert view.sources == [
        {"content": "内容 A", "source": "a.md", "score": 0.9},
        {"content": "内容 B", "source": "b.md", "score": 0.5},
    ]


def test_sources_cleared_when_no_retrieval():
    """寒暄路径不产生 documents，sources 保持为空。"""
    view = EventView()
    view.consume(chain_end("generate", {"answer": "你好！"}))
    assert view.sources == []


# ── 错误捕获 ──────────────────────────────────────────────────────


def test_chain_error_captured():
    view = EventView()
    view.consume({"event": "on_chain_error", "name": "generate",
                  "data": {"error": RuntimeError("LLM 宕机")}})
    assert "LLM 宕机" in (view.error or "")
    summary = view.summary()
    assert summary["error"] is not None


# ── 变更级别（界面重画节流依据）─────────────────────────────────


def test_consume_reports_change_levels():
    view = EventView()
    assert view.consume(chain_start("analyze")) == LEVEL_FULL
    assert view.consume(chain_start("LangGraph")) is None  # 非业务节点不触发重画
    assert view.consume(token_event("analyze", '{"a":1}')) is None
    assert view.consume(token_event("generate", "你")) == LEVEL_ANSWER
    assert view.consume(chain_end("decide", {})) == LEVEL_FULL


# ── 逐轮归集（自愈时间线）───────────────────────────────────────


def _reformulate_events() -> list[dict]:
    """完整重放一条固定 hybrid + 查询改写轨迹（两轮检索）。"""
    return [
        chain_start("analyze"),
        chain_end("analyze", {"query_type": "factual", "needs_retrieval": True}),
        chain_start("decide"),
        chain_end("decide", {"retrieval_strategy": "hybrid"}),
        chain_start("retrieve", {
            "query": "令牌桶限流怎么实现",
            "retrieval_strategy": "hybrid", "iteration_count": 0,
        }),
        chain_end("retrieve", {
            "documents": [doc("D:\\sample\\web_api_design.md", "粗排召回的无关内容", 0.021, 0)],
            "retrieval_scores": [0.021], "iteration_count": 1,
        }),
        chain_start("evaluate"),
        chain_end("evaluate", {
            "failure_mode": "low_recall", "suggested_action": "reformulate",
            "diagnosis_reason": "粗排未命中关键章节",
        }),
        chain_start("reformulate"),
        chain_end("reformulate", {
            "reformulated_query": "令牌桶算法如何实现？", "rewrite_effective": True,
        }),
        chain_start("retrieve", {
            "query": "令牌桶限流怎么实现",
            "reformulated_query": "令牌桶算法如何实现？",
            "retrieval_strategy": "hybrid", "iteration_count": 1,
        }),
        chain_end("retrieve", {
            "documents": [doc("data/sample/web_api_design.md", "令牌桶算法定义", 0.032, 3)],
            "retrieval_scores": [0.032], "iteration_count": 2,
        }),
        chain_start("evaluate"),
        chain_end("evaluate", {
            "failure_mode": "sufficient", "suggested_action": "proceed",
            "diagnosis_reason": "命中限流章节",
        }),
        chain_start("generate"),
        chain_end("generate", {"answer": "令牌桶…"}),
        chain_end("LangGraph", {
            "answer": "令牌桶…", "is_faithful": True, "verification_reason": "忠实",
            "iteration_count": 2, "verify_failures": 0,
        }),
    ]


def _replayed() -> EventView:
    view = EventView()
    for event in _reformulate_events():
        view.consume(event)
    return view


def test_rounds_capture_strategy_query_and_hits():
    first, second = _replayed().rounds
    assert (first.index, first.strategy, first.hits) == (1, "hybrid", 1)
    assert (second.index, second.strategy, second.hits) == (2, "hybrid", 1)
    assert second.query == "令牌桶算法如何实现？"
    assert (first.top_score, second.top_score) == (0.021, 0.032)


def test_second_round_does_not_overwrite_first_diagnosis():
    """核心回归：诊断字段只留终态，逐轮归档必须各自保留自己的结论。"""
    first, second = _replayed().rounds
    assert first.failure_mode == "low_recall"
    assert first.diagnosis_reason == "粗排未命中关键章节"
    assert second.failure_mode == "sufficient"
    # 顶层诊断面板仍展示终态（与时间线各取所需）
    assert second.diagnosis_reason == "命中限流章节"


def test_reformulate_action_records_rewritten_query_in_first_round():
    first, second = _replayed().rounds
    assert first.actions == [("reformulate", "令牌桶算法如何实现？")]
    assert first.healed is True
    assert second.actions == [] and second.healed is False


def test_reformulate_action_records_rewritten_query():
    view = EventView()
    view.consume(chain_start("retrieve", {"query": "什么是过拟合"}))
    view.consume(chain_end("retrieve", {
        "documents": [], "retrieval_scores": [], "iteration_count": 1,
    }))
    view.consume(chain_end("evaluate", {
        "failure_mode": "irrelevant", "suggested_action": "reformulate",
    }))
    view.consume(chain_end("reformulate", {
        "reformulated_query": "机器学习中什么是过拟合？如何解决？", "rewrite_effective": True,
    }))
    assert view.rounds[0].actions == [("reformulate", "机器学习中什么是过拟合？如何解决？")]
    assert view.rounds[0].rewrite_effective is True
    # 第二轮检索使用改写后的查询
    view.consume(chain_start("retrieve", {
        "query": "什么是过拟合", "reformulated_query": "机器学习中什么是过拟合？如何解决？",
        "retrieval_strategy": "hybrid", "iteration_count": 1,
    }))
    view.consume(chain_end("retrieve", {
        "documents": [], "retrieval_scores": [], "iteration_count": 2,
    }))
    assert view.rounds[1].query == "机器学习中什么是过拟合？如何解决？"


def test_round_strategy_falls_back_to_decide_output():
    """节点入参缺失（旧事件格式 / mock）时，策略回退到已捕获的 decide 值。"""
    view = EventView()
    view.consume(chain_end("decide", {"retrieval_strategy": "hybrid"}))
    view.consume(chain_start("retrieve"))
    view.consume(chain_end("retrieve", {"documents": [], "retrieval_scores": []}))
    assert view.rounds[0].strategy == "hybrid"
    assert view.rounds[0].index == 1  # 无 iteration_count 时按轮序兜底


def test_chitchat_has_no_rounds():
    view = EventView()
    view.consume(chain_end("analyze", {"query_type": "chitchat", "needs_retrieval": False}))
    view.consume(chain_end("decide", {"retrieval_strategy": "none"}))
    view.consume(chain_end("generate", {"answer": "你好！"}))
    assert view.rounds == []
    assert "未触发检索" in html_view.render_rounds(view)


# ── html_view 呈现装配 ────────────────────────────────────────────


def test_render_track_badges_and_self_heal_class():
    html = html_view.render_track(_replayed())
    assert "执行检索" in html and "retrieve" in html  # 中文职责名 + 原节点名双标
    assert "2 次" in html                              # 循环计数徽章
    assert 'class="rf-node is-heal"' in html           # 自愈节点独立配色
    assert html.count("rf-node ") == 7                 # 7 张节点卡片齐备


def test_render_diagnosis_groups_values_in_chinese():
    html = html_view.render_diagnosis(_replayed())
    assert "诊断结论" in html and "信息充足" in html  # 枚举值中文化
    assert "sufficient" in html                        # 原始值保留，便于对照日志
    assert "检索轮次" not in html                      # 归运行摘要，诊断区不重复展示


def test_render_diagnosis_hides_absent_fields():
    view = EventView()
    view.consume(chain_end("analyze", {"query_type": "factual"}))
    html = html_view.render_diagnosis(view)
    assert "查询意图" in html and "事实查询" in html
    assert "改写后查询" not in html  # 未发生的节点不留空行


def test_render_rounds_keeps_both_rounds_readable():
    html = html_view.render_rounds(_replayed())
    assert "第 1 轮" in html and "第 2 轮" in html
    assert "召回不足" in html and "改写查询" in html
    assert "令牌桶算法如何实现？" in html
    assert "is-healed" in html  # 触发自愈的行高亮


def test_render_sources_groups_by_round_and_shortens_path():
    html = html_view.render_sources(_replayed())
    assert "第 1 轮 · hybrid · 1 篇" in html
    assert "第 2 轮 · hybrid · 1 篇" in html
    # 可见文本只留文件名（绝对路径在窄列里会折行），完整路径移到 hover
    assert 'title="D:\\sample\\web_api_design.md">web_api_design.md<' in html
    assert html.count("rf-src-name") == 2


def test_render_sources_escapes_html_in_content():
    view = EventView()
    view.consume(chain_start("retrieve", {"query": "q", "retrieval_strategy": "hybrid"}))
    view.consume(chain_end("retrieve", {
        "documents": [doc("a.md", "<script>alert(1)</script>", 0.5, 0)],
        "retrieval_scores": [0.5], "iteration_count": 1,
    }))
    html = html_view.render_sources(view)
    assert "&lt;script&gt;" in html and "<script>alert" not in html


def test_render_sources_flows_paragraphs_without_markdown_lists():
    """正文行首的 `1. ` 必须被压进单行：Streamlit 会对注入的 HTML 再做 markdown 解析，
    否则正文被拆成 <ol>/<p>，叠加间距后实测多出约 58px 无信息空白。"""
    view = EventView()
    view.consume(chain_start("retrieve", {"query": "q", "retrieval_strategy": "hybrid"}))
    view.consume(chain_end("retrieve", {
        "documents": [doc("a.md", "方法：\n1. 早停\n2. 正则化\n\n另一段", 0.9, 0)],
        "retrieval_scores": [0.9], "iteration_count": 1,
    }))
    html = html_view.render_sources(view)
    assert "1. 早停<br>2. 正则化" in html
    assert "\n1. " not in html
    assert html.count('class="rf-para"') == 2


def test_render_rounds_notes_budget_cutoff():
    """跑满预算且末轮仍未达标：界面必须说明循环是被强制停掉的。"""
    view = EventView()
    for i in range(1, 4):
        view.consume(chain_start("retrieve", {
            "query": "q", "retrieval_strategy": "hybrid", "iteration_count": i - 1,
        }))
        view.consume(chain_end("retrieve", {
            "documents": [], "retrieval_scores": [], "iteration_count": i,
        }))
        view.consume(chain_end("evaluate", {
            "failure_mode": "irrelevant", "suggested_action": "reformulate",
        }))
    view.consume(chain_end("LangGraph", {"answer": "a", "max_iterations": 3,
                                         "iteration_count": 3}))
    html = html_view.render_rounds(view)
    assert "max_iterations=3 上限" in html and "强制生成" in html


def test_render_rounds_silent_when_loop_not_truncated():
    """未跑满预算（或末轮自己走通）时不得误报截断。"""
    assert "max_iterations" not in html_view.render_rounds(_replayed())


def test_render_summary_reports_heal_actions():
    html = html_view.render_summary(_replayed())
    assert "reformulate" in html  # 自愈动作实际触发情况
    assert "忠实" in html and "检索轮次" in html


def test_render_summary_flags_errors():
    view = EventView()
    view.consume({"event": "on_chain_error", "name": "generate",
                  "data": {"error": RuntimeError("LLM 宕机")}})
    assert "异常中止" in html_view.render_summary(view)
