"""Streamlit UI 单元测试（离线）：EventView 纯逻辑事件蒸馏

重点回归：
1. 节点三态与循环计数（自愈可视化的数据来源）
2. token 流按节点过滤（结构化节点的 JSON 碎片不得混入答案流——实测确认的泄漏）
3. 诊断字段提取 / 最终状态捕获 / 来源表格
"""

from types import SimpleNamespace

from src.ui.event_view import DONE, PENDING, RUNNING, EventView

# ── 合成事件工厂 ───────────────────────────────────────────────────


def chain_start(name: str) -> dict:
    return {"event": "on_chain_start", "name": name}


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
    """自愈序列：retrieve → evaluate(switch) → switch_strategy → retrieve ×2。"""
    view = EventView()
    view.consume(chain_start("analyze"))
    view.consume(chain_end("analyze", {"query_type": "complex"}))
    view.consume(chain_start("decide"))
    view.consume(chain_end("decide", {"retrieval_strategy": "hybrid"}))

    view.consume(chain_start("retrieve"))
    view.consume(chain_end("retrieve", {"documents": [], "retrieval_scores": []}))
    view.consume(chain_start("evaluate"))
    view.consume(chain_end("evaluate", {"failure_mode": "low_recall", "suggested_action": "switch_strategy"}))
    view.consume(chain_start("switch_strategy"))
    view.consume(chain_end("switch_strategy", {"retrieval_strategy": "hybrid"}))
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
        {"failure_mode": "low_recall", "suggested_action": "switch_strategy",
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
