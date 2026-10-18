"""Agent 决策层单元测试——全 mock，离线

覆盖：
1. 纯规则节点：decide（固定 hybrid / none）
2. 路由矩阵：evaluate（proceed / 达上限）/ reformulate（改写无效短路）/
   verify（None 不触发循环、双预算终止）
3. evaluate 节点：空结果改写、LLM 诊断、无绝对分数阈值
4. 结构化输出：枚举归一化（旧 switch 输出映射为 reformulate）、JSON 提取
5. 状态机端到端（Fake 组件跑真实图）：正常单轮、幻觉循环、
   改写无效短路、最坏路径终止性、mermaid 可视化
"""

import json

from langchain_core.documents import Document

from fakes import FakeLLM, FakeRetriever, make_script, scripted_verify
from src.agent.graph import (
    build_agent_graph,
    route_after_analyze,
    route_after_evaluate,
    route_after_reformulate,
    route_after_verify,
)
from src.agent.nodes import analyze_node, clarify_node, decide_node, evaluate_node
from src.agent.state import initial_state
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import (
    AnalyzeResult,
    EvaluateResult,
    ReformulateResult,
    VerifyResult,
)

PROMPTS = PromptManager()


def test_schema_aliases_for_local_model_variants():
    analyze = AnalyzeResult.model_validate({
        "query_type": "Question",
        "needs_retrieval": True,
        "needs_clarification": True,
        "ambiguity_type": "missing_entity",
    })
    assert analyze.query_type == "factual"
    assert analyze.ambiguity_type == "missing_slot"

    evaluate = EvaluateResult.model_validate({
        "failure_mode": "insufficient",
        "suggested_action": "continue",
    })
    assert evaluate.failure_mode == "low_recall"
    assert evaluate.suggested_action == "proceed"

    reformulate = ReformulateResult.model_validate({
        "reformulated_query": "具体问题",
        "rewrite_strategy": "paraphrase",
    })
    assert reformulate.rewrite_strategy == "synonym_replace"


# ── 纯规则节点 ────────────────────────────────────────────────────


def test_decide_rules():
    assert decide_node({"query_type": "chitchat"})["retrieval_strategy"] == "none"
    assert decide_node({"needs_retrieval": False})["retrieval_strategy"] == "none"
    assert decide_node({"query_type": "factual"})["retrieval_strategy"] == "hybrid"
    assert decide_node({"query_type": "reasoning"})["retrieval_strategy"] == "hybrid"
    assert decide_node({"query_type": "complex"})["retrieval_strategy"] == "hybrid"
    assert decide_node({})["retrieval_strategy"] == "hybrid"  # 未知类型安全默认


def test_initial_state_defaults():
    state = initial_state("q")
    assert state["iteration_count"] == 0
    assert state["verify_failures"] == 0
    assert state["max_iterations"] == 3


# ── clarify 终止分支 ──────────────────────────────────────────────


def test_route_after_analyze():
    assert route_after_analyze({"needs_clarification": True}) == "clarify"
    assert route_after_analyze({"needs_clarification": False}) == "decide"
    assert route_after_analyze({}) == "decide"


def test_analyze_clarification_fields():
    llm = FakeLLM(make_script(
        AnalyzeResult=AnalyzeResult(
            query_type="complex",
            needs_retrieval=True,
            needs_clarification=True,
            ambiguity_type="multiple_interpretations",
            clarification_question="你指的是 Python 的 GIL，还是 Java 的 GC？",
        )
    ))
    out = analyze_node({"query": "它怎么配置？"}, llm=llm, prompts=PROMPTS)
    assert out["needs_clarification"] is True
    assert out["ambiguity_type"] == "multiple_interpretations"
    assert "GIL" in out["clarification_question"]


def test_clarify_node_output():
    out = clarify_node({
        "clarification_question": "你指的是哪个对象？",
        "ambiguity_type": "missing_slot",
    })
    assert out["answer"] == ""
    assert out["clarification_question"] == "你指的是哪个对象？"
    assert "clarify" in out["messages"][0]


def test_graph_clarify_terminates(make_graph):
    script = make_script(
        AnalyzeResult=AnalyzeResult(
            query_type="complex",
            needs_retrieval=True,
            needs_clarification=True,
            ambiguity_type="missing_slot",
            clarification_question="你指的是哪个对象？",
        )
    )
    state = make_graph(script).invoke(initial_state("它怎么用？"), config={"recursion_limit": 50})
    assert state["answer"] == ""
    assert state["clarification_question"] == "你指的是哪个对象？"
    assert state["iteration_count"] == 0
    assert any(m.startswith("clarify:") for m in state.get("messages", []))


# ── 路由矩阵 ──────────────────────────────────────────────────────


def test_route_after_evaluate():
    assert route_after_evaluate({"suggested_action": "proceed"}) == "generate"
    assert route_after_evaluate(
        {"suggested_action": "reformulate", "iteration_count": 1, "max_iterations": 3}
    ) == "reformulate"
    assert route_after_evaluate(
        {"suggested_action": "reformulate", "iteration_count": 3, "max_iterations": 3}
    ) == "generate"
    # 旧 switch 输出在 schema 层已归一化为 reformulate；路由对非 proceed 统一改写
    assert route_after_evaluate({"suggested_action": "switch_strategy"}) == "reformulate"


def test_route_after_reformulate():
    assert route_after_reformulate({"rewrite_effective": True}) == "retrieve"
    assert route_after_reformulate({"rewrite_effective": False}) == "generate"
    assert route_after_reformulate({}) == "generate"  # 缺字段安全默认为无效


def test_route_after_verify():
    assert route_after_verify({"is_faithful": True}) == "end"
    assert route_after_verify({"is_faithful": None}) == "end"  # 未验证不触发循环
    assert route_after_verify(
        {"is_faithful": False, "verify_failures": 1, "max_iterations": 3}
    ) == "reformulate"
    assert route_after_verify(
        {"is_faithful": False, "verify_failures": 3, "max_iterations": 3}
    ) == "end"  # 幻觉重试上限
    assert route_after_verify(
        {"is_faithful": False, "verify_failures": 1, "iteration_count": 3, "max_iterations": 3}
    ) == "end"  # 检索预算耗尽


# ── evaluate 节点分数语义 ─────────────────────────────────────────


class SufficientLLM:
    """恒定返回 sufficient 的 LLM stub，用于检验规则层不越权。"""

    def invoke_structured(self, prompt, schema, **kwargs):
        assert schema is EvaluateResult
        return EvaluateResult(failure_mode="sufficient", suggested_action="proceed", reason="LLM 认为充足")


def test_evaluate_hybrid_rrf_scores_not_flagged():
    """hybrid 的 RRF 低分（≈0.03）不得套用绝对阈值；LLM 判 sufficient 即 proceed。"""
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "hybrid",
         "documents": [Document(page_content="doc")], "retrieval_scores": [0.033, 0.028]},
        llm=SufficientLLM(), prompts=PROMPTS,
    )
    assert out["suggested_action"] == "proceed"


def test_evaluate_empty_docs_reformulate():
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "hybrid", "documents": [], "retrieval_scores": []},
        llm=SufficientLLM(), prompts=PROMPTS,
    )
    assert out["failure_mode"] == "low_recall" and out["suggested_action"] == "reformulate"


class BrokenEvaluateLLM:
    def invoke_structured(self, prompt, schema, **kwargs):
        raise RuntimeError("judge down")


def test_evaluate_llm_failure_defaults_proceed():
    """LLM 诊断失败且无空结果时保守放行，避免误触发改写循环。"""
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "hybrid",
         "documents": [Document(page_content="doc")], "retrieval_scores": [0.03]},
        llm=BrokenEvaluateLLM(), prompts=PROMPTS,
    )
    assert out["failure_mode"] == "sufficient"
    assert out["suggested_action"] == "proceed"


# ── 结构化输出 ────────────────────────────────────────────────────


def test_schema_enum_normalization():
    assert AnalyzeResult(query_type="Factual", needs_retrieval=True).query_type == "factual"
    r = EvaluateResult(failure_mode="低召回", suggested_action="切换策略")
    assert r.failure_mode == "low_recall" and r.suggested_action == "reformulate"
    assert ReformulateResult(reformulated_query="x", rewrite_strategy="具体化").rewrite_strategy == "specify"


def test_extract_json_nested_and_noisy():
    extract = LLMClient._extract_json
    nested = '{"reason": "含 } 花括号", "is_faithful": true}'
    assert json.loads(extract(f"```json\n{nested}\n```"))["is_faithful"] is True
    assert json.loads(extract(f"前缀噪声 ```json\n{nested}\n``` 后缀"))["is_faithful"] is True
    assert json.loads(extract(nested))["is_faithful"] is True


# ── 状态机端到端（Fake 组件跑真实图）──────────────────────────────


def test_graph_normal_single_round(make_graph):
    state = make_graph().invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["iteration_count"] == 1, f"正常单轮应只检索 1 次: {state['iteration_count']}"


def test_graph_legacy_switch_action_is_reformulated(make_graph):
    """旧模型输出 switch_strategy 时，schema 归一化为 reformulate，主链路不会出现 switch 分支。"""
    script = make_script(
        AnalyzeResult=AnalyzeResult(query_type="complex", needs_retrieval=True),
        EvaluateResult=EvaluateResult(failure_mode="low_recall", suggested_action="切换策略"),
    )
    state = make_graph(script).invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["suggested_action"] == "reformulate"
    assert "switch_strategy" not in {m.split(":")[0] for m in state.get("messages", [])}


def test_graph_hallucination_loop_effective_rewrite(make_graph):
    """verify 首轮判幻觉、改写有效 → 重检索一次后结束。"""
    script = make_script(VerifyResult=scripted_verify([False, True]))
    state = make_graph(script).invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["is_faithful"] is True
    assert state["iteration_count"] == 2, f"幻觉循环应重检索 1 次: {state['iteration_count']}"


def test_graph_ineffective_rewrite_short_circuit(make_graph):
    """verify 恒幻觉 + 改写无效 → verify↔generate 循环由 verify_failures 兜底终止，
    不应产生第二次检索。"""
    script = make_script(
        VerifyResult=VerifyResult(is_faithful=False),
        ReformulateResult=lambda **kw: ReformulateResult(
            reformulated_query=kw.get("query", "测试问题"), rewrite_strategy="specify"
        ),
    )
    state = make_graph(script).invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["iteration_count"] == 1, f"改写无效应短路: iter={state['iteration_count']}"
    assert state["verify_failures"] == 3, f"幻觉重试上限应为 3: {state['verify_failures']}"


def test_graph_worst_path_terminates(make_graph):
    """最坏路径：恒幻觉 + 恒 reformulate → 双预算强制收敛，无死循环。"""
    script = make_script(
        AnalyzeResult=AnalyzeResult(query_type="factual", needs_retrieval=True),
        EvaluateResult=EvaluateResult(failure_mode="low_recall", suggested_action="reformulate"),
        VerifyResult=VerifyResult(is_faithful=False),
    )
    state = make_graph(script).invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["iteration_count"] <= 3, f"超出检索上限: {state['iteration_count']}"


def test_graph_generate_fallback(make_graph):
    """LLM 生成彻底失败：generate 节点降级兜底，状态机仍产出答案。"""
    script = make_script()

    class BrokenGenerateLLM(FakeLLM):
        def generate(self, query, context):
            raise RuntimeError("LLM 宕机")

    # 直接替换图中 generate 节点不可行，改为构造带故障 generate 的图
    broken = build_agent_graph(
        vector_store=FakeRetriever(),
        bm25_retriever=FakeRetriever(),
        llm_client=BrokenGenerateLLM(script),
        prompt_manager=PromptManager(),
        hybrid_retriever=None,
        reranker=None,
    )
    state = broken.invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"], "降级兜底应产出非空回答"


def test_graph_mermaid_visualization(make_graph):
    """mermaid 文本可生成且包含 7 主链路 + clarify 终止分支。"""
    mermaid = make_graph().get_graph().draw_mermaid()
    for node in ["analyze", "clarify", "decide", "retrieve", "evaluate",
                 "reformulate", "generate", "verify"]:
        assert node in mermaid, f"mermaid 缺少节点 {node}"
    assert "switch_strategy" not in mermaid
