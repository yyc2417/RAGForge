"""Agent 决策层单元测试——全 mock，离线

覆盖：
1. 纯规则节点：decide / switch_strategy
2. 路由矩阵：evaluate（hybrid 短路/达上限）/ reformulate（改写无效短路）/
   verify（None 不触发循环、双预算终止）
3. evaluate 节点分数语义：vector 绝对阈值、bm25/hybrid 不套用
4. 结构化输出：枚举归一化、JSON 提取（嵌套不截断）
5. 状态机端到端（Fake 组件跑真实图）：正常单轮、hybrid 空转短路、
   幻觉循环、改写无效短路、最坏路径终止性、mermaid 可视化
"""

import json

from langchain_core.documents import Document

from fakes import FakeLLM, FakeRetriever, make_script, scripted_verify
from src.agent.graph import (
    build_agent_graph,
    route_after_evaluate,
    route_after_reformulate,
    route_after_verify,
)
from src.agent.nodes import decide_node, evaluate_node, switch_strategy_node
from src.agent.state import initial_state
from src.generation.llm_client import LLMClient
from src.generation.schemas import (
    AnalyzeResult,
    EvaluateResult,
    ReformulateResult,
    VerifyResult,
)
from src.generation.prompts import PromptManager

PROMPTS = PromptManager()


# ── 纯规则节点 ────────────────────────────────────────────────────


def test_decide_rules():
    assert decide_node({"query_type": "chitchat"})["retrieval_strategy"] == "none"
    assert decide_node({"needs_retrieval": False})["retrieval_strategy"] == "none"
    assert decide_node({"query_type": "factual"})["retrieval_strategy"] == "vector"
    assert decide_node({"query_type": "reasoning"})["retrieval_strategy"] == "bm25"
    assert decide_node({"query_type": "complex"})["retrieval_strategy"] == "hybrid"
    assert decide_node({})["retrieval_strategy"] == "vector"  # 未知类型安全默认


def test_switch_strategy_upgrade_chain():
    assert switch_strategy_node({"retrieval_strategy": "vector"})["retrieval_strategy"] == "bm25"
    assert switch_strategy_node({"retrieval_strategy": "bm25"})["retrieval_strategy"] == "hybrid"
    assert switch_strategy_node({"retrieval_strategy": "hybrid"})["retrieval_strategy"] == "hybrid"


def test_initial_state_defaults():
    state = initial_state("q")
    assert state["iteration_count"] == 0
    assert state["verify_failures"] == 0
    assert state["max_iterations"] == 3


# ── 路由矩阵 ──────────────────────────────────────────────────────


def test_route_after_evaluate():
    # hybrid 上 switch 无效 → 短路直接 generate
    assert route_after_evaluate(
        {"suggested_action": "switch_strategy", "retrieval_strategy": "hybrid",
         "iteration_count": 1, "max_iterations": 3}
    ) == "generate"
    # vector 切换有效
    assert route_after_evaluate(
        {"suggested_action": "switch_strategy", "retrieval_strategy": "vector",
         "iteration_count": 1, "max_iterations": 3}
    ) == "switch_strategy"
    # proceed / 达上限
    assert route_after_evaluate({"suggested_action": "proceed"}) == "generate"
    assert route_after_evaluate(
        {"suggested_action": "reformulate", "iteration_count": 3, "max_iterations": 3}
    ) == "generate"


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


def test_evaluate_vector_low_score_triggers_switch():
    """vector 策略 + cosine 低分：即使 LLM 说 sufficient，规则层也覆盖为 switch。"""
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "vector",
         "documents": [Document(page_content="doc")], "retrieval_scores": [0.1, 0.12]},
        llm=SufficientLLM(), prompts=PROMPTS,
    )
    assert out["failure_mode"] == "low_recall"
    assert out["suggested_action"] == "switch_strategy"


def test_evaluate_hybrid_rrf_scores_not_flagged():
    """hybrid 策略 + RRF 低分（≈0.03）：不得套用 0.3 阈值误判（修复前必空转 3 轮的根因）。"""
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "hybrid",
         "documents": [Document(page_content="doc")], "retrieval_scores": [0.033, 0.028]},
        llm=SufficientLLM(), prompts=PROMPTS,
    )
    assert out["suggested_action"] == "proceed"


def test_evaluate_bm25_scores_not_flagged():
    """bm25 分数无上界：不参与绝对分数规则。"""
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "bm25",
         "documents": [Document(page_content="doc")], "retrieval_scores": [15.3, 9.2]},
        llm=SufficientLLM(), prompts=PROMPTS,
    )
    assert out["suggested_action"] == "proceed"


def test_evaluate_empty_docs_switch():
    out = evaluate_node(
        {"query": "q", "retrieval_strategy": "vector", "documents": [], "retrieval_scores": []},
        llm=SufficientLLM(), prompts=PROMPTS,
    )
    assert out["failure_mode"] == "low_recall" and out["suggested_action"] == "switch_strategy"


# ── 结构化输出 ────────────────────────────────────────────────────


def test_schema_enum_normalization():
    assert AnalyzeResult(query_type="Factual", needs_retrieval=True).query_type == "factual"
    r = EvaluateResult(failure_mode="低召回", suggested_action="切换策略")
    assert r.failure_mode == "low_recall" and r.suggested_action == "switch_strategy"
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


def test_graph_hybrid_switch_short_circuit(make_graph):
    """hybrid 策略 + evaluate 恒建议 switch：应短路生成，而非空转满 3 轮。"""
    script = make_script(
        AnalyzeResult=AnalyzeResult(query_type="complex", needs_retrieval=True),
        EvaluateResult=EvaluateResult(failure_mode="low_recall", suggested_action="switch_strategy"),
    )
    state = make_graph(script).invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["iteration_count"] == 1, f"hybrid 短路失败: iter={state['iteration_count']}"


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
    """最坏路径：恒幻觉 + 恒 switch（vector 起步）→ 双预算强制收敛，无死循环。"""
    script = make_script(
        AnalyzeResult=AnalyzeResult(query_type="factual", needs_retrieval=True),
        EvaluateResult=EvaluateResult(failure_mode="low_recall", suggested_action="switch_strategy"),
        VerifyResult=VerifyResult(is_faithful=False),
    )
    state = make_graph(script).invoke(initial_state("测试问题"), config={"recursion_limit": 50})
    assert state["answer"] == "测试回答"
    assert state["iteration_count"] <= 3, f"超出检索上限: {state['iteration_count']}"


def test_graph_generate_fallback(make_graph):
    """LLM 生成彻底失败：generate 节点降级兜底，状态机仍产出答案。"""
    script = make_script()
    graph = make_graph(script)

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
    """mermaid 文本可生成且包含全部 8 个节点（不再吞断言）。"""
    mermaid = make_graph().get_graph().draw_mermaid()
    for node in ["analyze", "decide", "retrieve", "evaluate",
                 "reformulate", "switch_strategy", "generate", "verify"]:
        assert node in mermaid, f"mermaid 缺少节点 {node}"
