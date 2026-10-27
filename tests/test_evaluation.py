"""评测 v2 schema 与校验测试（离线）"""

import json
import math
from pathlib import Path

from src.evaluation import EvalItem, dataset_hash, load_dataset, validate_dataset


def test_eval_item_normalizes_behavior_and_sources():
    item = EvalItem(
        query="远程办公几天？", expected_source="handbook.md", gold_keywords=["远程办公"]
    )
    assert item.expected_behavior == "answer"
    assert item.expected_sources == ["handbook.md"]
    assert item.answerable is True

    refuse = EvalItem(query="K8s 怎么滚动升级？", answerable=False)
    assert refuse.expected_behavior == "refuse"
    assert refuse.expected_sources == []

    chitchat = EvalItem(query="你好", query_type="chitchat")
    assert chitchat.expected_behavior == "chitchat"
    assert chitchat.answerable is False

    clarify = EvalItem(
        query="它怎么配置？",
        expected_behavior="clarify",
        clarification_keywords=["哪个", "产品"],
    )
    assert clarify.answerable is False
    assert clarify.expected_behavior == "clarify"
    assert clarify.clarification_keywords == ["哪个", "产品"]


def test_validate_dataset_catches_duplicates_and_missing_labels(tmp_path):
    items = [
        EvalItem(query="重复问题", expected_source="a.md", gold_keywords=["a"]),
        EvalItem(query="重复问题", expected_source="a.md", gold_keywords=["a"]),
        EvalItem(query="可答但缺来源", expected_behavior="answer"),
        EvalItem(query="不可答却带来源", answerable=False, expected_source="b.md"),
    ]
    errors = validate_dataset(items)
    text = "\n".join(errors)
    assert "重复" in text
    assert "缺少 expected_sources" in text
    assert "缺少 gold_keywords" in text
    assert "不应带 expected_sources" in text


def test_validate_dataset_requires_clarification_keywords():
    bad = EvalItem(query="它怎么用？", expected_behavior="clarify")
    errors = validate_dataset([bad])
    assert any("clarification_keywords" in error for error in errors)

    good = EvalItem(
        query="它怎么用？",
        expected_behavior="clarify",
        expected_source="a.md",
        clarification_keywords=["哪个产品"],
    )
    assert validate_dataset([good]) == []


def test_validate_dataset_allows_reference_answer_without_keywords():
    item = EvalItem(
        query="公开集问题",
        expected_source="doc.txt",
        reference_answer="参考回答",
        tags=["public_adapter"],
    )
    assert validate_dataset([item]) == []
    errors = validate_dataset([EvalItem(query="两者都缺", expected_source="doc.txt")])
    assert len(errors) == 1 and "缺少 gold_keywords 或 reference_answer" in errors[0]


def test_validate_dataset_checks_corpus_sources(tmp_path):
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    items = [
        EvalItem(query="q1", expected_source="a.md", gold_keywords=["a"]),
        EvalItem(query="q2", expected_source="missing.md", gold_keywords=["b"]),
    ]
    errors = validate_dataset(items, corpus_dir=tmp_path)
    assert any("missing.md" in e for e in errors)
    assert not any("a.md" in e for e in errors)


def test_multiturn_and_clarify_fixtures():
    root = Path(__file__).parent
    multiturn = json.loads((root / "eval_multiturn.json").read_text(encoding="utf-8"))
    clarify = json.loads((root / "eval_clarify.json").read_text(encoding="utf-8"))
    assert len(multiturn) >= 10
    assert all(len(conv["turns"]) >= 3 for conv in multiturn)
    assert all("resolved_query" in turn for conv in multiturn for turn in conv["turns"])
    assert len(clarify) >= 10
    behaviors = {item["expected_behavior"] for item in clarify}
    assert {"clarify", "answer"} <= behaviors
    assert all(
        item.get("clarification_keywords")
        for item in clarify
        if item["expected_behavior"] == "clarify"
    )


def test_eval_clarify_aggregates_clear_and_ambiguous():
    from types import SimpleNamespace

    from scripts.eval_interaction import eval_clarify
    from src.generation.schemas import AnalyzeResult
    from tests.fakes import FakeLLM

    llm = FakeLLM({
        "AnalyzeResult": AnalyzeResult(
            query_type="factual", needs_retrieval=True, needs_clarification=False
        )
    })
    result = eval_clarify(SimpleNamespace(llm=llm))
    assert result["samples"] == 16
    assert result["clarify_cases"] == 11
    assert result["clear_cases"] == 5
    assert result["clarify_accuracy"] == round(5 / 16, 4)
    assert result["over_clarification_rate"] == 0.0
    assert result["followup_resolution_rate"] is None
    assert len(result["details"]) == 16


def test_crud_rag_read_import(tmp_path):
    from scripts.import_crud_rag import build_dataset, extract_news

    data = {
        "questanswer_1doc": [
            {"questions": "q1", "answers": "a1", "news1": "共享文档"},
        ],
        "questanswer_2docs": [
            {"questions": "q2", "answers": "a2", "news1": "共享文档", "news2": "文档二"},
        ],
        "questanswer_3docs": [
            {
                "questions": "q3",
                "answers": "a3",
                "news1": "文档一",
                "news2": "文档二",
                "news3": "文档三",
            }
        ],
    }
    items, corpus = build_dataset(
        data,
        quotas={
            "questanswer_1doc": 1,
            "questanswer_2docs": 1,
            "questanswer_3docs": 1,
        },
    )
    assert len(items) == 3
    assert len(corpus) == 4  # 共享文档按内容去重
    assert extract_news(data["questanswer_3docs"][0]) == ["文档一", "文档二", "文档三"]
    assert items[0].reference_answer == "a1"
    assert items[0].id.startswith("crud-questanswer_1doc-")
    assert items[0].query_type == "factual"
    assert items[1].query_type == "complex"
    assert [len(item.expected_sources) for item in items] == [1, 2, 3]

    for name, text in corpus.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert validate_dataset(items, corpus_dir=tmp_path) == []


def test_frozen_dataset_hashes_are_stable():
    root = Path(__file__).parent.parent
    assert dataset_hash(load_dataset(root / "tests" / "eval_dataset.json")) == (
        "a07fba4c991ad03d"
    )
    assert dataset_hash(
        load_dataset(root / "tests" / "eval_dataset_multiformat.json")
    ) == "d59643ccfb6943a2"


def test_dataset_hash_ignores_empty_clarification_keywords():
    base = EvalItem(query="q", expected_source="a.md", gold_keywords=["a"])
    h1 = dataset_hash([base])
    assert dataset_hash([base.model_copy(update={"clarification_keywords": []})]) == h1
    assert dataset_hash(
        [base.model_copy(update={"clarification_keywords": ["哪个产品"]})]
    ) != h1


def test_multiformat_fixture_covers_demo_sources():
    root = Path(__file__).parent.parent
    items = load_dataset(root / "tests" / "eval_dataset_multiformat.json")
    assert len(items) >= 40
    assert validate_dataset(items, corpus_dir=root / "data" / "demo") == []
    sources = {Path(source).name for item in items for source in item.expected_sources}
    assert sources == {
        "company_handbook.md",
        "team_roles.md",
        "travel_guide.txt",
        "city_guide.csv",
        "faq.jsonl",
        "product_faq.html",
        "product_manual.pdf",
        "project_brief.docx",
    }
    assert sum("multi-hop" in item.tags for item in items) >= 4
    assert sum("cross-lingual" in item.tags for item in items) >= 8
    clarify = [item for item in items if item.expected_behavior == "clarify"]
    assert len(clarify) == 8
    assert all(item.answerable is False for item in clarify)
    assert all(item.clarification_keywords for item in clarify)


def test_ndcg_at_k_uses_unique_documents():
    from langchain_core.documents import Document

    from scripts.eval_v2 import _ndcg_at_k

    docs = [
        Document(page_content="x", metadata={"source": "a.md"}),
        Document(page_content="x", metadata={"source": "a.md"}),
        Document(page_content="x", metadata={"source": "b.md"}),
        Document(page_content="x", metadata={"source": "c.md"}),
    ]
    assert _ndcg_at_k(docs, ["a.md"], 5) == 1.0
    assert abs(_ndcg_at_k(docs, ["b.md"], 5) - 1 / math.log2(3)) < 1e-12
    assert _ndcg_at_k(docs, ["z.md"], 5) == 0.0


def test_retrieval_gate_thresholds():
    from scripts.eval_v2 import _gate_failures

    modes = [{"mode": "hybrid", "hit_any": 0.9, "mrr_any": 0.75}]
    assert _gate_failures(
        modes, mode="hybrid", min_hit_any=0.85, min_mrr=0.7
    ) == []
    failures = _gate_failures(
        modes, mode="hybrid", min_hit_any=0.95, min_mrr=0.8
    )
    assert len(failures) == 2
    assert _gate_failures(
        modes, mode="missing", min_hit_any=0.1, min_mrr=None
    ) == ["没有找到模式 missing"]


def test_eval_script_accepts_v1_and_v2_sources():
    from langchain_core.documents import Document

    from scripts.eval import _chunk_hit_rank, _expected_sources

    assert _expected_sources({"expected_source": "a.md"}) == ["a.md"]
    assert _expected_sources(
        {"expected_source": "a.md", "expected_sources": ["a.md", "b.md"]}
    ) == ["a.md", "b.md"]

    docs = [
        Document(page_content="alpha", metadata={"source": "a.md"}),
        Document(page_content="beta", metadata={"source": "b.md"}),
    ]
    assert _chunk_hit_rank(docs, ["b.md"], []) == 2
    assert _chunk_hit_rank(docs, ["b.md"], ["needle"]) == 0
    assert _chunk_hit_rank(docs, ["b.md"], ["beta"]) == 2


def test_answer_f1_lexical_metric():
    from scripts.eval import _answer_f1

    assert _answer_f1("USB-C charging takes 90 minutes", "USB-C charging takes 90 minutes") == 1.0
    assert _answer_f1("foo bar", "foo baz") == 0.5
    assert _answer_f1("", "参考回答") == 0.0
    assert _answer_f1("回答", None) is None


def _eval_result(**overrides) -> dict:
    """构造 _summarize_results 的最小结果行。"""
    base = {
        "query": "q",
        "query_type": "factual",
        "expected_behavior": "answer",
        "expected_sources": ["a.md"],
        "chunk_rank": 1,
        "hit": True,
        "clarified": False,
        "generation_failed": False,
        "empty_answer": False,
        "answered": True,
        "is_faithful": True,
        "answer_f1": 1.0,
        "answer_verdict": "correct",
        "clarification_keyword_ok": None,
        "refusal_verdict": None,
        "error": None,
        "e2e_ms": 100.0,
    }
    base.update(overrides)
    return base


def test_eval_summary_separates_clarify_degraded_and_empty():
    from scripts.eval import _summarize_results

    results = [
        _eval_result(),
        _eval_result(
            chunk_rank=0, hit=False, clarified=True, empty_answer=True,
            answered=False, is_faithful=None, answer_f1=None, answer_verdict=None,
        ),
        _eval_result(
            generation_failed=True, answered=False, is_faithful=None,
            answer_f1=None, answer_verdict=None, answer="降级文案",
        ),
        _eval_result(
            chunk_rank=3, is_faithful=False, answer_f1=0.0, answer_verdict="wrong",
        ),
        _eval_result(
            expected_sources=[], expected_behavior="refuse", query_type="factual",
            chunk_rank=0, hit=False, answered=False, is_faithful=None,
            answer_f1=None, answer_verdict=None, refusal_verdict="refused",
        ),
        _eval_result(error="boom", answered=False, is_faithful=None,
                     answer_f1=None, answer_verdict=None),
    ]
    summary = _summarize_results(results)
    assert summary["samples"] == 6
    assert summary["errors"] == 1
    assert summary["answerable_count"] == 4
    assert summary["retrieval_evaluable_count"] == 3
    assert summary["answered_count"] == 2
    assert summary["answer_rate"] == 0.5
    assert summary["clarified_count"] == 1
    assert summary["clarify_rate"] == 0.2
    assert summary["expected_clarify_count"] == 0
    assert summary["clarify_accuracy"] is None
    assert summary["over_clarification_count"] == 1
    assert summary["over_clarification_rate"] == 0.25
    assert summary["degraded_count"] == 1
    # clarify 的空回答归 clarified_count，不重复计入 empty_answer_count
    assert summary["empty_answer_count"] == 0
    assert summary["recall_top5"] == 1.0
    assert summary["mrr_at5"] == round((1 + 1 + 1 / 3) / 3, 4)
    assert summary["faithfulness_rate"] == 0.5
    assert summary["hallucination_rate"] == 0.5
    assert summary["answer_f1"] == 0.5
    assert summary["semantic_accuracy"] == 0.5
    assert summary["semantic_counts"] == {"correct": 1, "wrong": 1}
    assert summary["refusal_accuracy"] == 1.0


def test_evaluate_mode_integrates_layered_summary():
    from scripts.eval import Evaluator

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.dataset = [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]
    evaluator.semantic_judge = False
    evaluator.judge_limit = None
    results = {
        "q1": _eval_result(query="q1"),
        "q2": _eval_result(
            query="q2", chunk_rank=0, hit=False, clarified=True,
            empty_answer=True, answered=False, is_faithful=None,
            answer_f1=None, answer_verdict=None,
        ),
        "q3": _eval_result(
            query="q3", generation_failed=True, answered=False,
            is_faithful=None, answer_f1=None, answer_verdict=None,
        ),
    }
    evaluator._run_single = lambda mode, item: results[item["query"]]
    out = evaluator.evaluate_mode("agent")
    assert out["samples"] == 3
    assert out["answerable_count"] == 3
    assert out["retrieval_evaluable_count"] == 2
    assert out["answered_count"] == 1
    assert out["clarified_count"] == 1
    assert out["degraded_count"] == 1
    assert out["recall_top5"] == 1.0
    assert out["faithfulness_rate"] == 1.0
    assert out["over_clarification_rate"] == round(1 / 3, 4)


def test_eval_summary_scores_expected_clarify():
    from scripts.eval import _summarize_results

    summary = _summarize_results([
        _eval_result(
            expected_behavior="clarify", expected_sources=[],
            clarified=True, clarification_keyword_ok=True, empty_answer=True,
            answered=False, is_faithful=None, answer_f1=None,
            answer_verdict=None, chunk_rank=0, hit=False,
        ),
        _eval_result(
            expected_behavior="clarify", expected_sources=[],
            answered=False, is_faithful=None, answer_f1=None,
            answer_verdict=None, chunk_rank=0, hit=False,
        ),
        _eval_result(
            expected_behavior="clarify", expected_sources=[],
            clarified=True, clarification_keyword_ok=False, empty_answer=True,
            answered=False, is_faithful=None, answer_f1=None,
            answer_verdict=None, chunk_rank=0, hit=False,
        ),
    ])
    assert summary["expected_clarify_count"] == 3
    assert summary["clarified_count"] == 2
    assert summary["clarify_accuracy"] == round(1 / 3, 4)
    assert summary["under_clarification_count"] == 2
    assert summary["under_clarification_rate"] == round(2 / 3, 4)
    assert summary["clarification_keyword_hit_rate"] == 0.5
    assert summary["answerable_count"] == 0
    assert summary["over_clarification_rate"] is None
    assert summary["recall_top5"] is None
    assert summary["faithfulness_rate"] is None


def test_eval_summary_all_clarify_has_no_hallucination_rate():
    from scripts.eval import _summarize_results

    summary = _summarize_results([
        _eval_result(
            chunk_rank=0, hit=False, clarified=True, empty_answer=True,
            answered=False, is_faithful=None, answer_f1=None, answer_verdict=None,
        )
    ])
    assert summary["recall_top5"] is None
    assert summary["faithfulness_rate"] is None
    assert summary["hallucination_rate"] is None
    assert summary["clarify_rate"] == 1.0
    assert summary["over_clarification_rate"] == 1.0


def test_answer_judge_schema_aliases_do_not_break_refusal_judge():
    from src.generation.schemas import AnswerJudgeResult, RefusalJudgeResult

    assert AnswerJudgeResult(verdict="部分正确").verdict == "partial"
    assert AnswerJudgeResult(verdict="refused").verdict == "unanswered"
    assert AnswerJudgeResult(verdict="未回答").verdict == "unanswered"
    # RefusalJudgeResult 仍必须接受 refused（不能被答案判分别名劫持）
    assert RefusalJudgeResult(verdict="refused").verdict == "refused"


def test_eval_run_single_marks_expected_clarify():
    from scripts.eval import Evaluator

    class ClarifyGraph:
        def invoke(self, state, config=None):
            return {
                "answer": "",
                "clarification_question": "你指的是星云笔记还是 Nebula Note？",
                "documents": [],
                "retrieval_scores": [],
                "messages": ["clarify: ok"],
                "is_faithful": None,
                "generation_failed": False,
            }

    evaluator = Evaluator.__new__(Evaluator)
    evaluator._pipelines = {}
    evaluator.semantic_judge = False
    evaluator.judge_limit = None
    evaluator._judge_calls = 0
    evaluator._get_pipeline = lambda mode: ClarifyGraph()
    result = evaluator._run_single("agent", {
        "query": "忘记密码后如何重置？",
        "expected_behavior": "clarify",
        "expected_sources": ["faq.jsonl"],
        "clarification_keywords": ["星云笔记", "Nebula Note"],
    })
    assert result["clarified"] is True
    assert result["clarification_keyword_ok"] is True
    assert result["chunk_rank"] == 0
    assert result["answered"] is False
    assert result["answer_f1"] is None
    assert result["faithfulness_source"] == "clarify_expected"


def test_eval_semantic_judge_uses_reference_and_keywords():
    from scripts.eval import Evaluator
    from src.generation.schemas import AnswerJudgeResult
    from tests.fakes import FakeLLM

    llm = FakeLLM({
        "AnswerJudgeResult": AnswerJudgeResult(verdict="correct", reason="一致")
    })
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._llm = llm
    verdict, reason, source = evaluator._judge_answer(
        "q", "回答", "参考答案", ["关键事实"]
    )
    assert (verdict, reason, source) == ("correct", "一致", "llm_judge")
    assert llm.structured_kwargs[0]["reference"] == "参考答案"
    assert "关键事实" in llm.structured_kwargs[0]["keywords"]


def test_eval_empty_answer_is_not_judged_faithful():
    from scripts.eval import Evaluator
    from tests.fakes import FakeLLM

    llm = FakeLLM({})
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._llm = llm
    assert evaluator._judge_faithfulness("q", "", [], ["kw"]) == (
        None,
        "empty_answer",
    )
    assert evaluator._keyword_check("", ["kw"]) is False
    assert llm.structured_calls == []


def test_dataset_hash_stable():
    items = [EvalItem(query="q", expected_source="a.md", gold_keywords=["a"])]
    assert dataset_hash(items) == dataset_hash(items)
    assert len(dataset_hash(items)) == 16
