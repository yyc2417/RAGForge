"""评测 v2 schema 与校验测试（离线）"""

import json
from pathlib import Path

from src.evaluation import EvalItem, dataset_hash, validate_dataset


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
    assert len(multiturn) >= 5
    assert all(len(conv["turns"]) >= 2 for conv in multiturn)
    assert all("resolved_query" in turn for conv in multiturn for turn in conv["turns"])
    assert len(clarify) >= 8
    assert all(item["expected_behavior"] == "clarify" for item in clarify)


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


def test_dataset_hash_stable():
    items = [EvalItem(query="q", expected_source="a.md", gold_keywords=["a"])]
    assert dataset_hash(items) == dataset_hash(items)
    assert len(dataset_hash(items)) == 16
