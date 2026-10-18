"""评测 v2 schema 与校验测试（离线）"""

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


def test_validate_dataset_checks_corpus_sources(tmp_path):
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    items = [
        EvalItem(query="q1", expected_source="a.md", gold_keywords=["a"]),
        EvalItem(query="q2", expected_source="missing.md", gold_keywords=["b"]),
    ]
    errors = validate_dataset(items, corpus_dir=tmp_path)
    assert any("missing.md" in e for e in errors)
    assert not any("a.md" in e for e in errors)


def test_dataset_hash_stable():
    items = [EvalItem(query="q", expected_source="a.md", gold_keywords=["a"])]
    assert dataset_hash(items) == dataset_hash(items)
    assert len(dataset_hash(items)) == 16
