"""难例候选池与晋升工具测试（离线）"""

import json

from scripts.promote_hard_cases import promote
from src.knowledge.feedback import HardCaseStore


def _candidate(message_id="11111111-1111-1111-1111-111111111111") -> dict:
    return {
        "message_id": message_id,
        "session_id": "s1",
        "query": "远程办公每周几天？",
        "resolved_query": "星云科技远程办公每周最多几天？",
        "answer": "每周最多两天。",
        "sources": [{"source": "handbook.md", "score": 0.9}],
        "trace": ["retrieve: ..."],
    }


def test_hard_case_store_add_dedup_and_status(tmp_path):
    store = HardCaseStore(tmp_path / "hard_cases.jsonl")
    assert store.add(_candidate()) is True
    assert store.add(_candidate()) is False
    items = store.list_cases()
    assert len(items) == 1
    assert items[0]["status"] == "candidate"
    assert store.update_status(items[0]["message_id"], "triaged") is True
    assert store.list_cases(status="triaged")[0]["message_id"] == items[0]["message_id"]
    assert store.update_status("missing", "promoted") is False


def test_hard_case_store_skips_bad_lines(tmp_path):
    path = tmp_path / "hard_cases.jsonl"
    path.write_text('{"message_id": "x"}\nnot-json\n', encoding="utf-8")
    store = HardCaseStore(path)
    assert len(store.list_cases()) == 1


def test_promote_answerable_and_dedup(tmp_path):
    dataset = tmp_path / "eval_dataset_hard.json"
    item = promote(
        _candidate(),
        answerable=True,
        expected_sources=["handbook.md"],
        gold_keywords=["远程办公", "两天"],
        dataset_path=dataset,
    )
    assert item["expected_behavior"] == "answer"
    assert item["expected_sources"] == ["handbook.md"]
    data = json.loads(dataset.read_text(encoding="utf-8"))
    assert len(data) == 1
    # 相同 query 拒绝重复晋升
    try:
        promote(
            _candidate("22222222-2222-2222-2222-222222222222"),
            answerable=True,
            expected_sources=["handbook.md"],
            gold_keywords=["远程办公"],
            dataset_path=dataset,
        )
        raise AssertionError("重复 query 应被拒绝")
    except ValueError as e:
        assert "已存在" in str(e)


def test_promote_unanswerable_and_validation(tmp_path):
    dataset = tmp_path / "eval_dataset_hard.json"
    item = promote(
        _candidate(),
        answerable=False,
        expected_sources=[],
        gold_keywords=[],
        dataset_path=dataset,
    )
    assert item["expected_behavior"] == "refuse"

    try:
        promote(
            _candidate("33333333-3333-3333-3333-333333333333"),
            answerable=True,
            expected_sources=[],
            gold_keywords=["x"],
            dataset_path=tmp_path / "other.json",
        )
        raise AssertionError("可答缺 source 应被拒绝")
    except ValueError as e:
        assert "expected_source" in str(e)
