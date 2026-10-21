"""难例候选池半自动晋升工具

用法：
    uv run python scripts/promote_hard_cases.py --list
    uv run python scripts/promote_hard_cases.py --message-id <id> \
        --answerable --expected-source handbook.md --gold-keywords "远程办公,两天"
    uv run python scripts/promote_hard_cases.py --message-id <id> --unanswerable

规则：
- 默认写入 tests/eval_dataset_hard.json，不污染冻结的 v1 45 题；
- 可答必须提供 expected_source + gold_keywords；
- 不可答不能带 expected_source；
- 写入后把候选状态改为 promoted。
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.feedback import HardCaseStore  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "eval_dataset_hard.json"


def _normalize_query(query: str) -> str:
    return " ".join(str(query).split()).lower()


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"数据集必须是 JSON 数组：{path}")
    return data


def promote(
    candidate: dict,
    *,
    answerable: bool,
    expected_sources: list[str],
    gold_keywords: list[str],
    reference_answer: str = "",
    difficulty: str = "L2",
    dataset_path: Path = DEFAULT_DATASET,
) -> dict:
    query = str(candidate.get("resolved_query") or candidate.get("query") or "").strip()
    if not query:
        raise ValueError("候选缺少 query")
    if answerable and not expected_sources:
        raise ValueError("可答样本必须提供 expected_source")
    if answerable and not gold_keywords:
        raise ValueError("可答样本必须提供 gold_keywords")
    if not answerable and expected_sources:
        raise ValueError("不可答样本不能带 expected_source")

    data = load_dataset(dataset_path)
    normalized = _normalize_query(query)
    if any(_normalize_query(item.get("query", "")) == normalized for item in data):
        raise ValueError("数据集中已存在相同 query")

    item = {
        "id": f"hard-{candidate.get('message_id', '')[:8]}",
        "query": query,
        "query_type": "factual",
        "difficulty": difficulty,
        "answerable": answerable,
        "expected_behavior": "answer" if answerable else "refuse",
        "expected_sources": expected_sources,
        "expected_source": expected_sources[0] if expected_sources else None,
        "gold_keywords": gold_keywords,
        "reference_answer": reference_answer or None,
        "split": "test",
        "tags": ["hard_case", "user_feedback"],
        "origin": candidate.get("message_id"),
    }
    data.append(item)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="难例候选池查看 / 晋升")
    parser.add_argument("--list", action="store_true", help="列出候选")
    parser.add_argument("--message-id", help="要晋升的候选 message_id")
    parser.add_argument("--answerable", action="store_true")
    parser.add_argument("--unanswerable", action="store_true")
    parser.add_argument("--expected-source", action="append", default=[], help="可重复")
    parser.add_argument("--gold-keywords", default="", help="逗号分隔")
    parser.add_argument("--reference-answer", default="")
    parser.add_argument("--difficulty", choices=["L1", "L2", "L3", "L4"], default="L2")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    args = parser.parse_args()

    store = HardCaseStore()
    if args.list:
        items = store.list_cases()
        if not items:
            print("暂无难例候选")
            return
        for item in items:
            print(
                f"{item.get('message_id')} | {item.get('status')} | "
                f"{item.get('created_at')} | {item.get('query', '')[:50]}"
            )
        return

    if not args.message_id:
        parser.error("需要 --message-id 或 --list")
    if args.answerable == args.unanswerable:
        parser.error("必须二选一：--answerable 或 --unanswerable")

    candidate = store.get(args.message_id)
    if candidate is None:
        print(f"候选不存在：{args.message_id}")
        sys.exit(1)

    keywords = [k.strip() for k in args.gold_keywords.split(",") if k.strip()]
    try:
        item = promote(
            candidate,
            answerable=args.answerable,
            expected_sources=args.expected_source,
            gold_keywords=keywords,
            reference_answer=args.reference_answer,
            difficulty=args.difficulty,
            dataset_path=Path(args.dataset),
        )
    except ValueError as e:
        print(f"晋升失败：{e}")
        sys.exit(1)

    store.update_status(args.message_id, "promoted")
    print(f"已晋升：{item['id']} → {args.dataset}")
    print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
