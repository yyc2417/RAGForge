"""把公开 RAG 数据集映射成评测 v2 schema

适用于 CRUD-RAG / RGB / MultiHop-RAG 等已经转换为 JSON/JSONL 的记录；
字段名不同时用参数指定。输出统一 EvalItem JSON 数组。

示例：
    uv run python scripts/adapt_public_eval.py \
        --input data/public_eval/crud_rag_read.jsonl \
        --output tests/eval_dataset_public.json \
        --query-field question --answer-field answer \
        --source-field doc_id --keywords-field keywords --split test

说明：
- 只做字段映射与 schema 校验，不下载数据、不做语义改写；
- 多跳数据可用 --sources-field 指向列表字段；
- 无法确定 expected_behavior 时用 --default-behavior 兜底。
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import EvalItem, validate_dataset  # noqa: E402


def _load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{line_no} 不是合法 JSON: {e}") from e
        return rows
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        raise ValueError("输入必须是 JSON 数组或 JSONL")
    return data


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="公开 RAG 数据集 → 评测 v2 schema")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "tests" / "eval_dataset_public.json"))
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--sources-field", default="sources")
    parser.add_argument("--keywords-field", default="gold_keywords")
    parser.add_argument("--behavior-field", default="expected_behavior")
    parser.add_argument("--difficulty-field", default="difficulty")
    parser.add_argument(
        "--default-behavior",
        choices=["answer", "refuse", "clarify", "chitchat"],
        default="answer",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"输入不存在：{input_path}")
        sys.exit(1)
    try:
        records = _load_records(input_path)
    except ValueError as e:
        print(f"加载失败：{e}")
        sys.exit(1)

    items = []
    errors = []
    for index, row in enumerate(records):
        query = str(row.get(args.query_field) or "").strip()
        if not query:
            errors.append(f"#{index}: 缺少 {args.query_field}")
            continue
        sources = _as_list(row.get(args.sources_field)) or _as_list(row.get(args.source_field))
        keywords = _as_list(row.get(args.keywords_field))
        behavior = row.get(args.behavior_field) or args.default_behavior
        try:
            item = EvalItem(
                id=str(row.get(args.id_field) or f"pub-{index:04d}"),
                query=query,
                difficulty=str(row.get(args.difficulty_field) or "L2"),
                expected_behavior=behavior,
                expected_sources=sources,
                gold_keywords=keywords,
                reference_answer=str(row.get(args.answer_field) or "") or None,
                split=args.split,
                tags=["public_adapter"],
            )
        except Exception as e:  # noqa: BLE001 - 逐条报告
            errors.append(f"#{index}: {e}")
            continue
        items.append(item)

    dataset_errors = validate_dataset(items)
    if dataset_errors:
        print("映射后校验失败：")
        for error in dataset_errors[:20]:
            print(f"- {error}")
        sys.exit(1)
    if errors:
        print(f"跳过 {len(errors)} 条：")
        for error in errors[:10]:
            print(f"- {error}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写入 {len(items)} 条 → {output}")


if __name__ == "__main__":
    main()
