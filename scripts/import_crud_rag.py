"""把 CRUD-RAG Read 任务转换为评测 v2 数据集与本地语料

输入是官方 data/crud_split/split_merged.json；只处理 questanswer_1doc /
questanswer_2docs / questanswer_3docs（Read 任务），生成：

- data/public_eval/crud_read_eval.json  评测 v2 数据集
- data/public_eval/crud_corpus/*.txt    每条样本对应的新闻原文

公开语料与数据集都放在 data/public_eval/（gitignore），不提交仓库。
语料文件名按内容 hash 生成，重复文档自动去重；该数据集不使用
gold_keywords，校验时由 reference_answer 兜底。

用法：
    uv run python scripts/import_crud_rag.py --limit 120
    uv run python scripts/import_crud_rag.py --limit 90 --seed 7
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation import EvalItem, dataset_hash, validate_dataset  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data" / "public_eval" / "crud_split_merged.json"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "public_eval" / "crud_read_eval.json"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "public_eval" / "crud_corpus"

# 任务名 → 该任务应包含的新闻篇数
READ_TASKS = {
    "questanswer_1doc": 1,
    "questanswer_2docs": 2,
    "questanswer_3docs": 3,
}
_NEWS_RE = re.compile(r"^news(\d+)$")


def extract_news(record: dict) -> list[str]:
    """按 news1/news2/... 的数字顺序提取非空新闻正文。"""
    pairs: list[tuple[int, str]] = []
    for key, value in record.items():
        match = _NEWS_RE.match(key)
        if match and isinstance(value, str) and value.strip():
            pairs.append((int(match.group(1)), value.strip()))
    pairs.sort(key=lambda pair: pair[0])
    return [text for _, text in pairs]


def _doc_name(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"crud_{digest}.txt"


def build_dataset(
    data: dict, quotas: dict[str, int], seed: int = 42
) -> tuple[list[EvalItem], dict[str, str]]:
    """从官方 split 数据构建 EvalItem 列表与内容寻址语料。

    Args:
        data: split_merged.json 解析后的 dict。
        quotas: 每个 Read 任务的抽取题数，例如 {"questanswer_1doc": 40, ...}。
        seed: 抽样随机种子，保证可复现。

    Returns:
        (items, corpus)：corpus 为 文件名 → 正文。
    """
    items: list[EvalItem] = []
    corpus: dict[str, str] = {}
    seen_queries: set[str] = set()
    seen_ids: set[str] = set()

    for task, expected_docs in READ_TASKS.items():
        records = data.get(task) or []
        if not isinstance(records, list):
            continue
        limit = quotas.get(task, 0)
        if limit <= 0:
            continue
        indices = list(range(len(records)))
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(indices)
        picked = 0
        for index in indices:
            if picked >= limit:
                break
            record = records[index]
            if not isinstance(record, dict):
                continue
            query = str(record.get("questions") or "").strip()
            answer = str(record.get("answers") or "").strip()
            docs = extract_news(record)
            normalized = " ".join(query.split()).lower()
            if (
                not query
                or not answer
                or len(docs) != expected_docs
                or normalized in seen_queries
            ):
                continue

            sources = []
            for text in docs:
                name = _doc_name(text)
                corpus[name] = text
                if name not in sources:
                    sources.append(name)

            multi_doc = expected_docs > 1
            record_id = str(record.get("ID") or f"{picked:03d}")
            item_id = f"crud-{task}-{record_id}"
            if item_id in seen_ids:
                item_id = f"{item_id}-{picked:03d}"
            seen_ids.add(item_id)
            items.append(
                EvalItem(
                    id=item_id,
                    query=query,
                    query_type="complex" if multi_doc else "factual",
                    difficulty="L3" if multi_doc else "L2",
                    expected_behavior="answer",
                    expected_sources=sources,
                    reference_answer=answer,
                    split="test",
                    tags=["crud-rag", task],
                )
            )
            seen_queries.add(normalized)
            picked += 1

    return items, corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="CRUD-RAG Read → 评测 v2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"输入不存在：{args.input}")
        sys.exit(1)
    if args.limit < len(READ_TASKS):
        print(f"--limit 至少为 {len(READ_TASKS)}")
        sys.exit(1)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    tasks = list(READ_TASKS)
    base, remainder = divmod(args.limit, len(tasks))
    quotas = {
        task: base + (1 if index < remainder else 0)
        for index, task in enumerate(tasks)
    }

    items, corpus = build_dataset(data, quotas=quotas, seed=args.seed)
    if not items:
        print("没有可用样本，请检查输入文件是否为官方 split_merged.json")
        sys.exit(1)

    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    for name, text in corpus.items():
        path = args.corpus_dir / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")

    errors = validate_dataset(items, corpus_dir=args.corpus_dir)
    if errors:
        print("数据集校验失败：")
        for error in errors[:20]:
            print(f"- {error}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in items],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"已写入 {len(items)} 题 / {len(corpus)} 篇语料 → {args.output}；"
        f"dataset_hash={dataset_hash(items)}"
    )


if __name__ == "__main__":
    main()
