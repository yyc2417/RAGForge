"""评测集校验

用法：
    uv run python scripts/validate_eval_dataset.py
    uv run python scripts/validate_eval_dataset.py --dataset tests/eval_dataset_hard.json
    uv run python scripts/validate_eval_dataset.py --corpus-dir data/demo

检查：schema、重复 query/id、answer/refuse 标签一致性、来源文件是否存在。
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import dataset_hash, load_dataset, validate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="评测集 schema / 标签 / 来源校验")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "tests" / "eval_dataset.json"))
    parser.add_argument("--corpus-dir", default=None, help="可选：检查 expected_sources 是否存在")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"数据集不存在：{path}")
        sys.exit(1)
    try:
        items = load_dataset(path)
    except Exception as e:  # noqa: BLE001 - CLI 给出可读错误
        print(f"加载失败：{e}")
        sys.exit(1)

    errors = validate_dataset(items, corpus_dir=args.corpus_dir)
    behavior = Counter(item.expected_behavior for item in items)
    split = Counter(item.split for item in items)
    print(f"数据集：{path}")
    print(f"样本数：{len(items)} | hash：{dataset_hash(items)}")
    print(f"behavior：{dict(behavior)} | split：{dict(split)}")
    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    print("校验通过")


if __name__ == "__main__":
    main()
