"""交互评测：clarify 判定 + 多轮会话

需要真实 LLM（DeepSeek 或本地 OpenAI 兼容 endpoint）。它不是 CI 测试，
而是 Phase 4 的本地/手动评测入口。

用法：
    uv run python scripts/eval_interaction.py --mode clarify
    uv run python scripts/eval_interaction.py --mode multiturn --limit 5
    uv run python scripts/eval_interaction.py --mode both --limit 5
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK，中文 query / 澄清问题在打印 JSON 时可能编码失败
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent.nodes import analyze_node  # noqa: E402
from src.agent.state import initial_state  # noqa: E402
from src.config import settings  # noqa: E402
from src.conversation import ConversationService, ConversationStore  # noqa: E402
from src.generation.prompts import PromptManager  # noqa: E402
from src.pipeline import build_agent_runtime  # noqa: E402
from src.utils.logger import logger  # noqa: E402

CLARIFY_PATH = PROJECT_ROOT / "tests" / "eval_clarify.json"
MULTITURN_PATH = PROJECT_ROOT / "tests" / "eval_multiturn.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "eval_interaction_report.json"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _basename(source: str) -> str:
    return Path(str(source).replace("\\", "/")).name


def _source_hit(documents, expected: list[str]) -> bool:
    found = {
        _basename(getattr(doc, "metadata", {}).get("source", ""))
        for doc in documents
    }
    return all(Path(s).name in found for s in expected)


def eval_clarify(runtime, limit: int | None = None) -> dict:
    cases = json.loads(CLARIFY_PATH.read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]
    prompts = PromptManager()
    passed = keyword_hits = 0
    details = []
    for case in cases:
        out = analyze_node({"query": case["query"]}, llm=runtime.llm, prompts=prompts)
        question = str(out.get("clarification_question") or "")
        detected = bool(out.get("needs_clarification"))
        keyword_ok = any(k in question for k in case.get("clarification_keywords", []))
        ok = detected and keyword_ok
        passed += int(ok)
        keyword_hits += int(keyword_ok)
        details.append({
            "query": case["query"],
            "detected": detected,
            "question": question,
            "keyword_ok": keyword_ok,
            "passed": ok,
        })
    result = {
        "samples": len(cases),
        "clarify_accuracy": round(passed / len(cases), 4) if cases else 0.0,
        "keyword_hit_rate": round(keyword_hits / len(cases), 4) if cases else 0.0,
        "details": details,
    }
    return result


def eval_multiturn(runtime, limit: int | None = None) -> dict:
    conversations = json.loads(MULTITURN_PATH.read_text(encoding="utf-8"))
    if limit:
        conversations = conversations[:limit]
    store = ConversationStore(Path(tempfile.mkdtemp(prefix="ragforge-multiturn-")) / "sessions")
    service = ConversationService(store=store, llm=runtime.llm)
    turn_total = turn_hits = conversation_success = 0
    details = []
    for conv in conversations:
        session = service.start_session()
        hit_flags = []
        for turn in conv["turns"]:
            standalone, _ = service.prepare_query(session.session_id, turn["query"])
            service.add_user_message(session.session_id, turn["query"])
            state = runtime.graph.invoke(
                initial_state(standalone), config={"recursion_limit": 50}
            )
            service.record_assistant_state(
                session.session_id,
                state,
                original_query=turn["query"],
                resolved_query=standalone if standalone != turn["query"] else None,
            )
            hit = _source_hit(state.get("documents", []), turn.get("expected_sources", []))
            hit_flags.append(hit)
            turn_total += 1
            turn_hits += int(hit)
        success = all(hit_flags)
        conversation_success += int(success)
        details.append({
            "conversation_id": conv["conversation_id"],
            "turns": len(hit_flags),
            "turn_hits": sum(hit_flags),
            "success": success,
        })
    result = {
        "conversations": len(conversations),
        "turns": turn_total,
        "turn_hit_rate": round(turn_hits / turn_total, 4) if turn_total else 0.0,
        "conversation_success_rate": (
            round(conversation_success / len(conversations), 4) if conversations else 0.0
        ),
        "details": details,
    }
    return result


def _write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="clarify / 多轮交互评测")
    parser.add_argument("--mode", choices=["clarify", "multiturn", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    logger.info("[eval-interaction] 构建运行时（demo 语料）...")
    runtime = build_agent_runtime()
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "model": settings.deepseek_model,
        "base_url": settings.deepseek_base_url,
        "mode": args.mode,
    }
    if args.mode in ("clarify", "both"):
        report["clarify"] = eval_clarify(runtime, args.limit)
    if args.mode in ("multiturn", "both"):
        report["multiturn"] = eval_multiturn(runtime, args.limit)

    _write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"[eval-interaction] 报告已写入 {args.output}")


if __name__ == "__main__":
    main()
