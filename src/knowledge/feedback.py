"""点踩难例候选池（JSONL）

- 每个 message_id 只保留一条候选；
- 状态：candidate / triaged / promoted / dismissed；
- 不自动写入正式评测集；由 promote_hard_cases.py 人工晋升。
"""

import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, settings
from src.utils.logger import logger

STATUSES = {"candidate", "triaged", "promoted", "dismissed"}


class HardCaseStore:
    """难例候选池。"""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            self.path = PROJECT_ROOT / settings.feedback_dir / "hard_cases.jsonl"
        else:
            self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def add(self, record: dict[str, Any]) -> bool:
        """追加候选；message_id 重复返回 False。"""
        message_id = str(record.get("message_id") or "")
        if not message_id:
            return False
        with self._lock:
            if any(item.get("message_id") == message_id for item in self._read_all()):
                return False
            row = {
                "message_id": message_id,
                "session_id": record.get("session_id", ""),
                "query": record.get("query", ""),
                "resolved_query": record.get("resolved_query"),
                "answer": record.get("answer", ""),
                "sources": record.get("sources", []),
                "trace": record.get("trace", []),
                "model": record.get("model", settings.deepseek_model),
                "rating": "down",
                "status": "candidate",
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"[feedback] 难例候选 +1：{message_id}")
        return True

    def list_cases(self, status: str | None = None) -> list[dict[str, Any]]:
        items = self._read_all()
        if status is not None:
            items = [item for item in items if item.get("status") == status]
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    def get(self, message_id: str) -> dict[str, Any] | None:
        for item in self._read_all():
            if item.get("message_id") == message_id:
                return item
        return None

    def update_status(self, message_id: str, status: str) -> bool:
        if status not in STATUSES:
            return False
        with self._lock:
            items = self._read_all()
            found = False
            for item in items:
                if item.get("message_id") == message_id:
                    item["status"] = status
                    found = True
                    break
            if found:
                self._write_all(items)
        return found

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        items = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[feedback] 坏行跳过 {self.path.name}:{line_no}: {e}")
                continue
            if isinstance(obj, dict):
                items.append(obj)
        return items

    def _write_all(self, items: list[dict[str, Any]]) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".hard-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                for item in items:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
