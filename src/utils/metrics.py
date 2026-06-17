"""指标统计：延迟、token 消耗、命中率

MetricsCollector 为进程内单例，记录检索延迟、端到端延迟、token 消耗，
并通过 get_summary() 返回 P50/P95/P99 统计摘要。
export_report() 将统计摘要导出为带时间戳的 JSON 报告文件。
"""

import json
import statistics
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class MetricsCollector:
    """性能指标收集器（单例）。

    单例设计：一次评估/会话内全局共享同一份统计数据，
    reset() 可在测试间清空状态。线程安全（double-check locking）。
    """

    _instance: "MetricsCollector | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-check locking
                    instance = super().__new__(cls)
                    instance._data = defaultdict(list)
                    instance._token_totals = {"prompt": 0, "completion": 0, "calls": 0}
                    cls._instance = instance
        return cls._instance

    # ── 记录 ───────────────────────────────────────────────────────
    def record_retrieval_latency(self, latency_ms: float) -> None:
        """记录单次检索延迟（毫秒）。"""
        self._data["retrieval_latency_ms"].append(float(latency_ms))

    def record_e2e_latency(self, latency_ms: float) -> None:
        """记录单次端到端延迟（毫秒）。"""
        self._data["e2e_latency_ms"].append(float(latency_ms))

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """记录单次 LLM 调用的 token 消耗。"""
        self._token_totals["prompt"] += int(prompt_tokens)
        self._token_totals["completion"] += int(completion_tokens)
        self._token_totals["calls"] += 1

    # ── 摘要 ───────────────────────────────────────────────────────
    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        """计算 P50/P95/P99；样本不足时回退到均值/最大值。"""
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "max": 0.0, "count": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def _pct(p: float) -> float:
            # 简单线性插值百分位，样本极小时退化为端点值
            if n == 1:
                return sorted_vals[0]
            idx = p * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac

        return {
            "p50": round(_pct(0.50), 2),
            "p95": round(_pct(0.95), 2),
            "p99": round(_pct(0.99), 2),
            "avg": round(statistics.mean(values), 2),
            "max": round(max(values), 2),
            "count": n,
        }

    def get_summary(self) -> dict:
        """返回所有指标的统计摘要。"""
        summary: dict = {}
        for key, values in self._data.items():
            summary[key] = self._percentiles(values)
        token = self._token_totals
        summary["token_usage"] = {
            "total_prompt": token["prompt"],
            "total_completion": token["completion"],
            "total": token["prompt"] + token["completion"],
            "calls": token["calls"],
            "avg_per_call": round(
                (token["prompt"] + token["completion"]) / token["calls"], 2
            ) if token["calls"] else 0.0,
        }
        return summary

    def reset(self) -> None:
        """清空所有统计数据（用于新一轮评估）。"""
        self._data = defaultdict(list)
        self._token_totals = {"prompt": 0, "completion": 0, "calls": 0}

    def export_report(self, path: str | Path, label: str = "") -> Path:
        """导出统计摘要为 JSON 报告文件。

        Args:
            path: 输出文件路径（.json）
            label: 报告标签（如评估模式名 baseline/agent/hybrid），写入元数据

        Returns:
            实际写入的文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "label": label,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": self.get_summary(),
        }
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
