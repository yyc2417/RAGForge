"""指标统计：延迟、token 消耗、命中率

MetricsCollector 为普通实例类（非全局单例）：
- API 场景：每个请求创建独立实例，经 ContextVar（set_current_collector）
  注入请求上下文，LangGraph 在 executor 线程中以 copy_context 执行同步节点，
  因此节点内的 get_current_collector() 能取到请求级实例，并发请求互不污染
- 评估/CLI 场景：单线程，直接使用 get_current_collector() 返回的进程级默认实例

record/reset/get_summary 均为实例级线程安全。
"""

import json
import statistics
import threading
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path


class MetricsCollector:
    """性能指标收集器（每请求/每评估实例化使用）。"""

    def __init__(self) -> None:
        self._data: defaultdict[str, list[float]] = defaultdict(list)
        self._token_totals: dict[str, int] = {"prompt": 0, "completion": 0, "calls": 0}
        self._lock = threading.Lock()

    # ── 记录 ───────────────────────────────────────────────────────
    def record_retrieval_latency(self, latency_ms: float) -> None:
        """记录单次检索延迟（毫秒）。"""
        with self._lock:
            self._data["retrieval_latency_ms"].append(float(latency_ms))

    def record_e2e_latency(self, latency_ms: float) -> None:
        """记录单次端到端延迟（毫秒）。"""
        with self._lock:
            self._data["e2e_latency_ms"].append(float(latency_ms))

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """记录单次 LLM 调用的 token 消耗。"""
        with self._lock:
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
        with self._lock:
            data_snapshot = {key: list(values) for key, values in self._data.items()}
            token = dict(self._token_totals)
        summary: dict = {}
        for key, values in data_snapshot.items():
            summary[key] = self._percentiles(values)
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
        """清空本实例的所有统计数据（用于新一轮评估）。"""
        with self._lock:
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


# ── 请求级注入（ContextVar）──────────────────────────────────────
_current: ContextVar["MetricsCollector | None"] = ContextVar("ragforge_metrics", default=None)
_default_collector = MetricsCollector()


def get_current_collector() -> "MetricsCollector":
    """获取当前上下文的指标收集器。

    - API 请求上下文：返回请求级实例（routes.py 经 set_current_collector 注入）
    - 其余场景（eval/CLI 单线程）：返回进程级默认实例
    """
    collector = _current.get()
    return collector if collector is not None else _default_collector


def set_current_collector(collector: "MetricsCollector"):
    """设置当前上下文的请求级收集器，返回 token 供 reset_current_collector 恢复。"""
    return _current.set(collector)


def reset_current_collector(token) -> None:
    """恢复 ContextVar 到 set_current_collector 之前的状态。"""
    _current.reset(token)
