"""结构与管道构建测试（Task 1-4 传承，离线）

覆盖：
1. build_rag_pipeline() 返回正确类型（LLMClient 构造不发起网络请求）
2. MetricsCollector 实例独立性与统计有效性（不再依赖全局单例的执行顺序）
3. src/main.py 行数约束（保持精简入口）
"""

import time

from src.config import PROJECT_ROOT
from src.generation.llm_client import LLMClient
from src.main import build_rag_pipeline
from src.retrieval import VectorStore
from src.utils.metrics import MetricsCollector


def test_pipeline_build(chroma_dir):
    """管道构建：类型正确（索引构建在 tmp 目录，不污染真实 chroma_db）。"""
    vs, llm = build_rag_pipeline()
    assert isinstance(vs, VectorStore)
    assert isinstance(llm, LLMClient)
    assert vs.count() > 0, "索引应已构建"


def test_metrics_collector_isolated_and_valid():
    """每个实例独立统计；摘要包含延迟与 token 明细；reset 生效。"""
    m1, m2 = MetricsCollector(), MetricsCollector()
    m1.record_retrieval_latency(10.0)
    m1.record_e2e_latency(100.0)
    m1.record_token_usage(50, 25)

    summary = m1.get_summary()
    assert summary["retrieval_latency_ms"]["count"] == 1
    assert summary["e2e_latency_ms"]["count"] == 1
    assert summary["token_usage"]["calls"] == 1
    assert summary["token_usage"]["total"] == 75

    # 实例隔离：m2 不受 m1 影响
    assert m2.get_summary()["token_usage"]["calls"] == 0

    # reset 清空（空实例的摘要不含延迟键，count 视为 0）
    m1.reset()
    assert m1.get_summary().get("e2e_latency_ms", {}).get("count", 0) == 0


def test_metrics_export_report(tmp_path):
    m = MetricsCollector()
    m.record_e2e_latency(42.0)
    path = m.export_report(tmp_path / "metrics.json", label="unit")
    assert path.exists()
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["label"] == "unit"
    assert data["summary"]["e2e_latency_ms"]["count"] == 1


def test_main_line_count():
    main_path = PROJECT_ROOT / "src" / "main.py"
    n = len(main_path.read_text(encoding="utf-8").splitlines())
    assert n <= 90, f"src/main.py 应 ≤ 90 行，实际 {n}"
