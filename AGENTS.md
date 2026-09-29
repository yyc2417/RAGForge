# RAGForge — Agent Guidelines

## 验证命令

- 运行测试：`uv run pytest`
- 代码检查：`uv run ruff check src/ tests/ scripts/`

## 模块结构

- `src/agent` — LangGraph 8 节点状态机（analyze → decide → retrieve → evaluate → generate → verify + reformulate / switch_strategy）
- `src/api` — FastAPI 服务，SSE 流式输出
- `src/generation` — LLMClient（DeepSeek / OpenAI 兼容）
- `src/ingestion` — 文档解析 / 分块 / 嵌入
- `src/retrieval` — 混合检索（VectorStore + BM25 + RRF + Reranker）
- `src/ui` — Streamlit 可视化演示（可选依赖 `--extra ui`）
- `src/utils` — 日志 / 指标收集 / token 预算
- `src/pipeline.py` / `src/cli.py` / `src/main.py` — 管道组装 / 启动模式 / 精简入口
- `src/config.py` — 唯一配置入口（pydantic-settings），禁止散落 `os.getenv`

## 关键约束

- 敏感字段使用 `SecretStr`
- `chunk_overlap` 必须 < `chunk_size`
- 依赖方向：api → pipeline → agent → retrieval / generation → ingestion / utils
