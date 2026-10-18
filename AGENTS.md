# RAGForge — Agent Guidelines

## 验证命令

- 运行测试：`uv run pytest`
- 代码检查：`uv run ruff check src/ tests/ scripts/`
- 检索回归门禁（离线，CI 同款）：
  `uv run python scripts/eval_v2.py --dataset tests/eval_dataset_multiformat.json --corpus-dir data/demo --rebuild --skip-reranker --report reports/ci_retrieval_report.json --gate-mode hybrid --min-hit-any 0.85 --min-mrr 0.60`

## 模块结构

- `src/agent` — LangGraph 7 节点主链路 + clarify 终止分支（固定 hybrid：analyze 分叉 clarify / decide → retrieve → evaluate → reformulate → generate → verify）；旧 8 节点（含 `switch_strategy`）为历史口径
- `src/api` — FastAPI 服务，SSE 流式输出
- `src/conversation` — 会话 JSON 存储、多轮 condense、会话反馈（图外）
- `src/generation` — LLMClient（DeepSeek / OpenAI 兼容）
- `src/ingestion` — 多格式文档解析（MD/TXT/PDF/HTML/DOCX/CSV/JSONL）/ 分块 / 嵌入
- `src/knowledge` — 运行时知识库（demo + library 全量重建、上传、文件夹导入、来源删除、优质库归档/加权/移除、难例池）
- `src/evaluation` — 评测 v2 schema / 校验 / dataset hash
- `src/retrieval` — 混合检索（VectorStore + BM25 + RRF + Reranker）
- `src/ui` — Streamlit 聊天式会话界面（可选依赖 `--extra ui`）
- `src/utils` — 日志 / 指标收集 / token 预算
- `src/pipeline.py` / `src/cli.py` / `src/main.py` — 管道组装 / 启动模式 / 精简入口
- `src/config.py` — 唯一配置入口（pydantic-settings），禁止散落 `os.getenv`

## 关键约束

- 敏感字段使用 `SecretStr`
- `chunk_overlap` 必须 < `chunk_size`
- 依赖方向：api / ui → pipeline → agent → retrieval / generation → ingestion / utils；pipeline → knowledge → ingestion / retrieval；ui → conversation → generation（conversation 不反向依赖 ui / agent）
