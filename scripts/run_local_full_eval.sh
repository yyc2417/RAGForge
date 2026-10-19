#!/usr/bin/env bash
# 本地模型全量端到端评测（不调用云端 API）
#
# 前置：
# - 本地 OpenAI 兼容服务监听 127.0.0.1:1919
# - CUDA 版 torch 已安装，且使用 .venv（裸 uv run 会按 lock 恢复 CPU torch）
# - bge-m3 已下载到 data/models/bge-m3，.env 指向该目录
#
# 用法：
#   bash scripts/run_local_full_eval.sh all
#   bash scripts/run_local_full_eval.sh crud
#   bash scripts/run_local_full_eval.sh multiformat
#
# 这是长任务：CRUD-RAG 120 题 × 4 模式；建议后台运行并观察报告。
set -euo pipefail

cd "$(dirname "$0")/.."

export DEEPSEEK_BASE_URL=http://127.0.0.1:1919/v1
export DEEPSEEK_MODEL=Qwen3.6-35B-A3B-NVFP4
export DEEPSEEK_API_KEY=local
export LLM_TIMEOUT=180
export API_REQUEST_TIMEOUT=300
export PYTHONIOENCODING=utf-8

PY=".venv/Scripts/python.exe"
TARGET="${1:-all}"

if [[ ! -x "$PY" ]]; then
  echo "找不到 $PY；请先创建项目虚拟环境并安装 CUDA torch" >&2
  exit 1
fi

run_crud() {
  "$PY" scripts/eval.py \
    --dataset data/public_eval/crud_read_eval.json \
    --corpus-dir data/public_eval/crud_corpus \
    --rebuild --resume \
    --report reports/local_e2e_crud_full.json
}

run_multiformat() {
  "$PY" scripts/eval.py \
    --dataset tests/eval_dataset_multiformat.json \
    --corpus-dir data/demo \
    --rebuild --resume \
    --report reports/local_e2e_multiformat_full.json
}

case "$TARGET" in
  crud) run_crud ;;
  multiformat) run_multiformat ;;
  all)
    run_crud
    run_multiformat
    ;;
  *)
    echo "用法：$0 [all|crud|multiformat]" >&2
    exit 2
    ;;
esac
