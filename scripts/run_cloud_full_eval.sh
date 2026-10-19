#!/usr/bin/env bash
# 云端 API 全量评测（本地检索 + 云端生成）
#
# 前置：
# - .env.cloud 配置云端 endpoint / key / model（已被 gitignore）
# - 本地 bge-m3 embedding 与 eval 索引可用
# - 本地全量报告已存在（默认要求 local_e2e_*_full.json）
#
# 用法：
#   bash scripts/run_cloud_full_eval.sh all
#   bash scripts/run_cloud_full_eval.sh crud
#   bash scripts/run_cloud_full_eval.sh multiformat
#
# 说明：云端生成是付费调用；脚本不会自动启动，需显式执行。
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env.cloud ]]; then
  echo "缺少 .env.cloud（云端 endpoint / key / model）" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env.cloud
set +a

export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"
TARGET="${1:-all}"
MODES=(baseline linear_hybrid agent agent_reranker)
REQUIRED_LOCAL=(
  reports/local_e2e_crud_full.json
  reports/local_e2e_multiformat_full.json
  reports/local_full_eval.done
)

if [[ ! -x "$PY" ]]; then
  echo "找不到 $PY；请先创建项目虚拟环境" >&2
  exit 1
fi

if [[ "$TARGET" == "all" ]]; then
  for f in "${REQUIRED_LOCAL[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "本地全量报告未完成：$f；请先跑完本地矩阵" >&2
      exit 1
    fi
  done
fi

run_dataset() {
  local dataset="$1"
  local corpus="$2"
  local report="$3"
  local first=1
  for mode in "${MODES[@]}"; do
    local args=(
      scripts/eval.py
      --dataset "$dataset"
      --corpus-dir "$corpus"
      --mode "$mode"
      --resume
      --report "$report"
    )
    if [[ "$first" == "1" ]]; then
      args+=(--rebuild)
      first=0
    fi
    "$PY" "${args[@]}"
  done
}

run_crud() {
  run_dataset "data/public_eval/crud_read_eval.json" \
    "data/public_eval/crud_corpus" \
    "reports/cloud_e2e_crud_full.json"
}

run_multiformat() {
  run_dataset "tests/eval_dataset_multiformat.json" \
    "data/demo" \
    "reports/cloud_e2e_multiformat_full.json"
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
