#!/usr/bin/env bash
# 补跑评测报告中出现错误的模式（默认云端）
#
# 只重跑包含 error 的模式，--force 覆盖该模式结果；最多 2 轮。
# 用法：
#   bash scripts/repair_eval_errors.sh
#
# 依赖 .env.cloud（云端补跑）；如要补跑本地，可先 export 本地 LLM 环境再执行。
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env.cloud ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.cloud
  set +a
fi

export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"

if [[ ! -x "$PY" ]]; then
  echo "找不到 $PY" >&2
  exit 1
fi

report_errored_modes() {
  local report="$1"
  "$PY" - "$report" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    sys.exit(0)
for mode in data.get("modes", []):
    if mode.get("errors", 0) > 0:
        print(mode["mode"])
PY
}

repair_dataset() {
  local dataset="$1"
  local corpus="$2"
  local report="$3"

  if [[ ! -f "$report" ]]; then
    echo "跳过（报告不存在）：$report"
    return
  fi

  local attempt
  for attempt in 1 2; do
    local modes
    modes="$(report_errored_modes "$report" | tr '\n' ' ')"
    echo "[repair] $report attempt=$attempt errored_modes=${modes:-none}"
    [[ -z "${modes// /}" ]] && return

    local mode
    for mode in $modes; do
      echo "[repair] rerun mode=$mode"
      "$PY" scripts/eval.py \
        --dataset "$dataset" \
        --corpus-dir "$corpus" \
        --mode "$mode" \
        --force --resume \
        --report "$report"
      sleep 10
    done
  done

  local remaining
  remaining="$(report_errored_modes "$report" | tr '\n' ' ')"
  if [[ -n "${remaining// /}" ]]; then
    echo "[repair] 仍有错误模式：$remaining" >&2
    return 1
  fi
  echo "[repair] $report 已无错误模式"
}

repair_dataset \
  "data/public_eval/crud_read_eval.json" \
  "data/public_eval/crud_corpus" \
  "reports/cloud_e2e_crud_full.json"

repair_dataset \
  "tests/eval_dataset_multiformat.json" \
  "data/demo" \
  "reports/cloud_e2e_multiformat_full.json"
