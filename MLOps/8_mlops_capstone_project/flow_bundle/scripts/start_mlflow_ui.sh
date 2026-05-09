#!/usr/bin/env bash
set -euo pipefail

MlflowHost="${1:-127.0.0.1}"
Port="${2:-5000}"

if command -v mlflow >/dev/null 2>&1; then
  mlflow_cmd=mlflow
elif command -v python3 >/dev/null 2>&1; then
  mlflow_cmd="python3 -m mlflow"
else
  echo "ERROR: mlflow or python3 is not installed." >&2
  exit 1
fi

exec $mlflow_cmd ui --host "$MlflowHost" --port "$Port"
