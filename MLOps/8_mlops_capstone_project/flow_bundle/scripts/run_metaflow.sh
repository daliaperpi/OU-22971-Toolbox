#!/usr/bin/env bash
set -euo pipefail

TrackingUri="${1:-http://127.0.0.1:5000}"
RefMonth="${2:-01}"
BatchMonth="${3:-04}"
RefYear="${4:-2020}"
BatchYear="${5:-2020}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/.." && pwd)"

if command -v python3 >/dev/null 2>&1; then
  python_cmd=python3
elif command -v python >/dev/null 2>&1; then
  python_cmd=python
else
  echo "ERROR: Python is not installed or not on PATH." >&2
  exit 1
fi

"$python_cmd" "$root/src/metaflow_capstone_flow.py" run \
  --tracking-uri "$TrackingUri" \
  --ref-month "$RefMonth" \
  --ref-year "$RefYear" \
  --batch-month "$BatchMonth" \
  --batch-year "$BatchYear"
