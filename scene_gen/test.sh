#!/usr/bin/env bash
# Linux 全量离线测试，与 test.ps1 一致；通过后写 .cache/verification.json。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$PROJECT_ROOT"
export TMPDIR="$PROJECT_ROOT/.cache/test-temp" TEMP="$PROJECT_ROOT/.cache/test-temp" TMP="$PROJECT_ROOT/.cache/test-temp"
mkdir -p "$TMPDIR"
exec "$PYTHON" -B -P "$PROJECT_ROOT/src/runtime_bootstrap.py" --receipt "$PROJECT_ROOT/.cache/verification.json" \
    --module unittest -- discover -s "$PROJECT_ROOT/tests" -v "$@"
