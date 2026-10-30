#!/usr/bin/env bash
# 顺序场景实验入口（Linux）；默认仅准备清单，--execute 才调用模型。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
exec "$PYTHON" "$PROJECT_ROOT/scripts/run_scene_batch.py" "$@"
