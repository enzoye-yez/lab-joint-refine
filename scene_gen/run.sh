#!/usr/bin/env bash
# Linux 启动入口，与 run.ps1 子命令一致；所有本地产物限定在当前项目。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$PROJECT_ROOT"
bootstrap=("$PYTHON" -B -P "$PROJECT_ROOT/src/runtime_bootstrap.py" --module)
command="${1:-}"
case "$command" in
    prepare) shift; exec "${bootstrap[@]}" asset_prep.prepare_cli -- "$@" ;;
    operate) shift; exec "${bootstrap[@]}" robot_operation -- "$@" ;;
    preview) shift; exec "$PYTHON" -m scene_generation.preview "$@" ;;
    scene) shift; exec "${bootstrap[@]}" scene_generation -- "$@" ;;
    generate) shift; exec "$PYTHON" -m asset_generation "$@" ;;
    *) exec "$PYTHON" -m asset_prep "$@" ;;
esac
