# Linux 运行环境；由 run.sh、test.sh 和批次脚本 source。
# 渲染固定走 Mesa 软件 EGL（llvmpipe，CPU），并隐藏所有 GPU。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "缺少项目环境，请先按照 README.md 创建 .venv。" >&2
    exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none
