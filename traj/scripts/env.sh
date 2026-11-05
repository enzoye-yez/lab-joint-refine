# Linux environment for scripts and tests: CPU (Mesa llvmpipe) EGL rendering, no GPUs.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
