"""Apply simulation-verified edits to a static scene run as one deterministic repair round.

    python scripts/apply_directed_repair.py --source-run runs/<R_k> --run <R_k+1> --edits edits.json
        [--max-calls 160 --max-tokens 80000000] [--resume]

No model call plans the repair: layout edits are applied and re-arranged directly; joint-parameter
and geometry edits become requirements of their job, which is re-authored through the normal
generate/precheck/publish path. Runs on frozen source (src/runtime_bootstrap.py); see
src/scene_generation/directed_repair.py. Writes runs/<R_k+1>/directed_repair_result.json.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PATHS = {"--source-run", "--edits", "--config-dir", "--package", "--protocol-pdf"}

if __name__ == "__main__":
    python = PROJECT/".venv"/"bin"/"python"
    os.environ.update(PYTHONPATH=str(PROJECT/"src"), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                      MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none",
                      __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    # Paths are given relative to the caller's directory; the repair runs from the project root.
    arguments = sys.argv[1:]
    for index, value in enumerate(arguments[:-1]):
        if value in PATHS and Path(arguments[index+1]).exists():
            arguments[index+1] = str(Path(arguments[index+1]).resolve())
    os.chdir(PROJECT)
    os.execv(str(python), [str(python), "-B", "-P", str(PROJECT/"src"/"runtime_bootstrap.py"),
                           "--module", "scene_generation.directed_repair", "--", *arguments])
