"""Apply robot-trajectory feedback to a static scene run as one targeted repair round.

    python scripts/apply_robot_feedback.py --source-run runs/<R_k> --run <R_k+1> --feedback feedback.json
        [--max-calls 160 --max-tokens 80000000] [--resume]

The repair runs on frozen source (src/runtime_bootstrap.py) like `run.sh scene`; see
src/scene_generation/external_repair.py. Writes runs/<R_k+1>/robot_feedback_result.json.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PATHS = {"--source-run", "--feedback", "--config-dir", "--package", "--protocol-pdf"}

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
                           "--module", "scene_generation.external_repair", "--", *arguments])
