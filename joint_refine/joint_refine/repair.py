"""Scene Repair Agent: hand the feedback to the scene generator (np002_agent_joint worktree) and
collect what it changed. The generator re-authors only the jobs its repair plan marks dirty."""
import json
import shutil
from pathlib import Path

from .common import SCENE_GEN, git_identity, guarded, read, wait_for_load, write

PYTHON = SCENE_GEN / ".venv/bin/python"
REPAIR_CPUS, REPAIR_MEM = 32, "128G"
MAX_CALLS, MAX_TOKENS = 240, 120_000_000      # 80M ran out in the static review of round 0


def source_scene(run_dir):
    """(artifact dir, scene_sha256) of a generator run's current scene."""
    run_dir = Path(run_dir)
    result = read(run_dir / "robot_feedback_result.json") or read(run_dir / "result.json")
    artifact = result.get("scene_artifact") or result.get("artifact")
    return (run_dir / artifact).resolve(), result["scene_sha256"]


def repair(round_dir, source_run, target_name, log):
    """Run apply_robot_feedback.py; returns the generator's result record (resumable)."""
    round_dir = Path(round_dir)
    out = round_dir / "scene_repair"
    target = SCENE_GEN / "runs" / target_name
    result_path = target / "robot_feedback_result.json"
    previous = (read(result_path) or {}).get("status")
    if previous not in ("candidate", "deferred", "no_change"):
        wait_for_load(log=log)
        command = [PYTHON, "scripts/apply_robot_feedback.py", "--source-run", source_run, "--run", target_name,
                   "--feedback", round_dir / "feedback/feedback.json",
                   "--max-calls", MAX_CALLS, "--max-tokens", MAX_TOKENS]
        if target.exists():
            command.append("--resume")
        if previous == "assembly_failed":
            # Rebuild with the current generator; the re-authored assets are kept.
            command.append("--reassemble")
        code, seconds = guarded(command, cpus=REPAIR_CPUS, mem=REPAIR_MEM, cwd=SCENE_GEN,
                                log_path=out / "repair.log", env={"PYTHONPATH": str(SCENE_GEN / "src")})
        if code and not result_path.exists():
            raise RuntimeError(f"apply_robot_feedback failed ({code}); see {out / 'repair.log'}")
    result = read(result_path)
    result["generator_code"] = git_identity(SCENE_GEN)
    result["run"] = str(target)
    write(out / "result.json", result)
    for name in ("repair_action.json", "changes.json"):
        if (target / name).exists():
            shutil.copy(target / name, out / name)
    return result
