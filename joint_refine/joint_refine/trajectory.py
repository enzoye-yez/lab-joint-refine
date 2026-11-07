"""Scene Adapter + Trajectory Agent: assemble/settle a scene artifact with our robot, run the frozen
single-skill benchmark on every operable interface, classify failures and check the layout."""
import collections
import json
import shutil
from pathlib import Path

from .common import TRAJ, git_identity, guarded, read, sha256, wait_for_load, write

PYTHON = TRAJ / ".venv/bin/python"
BENCH_CPUS, BENCH_MEM, WORKERS = 64, "256G", 48


def _env(scene):
    return {"NP002_SCENE_SOURCE": str(scene), "PYTHONPATH": str(TRAJ / "src")}


def prepare(scene, out, log):
    """Assemble + settle (cached by scene digest); copy the settle report into the round."""
    scene = Path(scene)
    wait_for_load(log=log)
    code, seconds = guarded([PYTHON, "scripts/prepare_scene.py"], cpus=8, mem="64G", cwd=TRAJ,
                            log_path=out / "prepare.log", env=_env(scene))
    if code:
        raise RuntimeError(f"prepare_scene failed ({code}); see {out / 'prepare.log'}")
    lines = (out / "prepare.log").read_text().splitlines()
    cache = next(l.split()[-1] for l in reversed(lines) if "scene.mjb" in l or l.startswith("already prepared"))
    cache = Path(cache).parent if cache.endswith(".mjb") else Path(cache)
    for name in ("settle.json", "assembly.json"):
        if (cache / name).exists():
            shutil.copy(cache / name, out / name)
    record = {"artifact": str(scene), "model_sha256": sha256(scene / "model.xml"),
              "interfaces_sha256": sha256(scene / "interfaces.json"), "cache": str(cache), "prepare_seconds": seconds}
    write(out / "scene.json", record)
    return record


def benchmark(scene, run_name, out, log):
    """Full benchmark under the resource guard; results stay in np002_traj/runs/<run_name> and are copied."""
    target = TRAJ / "runs" / run_name
    if not (target / "results.jsonl").exists() or not _finished(target):
        if target.exists():
            shutil.rmtree(target)          # a half-written run is deterministic to redo
        wait_for_load(log=log)
        code, seconds = guarded([PYTHON, "scripts/run_benchmark.py", "--run", run_name, "--workers", WORKERS],
                                cpus=BENCH_CPUS, mem=BENCH_MEM, cwd=TRAJ, log_path=out / "benchmark.log", env=_env(scene))
        if code:
            raise RuntimeError(f"benchmark failed ({code}); guard: {read(target / 'guard.json')}")
        write(target / "wall.json", {"seconds": seconds})
    for script in ("classify_failures.py", "layout_feedback.py"):
        code, _ = guarded([PYTHON, f"scripts/{script}", target], cpus=8, mem="64G", cwd=TRAJ,
                          log_path=out / "analysis.log", env=_env(scene))
        if code:
            raise RuntimeError(f"{script} failed ({code})")
    for name in ("config.json", "results.jsonl", "issues.jsonl", "issues.md", "layout_feedback.json",
                 "layout_feedback.md", "wall.json", "guard.json"):
        if (target / name).exists():
            shutil.copy(target / name, out / name)
    summary = summarize(out)
    summary["code"] = git_identity(TRAJ)
    write(out / "summary.json", summary)
    return summary


def _finished(target):
    config = read(target / "config.json") or {}
    lines = sum(1 for _ in (target / "results.jsonl").open())
    return config.get("tasks") == lines


def summarize(out):
    results = [json.loads(l) for l in (out / "results.jsonl").open()]
    issues = {r["task"]: r for r in map(json.loads, (out / "issues.jsonl").open())} if (out / "issues.jsonl").exists() else {}
    config = read(out / "config.json", {})
    by = lambda key: {k: {"tasks": v[0], "success": v[1]} for k, v in sorted(_count(results, key).items())}
    categories = collections.Counter(i["category"] for i in issues.values())
    return {"tasks": len(results), "success": sum(r["success"] for r in results),
            "excluded": sum(config.get("excluded", {}).values()), "excluded_reasons": config.get("excluded", {}),
            "by_skill": by("skill"), "by_mechanism": by("mechanism"), "by_kind": by("kind"), "by_instance": by("instance"),
            "failure_categories": dict(categories.most_common()),
            "layout_violations": len((read(out / "layout_feedback.json") or {}).get("violations", [])),
            "wall_seconds": (read(out / "wall.json") or {}).get("seconds"),
            "peak_memory_gb": (read(out / "guard.json") or {}).get("peak_gb")}


def _count(results, key):
    counts = collections.defaultdict(lambda: [0, 0])
    for r in results:
        counts[str(r.get(key))][0] += 1
        counts[str(r.get(key))][1] += int(r["success"])
    return counts
