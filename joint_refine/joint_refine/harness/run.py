"""Repair-skill harness: failed task types of an eval -> counterfactual scene edits -> verdicts -> requirements.

For every failing task type, one representative failed task is taken. The first skill whose trigger
matches proposes up to MAX_VARIANTS edits; the task is rerun on the unchanged scene (control) and on each
edited scene through np002_traj's own task code (scripts/run_counterfactual.py). Verdicts:

  scene_verified  a realisable edit (move/turn an instance, joint parameters) makes the task succeed
  scene_located   only a locating probe (a blocker's collision switched off) makes it succeed: the
                  blocker is known and the requirement says how to clear it
  charged         a prerequisite/unlock failure whose blocker has its own failing type: fixed there
  improved        no edit succeeds, but one moves the failure to a later stage (access -> planning ->
                  execution); recorded, not sent
  unresolved      skill ran, nothing helped (the trajectory side, or a skill still missing)
  no_skill        no skill matches the failure (input for the self-improvement step)
  flaky           the control succeeds: the original failure does not reproduce

  python -m joint_refine.harness.run --exp joint_01 --eval S6_T6 [--types-limit N] [--workers 48]
"""
import argparse
import collections
import json
import os
import time
from pathlib import Path

from ..common import BASE, ROOT, guarded, read, wait_for_load, write
from ..judge import stage
from ..triage import _reason, task_type
from .scene import SceneView
from .skills import SKILLS, skill_for

# The trajectory checkout the counterfactuals run in (the harness branch until it is merged).
TRAJ = Path(os.environ.get("HARNESS_TRAJ", BASE / "np002_traj"))
PYTHON = TRAJ / ".venv/bin/python"
CPUS, MEM, WORKERS, TASK_WALL = 64, "256G", 48, 900


def failing_types(results):
    groups = collections.defaultdict(list)
    for r in results:
        groups[task_type(r["task"], r.get("instance", ""), r.get("skill", ""))].append(r)
    out = {}
    for key, rows in sorted(groups.items()):
        if any(r["success"] for r in rows):
            continue
        rep = sorted(rows, key=lambda r: r["task"])[0]
        out[key] = {"representative": rep, "tasks": [r["task"] for r in rows]}
    return out


def blocker_type(result, by_task):
    """The type of the opener a prerequisite/unlock failure names, when that opener has its own result."""
    step = (result.get("detail") or {}).get("step") or {}
    opener = by_task.get(step.get("interface"))
    return None if opener is None else task_type(opener["task"], opener.get("instance", ""), opener.get("skill", ""))


def plan(scene, types, by_task, failing):
    jobs, plans = [], {}
    for n, (key, entry) in enumerate(types.items()):
        r = entry["representative"]
        skill = skill_for(r)
        row = {"type": key, "task": r["task"], "reason": _reason(r), "tasks": len(entry["tasks"]),
               "skill": skill.name if skill else None, "variants": []}
        charged = blocker_type(r, by_task) if row["reason"] in ("prerequisite_failed", "unlock_failed") else None
        if charged and charged in failing and charged != key:
            row["charged_to"] = charged
        if skill is not None and not row.get("charged_to"):     # a charged type is fixed through its blocker
            try:
                row["variants"] = skill.variants(scene, r)
            except Exception as exc:                       # a skill bug must not stop the batch
                row["skill_error"] = f"{type(exc).__name__}: {exc}"
        if row["variants"]:
            jobs.append({"id": f"{n}:control", "edits": [], "tasks": [r["task"]], "meta": {"type": key}})
            for v in row["variants"]:
                jobs.append({"id": f"{n}:{v['label']}", "edits": v["edits"], "tasks": [r["task"]],
                             "meta": {"type": key, "label": v["label"]}})
        row["index"] = n
        plans[key] = row
    return jobs, plans


def run_jobs(jobs, run_name, scene_source, out, workers, cpus=CPUS):
    path = out / "jobs.json"
    write(path, jobs)
    wait_for_load()
    code, seconds = guarded([PYTHON, "scripts/run_counterfactual.py", "--jobs", path, "--run", run_name,
                             "--workers", workers, "--task-wall", TASK_WALL], cpus=cpus, mem=MEM, cwd=TRAJ,
                            log_path=out / "counterfactual.log",
                            env={"PYTHONPATH": "src", "NP002_SCENE_SOURCE": str(scene_source)})
    if code:
        raise RuntimeError(f"run_counterfactual failed ({code}); see {out / 'counterfactual.log'}")
    rows = [json.loads(l) for l in (TRAJ / "runs" / run_name / "jobs.jsonl").open()]
    return {r["id"]: r for r in rows}, seconds


def decide(row, done, original):
    n = row["index"]
    control = done.get(f"{n}:control")
    tried = []
    for v in row["variants"]:
        job = done.get(f"{n}:{v['label']}")
        result = (job or {}).get("results", [None])[0] if job and job.get("results") else None
        tried.append({"label": v["label"], "realisable": v["realisable"], "error": (job or {}).get("error"),
                      "success": bool(result and result["success"]),
                      "reason": None if result is None else _reason(result) if not result["success"] else None,
                      "stage": None if result is None else stage(result),
                      "settle_moving": len(((job or {}).get("perturb") or {}).get("settle", {}).get("moving", []))})
    row["tried"] = tried
    if row.get("charged_to"):
        row["verdict"] = "charged"
    elif row["skill"] is None:
        row["verdict"] = "no_skill"
    elif not row["variants"]:
        row["verdict"] = "no_variant"
    elif control and control.get("results") and control["results"][0]["success"]:
        row["verdict"] = "flaky"
    else:
        winners = [t for t in tried if t["success"]]
        real = [t for t in winners if t["realisable"]]
        pick = (real or winners or [None])[0]
        if pick is not None:
            row["verdict"] = "scene_verified" if pick["realisable"] else "scene_located"
            row["winner"] = next(v for v in row["variants"] if v["label"] == pick["label"])
        elif any(t["stage"] is not None and t["stage"] > stage(original) for t in tried):
            row["verdict"] = "improved"
            row["improved_by"] = [t["label"] for t in tried if t["stage"] is not None and t["stage"] > stage(original)]
        else:
            row["verdict"] = "unresolved"
    return row


def requirements(plans):
    """Feedback rows (planning_feedback shape plus the verified edit) for verified or located types."""
    items = []
    for key, row in plans.items():
        if row["verdict"] not in ("scene_verified", "scene_located"):
            continue
        w = row["winner"]
        items.append({"kind": w["kind"], "instances": [w["target"]["instance"]], "detail": w["requirement"],
                      "evidence": {"type": key, "task": row["task"], "failure": row["reason"], "tasks_of_type": row["tasks"],
                                   "verified": row["verdict"] == "scene_verified", "skill": row["skill"],
                                   "edit": w["edits"], "target": w["target"]}})
    return items


def harness(exp, eval_key, workers=WORKERS, types_limit=0, run_name=None, results_path=None, out=None, cpus=CPUS):
    exp_dir = ROOT / "experiments" / exp
    state = read(exp_dir / "cycles.json")
    ev = state["evals"][eval_key]["steps"]
    cache = ev["prepare"]["value"]["cache"]
    source = ev["prepare"]["value"]["artifact"]
    out = Path(out) if out else exp_dir / "evals" / eval_key / "harness"
    out.mkdir(parents=True, exist_ok=True)
    results = [json.loads(l) for l in Path(results_path or exp_dir / "evals" / eval_key / "trajectory/results.jsonl").open()]
    by_task = {r["task"]: r for r in results}
    types = failing_types(results)
    failing = set(types)
    if types_limit:
        types = dict(list(types.items())[:types_limit])
    scene = SceneView(cache, source)
    jobs, plans = plan(scene, types, by_task, failing)
    started = time.time()
    done, seconds = run_jobs(jobs, run_name or f"harness_{exp}_{eval_key}", source, out, workers, cpus) if jobs else ({}, 0)
    for row in plans.values():
        decide(row, done, by_task[row["task"]])
    items = requirements(plans)
    summary = {"eval": eval_key, "types_total": len({task_type(r["task"], r.get("instance", ""), r.get("skill", "")) for r in results}),
               "types_failing": len(failing), "types_examined": len(plans), "jobs": len(jobs), "wall_s": round(seconds),
               "verdicts": dict(collections.Counter(r["verdict"] for r in plans.values())),
               "by_skill": {s.name: dict(collections.Counter(r["verdict"] for r in plans.values() if r["skill"] == s.name))
                            for s in SKILLS},
               "requirements": len(items), "seconds": round(time.time() - started)}
    write(out / "plans.json", plans)
    write(out / "requirements.json", items)
    write(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--types-limit", type=int, default=0)
    parser.add_argument("--run")
    parser.add_argument("--results", help="results.jsonl to read instead of the eval's (e.g. a partial run)")
    parser.add_argument("--out", help="output directory instead of evals/<eval>/harness")
    parser.add_argument("--cpus", type=int, default=CPUS)
    args = parser.parse_args()
    print(json.dumps(harness(args.exp, args.eval, args.workers, args.types_limit, args.run, args.results, args.out,
                             args.cpus), indent=1))


if __name__ == "__main__":
    main()
