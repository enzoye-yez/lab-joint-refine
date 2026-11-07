"""Joint refinement loop: scene -> trajectories -> diagnosis -> feedback -> scene repair -> ...

  .venv-less: run with np002_traj's interpreter (it has mujoco and numpy):
  /home/coder/3d_asset_gen/np002_traj/.venv/bin/python -m joint_refine.orchestrate --exp joint_01 \
      --source-run /home/coder/3d_asset_gen/np002_agent_v3/runs/np002_sol_full_12_np002 --max-rounds 4

Each step records its start/end in ledger.jsonl and its outputs under round_<k>/; a finished step
is skipped on restart, so the command can simply be run again after an interruption.
"""
import argparse
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from . import diagnose, feedback, judge, quota, repair, trajectory
from .common import CORES, ROOT, SCENE_GEN, TRAJ, Ledger, git_identity, now, read, write

MIN_GAIN = 0.01        # stop when a round fixes fewer than 1% of the tasks ...
STOP_STATUSES = ("deferred", "no_change")


class Experiment:
    def __init__(self, name):
        self.dir = ROOT / "experiments" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.ledger = Ledger(self.dir)
        self.state_path = self.dir / "state.json"
        self.state = read(self.state_path, {"rounds": {}})

    def save(self):
        write(self.state_path, self.state)

    def log(self, text):
        line = f"[{now()}] {text}"
        print(line, flush=True)
        with (self.dir / "orchestrate.log").open("a") as f:
            f.write(line + "\n")

    def step(self, k, name, fn):
        steps = self.state["rounds"].setdefault(str(k), {}).setdefault("steps", {})
        if steps.get(name, {}).get("status") == "done":
            return steps[name].get("value")
        self.ledger.event(round=k, step=name, status="started")
        self.log(f"round {k}: {name} started")
        started = time.time()
        try:
            value = fn()
        except Exception as exc:
            self.ledger.event(round=k, step=name, status="failed", error=str(exc)[:2000], seconds=round(time.time() - started))
            steps[name] = {"status": "failed", "error": str(exc)[:2000], "trace": traceback.format_exc()[-4000:]}
            self.save()
            raise
        steps[name] = {"status": "done", "seconds": round(time.time() - started), "value": value}
        self.ledger.event(round=k, step=name, status="done", seconds=round(time.time() - started), value=value)
        self.log(f"round {k}: {name} done in {time.time() - started:.0f} s")
        self.save()
        return value


def manifest(exp, args):
    write(exp.dir / "manifest.json", {
        "experiment": exp.name, "created": now(), "source_run": args.source_run, "max_rounds": args.max_rounds,
        "trajectory_code": git_identity(TRAJ), "generator_code": git_identity(SCENE_GEN),
        "orchestrator_code": git_identity(ROOT), "llm": {"provider": "codex (ChatGPT login)", "model": "gpt-6.1-sol"},
        "concurrency": {"benchmark_workers": trajectory.WORKERS, "benchmark_cpus": trajectory.BENCH_CPUS,
                        "generator_parallel_authors": 8, "generator_cpus": repair.REPAIR_CPUS},
        "budgets_per_repair": {"max_calls": repair.MAX_CALLS, "max_tokens": repair.MAX_TOKENS},
        "machine": {"cores": CORES, "host": platform.node(), "python": sys.version.split()[0]}})


def quota_ok(exp, k):
    """Model steps run only while at least quota.STOP_BELOW percent of the Codex quota is left."""
    if quota.STOP_FILE.exists():
        exp.log(f"round {k}: {quota.STOP_FILE} exists (quota stop); not starting model steps")
        return False
    try:
        percent = quota.require()
    except quota.QuotaLow as exc:
        exp.ledger.event(round=k, step="quota", status="stopped", error=str(exc))
        exp.log(str(exc))
        return False
    exp.ledger.event(round=k, step="quota", status="ok", remaining_percent=percent)
    return True


def run(args):
    exp = Experiment(args.exp)
    if not (exp.dir / "manifest.json").exists():
        manifest(exp, args)
    source_run = exp.state.setdefault("source_runs", {}).setdefault("0", args.source_run)
    exp.save()
    k = 0
    while True:
        rdir = exp.dir / f"round_{k}"
        source_run = exp.state["source_runs"][str(k)]
        artifact, scene_sha = repair.source_scene(source_run)
        exp.step(k, "scene", lambda: {"run": source_run, "artifact": str(artifact), "scene_sha256": scene_sha})
        exp.step(k, "prepare", lambda: trajectory.prepare(artifact, rdir / "scene", exp.log))
        bench = exp.step(k, "benchmark", lambda: trajectory.benchmark(artifact, f"{exp.name}_r{k}", rdir / "trajectory", exp.log))
        cache = exp.state["rounds"][str(k)]["steps"]["prepare"]["value"]["cache"]

        def _diagnose():
            findings = diagnose.diagnose(rdir, diagnose.Measure(cache, artifact))
            sent = [f for f in findings if f["attribution"] != "method"]
            return {"findings": len(findings), "scene_findings": len(sent),
                    "blocked_scene_tasks": len({t for f in sent for t in f["blocked_tasks"]})}
        diag = exp.step(k, "diagnose", _diagnose)
        if args.until == "diagnose":
            break
        if k > 0:
            verdict = exp.step(k, "judge", lambda: judge.judge(exp.dir / f"round_{k-1}", rdir))
            gain = verdict["transitions"].get("fixed", 0) - verdict["transitions"].get("regressed", 0)
            if gain < MIN_GAIN * bench["tasks"] and not verdict["new_findings"]:
                exp.log(f"round {k}: net gain {gain} < {MIN_GAIN:.0%} of tasks and no new findings; converged")
                break
        if k >= args.max_rounds:
            exp.log(f"reached {args.max_rounds} feedback rounds")
            break
        if not diag["scene_findings"]:
            exp.log("no scene findings left to send")
            break
        if not quota_ok(exp, k):
            break
        exp.step(k, "feedback", lambda: {"failures": len(feedback.write_feedback(rdir, artifact, scene_sha)["failures"])})
        if args.until == "feedback":
            break
        if not quota_ok(exp, k):
            break
        result = exp.step(k, "scene_repair", lambda: repair.repair(rdir, source_run, f"{exp.name}_r{k + 1}", exp.log))
        if result.get("status") in STOP_STATUSES:
            exp.log(f"round {k}: scene repair ended {result.get('status')}: {result.get('reason')}")
            break
        exp.state["source_runs"][str(k + 1)] = result["run"]
        exp.save()
        k += 1
    exp.ledger.event(step="experiment", status="finished", rounds=k)
    exp.log("experiment finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--source-run", required=True, help="scene generator run of round 0")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--until", choices=["diagnose", "feedback"], help="stop round 0 after this step (dry runs)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
