"""Alternating joint refinement: each cycle refines the trajectory method, then the scene.

  cycle c (1..N), starting from scene S_c and the previous method T_{c-1}:
    trajectory step   T_c = the np002_traj commit registered for the cycle (generic method fixes made
                      from the failures of eval S_c_T{c-1}); eval S_c_T_c; judge S_c_T{c-1} -> S_c_T_c
    scene step        diagnose S_c_T_c -> feedback -> scene repair -> S_{c+1}; eval S_{c+1}_T_c;
                      judge S_c_T_c -> S_{c+1}_T_c

  So every eval is the "after" side of exactly one judge, and each judge changes only one side:
  trajectory gains are measured on a fixed scene, scene gains with fixed trajectory code.

  python -m joint_refine.cycle --exp joint_01 --register-traj 1 --note "..."   # after committing T_1
  python -m joint_refine.cycle --exp joint_01 --cycle 1                         # runs the rest of cycle 1

Eval directories (evals/S<i>_T<j>/) hold scene/, trajectory/, diagnosis/, judge/ and, when a scene
step starts from them, feedback/ and scene_repair/. Steps are resumable through state.json.
"""
import argparse
import time
import traceback

from . import diagnose, feedback, judge, quota, repair, trajectory
from .common import ROOT, TRAJ, Ledger, git_identity, now, read, write

STOP_STATUSES = ("deferred", "no_change")


class Cycles:
    def __init__(self, name):
        self.dir = ROOT / "experiments" / name
        self.name = name
        self.ledger = Ledger(self.dir)
        self.state_path = self.dir / "cycles.json"
        self.state = read(self.state_path, {"scenes": {}, "trajs": {}, "evals": {}, "cycles": {}})

    def save(self):
        write(self.state_path, self.state)

    def log(self, text):
        line = f"[{now()}] {text}"
        print(line, flush=True)
        with (self.dir / "orchestrate.log").open("a") as f:
            f.write(line + "\n")

    def step(self, key, name, fn):
        steps = self.state["evals"].setdefault(key, {}).setdefault("steps", {})
        if steps.get(name, {}).get("status") == "done":
            return steps[name].get("value")
        self.ledger.event(eval=key, step=name, status="started")
        self.log(f"{key}: {name} started")
        started = time.time()
        try:
            value = fn()
        except Exception as exc:
            self.ledger.event(eval=key, step=name, status="failed", error=str(exc)[:2000], seconds=round(time.time() - started))
            steps[name] = {"status": "failed", "error": str(exc)[:2000], "trace": traceback.format_exc()[-4000:]}
            self.save()
            raise
        steps[name] = {"status": "done", "seconds": round(time.time() - started), "value": value}
        self.ledger.event(eval=key, step=name, status="done", seconds=round(time.time() - started), value=value)
        self.log(f"{key}: {name} done in {time.time() - started:.0f} s")
        self.save()
        return value

    def eval_dir(self, s, t):
        return self.dir / "evals" / f"S{s}_T{t}"

    def evaluate(self, s, t):
        """Assemble scene S_s, run trajectory code T_t on it and diagnose; returns the eval directory."""
        key, out = f"S{s}_T{t}", self.eval_dir(s, t)
        scene = self.state["scenes"][str(s)]
        code = git_identity(TRAJ)
        if code["commit"] != self.state["trajs"][str(t)]["commit"] or code["dirty"]:
            raise RuntimeError(f"np002_traj is at {code['commit'][:7]} (dirty={code['dirty']}), "
                               f"but {key} needs T{t} = {self.state['trajs'][str(t)]['commit'][:7]}")
        self.step(key, "prepare", lambda: trajectory.prepare(scene["artifact"], out / "scene", self.log))
        self.step(key, "benchmark", lambda: trajectory.benchmark(scene["artifact"], f"{self.name}_S{s}_T{t}",
                                                                 out / "trajectory", self.log))
        cache = self.state["evals"][key]["steps"]["prepare"]["value"]["cache"]

        def _diagnose():
            findings = diagnose.diagnose(out, diagnose.Measure(cache, scene["artifact"]))
            sent = [f for f in findings if f["attribution"] != "method"]
            return {"findings": len(findings), "scene_findings": len(sent),
                    "blocked_scene_tasks": len({t for f in sent for t in f["blocked_tasks"]})}
        self.step(key, "diagnose", _diagnose)
        return out

    def judge(self, before, after, label):
        b, a = self.eval_dir(*before), self.eval_dir(*after)
        return self.step(f"S{after[0]}_T{after[1]}", "judge", lambda: {"against": f"S{before[0]}_T{before[1]}",
                                                                       "effect_of": label, **judge.judge(b, a)})

    def quota_ok(self, where):
        if quota.STOP_FILE.exists():
            self.log(f"{where}: {quota.STOP_FILE} exists; not starting model steps")
            return False
        try:
            percent = quota.require()
        except quota.QuotaLow as exc:
            self.ledger.event(eval=where, step="quota", status="stopped", error=str(exc))
            self.log(str(exc))
            return False
        self.ledger.event(eval=where, step="quota", status="ok", remaining_percent=percent)
        return True


def attempted(cy, c):
    """Findings sent to the scene generator in earlier cycles (ids), for feedback.select()."""
    sent = set()
    for k in range(c):
        out = cy.eval_dir(k, k) if (cy.eval_dir(k, k) / "feedback").exists() else cy.eval_dir(k, max(k - 1, 0))
        selection = read(out / "feedback/selection.json", None)
        if selection:
            sent |= set(selection["sent"])
            continue
        message = read(out / "feedback/feedback.json", None)
        if message:                         # before selection.json existed: every finding on a named instance
            named = {i for f in message["failures"] for i in f["instances"]}
            sent |= {f["id"] for f in read(out / "diagnosis/findings.json")["findings"] if f["instance"] in named}
    return sent


def register_traj(cy, t, note):
    code = git_identity(TRAJ)
    if code["dirty"]:
        raise SystemExit("commit the trajectory changes first")
    cy.state["trajs"][str(t)] = {**code, "registered": now(), "note": note}
    cy.save()
    cy.ledger.event(step="register_traj", traj=t, commit=code["commit"], note=note)
    cy.log(f"T{t} = {code['commit'][:7]}: {note}")


def run_cycle(cy, c):
    """Trajectory step then scene step of cycle c; returns False when the experiment must stop."""
    if str(c) not in cy.state["trajs"]:
        raise SystemExit(f"register T{c} first (--register-traj {c})")
    record = cy.state["cycles"].setdefault(str(c), {"started": now()})
    cy.save()
    # Trajectory step: same scene, new method.
    cy.evaluate(c, c)
    record["trajectory_judge"] = cy.judge((c, c - 1), (c, c), "trajectory")
    cy.save()
    # Scene step: same method, repaired scene.
    out = cy.eval_dir(c, c)
    if not cy.state["evals"][f"S{c}_T{c}"]["steps"]["diagnose"]["value"]["scene_findings"]:
        cy.log(f"cycle {c}: no scene findings to send")
        return False
    if not cy.quota_ok(f"S{c}_T{c}"):
        return False
    scene = cy.state["scenes"][str(c)]
    cy.step(f"S{c}_T{c}", "feedback", lambda: {"failures": len(feedback.write_feedback(out, scene["artifact"], scene["sha256"], attempted=attempted(cy, c))["failures"])})
    if not cy.quota_ok(f"S{c}_T{c}"):
        return False
    result = cy.step(f"S{c}_T{c}", "scene_repair", lambda: repair.repair(out, scene["run"], f"{cy.name}_S{c + 1}", cy.log))
    if result.get("status") in STOP_STATUSES:
        cy.log(f"cycle {c}: scene repair ended {result.get('status')}: {result.get('reason')}")
        return False
    artifact, sha = repair.source_scene(result["run"])
    cy.state["scenes"][str(c + 1)] = {"run": result["run"], "artifact": str(artifact), "sha256": sha,
                                      "from": f"S{c}_T{c}", "repair_status": result.get("status")}
    cy.save()
    cy.evaluate(c + 1, c)
    record["scene_judge"] = cy.judge((c, c), (c + 1, c), "scene")
    record["finished"] = now()
    cy.save()
    cy.log(f"cycle {c} finished")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--register-traj", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--cycle", type=int)
    args = parser.parse_args()
    cy = Cycles(args.exp)
    if args.register_traj is not None:
        register_traj(cy, args.register_traj, args.note)
    if args.cycle is not None:
        ok = run_cycle(cy, args.cycle)
        cy.ledger.event(step="cycle", cycle=args.cycle, status="finished" if ok else "stopped")


if __name__ == "__main__":
    main()
