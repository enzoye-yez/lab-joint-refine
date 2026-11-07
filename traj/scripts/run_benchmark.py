"""Single-skill benchmark over every operable interface of the assembled room.

Every task starts from the same settled state with the robot at its home pose, runs the generic
skill for its interface type and mechanism, and is judged by the mechanism-derived goal plus
"no other instance moved" and the robot contact-force limit. No per-interface parameters exist.

  source scripts/env.sh && $PYTHON scripts/run_benchmark.py --run bench_01 [--limit 40] [--workers 32]
"""
import os

# One BLAS thread per worker, set before numpy loads (each worker is its own process).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import collections
import json
import multiprocessing as mp
import time
import traceback

import mujoco
import numpy as np

from np002_traj import guard
from np002_traj.executor.session import Abort, Session
from np002_traj.robot import ROOT, Robot
from np002_traj.scene.assemble import cache_for, config as scene_config
from np002_traj.scene.interfaces import load
from np002_traj.scene.settle import enable_sleep
from np002_traj.skills.dock import Kinematics
from np002_traj.skills.library import SKILLS, Context
from np002_traj.planning.base import OccupancyMap
from np002_traj.tasks.benchmark import task_for
from np002_traj.tasks.semantics import Semantics

CONFIG = scene_config()
STATE = {}


def scene_directory():
    directory = cache_for(CONFIG["scene_source"])
    if not (directory / "scene.mjb").exists():
        raise SystemExit(f"scene not prepared: run scripts/prepare_scene.py for {CONFIG['scene_source']}")
    return directory


def instance_roots(model, robot):
    """Top-level bodies of every instance except the robot: their poses must not change."""
    robot_bodies = set(robot.bodies)
    return [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and b not in robot_bodies]


def init():
    directory = scene_directory()
    model = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    enable_sleep(model)
    robot = Robot(model, prefix=CONFIG["robot_prefix"])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("assembled").id)
    mujoco.mj_forward(model, data)
    interfaces, _ = load(model, CONFIG["scene_source"], data)
    STATE.update(semantics=Semantics(model, CONFIG["scene_source"]), active=data.eq_active.copy())
    STATE.update(model=model, robot=robot, interfaces={i.id: i for i in interfaces}, kinematics=Kinematics(),
                 occupancy=OccupancyMap(model, data, robot), roots=instance_roots(model, robot), directory=directory)


MAX_PREREQUISITES = 2


def prerequisite(abort, model, payload):
    """The interface (and its task) that opens the movable part an enclosure abort names, if any."""
    if abort.reason != "enclosed" or not abort.detail.get("joint"):
        return None
    joint = model.joint(abort.detail["joint"]).id
    if abort.detail.get("movable") == "free":
        # A loose part lies in the way: pick it by its own grasp interface and put it down aside.
        for candidate in STATE["interfaces"].values():
            if candidate.kind == "grasp" and candidate.mechanism.joint == joint:
                task = make_task(candidate, model, payload)
                if not task.excluded and task.skill == "pick":
                    task.skill = "set_aside"
                    return candidate, task
        return None
    for candidate in STATE["interfaces"].values():
        if candidate.kind == "grasp" and candidate.mechanism.joint == joint:
            task = make_task(candidate, model, payload)
            if not task.excluded:
                return candidate, task
    return None


def make_task(interface, model, payload):
    return task_for(interface, model, payload, semantics=STATE["semantics"], active=STATE["active"])


def unlock(ctx, session, task, model, payload, result):
    """Release the locks the asset declares for this part, each with its own interface's skill first."""
    for lock in task.unlock:
        opener = STATE["interfaces"].get(lock["interface"])
        step = {"interface": lock["interface"], "for": lock["connection"], "unlock": True}
        opener_task = make_task(opener, model, payload) if opener else None
        if opener_task is None or opener_task.excluded or not opener_task.skill:
            step["failed"] = "release interface has no runnable task"
            result["prerequisites"].append(step)
            raise Abort("unlock_failed", step=step)
        opener_task.key = "pre:" + opener.id
        try:
            session.monitor.evaluate(opener_task.goal, opener_task.key)
            SKILLS[opener_task.skill](ctx, opener, opener_task)
        except Abort as inner:
            step["failed"] = inner.reason
            result["prerequisites"].append(step)
            raise Abort("unlock_failed", step=step)
        session.data.eq_active[lock["equality"]] = 0          # the release event of that interface
        result["prerequisites"].append(step)


WORKER_BYTES = guard.default_limits()["worker_bytes"]
TASKS_PER_WORKER = 20          # workers are recycled so a slow leak cannot accumulate


def run_task(task_id):
    try:
        guard.worker_check(WORKER_BYTES)
    except MemoryError as exc:
        return {"task": task_id, "skill": "", "kind": "", "mechanism": "", "instance": "", "reason": "worker_memory",
                "detail": {"error": str(exc)}, "success": False, "achieved": False, "disturbed": {}, "log": [],
                "prerequisites": [], "wall_s": 0.0}
    model, robot = STATE["model"], STATE["robot"]
    interface = STATE["interfaces"][task_id]
    payload = robot.profile["arms"]["left"]["payload_kg"]
    task = make_task(interface, model, payload)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("assembled").id)
    mujoco.mj_forward(model, data)
    session = Session(model, data, robot)
    session.monitor.evaluate(task.goal, "goal")                      # anchor relative goals at the start
    start = {b: data.xpos[b].copy() for b in STATE["roots"]}
    own = set()
    b = interface.body
    while b > 0:
        own.add(b)
        b = model.body_parentid[b]
    ctx = Context(session, STATE["kinematics"], STATE["occupancy"], [], keep_closest=True)
    started = time.time()
    result = {"task": task_id, "skill": task.skill, "kind": interface.kind, "mechanism": interface.mechanism.kind,
              "instance": interface.instance}
    result["prerequisites"] = []
    try:
        unlock(ctx, session, task, model, payload, result)
        for attempt in range(MAX_PREREQUISITES + 1):
            try:
                SKILLS[task.skill](ctx, interface, task)
                break
            except Abort as exc:
                opener = prerequisite(exc, model, payload) if attempt < MAX_PREREQUISITES else None
                if opener is None:
                    raise
                # Something movable is in the way (a lid, a door): open it with its own interface first.
                step = {"interface": opener[0].id, "for": exc.detail.get("geom")}
                opener[1].key = "pre:" + opener[0].id
                session.monitor.evaluate(opener[1].goal, "pre:" + opener[0].id)      # anchor its start
                try:
                    SKILLS[opener[1].skill](ctx, opener[0], opener[1])
                    step["opened"] = session.monitor.evaluate(opener[1].goal, "pre:" + opener[0].id)
                    if opener[1].skill != "set_aside":         # even part-way: let the exact checks decide
                        ctx.opened = (ctx.opened or set()) | {opener[0].mechanism.joint}
                except Abort as inner:
                    step["failed"] = inner.reason
                    result["prerequisites"].append(step)
                    raise Abort("prerequisite_failed", step=step, original=exc.reason)
                result["prerequisites"].append(step)
        session.wait(0.5)
        result["reason"] = None
    except Abort as exc:
        result["reason"], result["detail"] = exc.reason, exc.detail
    except Exception as exc:                                         # a code bug, never fed to a model
        result["reason"], result["detail"] = "code_error", {"error": repr(exc), "trace": traceback.format_exc()[-1500:]}
    own |= ctx.moved or set()                                       # set aside on purpose by a prerequisite
    moved = {model.body(b).name: float(np.linalg.norm(data.xpos[b] - p)) for b, p in start.items() if b not in own}
    disturbed = {k: v for k, v in moved.items() if v > 0.005}
    goal = session.monitor.evaluate(task.goal, "goal")
    result.update(goal=goal, goal_detail=session.monitor.details.get("goal"), disturbed=disturbed,
                  peak_robot_force_n=session.peak_force, sim_s=session.time, wall_s=time.time() - started, log=ctx.log)
    achieved = goal or (task.momentary and "goal" in session.monitor.reached)
    result["achieved"] = bool(achieved)
    result["success"] = bool(achieved) and not disturbed and result["reason"] is None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int, default=0, help="stratified sample of N tasks (0 = all)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--only", nargs="*", default=None, help="interface ids")
    args = parser.parse_args()
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=False)
    init()
    model, robot = STATE["model"], STATE["robot"]
    payload = robot.profile["arms"]["left"]["payload_kg"]
    tasks = {i: make_task(f, model, payload) for i, f in STATE["interfaces"].items()}
    runnable = sorted(i for i, t in tasks.items() if not t.excluded)
    if args.only:
        runnable = [i for i in runnable if i in set(args.only)]
    elif args.limit:
        # Round-robin over (skill, instrument) so one instrument with many parts does not dominate.
        groups = collections.defaultdict(list)
        for i in runnable:
            groups[(tasks[i].skill, STATE["interfaces"][i].instance)].append(i)
        rng = np.random.default_rng(0)
        for g in groups.values():
            rng.shuffle(g)
        order, depth = [], 0
        while len(order) < args.limit and any(len(g) > depth for g in groups.values()):
            order += [g[depth] for _, g in sorted(groups.items()) if len(g) > depth]
            depth += 1
        runnable = sorted(order[:args.limit])
    from np002_traj.record.run import code_identity
    (out / "config.json").write_text(json.dumps({"scene": str(STATE["directory"]), "scene_source": CONFIG["scene_source"], "code": code_identity(),
                                                 "tasks": len(runnable), "excluded": collections.Counter(
                                                     t.excluded.split(" needs")[0] if t.excluded else "" for t in tasks.values() if t.excluded)},
                                                indent=1), encoding="utf-8")
    limits = guard.default_limits()
    workers = guard.safe_workers(args.workers, limits=limits)
    if workers < args.workers:
        print(f"workers capped from {args.workers} to {workers} by the resource guard", flush=True)
    print(f"{len(runnable)} tasks, {workers} workers; guard: run memory <= {limits['tree_bytes'] / guard.GB:.0f} GB, "
          f"machine available >= {limits['available_floor_bytes'] / guard.GB:.0f} GB, load <= {limits['load_limit']:.0f}",
          flush=True)
    watchdog = guard.Guard(out, limits)
    started = time.time()
    with mp.get_context("fork").Pool(workers, initializer=init, maxtasksperchild=TASKS_PER_WORKER) as pool, \
            (out / "results.jsonl").open("w") as sink:
        watchdog.start(on_trip=pool.terminate)
        done = collections.Counter()
        for result in pool.imap_unordered(run_task, runnable):
            sink.write(json.dumps(result, default=float) + "\n")
            sink.flush()
            done[(result["skill"], result["success"])] += 1
            if sum(done.values()) % 10 == 0:
                print(f"{sum(done.values())}/{len(runnable)} in {time.time()-started:.0f} s: {dict(done)}", flush=True)
    watchdog.stop()
    print("finished", dict(done), f"peak run memory {watchdog.peak / guard.GB:.1f} GB")


if __name__ == "__main__":
    main()
