"""Single-skill benchmark over every operable interface of the assembled room.

Every task starts from the same settled state with the robot at its home pose, runs the generic
skill for its interface type and mechanism, and is judged by the mechanism-derived goal plus
"no other instance moved" and the robot contact-force limit. No per-interface parameters exist.

  source scripts/env.sh && $PYTHON scripts/run_benchmark.py --run bench_01 [--limit 40] [--workers 32]
"""
import argparse
import collections
import json
import multiprocessing as mp
import time
import traceback

import mujoco
import numpy as np

from np002_traj.executor.session import Abort, Session
from np002_traj.robot import ROOT, Robot
from np002_traj.scene.interfaces import load
from np002_traj.scene.settle import enable_sleep
from np002_traj.skills.dock import Kinematics
from np002_traj.skills.library import SKILLS, Context
from np002_traj.planning.base import OccupancyMap
from np002_traj.tasks.benchmark import task_for

CONFIG = json.loads((ROOT / "config/np002.json").read_text())
STATE = {}


def scene_directory():
    caches = sorted((ROOT / "runs/scene_cache").glob("*/scene.mjb"), key=lambda p: p.stat().st_mtime)
    return caches[-1].parent


def instance_roots(model, robot):
    """Top-level bodies of every instance except the robot: their poses must not change."""
    robot_bodies = set(robot.bodies)
    return [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and b not in robot_bodies]


def init():
    directory = scene_directory()
    model = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    enable_sleep(model)
    robot = Robot(model, prefix=CONFIG["robot_prefix"])
    interfaces, _ = load(model, CONFIG["scene_source"])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("assembled").id)
    mujoco.mj_forward(model, data)
    STATE.update(model=model, robot=robot, interfaces={i.id: i for i in interfaces}, kinematics=Kinematics(),
                 occupancy=OccupancyMap(model, data, robot), roots=instance_roots(model, robot), directory=directory)


def run_task(task_id):
    model, robot = STATE["model"], STATE["robot"]
    interface = STATE["interfaces"][task_id]
    payload = robot.profile["arms"]["left"]["payload_kg"]
    task = task_for(interface, model, payload)
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
    ctx = Context(session, STATE["kinematics"], STATE["occupancy"], [])
    started = time.time()
    result = {"task": task_id, "skill": task.skill, "kind": interface.kind, "mechanism": interface.mechanism.kind,
              "instance": interface.instance}
    try:
        SKILLS[task.skill](ctx, interface, task)
        session.wait(0.5)
        result["reason"] = None
    except Abort as exc:
        result["reason"], result["detail"] = exc.reason, exc.detail
    except Exception as exc:                                         # a code bug, never fed to a model
        result["reason"], result["detail"] = "code_error", {"error": repr(exc), "trace": traceback.format_exc()[-1500:]}
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
    tasks = {i: task_for(f, model, payload) for i, f in STATE["interfaces"].items()}
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
    (out / "config.json").write_text(json.dumps({"scene": str(STATE["directory"]), "code": code_identity(),
                                                 "tasks": len(runnable), "excluded": collections.Counter(
                                                     t.excluded.split(" needs")[0] if t.excluded else "" for t in tasks.values() if t.excluded)},
                                                indent=1), encoding="utf-8")
    print(f"{len(runnable)} tasks, {args.workers} workers", flush=True)
    started = time.time()
    with mp.get_context("fork").Pool(args.workers, initializer=init) as pool, (out / "results.jsonl").open("w") as sink:
        done = collections.Counter()
        for result in pool.imap_unordered(run_task, runnable):
            sink.write(json.dumps(result, default=float) + "\n")
            sink.flush()
            done[(result["skill"], result["success"])] += 1
            if sum(done.values()) % 10 == 0:
                print(f"{sum(done.values())}/{len(runnable)} in {time.time()-started:.0f} s: {dict(done)}", flush=True)
    print("finished", dict(done))


if __name__ == "__main__":
    main()
