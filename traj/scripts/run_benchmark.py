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

import numpy as np

from np002_traj import guard
from np002_traj.robot import ROOT
from np002_traj.scene.assemble import cache_for, config as scene_config
from np002_traj.tasks import run
from np002_traj.tasks.run import TASKS_PER_WORKER
from np002_traj.tasks.types import representatives, task_type

CONFIG = scene_config()
STATE = {}


def scene_directory():
    directory = cache_for(CONFIG["scene_source"])
    if not (directory / "scene.mjb").exists():
        raise SystemExit(f"scene not prepared: run scripts/prepare_scene.py for {CONFIG['scene_source']}")
    return directory


def init():
    STATE.clear()
    STATE.update(run.build_state(scene_directory(), CONFIG["scene_source"], CONFIG["robot_prefix"]))


def make_task(interface, model, payload):
    return run.make_task(STATE, interface, model, payload)


def run_task(task_id):
    return run.run_task(STATE, task_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int, default=0, help="stratified sample of N tasks (0 = all)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--only", nargs="*", default=None, help="interface ids")
    parser.add_argument("--per-type", type=int, default=0,
                        help="run only N representatives of each task type (0 = every task)")
    args = parser.parse_args()
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=False)
    init()
    model, robot = STATE["model"], STATE["robot"]
    payload = robot.capability["arms"]["left"]["payload_kg"]
    tasks = {i: make_task(f, model, payload) for i, f in STATE["interfaces"].items()}
    runnable = sorted(i for i, t in tasks.items() if not t.excluded)
    if args.only:
        runnable = [i for i in runnable if i in set(args.only)]
    if args.per_type:
        members = collections.defaultdict(list)
        for i in runnable:
            members[task_type(i, STATE["interfaces"][i].instance, tasks[i].skill)].append(i)
        runnable = sorted(j for g in members.values() for j in representatives(g, args.per_type))
    elif args.limit and not args.only:
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
                                                 "unsettled_not_charged": sorted(model.body(b).name for b in STATE["unsettled"]),
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
