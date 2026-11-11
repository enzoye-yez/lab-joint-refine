"""Run counterfactual jobs (scene edits + tasks to rerun) in a worker pool.

  PYTHONPATH=src $PYTHON scripts/run_counterfactual.py --jobs jobs.json --run cf_01 [--workers 16] [--task-wall 900]

jobs.json is a list of {"id", "edits", "tasks", "meta"?}; results stream to runs/<run>/jobs.jsonl.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from np002_traj import guard
from np002_traj.robot import ROOT
from np002_traj.scene.assemble import cache_for, config as scene_config
from np002_traj.tasks import counterfactual, run

CONFIG = scene_config()


def init(wall):
    run.TASK_WALL_S = wall
    directory = cache_for(CONFIG["scene_source"])
    counterfactual.load_base(directory, CONFIG["scene_source"], CONFIG["robot_prefix"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--task-wall", type=int, default=900)
    args = parser.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in (out / "jobs.jsonl").open()} if (out / "jobs.jsonl").exists() else set()
    todo = [j for j in jobs if j["id"] not in done]          # resumable
    directory = cache_for(CONFIG["scene_source"])
    if not (directory / "scene.mjb").exists():
        raise SystemExit(f"scene not prepared: {directory}")
    (out / "config.json").write_text(json.dumps({"scene": str(directory), "scene_source": CONFIG["scene_source"],
                                                 "jobs": len(jobs), "task_wall_s": args.task_wall}, indent=1))
    limits = guard.default_limits()
    workers = min(len(todo) or 1, guard.safe_workers(args.workers, limits=limits))
    print(f"{len(todo)} of {len(jobs)} jobs to run, {workers} workers", flush=True)
    watchdog = guard.Guard(out, limits)
    started = time.time()
    with mp.get_context("fork").Pool(workers, initializer=init, initargs=(args.task_wall,), maxtasksperchild=4) as pool, \
            (out / "jobs.jsonl").open("a") as sink:
        watchdog.start(on_trip=pool.terminate)
        for n, result in enumerate(pool.imap_unordered(counterfactual.run_job, todo), 1):
            sink.write(json.dumps(result, default=float) + "\n")
            sink.flush()
            ok = sum(r["success"] for r in result["results"])
            print(f"{n}/{len(todo)} {result['id']}: {ok}/{len(result['results'])} succeeded "
                  f"({result['wall_s']:.0f} s){' ' + result['error'] if result.get('error') else ''}", flush=True)
    watchdog.stop()
    print(f"finished in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
