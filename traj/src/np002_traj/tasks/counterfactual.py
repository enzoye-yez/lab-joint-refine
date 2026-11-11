"""Rerun tasks on a perturbed copy of the scene (see scene/perturb.py).

A job is {"id", "edits": [edit, ...], "tasks": [interface id, ...]}; an empty edit list reruns the tasks
on the unchanged scene (the control that the failure reproduces). Every task goes through
tasks/run.run_task, so a counterfactual success means the benchmark itself would succeed there.
"""
import time

import mujoco

from ..robot import Robot
from ..scene import perturb
from . import run

BASE = {}


def load_base(directory, scene_source, robot_prefix):
    """The settled scene every job starts from (loaded once per process)."""
    model = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    BASE.update(model=model, robot=Robot(model, prefix=robot_prefix), directory=directory,
                scene_source=scene_source, robot_prefix=robot_prefix)
    BASE["state"] = run.build_state(directory, scene_source, robot_prefix, model=model)


def run_job(job):
    started = time.time()
    out = {"id": job["id"], "edits": job.get("edits", []), "meta": job.get("meta", {})}
    try:
        if job.get("edits"):
            model, report = perturb.apply(BASE["model"], job["edits"], BASE["robot"])
            state = run.build_state(BASE["directory"], BASE["scene_source"], BASE["robot_prefix"], model=model)
            out["perturb"] = report
        else:
            state = BASE["state"]
    except (perturb.EditError, KeyError, ValueError) as exc:
        out.update(error=f"{type(exc).__name__}: {exc}", results=[], wall_s=time.time() - started)
        return out
    out["results"] = [run.run_task(state, t) for t in job["tasks"]]
    out["wall_s"] = time.time() - started
    return out
