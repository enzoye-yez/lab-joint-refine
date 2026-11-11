"""Run one single-skill task in-process: the scene state every task starts from, prerequisite and
unlock handling, tools and hold-up, and the success judgement.

  state = build_state(directory, scene_source, robot_prefix)        # once per process
  result = run_task(state, interface_id)

`build_state` accepts an already compiled (e.g. perturbed) model with an "assembled" keyframe, so
counterfactual scene edits run through exactly the same task code as the benchmark.
"""
import signal
import json
import time
import traceback

import mujoco
import numpy as np

from .. import guard
from ..executor.session import Abort, Session
from ..robot import Robot
from ..scene.interfaces import load
from ..scene.locality import Locality
from ..scene.settle import enable_sleep
from ..skills.dock import Kinematics
from ..skills.library import SKILLS, Context, hold_up, mount_fit, present_again, tool_pick
from ..planning.base import OccupancyMap
from .benchmark import task_for
from .semantics import Semantics
from .types import task_type


def build_state(directory, scene_source, robot_prefix, model=None, capability=None):
    """Everything a task needs, built from the settled scene in `directory` (or from `model`, which
    must carry the same "assembled" keyframe)."""
    if model is None:
        model = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    enable_sleep(model)
    robot = Robot(model, prefix=robot_prefix, capability=capability)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("assembled").id)
    mujoco.mj_forward(model, data)
    interfaces, _ = load(model, scene_source, data)
    return {"semantics": Semantics(model, scene_source), "active": data.eq_active.copy(), "model": model,
            "robot": robot, "interfaces": {i.id: i for i in interfaces}, "kinematics": Kinematics(capability=robot.capability),
            "occupancy": OccupancyMap(model, data, robot), "roots": instance_roots(model, robot),
            "directory": directory, "unsettled": unsettled_bodies(model, directory),
            "locality": Locality(model, data), "site_xpos": data.site_xpos.copy()}


def instance_roots(model, robot):
    """Top-level bodies of every instance except the robot: their poses must not change."""
    robot_bodies = set(robot.bodies)
    return [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and b not in robot_bodies]


def unsettled_bodies(model, directory):
    """Bodies still moving at the end of settling: they move without the robot, so their motion is
    not charged to a task (the scene side reports them as support findings)."""
    report = directory / "settle.json"
    names = [r["body"] for r in json.loads(report.read_text())["unsettled"]] if report.exists() else []
    return {model.body(n).id for n in names}


MAX_PREREQUISITES = 2
TASK_WALL_S = 3600             # a task still running after this is recorded as task_timeout


class TaskTimeout(Exception):
    pass


def _timeout(signum, frame):
    raise TaskTimeout()
OPENING_SKILLS = ("actuate", "rotate")
OPEN_TRAVEL = 0.8             # fraction of the joint range an opener prerequisite drives to
OPEN_TRAVEL_MAX = 0.95        # an actuated joint is driven to 0.97 of its range: ask no more than it reaches


def declared_travel(target, opener, model, data):
    """Fraction of the way to its far end that the target's declared prerequisite asks the opener's
    joint to go (an asset says a tip box lid must stand at 1.98 rad for its tips), or None."""
    for need in (target.extra.get("prerequisites") or []) if target is not None else ():
        if need.get("interface") != opener.id.split("__", 1)[-1] or need.get("target_position") is None:
            continue
        joint = opener.mechanism.joint
        now = float(data.qpos[model.jnt_qposadr[joint]])
        low, high = model.jnt_range[joint]
        far = high if high - now >= now - low else low
        if abs(far - now) > 1e-6:
            return min(OPEN_TRAVEL_MAX, abs(float(need["target_position"]) - now) / abs(far - now))
    return None


def prerequisite(state, abort, model, payload, target=None, data=None):
    """The interface (and its task) that opens the movable part an enclosure abort names, if any."""
    if abort.reason != "enclosed" or not abort.detail.get("joint"):
        return None
    joint = model.joint(abort.detail["joint"]).id
    if abort.detail.get("movable") == "free":
        # A loose part lies in the way: pick it by its own grasp interface and put it down aside.
        for candidate in state["interfaces"].values():
            if candidate.kind == "grasp" and candidate.mechanism.joint == joint:
                task = make_task(state, candidate, model, payload)
                if not task.excluded and task.skill == "pick":
                    task.skill = "set_aside"
                    return candidate, task
        return None
    # Any interface that drives that joint can open it: a handle (actuate) first, then a rotate grip.
    openers = []
    for candidate in state["interfaces"].values():
        if candidate.mechanism.joint == joint:
            task = make_task(state, candidate, model, payload)
            if not task.excluded and task.skill in OPENING_SKILLS:
                if "travel" in task.parameters:         # open it far enough to clear the part behind it
                    declared = declared_travel(target, candidate, model, data) if data is not None else None
                    task.parameters["travel"] = task.goal["travel"] = max(task.parameters["travel"], OPEN_TRAVEL,
                                                                          declared or 0.0)
                openers.append((OPENING_SKILLS.index(task.skill), candidate.id, candidate, task))
    return min(openers, key=lambda o: o[:2])[2:] if openers else None


def make_task(state, interface, model, payload):
    return task_for(interface, model, payload, semantics=state["semantics"], active=state["active"])


def unlock(state, ctx, session, task, model, payload, result):
    """Release the locks the asset declares for this part, each with its own interface's skill first."""
    for lock in task.unlock:
        opener = state["interfaces"].get(lock["interface"])
        step = {"interface": lock["interface"], "for": lock["connection"], "unlock": True}
        opener_task = make_task(state, opener, model, payload) if opener else None
        if opener_task is None or opener_task.excluded or not opener_task.skill:
            step["failed"] = "release interface has no runnable task"
            result["prerequisites"].append(step)
            raise Abort("unlock_failed", step=step)
        opener_task.key = "pre:" + opener.id
        try:
            session.monitor.evaluate(opener_task.goal, opener_task.key)
            attempt(state, ctx, session, opener, opener_task, model, payload, result)      # it may be enclosed too
        except Abort as inner:
            step["failed"] = inner.reason
            result["prerequisites"].append(step)
            raise Abort("unlock_failed", step=step)
        session.data.eq_active[lock["equality"]] = 0          # the release event of that interface
        result["prerequisites"].append(step)


def tool_for(state, interface, session, model, payload):
    """For a part the fingers cannot hold but that declares an orifice (a pipette tip), the tool with a
    declared mount point from the same catalogue asset (the pipettes of a tip set) whose mount end fits
    the part's opening best, the nearest among equals, with the tool's own grasp interface and task."""
    interfaces = state["interfaces"].values()
    orifice = next((i for i in interfaces if i.kind == "insert" and i.body == interface.body and i.site >= 0
                    and i.approach is not None), None)
    if orifice is None:
        return None
    where = session.data.site_xpos[orifice.site]
    tools = []
    for mount in interfaces:
        if mount.kind != "mount" or mount.site < 0 or mount.approach is None or mount.instance == interface.instance \
                or mount.extra.get("asset_id") != orifice.extra.get("asset_id"):
            continue
        for grasp in interfaces:
            if grasp.kind == "grasp" and grasp.body == mount.body:
                grasp_task = make_task(state, grasp, model, payload)
                if not grasp_task.excluded and grasp_task.skill == "pick":
                    fit = round(mount_fit(model, session.data, mount, orifice), 4)      # 0.1 mm classes
                    tools.append((fit, float(np.linalg.norm(session.data.site_xpos[mount.site] - where)), mount.id,
                                  {"mount": mount, "grasp": grasp}, grasp_task))
                break
    if not tools:
        return None
    _, _, _, tool, grasp_task = min(tools, key=lambda t: t[:3])
    grasp_task.key = "tool:" + tool["grasp"].id
    return tool, grasp_task, orifice


FRESH_GRASPS = 2           # fresh grasps after a regrasp closed on nothing while driving a mechanism
HELD_SKILLS = ("press", "actuate", "rotate")


def holder_for(state, interface, model, payload):
    """A loose, light part's own grasp interface (a pipette's handle for its knob, plunger or ejector)
    and its pick task, or None when the part is fixed, heavy or has no other grasp."""
    root = interface.body
    while model.body_parentid[root] > 0:
        root = model.body_parentid[root]
    free = any(int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(model.njnt) if model.jnt_bodyid[j] == root)
    if not free or model.body_subtreemass[root] > state["robot"].capability["hold_up_max_kg"]:
        return None
    for grasp in state["interfaces"].values():
        if grasp.kind != "grasp" or grasp.id == interface.id:
            continue
        b = grasp.body
        while model.body_parentid[b] > 0:
            b = model.body_parentid[b]
        if b != root:
            continue
        grasp_task = make_task(state, grasp, model, payload)
        if not grasp_task.excluded and grasp_task.skill == "pick":
            grasp_task.key = "hold:" + grasp.id
            return grasp, grasp_task
    return None


def _opened_enough(session, opener, task):
    """An opener's joint has gone at least OPEN_TRAVEL of the way from where it started to its far end."""
    joint = opener.mechanism.joint
    start = session.monitor.origins.get(task.key)
    if joint is None or joint < 0 or start is None or np.size(start) != 1 or \
            session.model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return False                       # only a 1-DoF opener has a "fraction of the way"
    start = float(np.ravel(start)[0])
    low, high = session.model.jnt_range[joint]
    far = high if high - start >= start - low else low
    now = float(session.data.qpos[session.model.jnt_qposadr[joint]])
    return abs(far - start) > 1e-6 and (now - start) / (far - start) >= OPEN_TRAVEL


def _achieved(session, task):
    return session.monitor.evaluate(task.goal, task.key) is True or (task.momentary and task.key in session.monitor.reached)


def attempt(state, ctx, session, interface, task, model, payload, result):
    """Run the task's skill; when a movable part encloses the target, open it first and retry."""
    for tries in range(MAX_PREREQUISITES + 1):
        try:
            failure = None
            for again in range(FRESH_GRASPS + 1):
                try:
                    SKILLS[task.skill](ctx, interface, task)
                    failure = None
                except Abort as exc:
                    if exc.reason == "enclosed":
                        raise
                    failure = exc
                # A regrasp that closed on nothing mid-drive: start over with a fresh grasp where it now is.
                if failure is None or failure.reason not in ("regrasp_missed", "regrasp_failed"):
                    break
                result.setdefault("fresh_grasps", 0)
                result["fresh_grasps"] += 1
            if failure is None and (task.skill not in HELD_SKILLS or _achieved(session, task)):
                return
            if task.skill == "pick" and failure.reason == "no_grasp":
                found = tool_for(state, interface, session, model, payload)
                if found is None:
                    raise failure
                # The fingers cannot hold it: use the tool made for it (a pipette for its tip).
                result["tool"] = found[0]["mount"].id
                tool_pick(ctx, interface, task, *found)
                return
            holder = holder_for(state, interface, model, payload) if task.skill in HELD_SKILLS and ctx.holding is None else None
            if holder is None:
                if failure is not None:
                    raise failure
                return                            # the goal check after the task decides
            # It slides or turns on the bench under the hand: hold it up with one hand, work it with the other.
            result["held_by"] = holder[0].id
            hold_up(ctx, *holder)
            for spot in range(1, 99):
                try:
                    SKILLS[task.skill](ctx, interface, task)
                except Abort as exc:
                    if exc.reason == "enclosed":
                        raise
                    failure = exc
                else:
                    failure = None
                    if _achieved(session, task):
                        return
                # The other arm cannot reach or work it there: present it elsewhere and try again.
                if not present_again(ctx, holder[0], spot):
                    if failure is not None:
                        raise failure
                    return
            return
        except Abort as exc:
            opener = prerequisite(state, exc, model, payload, interface, session.data) if tries < MAX_PREREQUISITES else None
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
                if not _opened_enough(session, opener[0], opener[1]):
                    step["failed"] = inner.reason
                    result["prerequisites"].append(step)
                    raise Abort("prerequisite_failed", step=step, original=exc.reason)
                # Stopped short of the declared position but past the usual opening: go on, the checks decide.
                step.update(opened=True, stopped=inner.reason)
                ctx.opened = (ctx.opened or set()) | {opener[0].mechanism.joint}
            result["prerequisites"].append(step)


WORKER_BYTES = guard.default_limits()["worker_bytes"]
TASKS_PER_WORKER = 20          # workers are recycled so a slow leak cannot accumulate


def run_task(state, task_id):
    try:
        guard.worker_check(WORKER_BYTES)
    except MemoryError as exc:
        return {"task": task_id, "skill": "", "kind": "", "mechanism": "", "instance": "", "reason": "worker_memory",
                "detail": {"error": str(exc)}, "success": False, "achieved": False, "disturbed": {}, "log": [],
                "prerequisites": [], "wall_s": 0.0}
    model, robot = state["model"], state["robot"]
    interface = state["interfaces"][task_id]
    payload = robot.capability["arms"]["left"]["payload_kg"]
    task = make_task(state, interface, model, payload)
    keep = {interface.instance} | {state["interfaces"][u["interface"]].instance for u in task.unlock
                                   if u["interface"] in state["interfaces"]}
    saved, local = state["locality"].apply(state["site_xpos"][interface.site], keep)
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(TASK_WALL_S)
        result = _run_task(state, task_id, interface, task, model, robot, payload)
    except TaskTimeout:                 # fired outside the skill (goal checks after it)
        result = {"task": task_id, "skill": task.skill, "kind": interface.kind, "mechanism": interface.mechanism.kind,
                  "instance": interface.instance, "type": task_type(interface.id, interface.instance, task.skill), "prerequisites": [], "reason": "task_timeout",
                  "detail": {"wall_limit_s": TASK_WALL_S}, "goal": False, "disturbed": {}, "log": [],
                  "wall_s": float(TASK_WALL_S), "achieved": False, "success": False}
    finally:
        signal.alarm(0)
        state["locality"].restore(saved)
    result["locality"] = local
    return result


DISTURB_M = 0.005          # another instance moved further than this: a side effect
OWN_INSTANCE_M = 0.02      # the operated instance's other parts may shift a little while it is worked on


def _disturbed(state, model, data, start, own, instance):
    """Roots that moved as a side effect. A root that kept its offset to the operated root (a cap lifted
    with its tube) was carried, not disturbed."""
    anchor = next((b for b in own if model.body_parentid[b] == 0), 0)
    disturbed = {}
    for b, p in start.items():
        if b in own or b in state["unsettled"]:
            continue
        moved = float(np.linalg.norm(data.xpos[b] - p))
        if anchor in start and np.linalg.norm((data.xpos[b] - data.xpos[anchor]) - (p - start[anchor])) < DISTURB_M:
            continue
        name = model.body(b).name
        if moved > (OWN_INSTANCE_M if name.startswith(instance + "__") else DISTURB_M):
            disturbed[name] = moved
    return disturbed


def _run_task(state, task_id, interface, task, model, robot, payload):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("assembled").id)
    mujoco.mj_forward(model, data)
    session = Session(model, data, robot)
    session.monitor.evaluate(task.goal, "goal")                      # anchor relative goals at the start
    start = {b: data.xpos[b].copy() for b in state["roots"]}
    own = set()
    b = interface.body
    while b > 0:
        own.add(b)
        b = model.body_parentid[b]
    ctx = Context(session, state["kinematics"], state["occupancy"], [], keep_closest=True)
    equalities = (model.eq_obj1id.copy(), model.eq_obj2id.copy(), model.eq_data.copy())   # skills re-target welds
    started = time.time()
    result = {"task": task_id, "skill": task.skill, "kind": interface.kind, "mechanism": interface.mechanism.kind,
              "instance": interface.instance, "type": task_type(interface.id, interface.instance, task.skill)}
    result["prerequisites"] = []
    try:
        unlock(state, ctx, session, task, model, payload, result)
        attempt(state, ctx, session, interface, task, model, payload, result)
        session.wait(0.5)
        result["reason"] = None
    except Abort as exc:
        result["reason"], result["detail"] = exc.reason, exc.detail
    except TaskTimeout:
        result["reason"], result["detail"] = "task_timeout", {"wall_limit_s": TASK_WALL_S}
    except Exception as exc:                                         # a code bug, never fed to a model
        result["reason"], result["detail"] = "code_error", {"error": repr(exc), "trace": traceback.format_exc()[-1500:]}
    model.eq_obj1id[:], model.eq_obj2id[:], model.eq_data[:] = equalities
    own |= ctx.moved or set()                                       # set aside on purpose by a prerequisite
    disturbed = _disturbed(state, model, data, start, own, interface.instance)
    goal = session.monitor.evaluate(task.goal, "goal")
    result.update(goal=goal, goal_detail=session.monitor.details.get("goal"), disturbed=disturbed,
                  peak_robot_force_n=session.peak_force, sim_s=session.time, wall_s=time.time() - started, log=ctx.log)
    achieved = goal or (task.momentary and "goal" in session.monitor.reached)
    result["achieved"] = bool(achieved)
    result["success"] = bool(achieved) and not disturbed and result["reason"] is None
    return result


