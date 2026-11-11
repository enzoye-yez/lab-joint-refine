"""Settle the assembled room once with island sleeping; the settled state becomes the task initial state.

Bodies that never come to rest are reported as asset problems instead of being hidden.
"""
import json

import mujoco
import numpy as np

from ..control.joint import JointPD
from ..record.run import get_state
from ..robot import Robot

SLEEP = int(mujoco.mjtEnableBit.mjENBL_SLEEP)


def enable_sleep(model):
    """Let resting trees sleep, unless a tendon equality couples joints (MuJoCo cannot sleep with those)."""
    if any(model.eq_type[i] == mujoco.mjtEq.mjEQ_TENDON for i in range(model.neq)):
        model.opt.enableflags &= ~SLEEP
        return False
    model.opt.enableflags |= SLEEP
    return True


def body_speeds(model, data, bodies):
    out, v = {}, np.zeros(6)
    for b in bodies:
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(b), v, 0)
        out[int(b)] = (float(np.linalg.norm(v[3:])), float(np.linalg.norm(v[:3])))
    return out


def settle(model, robot, max_seconds=3.0, linear=1e-3, angular=0.01, key="assembled"):
    """Run with the robot holding its pose until every other tree sleeps or is slower than the limits."""
    sleeping = enable_sleep(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    hold = [JointPD(model, arm) for arm in robot.arms.values()]
    for c in hold:
        c.hold(data)
    data.ctrl[robot.base_actuators] = 0.0
    robot_trees = {int(model.body_treeid[b]) for b in robot.bodies if model.body_treeid[b] >= 0}
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and model.body_treeid[b] >= 0
             and int(model.body_treeid[b]) not in robot_trees]
    steps = int(max_seconds / model.opt.timestep)
    for step in range(steps):
        for c in hold:
            c.apply(data)
        mujoco.mj_step(model, data)
        if step % 50 == 49:
            moving = [b for b, (lin, ang) in body_speeds(model, data, roots).items()
                      if data.body_awake[b] and (lin > linear or ang > angular)]
            if not moving:
                break
    speeds = body_speeds(model, data, roots)
    unsettled = sorted(({"body": model.body(b).name, "linear_mps": lin, "angular_radps": ang}
                        for b, (lin, ang) in speeds.items() if lin > linear or ang > angular),
                       key=lambda r: -r["linear_mps"])
    report = {"seconds": float(data.time), "trees": len(roots), "awake_trees": int(data.ntree_awake),
              "sleep": sleeping, "unsettled": unsettled}
    return data, report
