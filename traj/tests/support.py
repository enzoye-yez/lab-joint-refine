"""Shared test worlds: the robot on a floor, optionally with extra bodies."""
import mujoco
import numpy as np

from np002_traj.control.joint import JointPD
from np002_traj.robot import ROOT, Robot


def world(box=None, extra=None):
    """The robot on a floor; optionally a free box at the left TCP held by a test-only weld."""
    spec = mujoco.MjSpec.from_file(str(ROOT / "data/robot/model.xml"))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    if box:
        body = spec.worldbody.add_body(name="box", pos=box["pos"])
        body.add_freejoint(name="box")
        body.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[box["half"]] * 3, mass=box["mass"],
                      friction=[0.9, 0.02, 0.001], condim=4)
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="box", objtype=mujoco.mjtObj.mjOBJ_BODY, name="box_fixture")
        for key in spec.keys:  # the extra free joint extends qpos
            key.qpos = list(key.qpos) + list(box["pos"]) + [1, 0, 0, 0]
    if extra:
        extra(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, Robot(model)


def controllers(model, data, robot):
    result = {side: JointPD(model, arm) for side, arm in robot.arms.items()}
    for c in result.values():
        c.hold(data)
    return result


def run(model, data, control, seconds, record=None):
    out = []
    for _ in range(int(seconds / model.opt.timestep)):
        for c in control.values():
            c.apply(data)
        mujoco.mj_step(model, data)
        if record:
            out.append(record(data))
    return np.array(out)
