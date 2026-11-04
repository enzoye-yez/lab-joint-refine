"""Choose where to park, which arm to use, the lift height and the arm postures for an interface.

Inverse kinematics runs on the robot-only model (fast); every solution is then checked against the
room with the collision world. Candidates are ordered by how directly the base faces the part.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..planning.collision import CollisionWorld
from ..planning.ik import IK, IKFailure, Target
from ..robot import ROOT, Robot


@dataclass
class Dock:
    base: np.ndarray        # world x, y, yaw
    lift: float
    side: str
    frame: object           # skills.grasp.Frame used
    q_pre: np.ndarray
    q_goal: np.ndarray
    tried: int


class Kinematics:
    """The robot alone: base, lift and arms, for fast IK in world coordinates."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "data/robot/model.xml"))
        self.robot = Robot(self.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.ik = {s: IK(self.model, a, iterations=200) for s, a in self.robot.arms.items()}

    def qpos(self, base, lift, arms):
        q = self.data.qpos.copy()
        q[self.robot.base_qpos] = base
        q[self.robot.lift_qpos] = lift
        for side, values in arms.items():
            q[self.robot.arms[side].qpos] = values
        return q

    def shoulder_height(self, lift):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.qpos([0, 0, 0], lift, {})
        mujoco.mj_kinematics(self.model, d)
        link = self.model.body(self.robot.profile["arms"]["left"]["joints"][1].replace("joint", "link")).id
        return float(d.xpos[link][2])


def footprint_free(occupancy, pose, half, margin=0.03):
    """Exact base rectangle against the occupancy grid."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    xs = np.arange(-half[0] - margin, half[0] + margin + 1e-9, occupancy.resolution / 2)
    ys = np.arange(-half[1] - margin, half[1] + margin + 1e-9, occupancy.resolution / 2)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([pose[0] + c * gx.ravel() - s * gy.ravel(), pose[1] + s * gx.ravel() + c * gy.ravel()])
    cells = np.floor((points - occupancy.origin) / occupancy.resolution).astype(int)
    if np.any(cells < 0) or np.any(cells >= occupancy.occupied.shape):
        return False
    return not occupancy.occupied[cells[:, 0], cells[:, 1]].any()


def find_dock(session, kinematics, occupancy, frames, pre_distance=0.10, allowed=(), distances=(0.55, 0.65, 0.75, 0.85),
              bearings=24, lift_offsets=(0.15, 0.0, 0.3), max_candidates=400):
    robot, m = session.robot, session.model
    half = robot.profile["base"]["footprint_half_m"]
    lift_range = robot.profile["lift"]["range_m"]
    shoulder0 = kinematics.shoulder_height(0.0)
    reach = robot.profile["arms"]["left"]["reach_m"]
    worlds = {}
    tried = 0
    reasons = {"footprint": 0, "reach": 0, "pre_ik": 0, "pre_collision": 0, "goal_ik": 0, "goal_collision": 0, "approach": 0}
    blocking = {}
    for frame in frames:
        target = frame.position
        heading = frame.rotation[:, 2][:2]                     # tool pointing, projected on the floor
        preferred = np.arctan2(heading[1], heading[0]) if np.linalg.norm(heading) > 0.3 else None
        options = []
        for r in distances:
            for k in range(bearings):
                phi = 2 * np.pi * k / bearings
                score = r * 0.5 + (0.0 if preferred is None else 1 - np.cos(phi - preferred))
                options.append((score, r, phi))
        for _, r, phi in sorted(options):
            u = np.array([np.cos(phi), np.sin(phi)])
            for side in robot.arms:
                lateral = robot.profile["arms"][side].get("lateral_m", 0.22 if side == "left" else -0.22)
                normal = np.array([-u[1], u[0]])
                centre = target[:2] - r * u - lateral * normal  # put this arm's shoulder in line with the part
                base = np.array([*centre, phi])
                if not footprint_free(occupancy, base, half):
                    reasons["footprint"] += 1
                    continue
                for offset in lift_offsets:
                    lift = float(np.clip(target[2] + offset - shoulder0, *lift_range))
                    shoulder = np.array([*centre, shoulder0 + lift])
                    if np.linalg.norm(target - shoulder) > reach + 0.15:
                        reasons["reach"] += 1
                        continue
                    tried += 1
                    if tried > max_candidates:
                        return None, {"tried": tried, "reasons": reasons, "blocking": _top(blocking)}
                    plan = _solve(session, kinematics, worlds, side, base, lift, frame, pre_distance, allowed, reasons, blocking)
                    if plan is not None:
                        return Dock(base, lift, side, frame, *plan, tried), {"tried": tried, "reasons": reasons}
    return None, {"tried": tried, "reasons": reasons, "blocking": _top(blocking)}


def _top(counter, n=5):
    return dict(sorted(counter.items(), key=lambda kv: -kv[1])[:n])


def _solve(session, kinematics, worlds, side, base, lift, frame, pre_distance, allowed, reasons, blocking):
    robot, kin = session.robot, kinematics
    q_base = kin.qpos(kin.robot.base_joint_values(base), lift, {})
    world = worlds.get(side)
    if world is None:
        world = worlds[side] = CollisionWorld(session.model, robot, robot.arms[side].qpos, margin=0.01)
        world.sync(session.data)
    world.set_joints(np.concatenate([robot.base_qpos, [robot.lift_qpos]]), [*robot.base_joint_values(base), lift])
    world.select()
    pre = frame.offset(pre_distance)
    ik = kin.ik[side]

    def solve(target, stage, **kw):
        try:
            solution = ik.solve(q_base, target, **kw).q
        except IKFailure:
            reasons[stage + "_ik"] += 1
            return None
        hit = world.blocking(solution)
        if hit is not None:
            reasons[stage + "_collision"] += 1
            name = "/".join(hit.get("geoms", ["self"]))
            blocking[name] = blocking.get(name, 0) + 1
            return None
        return solution

    world.allow(())
    q_pre = solve(Target(pre.position, pre.rotation), "pre", restarts=3)
    if q_pre is None:
        return None
    world.allow(allowed)
    q_goal = solve(Target(frame.position, frame.rotation, position_tolerance=1e-3), "goal", seed=q_pre, restarts=1)
    world.allow(())
    if q_goal is None:
        return None
    world.allow(allowed)
    ok = world.edge_valid(q_pre, q_goal)
    world.allow(())
    if not ok:
        reasons["approach"] += 1
        return None
    return q_pre, q_goal
