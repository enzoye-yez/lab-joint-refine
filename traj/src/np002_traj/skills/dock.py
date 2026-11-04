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


SHOULDER_FORWARD = 0.08    # m, arm mounts ahead of the base centre (data/robot build)
FREE_MARGIN = 0.005        # m, clearance in free motion
GOAL_MARGIN = 0.002        # m, clearance to neighbours at the grasp itself (FR3 repeatability is 0.1 mm)


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


def nearest_free(occupancy, target_xy, u, normal, lateral, phi, half, start=0.3, stop=1.3, step=0.02):
    """Smallest distance along bearing u at which the base rectangle no longer overlaps anything."""
    for r in np.arange(start, stop, step):
        centre = target_xy - r * u - lateral * normal
        if footprint_free(occupancy, np.array([*centre, phi]), half):
            return float(r)
    return None


def find_dock(session, kinematics, occupancy, frames, pre_distance=0.10, allowed=(), extra=(0.0, 0.06, 0.15),
              bearings=36, lift_offsets=(0.15, 0.0, 0.3), max_candidates=300):
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
        for k in range(bearings):
            phi = 2 * np.pi * k / bearings
            u = np.array([np.cos(phi), np.sin(phi)])
            normal = np.array([-u[1], u[0]])
            for side in robot.arms:
                lateral = robot.profile["arms"][side]["lateral_m"]   # this arm's shoulder in line with the part
                r0 = nearest_free(occupancy, target[:2], u, normal, lateral, phi, half)
                if r0 is None:
                    reasons["footprint"] += 1
                    continue
                for dr in extra:
                    r = r0 + dr
                    score = r + (0.0 if preferred is None else 0.6 * (1 - np.cos(phi - preferred)))
                    options.append((score, r, phi, side, lateral))
        for _, r, phi, side, lateral in sorted(options, key=lambda o: o[0]):
            u = np.array([np.cos(phi), np.sin(phi)])
            normal = np.array([-u[1], u[0]])
            centre = target[:2] - r * u - lateral * normal
            base = np.array([*centre, phi])
            if not footprint_free(occupancy, base, half):
                reasons["footprint"] += 1
                continue
            for offset in lift_offsets:
                lift = float(np.clip(target[2] + offset - shoulder0, *lift_range))
                shoulder = np.array([*(target[:2] - (r - SHOULDER_FORWARD) * u), shoulder0 + lift])
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
    return dict(sorted(counter.items(), key=lambda kv: -kv[1][0])[:n])


def _solve(session, kinematics, worlds, side, base, lift, frame, pre_distance, allowed, reasons, blocking):
    robot, kin = session.robot, kinematics
    q_base = kin.qpos(kin.robot.base_joint_values(base), lift, {})
    world = worlds.get(side)
    if world is None:
        world = worlds[side] = CollisionWorld(session.model, robot, robot.arms[side].qpos, margin=FREE_MARGIN)
        world.sync(session.data)
    # The other arm waits folded (as it does during execution).
    other = [o for o in robot.arms if o != side]
    addresses = np.concatenate([robot.base_qpos, [robot.lift_qpos]] + [robot.arms[o].qpos for o in other])
    values = np.concatenate([robot.base_joint_values(base), [lift]] +
                            [np.array(robot.profile["arms"][o]["travel"], float) for o in other])
    world.set_joints(addresses, values)
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
            closest = reasons.setdefault("_closest", {"distance_m": -1e9})
            if hit["distance_m"] > closest["distance_m"]:
                reasons["_closest"] = {"distance_m": hit["distance_m"], "stage": stage, "side": side, "base": base.tolist(),
                                       "lift": lift, "q": solution.tolist(), "geoms": hit.get("geoms")}
            name = "/".join(hit.get("geoms", ["self"]))
            count, closest = blocking.get(name, (0, 1.0))
            blocking[name] = (count + 1, min(closest, round(hit["distance_m"], 4)))
            return None
        return solution

    world.allow(())
    q_pre = solve(Target(pre.position, pre.rotation), "pre", restarts=5)
    if q_pre is None:
        return None
    world.allow(allowed)
    world.margin, margin = GOAL_MARGIN, world.margin
    q_goal = solve(Target(frame.position, frame.rotation, position_tolerance=1e-3), "goal", seed=q_pre, restarts=1)
    world.allow(())
    world.margin = margin
    if q_goal is None:
        return None
    world.allow(allowed)
    world.margin, margin = GOAL_MARGIN, world.margin
    ok = world.edge_valid(q_pre, q_goal)
    world.allow(())
    world.margin = margin
    if not ok:
        reasons["approach"] += 1
        return None
    return q_pre, q_goal
