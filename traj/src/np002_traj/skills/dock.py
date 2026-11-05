"""Choose where to park, which arm to use, the lift height and the arm postures for an interface.

Inverse kinematics runs on the robot-only model (fast); every solution is then checked against the
room with the collision world. Candidates are ordered by how directly the base faces the part.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..planning.collision import CollisionWorld
from ..planning.ik import IK, IKFailure, Target
from ..planning.wholebody import WholeBody
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
    other_posture: str = "travel"   # profile posture the arm that is not used started from
    q_other: np.ndarray = None      # its posture at the dock (may be repaired away from the active arm)
    pre_distance: float = 0.10      # the pre-grasp distance q_pre was solved for (shorter in tight places)


SHOULDER_FORWARD = 0.08    # m, arm mounts ahead of the base centre (data/robot build)
SHORT_PRE = 0.05           # m, fallback pre-grasp distance
RANK_PENALTY = 0.15
VARIANT_PENALTY = 0.05     # shifted or shallower grasps after the centred one        # declared approach first, then the main sides, then oblique ones
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


MAX_REPAIRS = 40          # whole-body repairs per dock search (each costs tens of distance sweeps)


def find_dock(session, kinematics, occupancy, frames, pre_distance=0.10, allowed=(), extra=(0.0, 0.06, 0.15),
              bearings=36, lift_offsets=(0.15, 0.0, 0.3), max_candidates=600, hand_state=None):
    """hand_state(side, frame) -> (qpos addresses, values) of the tool joints during the operation
    (pre-shaped fingers, press pin). Default: the fingers at the frame's approach opening."""
    if hand_state is None:
        hand_state = lambda side, frame: _preshaped(session, side, frame)
    robot, m = session.robot, session.model
    half = robot.profile["base"]["footprint_half_m"]
    lift_range = robot.profile["lift"]["range_m"]
    shoulder0 = kinematics.shoulder_height(0.0)
    reach = robot.profile["arms"]["left"]["reach_m"]
    worlds = {}
    tried = 0
    reasons = {"footprint": 0, "reach": 0, "pre_ik": 0, "pre_collision": 0, "goal_ik": 0, "goal_collision": 0, "approach": 0}
    blocking = {}
    repairs = [MAX_REPAIRS]
    # Rank every (grasp frame, base pose) pair together so that no approach direction is starved.
    distances = {}
    options = []
    for index, frame in enumerate(frames):
        target = frame.position
        heading = frame.rotation[:, 2][:2]                     # tool pointing, projected on the floor
        preferred = np.arctan2(heading[1], heading[0]) if np.linalg.norm(heading) > 0.3 else None
        for k in range(bearings):
            phi = 2 * np.pi * k / bearings
            u = np.array([np.cos(phi), np.sin(phi)])
            normal = np.array([-u[1], u[0]])
            for side in robot.arms:
                lateral = robot.profile["arms"][side]["lateral_m"]   # this arm's shoulder in line with the part
                key = (round(target[0], 3), round(target[1], 3), k, side)
                if key not in distances:
                    distances[key] = nearest_free(occupancy, target[:2], u, normal, lateral, phi, half)
                r0 = distances[key]
                if r0 is None:
                    reasons["footprint"] += 1
                    continue
                for dr in extra:
                    r = r0 + dr
                    score = (r + (0.0 if preferred is None else 0.6 * (1 - np.cos(phi - preferred)))
                             + RANK_PENALTY * frame.rank + VARIANT_PENALTY * frame.variant + 0.01 * index / len(frames))
                    options.append((score, index, r, phi, side, lateral))
    for _, index, r, phi, side, lateral in sorted(options, key=lambda o: o[0]):
        frame = frames[index]
        target = frame.position
        u = np.array([np.cos(phi), np.sin(phi)])
        normal = np.array([-u[1], u[0]])
        centre = target[:2] - r * u - lateral * normal
        base = np.array([*centre, phi])
        for offset in lift_offsets:
            lift = float(np.clip(target[2] + offset - shoulder0, *lift_range))
            shoulder = np.array([*(target[:2] - (r - SHOULDER_FORWARD) * u), shoulder0 + lift])
            if np.linalg.norm(target - shoulder) > reach + 0.15:
                reasons["reach"] += 1
                continue
            tried += 1
            if tried > max_candidates:
                return None, {"tried": tried, "reasons": reasons, "blocking": _top(blocking)}
            for posture in ("travel", "stow"):
                plan = _solve(session, kinematics, worlds, side, base, lift, frame, pre_distance, allowed, reasons, blocking,
                              posture, hand_state, repairs)
                if plan is not None:
                    q_pre, q_goal, lift_used, q_other, pre_used = plan
                    return Dock(base, lift_used, side, frame, q_pre, q_goal, tried, posture, q_other, pre_used), \
                        {"tried": tried, "reasons": reasons}
                if not reasons.pop("_other_arm", False):
                    break                     # only retry when the other arm was what blocked
    return None, {"tried": tried, "reasons": reasons, "blocking": _top(blocking)}


def _top(counter, n=5):
    return dict(sorted(counter.items(), key=lambda kv: -kv[1][0])[:n])


def _solve(session, kinematics, worlds, side, base, lift, frame, pre_distance, allowed, reasons, blocking,
           posture="travel", hand_state=None, repairs=None):
    robot, kin = session.robot, kinematics
    world = worlds.get(side)
    if world is None:
        world = worlds[side] = CollisionWorld(session.model, robot, robot.arms[side].qpos, margin=FREE_MARGIN)
        world.sync(session.data)
        worlds[side + ":body"] = WholeBody(kin, world, robot, side)
    body = worlds[side + ":body"]
    other = [o for o in robot.arms if o != side][0]
    state = {"lift": lift, "other": np.array(robot.profile["arms"][other][posture], float)}
    addresses = np.concatenate([robot.base_qpos, [robot.lift_qpos], robot.arms[other].qpos])
    world.set_joints(addresses, np.concatenate([robot.base_joint_values(base), [lift], state["other"]]))
    if hand_state is not None:
        world.set_joints(*hand_state(side, frame))
    world.select()
    pre = frame.offset(pre_distance)
    ik = kin.ik[side]
    other_prefix = other + "_"

    def record(hit, stage, solution):
        reasons[stage + "_collision"] += 1
        if hit.get("kind") == "self" and any(other_prefix in g for g in hit.get("geoms", [])):
            reasons["_other_arm"] = True
        closest = reasons.setdefault("_closest", {"distance_m": -1e9})
        if hit["distance_m"] > closest["distance_m"]:
            reasons["_closest"] = {"distance_m": hit["distance_m"], "stage": stage, "side": side, "base": base.tolist(),
                                   "lift": state["lift"], "q": solution.tolist(), "geoms": hit.get("geoms")}
        name = "/".join(hit.get("geoms", ["self"]))
        count, closest = blocking.get(name, (0, 1.0))
        blocking[name] = (count + 1, min(closest, round(hit["distance_m"], 4)))

    def solve(target, stage, free, **kw):
        q_base = kin.qpos(kin.robot.base_joint_values(base), state["lift"], {})
        try:
            solution = ik.solve(q_base, target, **kw).q
        except IKFailure:
            reasons[stage + "_ik"] += 1
            return None
        hit = world.blocking(solution)
        if hit is None:
            return solution
        record(hit, stage, solution)
        if repairs is None or repairs[0] <= 0:
            return None
        # Push every link out of collision along the distance gradient, keeping the TCP on target.
        repairs[0] -= 1
        fixed = body.repair(base, state["lift"], solution, state["other"], target, free=free)
        if fixed is None:
            world.set_joints(addresses, np.concatenate([robot.base_joint_values(base), [state["lift"]], state["other"]]))
            return None
        q, state["other"], state["lift"] = fixed
        world.set_joints(addresses, np.concatenate([robot.base_joint_values(base), [state["lift"]], state["other"]]))
        if world.blocking(q) is not None:
            return None
        reasons["repaired_" + stage] = reasons.get("repaired_" + stage, 0) + 1
        return q

    world.allow(())
    pre_used = pre_distance
    q_pre = solve(Target(pre.position, pre.rotation), "pre", ("arm", "other", "lift"), restarts=5)
    if q_pre is None and pre_distance > SHORT_PRE:
        # In tight places (inside an enclosure, beside a low box) back off less before the grasp.
        pre, pre_used = frame.offset(SHORT_PRE), SHORT_PRE
        q_pre = solve(Target(pre.position, pre.rotation), "pre", ("arm", "other", "lift"), restarts=3)
    if q_pre is None:
        return None
    world.allow(allowed)
    world.margin, margin = GOAL_MARGIN, world.margin
    q_goal = solve(Target(frame.position, frame.rotation, position_tolerance=1e-3), "goal", ("arm",), seed=q_pre, restarts=1)
    if q_goal is not None and np.isfinite(frame.width) and frame.width > 0:
        # The fingers also sweep inwards when they close: check them closed onto the part.
        fingers = session.model.jnt_qposadr[[session.model.joint(robot.prefix + j).id
                                             for j in robot.profile["arms"][side]["gripper"]["joints"]]]
        world.set_joints(fingers, [max(frame.width / 2 - 0.001, 0.0)] * len(fingers))
        hit = world.blocking(q_goal)
        world.set_joints(*hand_state(side, frame))
        if hit is not None:
            reasons["closing_collision"] = reasons.get("closing_collision", 0) + 1
            name = "/".join(hit.get("geoms", ["self"]))
            count, closest = blocking.get(name, (0, 1.0))
            blocking[name] = (count + 1, min(closest, round(hit["distance_m"], 4)))
            q_goal = None
    if q_goal is not None and not world.edge_valid(q_pre, q_goal):
        reasons["approach"] += 1
        q_goal = None
    world.allow(())
    world.margin = margin
    if hand_state is not None:
        world.set_joints(*_open_hand(session, side))
    if q_goal is None:
        return None
    return q_pre, q_goal, state["lift"], state["other"], pre_used


def _preshaped(session, side, frame):
    robot, arm = session.robot, session.robot.arms[side]
    addresses, values = _open_hand(session, side)
    if frame.opening is not None:
        values[:-1 if arm.pin_qpos >= 0 else None] = frame.opening
    return addresses, values


def _open_hand(session, side):
    """Tool joints back to their synced state: fingers open, press pin retracted."""
    robot, arm = session.robot, session.robot.arms[side]
    fingers = [int(session.model.jnt_qposadr[session.model.joint(robot.prefix + j).id])
               for j in robot.profile["arms"][side]["gripper"]["joints"]]
    addresses, values = fingers, [arm.gripper_open] * len(fingers)
    if arm.pin_qpos >= 0:
        addresses, values = addresses + [arm.pin_qpos], values + [0.0]
    return np.array(addresses), np.array(values)
