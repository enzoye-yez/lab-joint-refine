"""Skills composed from motion primitives. The same code serves every interface of a given kind.

actuate: grasp a part of a 1-DoF mechanism and drive the joint; the TCP path is the part's own
         motion, computed by forward kinematics of the mechanism (works for slides and hinges).
pick:    grasp a loose object and lift it.
press:   push a part along its approach axis with the closed fingertips.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..executor.session import Abort
from ..geometry import body_geoms, subtree_bodies
from . import motion
from .dock import find_dock
from .grasp import contact_frames, enclosure, grasp_frames

PRE_DISTANCE = 0.10        # m behind the grasp/contact frame
RETREAT = 0.12             # m


@dataclass
class Context:
    session: object
    kinematics: object
    occupancy: object
    log: list
    keep_closest: bool = False


def _note(ctx, step, **detail):
    ctx.log.append({"t": round(ctx.session.time, 3), "step": step, **detail})


def _approach(ctx, interface, frames, allowed):
    s = ctx.session
    own = subtree_bodies(s.model, interface.mechanism.body if interface.mechanism.body >= 0 else interface.body)
    from ..scene.interfaces import mechanism
    movable = lambda g: mechanism(s.model, int(s.model.geom_bodyid[g])).kind in ("slide", "hinge")
    blocker = enclosure(s.model, s.data, interface, own | subtree_bodies(s.model, interface.body), movable=movable)
    if blocker is not None:
        mech = mechanism(s.model, int(s.model.geom_bodyid[blocker]))
        raise Abort("enclosed", geom=s.model.geom(blocker).name, movable=mech.kind,
                    joint=s.model.joint(mech.joint).name if mech.joint >= 0 else None)
    dock, search = find_dock(s, ctx.kinematics, ctx.occupancy, frames, PRE_DISTANCE, allowed)
    if dock is None:
        closest = search["reasons"].pop("_closest", None)
        if ctx.keep_closest and closest:
            search["closest"] = {k: v for k, v in closest.items() if k != "q"}
        raise Abort("no_dock", frames=len(frames), **search)
    search["reasons"].pop("_closest", None)
    _note(ctx, "dock", base=dock.base.round(3).tolist(), side=dock.side, lift=round(dock.lift, 3), **search)
    for side in s.robot.arms:
        motion.move_arm(s, side, motion.travel_posture(s.robot, side))
    motion.drive(s, dock.base, ctx.occupancy)
    motion.set_lift(s, dock.lift)
    _replan_at_measured_base(ctx, dock, allowed)
    motion.move_arm(s, dock.side, dock.q_pre)
    _note(ctx, "pre_pose", side=dock.side)
    return dock


def _final_approach(ctx, dock, speed=0.03):
    """Precise last centimetres: joint-space interpolation to the collision-checked IK solution."""
    s, arm = ctx.session, ctx.session.robot.arms[dock.side]
    start = s.data.qpos[arm.qpos].copy()
    distance = float(np.linalg.norm(s.data.site_xpos[arm.tcp] - dock.frame.position))
    steps = max(50, int(distance / speed / s.model.opt.timestep))
    for k in range(1, steps + 1):
        t = k / steps
        smooth = t * t * (3 - 2 * t)
        s.use_joint(dock.side, start + (dock.q_goal - start) * smooth)
        s.step()
    s.use_joint(dock.side, dock.q_goal)
    s.wait(0.5, lambda: np.max(np.abs(s.data.qvel[arm.dofs])) < 0.01)
    error = float(np.linalg.norm(s.data.site_xpos[arm.tcp] - dock.frame.position))
    _note(ctx, "at_grasp", tcp_error_mm=round(error * 1e3, 2))
    return error


def _replan_at_measured_base(ctx, dock, allowed):
    """Re-solve the arm postures for where the base and lift actually stopped (mm-level parking error
    becomes cm at the hand), keeping the planned solutions as seeds."""
    from ..planning.ik import IKFailure, Target
    from .dock import FREE_MARGIN, GOAL_MARGIN
    s, kin = ctx.session, ctx.kinematics
    base = s.robot.base_pose(s.data.qpos)
    lift = float(s.data.qpos[s.robot.lift_qpos])
    q_base = kin.qpos(kin.robot.base_joint_values(base), lift, {})
    world = motion.arm_world(s, dock.side, margin=FREE_MARGIN)
    pre = dock.frame.offset(PRE_DISTANCE)
    try:
        q_pre = kin.ik[dock.side].solve(q_base, Target(pre.position, pre.rotation), seed=dock.q_pre,
                                        valid=world.valid, restarts=2).q
        world.allow(allowed)
        world.margin = GOAL_MARGIN
        q_goal = kin.ik[dock.side].solve(q_base, Target(dock.frame.position, dock.frame.rotation, position_tolerance=5e-4),
                                         seed=dock.q_goal, valid=world.valid, restarts=1).q
    except IKFailure as exc:
        raise Abort("replan_at_dock", base=base.round(4).tolist(), residual_mm=round(exc.best.position_error * 1e3, 2))
    _note(ctx, "replanned", base_error_mm=round(float(np.linalg.norm(base[:2] - dock.base[:2])) * 1e3, 2),
          yaw_error_rad=round(float(base[2] - dock.base[2]), 4), joint_change=round(float(np.max(np.abs(q_goal - dock.q_goal))), 3))
    dock.q_pre, dock.q_goal = q_pre, q_goal


def _part_geoms(model, body):
    return set(body_geoms(model, subtree_bodies(model, body)).tolist())


def _pads_touch(ctx, side, geoms):
    s = ctx.session
    arm = s.robot.arms[side]
    rows = [s.monitor.contacts(np.array([p]), np.array(sorted(geoms))) for p in arm.pads]
    # What each finger actually touches (for failure summaries)
    fingers = [g for g in s.robot.geoms if s.model.geom_bodyid[g] in arm.fingers]
    others = np.array([g for g in range(s.model.ngeom) if g not in set(s.robot.geoms.tolist())])
    touched = sorted({s.model.geom(r[1]).name for r in s.monitor.contacts(np.array(fingers), others)})[:6]
    opening = float(s.data.qpos[s.model.jnt_qposadr[s.model.actuator_trnid[arm.gripper, 0]]])
    _note(ctx, "fingers", opening_mm=round(opening * 2e3, 2), touching=touched)
    return all(rows), [sum(r[2] for r in row) for row in rows]


def _retreat(ctx, dock):
    s, arm = ctx.session, ctx.session.robot.arms[dock.side]
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    back = s.data.site_xpos[arm.tcp] - rotation[:, 2] * RETREAT
    motion.move_tcp(s, dock.side, back, rotation, speed=0.08)
    s.use_joint(dock.side)


def mechanism_poses(model, data, interface, joint, target, tcp_position, tcp_rotation, steps):
    """TCP poses that keep the grasp fixed on the part while its joint goes from now to `target`."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, scratch)
    body = interface.body
    start_pos, start_rot = scratch.xpos[body].copy(), scratch.xmat[body].reshape(3, 3).copy()
    local_pos = start_rot.T @ (tcp_position - start_pos)
    local_rot = start_rot.T @ tcp_rotation
    address = model.jnt_qposadr[joint]
    poses = []
    for value in np.linspace(data.qpos[address], target, steps + 1)[1:]:
        scratch.qpos[address] = value
        mujoco.mj_kinematics(model, scratch)
        rot = scratch.xmat[body].reshape(3, 3)
        poses.append((scratch.xpos[body] + rot @ local_pos, rot @ local_rot))
    return poses


def actuate(ctx, interface, task):
    s, m = ctx.session, ctx.session.model
    width = 2 * s.robot.profile["arms"]["left"]["gripper"]["open"] - 0.004
    frames = grasp_frames(m, s.data, interface, width)
    if not frames:
        raise Abort("part_wider_than_gripper")
    moving = _part_geoms(m, interface.mechanism.body)
    s.allowed_contact = moving
    dock = _approach(ctx, interface, frames, moving)
    arm = s.robot.arms[dock.side]
    motion.gripper(s, dock.side, arm.gripper_open)
    _final_approach(ctx, dock)
    motion.gripper(s, dock.side, 0.0, force=40.0)
    touching, forces = _pads_touch(ctx, dock.side, moving)
    _note(ctx, "grasp", pad_force_n=[round(f, 2) for f in forces])
    if not touching:
        raise Abort("grasp_missed", pad_force_n=forces)
    joint = interface.mechanism.joint
    low, high = m.jnt_range[joint]
    now = s.data.qpos[m.jnt_qposadr[joint]]
    far = high if high - now >= now - low else low
    target = now + (far - now) * min(1.0, task.parameters.get("travel", 0.8) + 0.1)
    done = lambda: s.monitor.evaluate(task.goal, "goal") is True
    hinge = interface.mechanism.kind == "hinge"
    address = m.jnt_qposadr[joint]
    reached = False
    for cycle in range(MAX_REGRASPS + 1):
        now = s.data.qpos[address]
        stroke = target - now
        if hinge:                                    # one wrist turn at a time, then regrasp
            stroke = np.sign(stroke) * min(abs(stroke), TURN_PER_GRASP)
        tcp_pos, tcp_rot = s.data.site_xpos[arm.tcp].copy(), s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
        steps = max(4, int(abs(stroke) / (0.05 if hinge else 0.01)))
        poses = mechanism_poses(m, s.data, interface, joint, now + stroke, tcp_pos, tcp_rot, steps)
        reached = motion.follow(s, dock.side, poses, speed=0.05, until=done)
        _note(ctx, "drive_mechanism", cycle=cycle, reached=reached, detail=s.monitor.details.get("goal"))
        if reached or not hinge or abs(target - s.data.qpos[address]) < 0.05:
            break
        _regrasp(ctx, dock, interface, moving, width)
    if not reached:
        s.wait(1.0, done)
    motion.gripper(s, dock.side, arm.gripper_open, seconds=0.8)
    _retreat(ctx, dock)


MAX_REGRASPS, TURN_PER_GRASP = 6, 2.0        # rad of mechanism rotation per grasp (FR3 wrist: +-2.9 rad)


def _regrasp(ctx, dock, interface, allowed, width):
    """Release, back off and grasp the part again where it now is, with the wrist wound back."""
    from ..planning.ik import IKFailure, Target
    from .dock import FREE_MARGIN, GOAL_MARGIN
    s, kin, arm = ctx.session, ctx.kinematics, ctx.session.robot.arms[dock.side]
    motion.gripper(s, dock.side, arm.gripper_open, seconds=0.6)
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    motion.move_tcp(s, dock.side, s.data.site_xpos[arm.tcp] - rotation[:, 2] * 0.03, rotation, speed=0.05)
    base = s.robot.base_pose(s.data.qpos)
    q_base = kin.qpos(kin.robot.base_joint_values(base), float(s.data.qpos[s.robot.lift_qpos]), {})
    world = motion.arm_world(s, dock.side, margin=FREE_MARGIN)
    low, high = s.model.jnt_range[arm.joints[-1]]
    best = None
    for frame in grasp_frames(s.model, s.data, interface, width):
        try:
            world.allow(())
            world.margin = FREE_MARGIN
            pre = frame.offset(0.03)
            q_pre = kin.ik[dock.side].solve(q_base, Target(pre.position, pre.rotation), seed=s.data.qpos[arm.qpos],
                                            valid=world.valid, restarts=1).q
            world.allow(allowed)
            world.margin = GOAL_MARGIN
            q_goal = kin.ik[dock.side].solve(q_base, Target(frame.position, frame.rotation, position_tolerance=5e-4),
                                             seed=q_pre, valid=world.valid, restarts=0).q
        except IKFailure:
            continue
        room = min(q_goal[-1] - low, high - q_goal[-1])        # wrist travel left on the tighter side
        if best is None or room > best[0]:
            best = (room, frame, q_pre, q_goal)
    if best is None:
        raise Abort("regrasp_failed")
    _, dock.frame, dock.q_pre, dock.q_goal = best
    motion.move_arm(s, dock.side, dock.q_pre)
    _final_approach(ctx, dock)
    motion.gripper(s, dock.side, 0.0, force=40.0)
    touching, forces = _pads_touch(ctx, dock.side, allowed)
    if not touching:
        raise Abort("regrasp_missed", pad_force_n=forces)


def pick(ctx, interface, task):
    s, m = ctx.session, ctx.session.model
    width = 2 * s.robot.profile["arms"]["left"]["gripper"]["open"] - 0.004
    frames = grasp_frames(m, s.data, interface, width)
    if not frames:
        raise Abort("part_wider_than_gripper")
    part = _part_geoms(m, interface.mechanism.body)
    s.allowed_contact = part
    dock = _approach(ctx, interface, frames, part)
    arm = s.robot.arms[dock.side]
    motion.gripper(s, dock.side, arm.gripper_open)
    _final_approach(ctx, dock)
    motion.gripper(s, dock.side, 0.0, force=30.0)
    touching, forces = _pads_touch(ctx, dock.side, part)
    _note(ctx, "grasp", pad_force_n=[round(f, 2) for f in forces])
    if not touching:
        raise Abort("grasp_missed", pad_force_n=forces)
    s.monitor.evaluate(task.goal, "goal")                   # anchors the lift height at the grasp
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    up = s.data.site_xpos[arm.tcp] + [0, 0, task.parameters.get("lift_m", 0.08)]
    motion.move_tcp(s, dock.side, up, rotation, speed=0.04)
    s.wait(1.0, lambda: s.monitor.evaluate(task.goal, "goal") is True)


def press(ctx, interface, task):
    s, m = ctx.session, ctx.session.model
    reach = s.robot.profile["arms"]["left"]["gripper"]["fingertip_from_tcp_m"]
    frames = contact_frames(m, s.data, interface, reach)
    part = _part_geoms(m, interface.mechanism.body if interface.mechanism.body >= 0 else interface.body)
    s.allowed_contact = part
    dock = _approach(ctx, interface, frames, part)
    motion.gripper(s, dock.side, 0.0, force=40.0, seconds=0.6)
    arm = s.robot.arms[dock.side]
    frame = dock.frame
    _final_approach(ctx, dock)
    done = lambda: s.monitor.evaluate(task.goal, "goal") is True
    s.monitor.evaluate(task.goal, "goal")                   # anchor the joint's starting position
    depth = 0.03
    motion.move_tcp(s, dock.side, frame.position + frame.rotation[:, 2] * depth, frame.rotation, speed=0.02, until=done)
    _note(ctx, "press", reached=done(), detail=s.monitor.details.get("goal"))
    _retreat(ctx, dock)


SKILLS = {"actuate": actuate, "rotate": actuate, "pick": pick, "press": press}
