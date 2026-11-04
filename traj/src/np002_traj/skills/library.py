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
from .grasp import contact_frames, grasp_frames

PRE_DISTANCE = 0.10        # m behind the grasp/contact frame
RETREAT = 0.12             # m


@dataclass
class Context:
    session: object
    kinematics: object
    occupancy: object
    log: list


def _note(ctx, step, **detail):
    ctx.log.append({"t": round(ctx.session.time, 3), "step": step, **detail})


def _approach(ctx, interface, frames, allowed):
    s = ctx.session
    dock, search = find_dock(s, ctx.kinematics, ctx.occupancy, frames, PRE_DISTANCE, allowed)
    if dock is None:
        raise Abort("no_dock", frames=len(frames), **search)
    _note(ctx, "dock", base=dock.base.round(3).tolist(), side=dock.side, lift=round(dock.lift, 3), **search)
    for side in s.robot.arms:
        motion.move_arm(s, side, motion.travel_posture(s.robot, side))
    motion.drive(s, dock.base, ctx.occupancy)
    motion.set_lift(s, dock.lift)
    motion.move_arm(s, dock.side, dock.q_pre)
    _note(ctx, "pre_pose", side=dock.side)
    return dock


def _part_geoms(model, body):
    return set(body_geoms(model, subtree_bodies(model, body)).tolist())


def _pads_touch(ctx, side, geoms):
    arm = ctx.session.robot.arms[side]
    rows = [ctx.session.monitor.contacts(np.array([p]), np.array(sorted(geoms))) for p in arm.pads]
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
    width = 2 * s.robot.profile["arms"]["left"]["gripper"]["open"] - 0.006
    frames = grasp_frames(m, s.data, interface, width)
    if not frames:
        raise Abort("part_wider_than_gripper")
    moving = _part_geoms(m, interface.mechanism.body)
    s.allowed_contact = moving
    dock = _approach(ctx, interface, frames, moving)
    arm = s.robot.arms[dock.side]
    motion.gripper(s, dock.side, arm.gripper_open)
    motion.move_tcp(s, dock.side, dock.frame.position, dock.frame.rotation, speed=0.04)
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
    tcp_pos, tcp_rot = s.data.site_xpos[arm.tcp].copy(), s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    steps = max(4, int(abs(target - now) / (0.01 if interface.mechanism.kind == "slide" else 0.05)))
    poses = mechanism_poses(m, s.data, interface, joint, target, tcp_pos, tcp_rot, steps)
    done = lambda: s.monitor.evaluate(task.goal, "goal") is True
    reached = motion.follow(s, dock.side, poses, speed=0.05, until=done)
    _note(ctx, "drive_mechanism", reached=reached, detail=s.monitor.details.get("goal"))
    if not reached:
        s.wait(1.0, done)
    motion.gripper(s, dock.side, arm.gripper_open, seconds=0.8)
    _retreat(ctx, dock)


def pick(ctx, interface, task):
    s, m = ctx.session, ctx.session.model
    width = 2 * s.robot.profile["arms"]["left"]["gripper"]["open"] - 0.006
    frames = grasp_frames(m, s.data, interface, width)
    if not frames:
        raise Abort("part_wider_than_gripper")
    part = _part_geoms(m, interface.mechanism.body)
    s.allowed_contact = part
    dock = _approach(ctx, interface, frames, part)
    arm = s.robot.arms[dock.side]
    motion.gripper(s, dock.side, arm.gripper_open)
    motion.move_tcp(s, dock.side, dock.frame.position, dock.frame.rotation, speed=0.04)
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
    reach = 0.012                                            # fingertip ahead of the TCP (pad half-height)
    frames = contact_frames(m, s.data, interface, reach)
    part = _part_geoms(m, interface.mechanism.body if interface.mechanism.body >= 0 else interface.body)
    s.allowed_contact = part
    dock = _approach(ctx, interface, frames, part)
    motion.gripper(s, dock.side, 0.0, force=40.0, seconds=0.6)
    arm = s.robot.arms[dock.side]
    frame = dock.frame
    motion.move_tcp(s, dock.side, frame.position, frame.rotation, speed=0.03)
    done = lambda: s.monitor.evaluate(task.goal, "goal") is True
    s.monitor.evaluate(task.goal, "goal")                   # anchor the joint's starting position
    depth = 0.03
    motion.move_tcp(s, dock.side, frame.position + frame.rotation[:, 2] * depth, frame.rotation, speed=0.02, until=done)
    _note(ctx, "press", reached=done(), detail=s.monitor.details.get("goal"))
    _retreat(ctx, dock)


SKILLS = {"actuate": actuate, "rotate": actuate, "pick": pick, "press": press}
