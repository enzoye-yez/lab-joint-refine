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
from .grasp import contact_frames, enclosure, grasp_frames, part_geoms, suction_frames
from .hand import HandProbe

PRE_DISTANCE = 0.10        # m behind the grasp/contact frame
RETREAT = 0.12             # m


@dataclass
class Context:
    session: object
    kinematics: object
    occupancy: object
    log: list
    keep_closest: bool = False
    probes: dict = None
    opened: set = None        # joints a prerequisite already opened: no longer count as blockers
    moved: set = None         # bodies a prerequisite moved on purpose (set aside)

    def probe(self, side=None):
        """Hand geometry for grasp synthesis (both hands are built alike)."""
        side = side or next(iter(self.session.robot.arms))
        if self.probes is None:
            self.probes = {}
        if side not in self.probes:
            self.probes[side] = HandProbe(self.session.model, self.session.robot, self.kinematics, side)
        return self.probes[side]


def _note(ctx, step, **detail):
    ctx.log.append({"t": round(ctx.session.time, 3), "step": step, **detail})


def check_enclosure(ctx, interface, hand_radius=0.035):
    """Abort with the blocking movable part (lid, door) if no straight way in to the site is free."""
    s = ctx.session
    own = subtree_bodies(s.model, interface.mechanism.body if interface.mechanism.body >= 0 else interface.body)
    from ..scene.interfaces import mechanism
    opened = ctx.opened or set()
    payload = s.robot.profile["arms"]["left"]["payload_kg"]

    def movable(g):
        mech = mechanism(s.model, int(s.model.geom_bodyid[g]), s.data.eq_active)
        if mech.kind in ("slide", "hinge"):
            return mech.joint not in opened
        return mech.kind == "free" and mech.mass_kg <= payload        # a loose cover: set it aside
    blocker = enclosure(s.model, s.data, interface, own | subtree_bodies(s.model, interface.body), movable=movable,
                        hand_radius=hand_radius)
    if blocker is None:
        return
    mech = mechanism(s.model, int(s.model.geom_bodyid[blocker]), s.data.eq_active)
    if not movable(blocker):
        # Bounding boxes of fixed meshes are conservative: let the exact hand check decide.
        _note(ctx, "enclosure", fixed=s.model.geom(blocker).name)
        return
    raise Abort("enclosed", geom=s.model.geom(blocker).name, movable=mech.kind, joint=s.model.joint(mech.joint).name,
                body=s.model.body(mech.body).name)


def no_frames(ctx, interface, reason, report):
    """No grasp or press pose: if a movable part (a lid, a door) blocks most candidates, open it first."""
    from ..scene.interfaces import mechanism
    s = ctx.session
    own = subtree_bodies(s.model, interface.mechanism.body if interface.mechanism.body >= 0 else interface.body)
    payload = s.robot.profile["arms"]["left"]["payload_kg"]
    for name in report.get("blocking", {}):
        g = s.model.geom(name).id
        if int(s.model.geom_bodyid[g]) in own:
            continue
        mech = mechanism(s.model, int(s.model.geom_bodyid[g]), s.data.eq_active)
        if mech.kind in ("slide", "hinge") and mech.joint != interface.mechanism.joint and mech.joint not in (ctx.opened or set()):
            raise Abort("enclosed", geom=name, movable=mech.kind, joint=s.model.joint(mech.joint).name, via=reason,
                        body=s.model.body(mech.body).name)
        if mech.kind == "free" and mech.mass_kg <= payload and mech.body not in (ctx.moved or set()):
            raise Abort("enclosed", geom=name, movable="free", joint=s.model.joint(mech.joint).name, via=reason,
                        body=s.model.body(mech.body).name)
        break
    raise Abort(reason, **report)


def _approach(ctx, interface, frames, allowed, hand_state=None, prepare=None, path=None):
    s = ctx.session
    dock, search = find_dock(s, ctx.kinematics, ctx.occupancy, frames, PRE_DISTANCE, allowed, hand_state=hand_state,
                             path=path)
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
    for side in s.robot.arms:
        if side != dock.side and np.max(np.abs(dock.q_other - motion.travel_posture(s.robot, side))) > 1e-3:
            motion.move_arm(s, side, dock.q_other)
    _replan_at_measured_base(ctx, dock, allowed, hand_state)
    if prepare is not None:
        prepare(dock)                 # tool state for the operation (pin out) before the arm moves in
    elif dock.frame.opening is not None:
        motion.gripper(s, dock.side, dock.frame.opening, seconds=0.6)      # pre-shape the fingers
    motion.move_arm(s, dock.side, dock.q_pre)
    _note(ctx, "pre_pose", side=dock.side, repaired={k: v for k, v in search["reasons"].items() if k.startswith("repaired")})
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


def _replan_at_measured_base(ctx, dock, allowed, hand_state=None):
    """Re-solve the arm postures for where the base and lift actually stopped (mm-level parking error
    becomes cm at the hand), keeping the planned solutions as seeds."""
    from ..planning.ik import IKFailure, Target
    from .dock import FREE_MARGIN, GOAL_MARGIN
    s, kin = ctx.session, ctx.kinematics
    base = s.robot.base_pose(s.data.qpos)
    lift = float(s.data.qpos[s.robot.lift_qpos])
    q_base = kin.qpos(kin.robot.base_joint_values(base), lift, {})
    world = motion.arm_world(s, dock.side, margin=FREE_MARGIN)
    from .dock import _preshaped
    world.set_joints(*(hand_state or (lambda side, frame: _preshaped(s, side, frame)))(dock.side, dock.frame))
    from ..planning.wholebody import WholeBody
    body = WholeBody(kin, world, s.robot, dock.side)
    other = [o for o in s.robot.arms if o != dock.side][0]
    q_other = s.data.qpos[s.robot.arms[other].qpos].copy()

    def solve(target, seed, restarts):
        # The continuous solution from the planned one; if the parking error makes it collide, repair the arm.
        try:
            return kin.ik[dock.side].solve(q_base, target, seed=seed, valid=world.valid, restarts=restarts).q
        except IKFailure as exc:
            try:
                q = kin.ik[dock.side].solve(q_base, target, seed=seed, restarts=3).q
            except IKFailure:
                raise Abort("replan_at_dock", base=base.round(4).tolist(), residual_mm=round(exc.best.position_error * 1e3, 2))
            fixed = body.repair(base, lift, q, q_other, target, free=("arm",))
            if fixed is None:
                raise Abort("replan_at_dock", base=base.round(4).tolist(), residual_mm=round(exc.best.position_error * 1e3, 2),
                            blocking=world.blocking(q), lift_error_m=round(lift - dock.lift, 4),
                            base_error_m=round(float(np.linalg.norm(base[:2] - dock.base[:2])), 4),
                            joint_change=round(float(np.max(np.abs(q - seed))), 3))
            return fixed[0]

    pre = dock.frame.offset(dock.pre_distance)
    q_pre = solve(Target(pre.position, pre.rotation), dock.q_pre, 2)
    world.allow(allowed)
    world.margin = GOAL_MARGIN
    q_goal = solve(Target(dock.frame.position, dock.frame.rotation, position_tolerance=5e-4), dock.q_goal, 1)
    _note(ctx, "replanned", base_error_mm=round(float(np.linalg.norm(base[:2] - dock.base[:2])) * 1e3, 2),
          yaw_error_rad=round(float(base[2] - dock.base[2]), 4), joint_change=round(float(np.max(np.abs(q_goal - dock.q_goal))), 3))
    dock.q_pre, dock.q_goal = q_pre, q_goal


def _part_geoms(model, body, data=None, centre=None, radius=0.015):
    """The part's geoms the hand may touch: only those near the interface, not a whole door or lid."""
    geoms = body_geoms(model, subtree_bodies(model, body))
    if data is None or centre is None:
        return set(geoms.tolist())
    near = set()
    for g in geoms:
        rot = data.geom_xmat[g].reshape(3, 3)
        local = rot.T @ (centre - data.geom_xpos[g]) - model.geom_aabb[g, :3]
        if np.linalg.norm(np.maximum(np.abs(local) - model.geom_aabb[g, 3:], 0)) <= radius:
            near.add(int(g))
    return near


def _site(s, interface):
    return s.data.site_xpos[interface.site].copy() if interface.site >= 0 else s.data.xpos[interface.body].copy()


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
    """Back the hand out along its tool axis as far as the arm stays clear of everything (checked)."""
    from ..planning.ik import IKFailure, Target
    s, arm, kin = ctx.session, ctx.session.robot.arms[dock.side], ctx.kinematics
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    here = s.data.qpos[arm.qpos].copy()
    base = s.robot.base_pose(s.data.qpos)
    q_base = kin.qpos(kin.robot.base_joint_values(base), float(s.data.qpos[s.robot.lift_qpos]), {})
    world = motion.arm_world(s, dock.side, margin=0.003)
    world.allow(s.allowed_contact)                 # the part just released may still touch the pads
    axis = rotation @ np.asarray(getattr(getattr(dock, "frame", None), "axis", (0.0, 0.0, 1.0)))
    for distance in (RETREAT, RETREAT / 2, 0.03):
        back = s.data.site_xpos[arm.tcp] - axis * distance
        try:
            q = kin.ik[dock.side].solve(q_base, Target(back, rotation, position_tolerance=2e-3), seed=here, restarts=0).q
        except IKFailure:
            continue
        if world.edge_valid(here, q):
            motion.move_tcp(s, dock.side, back, rotation, speed=0.08)
            break
    s.use_joint(dock.side)


def operation_path(ctx, interface, samples=10, limit=None):
    """For the dock search: the TCP poses that stay rigid on the part while its joint goes to its goal."""
    s, m = ctx.session, ctx.session.model
    joint = interface.mechanism.joint
    low, high = m.jnt_range[joint]
    now = s.data.qpos[m.jnt_qposadr[joint]]
    far = high if high - now >= now - low else low
    target = now + (far - now) * DRIVE_FRACTION
    if limit is not None:
        target = now + np.sign(target - now) * min(abs(target - now), limit)
    return lambda frame: mechanism_poses(m, s.data, interface, joint, target, frame.position, frame.rotation, samples)


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
    check_enclosure(ctx, interface)
    frames, report = grasp_frames(m, s.data, interface, ctx.probe(), s.robot.geoms)
    _note(ctx, "grasp_synthesis", **report)
    if not frames:
        # Nothing to hold on to (a tab flush with a wall, a lid edge): push the part along its motion.
        try:
            return push(ctx, interface, task)
        except Abort as exc:
            if exc.reason not in ("no_push_pose",):
                raise
            report["push"] = {k: v for k, v in exc.detail.items() if k != "blocking"}
    if not frames:
        no_frames(ctx, interface, "no_grasp", report)
    moving = set(part_geoms(m, s.data, interface, _site(s, interface)).tolist())
    s.allowed_contact = moving
    hinge = interface.mechanism.kind == "hinge"
    dock = _approach(ctx, interface, frames, moving, path=operation_path(ctx, interface, limit=TURN_PER_GRASP if hinge else None))
    arm = s.robot.arms[dock.side]
    _final_approach(ctx, dock)
    motion.gripper(s, dock.side, 0.0, force=40.0)
    touching, forces = _pads_touch(ctx, dock.side, moving)
    _note(ctx, "grasp", pad_force_n=[round(f, 2) for f in forces])
    if not touching:
        raise Abort("grasp_missed", pad_force_n=forces)
    release_events(ctx, task, "grasped")
    joint = interface.mechanism.joint
    low, high = m.jnt_range[joint]
    now = s.data.qpos[m.jnt_qposadr[joint]]
    far = high if high - now >= now - low else low
    # Drive close to the end stop so the part stays where it is left (a lid past vertical, a door open).
    target = now + (far - now) * DRIVE_FRACTION
    done = lambda: s.monitor.evaluate(task.goal, task.key) is True
    hinge = interface.mechanism.kind == "hinge"
    address = m.jnt_qposadr[joint]
    reached = False
    cartesian = s.cartesian[dock.side]
    cartesian.kp[:3] = FOLLOW_STIFFNESS
    cartesian.kd[:3] = 2 * np.sqrt(FOLLOW_STIFFNESS * 3.0)
    for cycle in range(MAX_REGRASPS + 1):
        now = s.data.qpos[address]
        stroke = target - now
        if hinge:                                    # one wrist turn at a time, then regrasp
            stroke = np.sign(stroke) * min(abs(stroke), TURN_PER_GRASP)
        tcp_pos, tcp_rot = s.data.site_xpos[arm.tcp].copy(), s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
        steps = max(4, int(abs(stroke) / (0.05 if hinge else 0.01)))
        poses = mechanism_poses(m, s.data, interface, joint, now + stroke, tcp_pos, tcp_rot, steps)
        reached = motion.follow(s, dock.side, poses, speed=0.05, until=done)
        _note(ctx, "drive_mechanism", cycle=cycle, reached=reached, detail=s.monitor.details.get(task.key))
        if reached or not hinge or abs(target - s.data.qpos[address]) < 0.05:
            break
        _regrasp(ctx, dock, interface, moving)
    release_events(ctx, task, "driven")
    if not reached:
        s.wait(1.0, done)
    motion.gripper(s, dock.side, arm.gripper_open, seconds=0.8)
    _retreat(ctx, dock)


DRIVE_FRACTION = 0.97                      # of the remaining joint range
FOLLOW_STIFFNESS = 2000.0                  # N/m while driving a mechanism
MAX_REGRASPS, TURN_PER_GRASP = 6, 2.0        # rad of mechanism rotation per grasp (FR3 wrist: +-2.9 rad)


def _regrasp(ctx, dock, interface, allowed):
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
    for frame in grasp_frames(s.model, s.data, interface, ctx.probe(), s.robot.geoms)[0]:
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
    check_enclosure(ctx, interface)
    frames, report = grasp_frames(m, s.data, interface, ctx.probe(), s.robot.geoms)
    _note(ctx, "grasp_synthesis", **report)
    suction = False
    if not frames:
        # Nothing to hold between the fingers (a flat cover, a plate): lift it by suction if it is light.
        cup = s.robot.arms[next(iter(s.robot.arms))].suction
        if cup is not None and interface.mechanism.mass_kg <= cup["max_mass_kg"]:
            frames, suction_report = suction_frames(m, s.data, interface, ctx.probe(), s.robot.geoms, cup)
            _note(ctx, "suction_synthesis", **suction_report)
            report["suction"] = {k: v for k, v in suction_report.items() if k != "blocking"}
            suction = bool(frames)
    if not frames:
        no_frames(ctx, interface, "no_grasp", report)
    root = interface.mechanism.body
    part = set(part_geoms(m, s.data, interface, _site(s, interface)).tolist())
    if suction:
        part = set(body_geoms(m, subtree_bodies(m, root)).tolist())
    s.allowed_contact = part
    hand = _closed_hand(s) if suction else None
    dock = _approach(ctx, interface, frames, part, hand_state=hand,
                     prepare=(lambda d: motion.gripper(s, d.side, 0.0, force=40.0, seconds=0.6)) if suction else None)
    dock.suction = suction
    arm = s.robot.arms[dock.side]
    _final_approach(ctx, dock)
    if suction:
        rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
        normal = rotation[:, 0]                                  # tool x = the face's outward normal
        motion.move_tcp(s, dock.side, s.data.site_xpos[arm.tcp] - normal * 0.002, rotation, speed=0.01)
        suction_on(s, dock.side, root)
        _note(ctx, "suction_on", body=m.body(root).name)
    else:
        motion.gripper(s, dock.side, 0.0, force=30.0)
        touching, forces = _pads_touch(ctx, dock.side, part)
        _note(ctx, "grasp", pad_force_n=[round(f, 2) for f in forces])
        if not touching:
            raise Abort("grasp_missed", pad_force_n=forces)
    release_events(ctx, task, "grasped")                    # e.g. a retention clip lets go once the object is held
    s.monitor.evaluate(task.goal, task.key)                   # anchors the lift height at the grasp
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    up = s.data.site_xpos[arm.tcp] + [0, 0, task.parameters.get("lift_m", 0.08)]
    motion.move_tcp(s, dock.side, up, rotation, speed=0.04)
    s.wait(1.0, lambda: s.monitor.evaluate(task.goal, task.key) is True)
    return dock


def suction_on(s, side, body):
    """Hold `body` on the suction cup: a weld from the cup to the body at their current relative pose."""
    m, d, cup = s.model, s.data, s.robot.arms[side].suction
    e, own = cup["equality"], cup["body"]
    rotation = d.xmat[own].reshape(3, 3)
    m.eq_obj2id[e] = body
    m.eq_data[e, :3] = 0.0
    m.eq_data[e, 3:6] = rotation.T @ (d.xpos[body] - d.xpos[own])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rotation.T @ d.xmat[body].reshape(3, 3)).flatten())
    m.eq_data[e, 6:10] = quat
    m.eq_data[e, 10] = 1.0
    d.eq_active[e] = 1


def suction_off(s, side):
    s.data.eq_active[s.robot.arms[side].suction["equality"]] = 0


def _closed_hand(s):
    def state(side, frame=None):
        addresses, _ = _hand_joints(s, side)
        return np.array(addresses), np.zeros(len(addresses))
    return state


def press(ctx, interface, task):
    """Push the part along its approach axis with the extended press pin (fingers closed beside it)."""
    s, m = ctx.session, ctx.session.model
    check_enclosure(ctx, interface, hand_radius=0.012)     # only the pin goes in
    probe = ctx.probe()
    side0 = next(iter(s.robot.arms))
    frames, report = contact_frames(m, s.data, interface, probe, s.robot.geoms, s.robot.arms[side0].pin_tip)
    _note(ctx, "press_synthesis", **report)
    if not frames:
        no_frames(ctx, interface, "no_press_pose", report)
    part = set(part_geoms(m, s.data, interface, _site(s, interface), radius=0.01).tolist())
    s.allowed_contact = part

    dock = _approach(ctx, interface, frames, part, hand_state=_pin_hand(s), prepare=_pin_deploy(s))
    frame = dock.frame
    _final_approach(ctx, dock)
    done = lambda: s.monitor.evaluate(task.goal, task.key) is True
    s.monitor.evaluate(task.goal, task.key)                   # anchor the joint's starting position
    depth = 0.03
    motion.move_tcp(s, dock.side, frame.position + frame.rotation[:, 2] * depth, frame.rotation, speed=0.02, until=done)
    _note(ctx, "press", reached=done(), detail=s.monitor.details.get(task.key))
    _retreat(ctx, dock)
    s.data.ctrl[s.robot.arms[dock.side].pin_actuator] = 0.0
    s.wait(0.5)


def _element(arm, element):
    return (arm.pin_qpos, arm.pin_actuator, arm.pin_extended) if element == "pin" else \
        (arm.blade_qpos, arm.blade_actuator, arm.blade_extended)


def _pin_hand(s, element="pin"):
    """Tool state while using the press pin (or the pry blade): fingers closed beside it, it extended."""
    def state(side, frame=None):
        qpos, _, extended = _element(s.robot.arms[side], element)
        addresses, _ = _hand_joints(s, side)
        return np.array(addresses + [qpos]), np.array([0.0] * len(addresses) + [extended])
    return state


def _pin_deploy(s, element="pin"):
    def deploy(dock):
        qpos, actuator, extended = _element(s.robot.arms[dock.side], element)
        motion.gripper(s, dock.side, 0.0, force=40.0, seconds=0.6)
        s.data.ctrl[actuator] = extended
        s.wait(1.0, lambda: abs(s.data.qpos[qpos] - extended) < 5e-4)
    return deploy


PUSH_LEAD = 0.003           # m, the pin set-point runs this far ahead of the contact along the motion
MAX_PUSH_STEPS = 80
PUSH_INSERT = 0.0015        # m, along the tool's way in
PUSH_STIFFNESS = 5000.0     # N/m: the contact overlap is a few millimetres


def push(ctx, interface, task):
    """Drive a 1-DoF mechanism by pushing it with the side of the press pin along its own motion."""
    from .grasp import push_frames
    s, m = ctx.session, ctx.session.model
    joint = interface.mechanism.joint
    address = m.jnt_qposadr[joint]
    low, high = m.jnt_range[joint]
    now = s.data.qpos[address]
    far = high if high - now >= now - low else low
    target = now + (far - now) * DRIVE_FRACTION
    site = _site(s, interface)
    first = mechanism_poses(m, s.data, interface, joint, now + (target - now) * 0.02, site, np.eye(3), 1)[0][0]
    if np.linalg.norm(first - site) < 1e-6:
        raise Abort("no_push_pose", why="the site does not move with the joint")
    side0 = next(iter(s.robot.arms))
    report = {}
    for element in ("pin", "blade"):
        arm0 = s.robot.arms[side0]
        tip = arm0.pin_tip if element == "pin" else arm0.blade_tip
        if tip is None:
            continue
        # The blade goes only under an overhanging edge: try holding its front edge back from the site.
        depths = (0.0,) if element == "pin" else (0.001, 0.002, 0.004)
        frames, found = push_frames(m, s.data, interface, ctx.probe(), s.robot.geoms, tip, first - site,
                                    element=element, depths=depths)
        _note(ctx, "push_synthesis", element=element, **found)
        report[element] = {k: v for k, v in found.items() if k != "blocking"}
        report.setdefault("blocking", {}).update(found["blocking"])
        if frames:
            pin_tip = tip
            break
    else:
        raise Abort("no_push_pose", **report)
    part = set(part_geoms(m, s.data, interface, site, radius=0.015).tolist())
    s.allowed_contact = part
    dock = _approach(ctx, interface, frames, part, hand_state=_pin_hand(s, element), prepare=_pin_deploy(s, element),
                     path=operation_path(ctx, interface))
    _final_approach(ctx, dock)
    release_events(ctx, task, "grasped")          # the pin is at the part: its declared release (unlatch)
    arm = s.robot.arms[dock.side]
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    offset = rotation @ pin_tip
    done = lambda: s.monitor.evaluate(task.goal, task.key) is True
    s.monitor.evaluate(task.goal, task.key)
    cartesian = s.cartesian[dock.side]
    cartesian.kp[:3] = PUSH_STIFFNESS
    cartesian.kd[:3] = 2 * np.sqrt(PUSH_STIFFNESS * 3.0)
    # Closed loop: from wherever the part is now, push the contact point a little further along the
    # part's own motion at that point, so the tool stays on the part even when it lags or speeds up.
    step = 0.05 if interface.mechanism.kind == "hinge" else 0.005
    reached, stalled, best = False, 0, s.data.qpos[address]
    for _ in range(MAX_PUSH_STEPS):
        now = s.data.qpos[address]
        if (target - now) * np.sign(target - best + 1e-12) <= 0:
            break
        # The tool keeps its pose relative to the part (it turns with a lid), pushing slightly ahead.
        tcp_pos, tcp_rot = s.data.site_xpos[arm.tcp].copy(), s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
        tip = tcp_pos + tcp_rot @ pin_tip
        ahead = now + np.sign(target - now) * min(step, abs(target - now))
        (moved_tcp, moved_rot), = mechanism_poses(m, s.data, interface, joint, ahead, tcp_pos, tcp_rot, 1)
        moved_tip = moved_tcp + moved_rot @ pin_tip
        lead = (moved_tip - tip) / max(np.linalg.norm(moved_tip - tip), 1e-9) * PUSH_LEAD
        # A little further in along the way in, so a blade stays under the edge it lifts.
        motion.move_tcp(s, dock.side, moved_tcp + lead + moved_rot[:, 2] * PUSH_INSERT, moved_rot, speed=0.04,
                        settle=0.0, until=done)
        if done():
            reached = True
            break
        progress = (s.data.qpos[address] - best) * np.sign(target - best + 1e-12)
        if progress > 1e-3:
            best, stalled = s.data.qpos[address], 0
        else:
            stalled += 1
            if stalled >= 4:
                break
    _note(ctx, "push", reached=reached, detail=s.monitor.details.get(task.key))
    if not reached:
        s.wait(1.0, done)
    _retreat(ctx, dock)
    s.data.ctrl[arm.pin_actuator if element == "pin" else arm.blade_actuator] = 0.0
    s.wait(0.5)


def _hand_joints(s, side):
    return [int(s.model.jnt_qposadr[s.model.joint(s.robot.prefix + j).id])
            for j in s.robot.profile["arms"][side]["gripper"]["joints"]], None


def release_events(ctx, task, when):
    """Fire the declared mechanical release events of this task at the moment the skill reaches `when`.

    The asset declares which interface releases a connection (a clip, a latch, a seated cover); the
    controller sequences it once that interface is in hand, as the scene's approximations specify.
    """
    s = ctx.session
    for event in task.releases:
        if event["when"] != when or s.data.eq_active[event["equality"]] == 0:
            continue
        joint = event.get("joint")
        if joint is not None and event.get("at") is not None:
            value = s.data.qpos[s.model.jnt_qposadr[joint]]
            if abs(value) + 1e-6 < abs(event["at"]):
                continue
        s.data.eq_active[event["equality"]] = 0
        _note(ctx, "release_event", event=event["id"], when=when)


def set_aside(ctx, interface, task, distances=(0.15, 0.25), clearance=0.004):
    """Pick a loose blocker (a cover, a cap), carry it sideways to a free spot and put it down."""
    from .grasp import scratch
    s, m = ctx.session, ctx.session.model
    dock = pick(ctx, interface, task)
    side = dock.side
    arm = s.robot.arms[side]
    root = interface.mechanism.body
    own = np.flatnonzero(np.isin(m.geom_bodyid, list(subtree_bodies(m, root))))
    own = own[(m.geom_contype[own] != 0) | (m.geom_conaffinity[own] != 0)]
    address = m.jnt_qposadr[interface.mechanism.joint]
    world = scratch(m, s.data)
    start = world.qpos[address:address + 3].copy()
    lowered = -task.parameters.get("lift_m", 0.08) + 0.01
    exclude = set(s.robot.geoms.tolist()) | set(own.tolist())
    from .grasp import nearby
    choice = None
    fromto = np.zeros(6)
    for distance in distances:
        for k in range(8):
            phi = 2 * np.pi * k / 8
            offset = np.array([np.cos(phi) * distance, np.sin(phi) * distance, 0.0])
            ok = True
            for dz in (0.0, lowered):
                world.qpos[address:address + 3] = start + offset + [0, 0, dz]
                mujoco.mj_kinematics(m, world)
                centre = world.xpos[root]
                for g in own:
                    for h in nearby(m, world, world.geom_xpos[g], exclude, 0.05):
                        if mujoco.mj_geomDistance(m, world, int(g), int(h), clearance, fromto) < clearance:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                choice = offset
                break
        if choice is not None:
            break
    if choice is None:
        raise Abort("no_place_aside")
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    above = s.data.site_xpos[arm.tcp] + choice
    motion.move_tcp(s, side, above, rotation, speed=0.05)
    others = np.array(sorted(set(range(m.ngeom)) - exclude))
    touching = lambda: bool(len(s.monitor.contacts(own, others)))
    motion.move_tcp(s, side, above + [0, 0, lowered], rotation, speed=0.02, until=touching)
    if getattr(dock, "suction", False):
        suction_off(s, side)
        s.wait(0.5)
    else:
        motion.gripper(s, side, arm.gripper_open, seconds=0.8)
    if ctx.moved is None:
        ctx.moved = set()
    ctx.moved |= set(subtree_bodies(m, root))
    _note(ctx, "set_aside", offset=choice.round(3).tolist())
    _retreat(ctx, dock)


SKILLS = {"actuate": actuate, "rotate": actuate, "pick": pick, "press": press, "set_aside": set_aside}
