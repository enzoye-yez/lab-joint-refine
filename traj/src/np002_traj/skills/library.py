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
    holding: str = None       # arm holding the part up: the other arm works from where the robot stands
    tool_check: object = None  # set while a tool is carried: raises if it has left the hand

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


def _approach(ctx, interface, frames, allowed, hand_state=None, prepare=None, path=None, tool=None, sides=None):
    s = ctx.session
    if ctx.holding is not None:
        return _approach_held(ctx, frames, allowed, hand_state, prepare, path, tool)
    dock, search = find_dock(s, ctx.kinematics, ctx.occupancy, frames, PRE_DISTANCE, allowed, hand_state=hand_state,
                             path=path, tool=tool, sides=sides)
    if dock is None:
        closest = search["reasons"].pop("_closest", None)
        if ctx.keep_closest and closest:
            search["closest"] = {k: v for k, v in closest.items() if k != "q"}
        raise Abort("no_dock", frames=len(frames), **search)
    search["reasons"].pop("_closest", None)
    _note(ctx, "dock", base=dock.base.round(3).tolist(), side=dock.side, lift=round(dock.lift, 3), **search)
    check = getattr(ctx, "tool_check", None) or (lambda step: None)
    for side in s.robot.arms:
        _let_go(ctx, side, keep=sides)
        motion.move_arm(s, side, _carry_posture(s, side))
        check("fold_" + side)
    motion.drive(s, dock.base, ctx.occupancy)
    check("drive")
    motion.set_lift(s, dock.lift)
    check("lift")
    for side in s.robot.arms:
        if side != dock.side and np.max(np.abs(dock.q_other - motion.travel_posture(s.robot, side))) > 1e-3:
            motion.move_arm(s, side, dock.q_other)
    dock.tool = tool
    _replan_at_measured_base(ctx, dock, allowed, hand_state)
    if prepare is not None:
        prepare(dock)                 # tool state for the operation (pin out) before the arm moves in
    elif dock.frame.opening is not None:
        motion.gripper(s, dock.side, dock.frame.opening, seconds=0.6)      # pre-shape the fingers
    motion.move_arm(s, dock.side, dock.q_pre)
    _note(ctx, "pre_pose", side=dock.side, repaired={k: v for k, v in search["reasons"].items() if k.startswith("repaired")})
    return dock


def _approach_held(ctx, frames, allowed, hand_state, prepare, path, tool):
    """The free arm's approach while the other arm holds the part up: base, lift and holding arm stay."""
    from .dock import relay_dock
    s = ctx.session
    side = next(o for o in s.robot.arms if o != ctx.holding)
    dock, search = relay_dock(s, ctx.kinematics, frames, side, allowed, pre_distance=PRE_DISTANCE,
                              hand_state=hand_state, tool=tool, path=path)
    if dock is None:
        raise Abort("no_dock_held", frames=len(frames), **search)
    _note(ctx, "dock_held", side=side, **search)
    dock.tool = tool
    if prepare is not None:
        prepare(dock)
    elif dock.frame.opening is not None:
        motion.gripper(s, side, dock.frame.opening, seconds=0.6)
    motion.move_arm(s, side, dock.q_pre)
    _note(ctx, "pre_pose", side=side, held_by=ctx.holding)
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
    if getattr(dock, "tool", None) is not None:
        world.tool(*dock.tool(dock.side))
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


CARRY_FOLDS = (1.0, 0.75, 0.5, 0.25, 0.0)   # fractions of the way to the travel posture tried with a tool


def _carry_posture(s, side):
    """The travel posture, or, for the arm carrying a tool that would strike the robot folded fully, the
    most folded posture on the way there in which the tool is clear (checked with the tool along)."""
    travel = motion.travel_posture(s.robot, side)
    held = getattr(s, "carried_root", None)
    if held is None or held[1] != side:
        return travel
    world = motion.arm_world(s, side)
    here = s.data.qpos[s.robot.arms[side].qpos].copy()
    for fold in CARRY_FOLDS:
        q = here + (travel - here) * fold
        if world.valid(q):
            return q
    return here


LET_GO_BACK_M = 0.05      # m an idle hand backs out along its tool axis after opening


def _let_go(ctx, side, keep=None):
    """An arm left on a part by an earlier step (a lid it opened, a button it pressed) opens its fingers and
    backs out before the robot folds and drives; the arm holding a part up, or the tool arm, keeps it."""
    s, m, d = ctx.session, ctx.session.model, ctx.session.data
    if side == ctx.holding or (keep is not None and side in keep):
        return
    arm = s.robot.arms[side]
    own = set(body_geoms(m, subtree_bodies(m, int(m.jnt_bodyid[arm.joints[0]]))).tolist())
    robot = set(s.robot.geoms.tolist())
    touching = any((c.geom1 in own and c.geom2 not in robot) or (c.geom2 in own and c.geom1 not in robot)
                   for c in d.contact[:d.ncon])
    if not touching:
        return
    motion.gripper(s, side, arm.gripper_open, seconds=0.6)
    rotation = d.site_xmat[arm.tcp].reshape(3, 3).copy()
    motion.move_tcp(s, side, d.site_xpos[arm.tcp] - rotation[:, 2] * LET_GO_BACK_M, rotation, speed=0.08)
    s.use_joint(side)
    _note(ctx, "let_go", side=side)


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
    # A dock is first asked to follow a whole wrist turn (a hinge) or the whole stroke (a slide); when none
    # can, a shorter first stage is enough: the part is regrasped where it stopped and driven on.
    stages = DOCK_STAGES["hinge" if hinge else "slide"]
    for i, limit in enumerate(stages):
        try:
            dock = _approach(ctx, interface, frames, moving, path=operation_path(ctx, interface, limit=limit))
            break
        except Abort as exc:
            if exc.reason != "no_dock" or not exc.detail.get("reasons", {}).get("path") or i == len(stages) - 1:
                raise
            _note(ctx, "dock_stage", limit=limit, rejected_paths=exc.detail["reasons"]["path"])
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
    stalled = 0
    for cycle in range(MAX_DRIVE_CYCLES):
        now = s.data.qpos[address]
        stroke = target - now
        if hinge:                                    # one wrist turn at a time, then regrasp
            stroke = np.sign(stroke) * min(abs(stroke), TURN_PER_GRASP)
        tcp_pos, tcp_rot = s.data.site_xpos[arm.tcp].copy(), s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
        steps = max(4, int(abs(stroke) / (0.05 if hinge else 0.01)))
        poses = mechanism_poses(m, s.data, interface, joint, now + stroke, tcp_pos, tcp_rot, steps)
        reached = motion.follow(s, dock.side, poses, speed=0.05, until=done)
        _note(ctx, "drive_mechanism", cycle=cycle, reached=reached, detail=s.monitor.details.get(task.key))
        # A slide the arm could not follow to the end is regrasped where it stopped, like a hinge, if it moved.
        progress = abs(s.data.qpos[address] - now)
        # Regrasp and go on while each grasp still moves it; stop after two grasps that do not.
        stalled = stalled + 1 if progress < (STALL_HINGE if hinge else STALL_SLIDE) else 0
        if reached or abs(target - s.data.qpos[address]) < (0.05 if hinge else 0.005) or stalled >= 2 \
                or cycle == MAX_DRIVE_CYCLES - 1:
            break
        _regrasp(ctx, dock, interface, moving)
    release_events(ctx, task, "driven")
    if not reached:
        s.wait(1.0, done)
    motion.gripper(s, dock.side, arm.gripper_open, seconds=0.8)
    _retreat(ctx, dock)


DOCK_STAGES = {"hinge": (2.0, 1.0, 0.5), "slide": (None, 0.10, 0.04)}   # rad / m
DRIVE_FRACTION = 0.97                      # of the remaining joint range
FOLLOW_STIFFNESS = 2000.0                  # N/m while driving a mechanism
TURN_PER_GRASP = 2.0                     # rad of mechanism rotation per grasp (FR3 wrist: +-2.9 rad)
MAX_DRIVE_CYCLES = 16                        # grasps while driving one mechanism, as long as each makes progress
STALL_HINGE, STALL_SLIDE = 0.02, 0.002       # rad / m: a grasp that moves it less makes no progress


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
    for raise_m in GRASP_RAISES if not frames else ():
        # Neighbours (a rack, the next tube) close round the declared point: try the part higher up.
        frames, higher = grasp_frames(m, s.data, interface, ctx.probe(), s.robot.geoms, raise_m=raise_m)
        if frames:
            _note(ctx, "grasp_synthesis", raised_m=raise_m, **higher)
            break
    if not frames:
        no_frames(ctx, interface, "no_grasp", report)
    root = interface.mechanism.body
    part = set(part_geoms(m, s.data, interface, _site(s, interface)).tolist())
    if suction:
        part = set(body_geoms(m, subtree_bodies(m, root)).tolist())
    s.allowed_contact = part
    hand = _closed_hand(s) if suction else None
    lift = task.parameters.get("lift_m", 0.08)
    tried = []
    for attempt in range(1 if suction else GRASP_ATTEMPTS):
        candidates = [f for f in frames if not any(f is g for g in tried)]
        if not candidates:
            break
        dock = _approach(ctx, interface, candidates, part, hand_state=hand,
                         prepare=(lambda d: motion.gripper(s, d.side, 0.0, force=40.0, seconds=0.6)) if suction else None)
        tried.append(dock.frame)
        last = attempt == GRASP_ATTEMPTS - 1 or len(candidates) == 1
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
                if last:
                    raise Abort("grasp_missed", pad_force_n=forces)
                _let_go_part(ctx, dock)
                continue
        release_events(ctx, task, "grasped")                    # e.g. a retention clip lets go once the object is held
        s.monitor.evaluate(task.goal, task.key)                   # anchors the lift height at the grasp
        start = s.data.xpos[root].copy()
        extract(ctx, dock.side, root, np.array([0.0, 0.0, 1.0]), lift)
        rise = float(s.data.xpos[root][2] - start[2])
        if suction or last or rise >= GRASP_HELD_FRACTION * lift:
            break
        # It stayed behind (slipped out of a weak or lopsided grasp): put the hand back and try another grasp.
        _note(ctx, "grasp_slipped", rise_mm=round(rise * 1e3, 1), attempt=attempt)
        _let_go_part(ctx, dock)
    s.wait(1.0, lambda: s.monitor.evaluate(task.goal, task.key) is True)
    return dock


GRASP_ATTEMPTS = 3           # grasps tried on one part when the fingers miss it or it slips out
GRASP_HELD_FRACTION = 0.5    # of the lift the part must rise with the hand to count as held


def _let_go_part(ctx, dock):
    s = ctx.session
    motion.gripper(s, dock.side, s.robot.arms[dock.side].gripper_open, seconds=0.6)
    _retreat(ctx, dock)


GRASP_RAISES = (0.006, 0.012, 0.018, 0.024)   # m above the declared grasp point tried when it has no grasp
EXTRACT_STEP = 0.005        # m of travel between clearance measurements
EXTRACT_CLEAR = 0.004       # m the held object is kept from its neighbours, sideways
EXTRACT_SHIFT = 0.002       # m largest sideways correction per step
EXTRACT_MAX_SHIFT = 0.015   # m largest sideways correction in total


def extract(ctx, side, root, direction, distance, speed=0.04):
    """Move the held object `distance` along `direction` (out of a rack, off a deck), steering it
    sideways away from whatever it comes close to, so its walls do not drag neighbours along.

    Closed loop on the simulator's own geometry: at every step the signed distances from the held
    object's geoms to nearby geoms give a sideways correction (only the part across `direction`).
    """
    from .grasp import nearby
    s, m = ctx.session, ctx.session.model
    arm = s.robot.arms[side]
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    across = np.eye(3) - np.outer(direction, direction)
    held = np.flatnonzero(np.isin(m.geom_bodyid, list(subtree_bodies(m, root))))
    held = held[(m.geom_contype[held] != 0) | (m.geom_conaffinity[held] != 0)]
    exclude = set(s.robot.geoms.tolist()) | set(held.tolist())
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    start = s.data.site_xpos[arm.tcp].copy()
    shift, fromto, closest = np.zeros(3), np.zeros(6), np.inf
    for k in range(1, int(np.ceil(distance / EXTRACT_STEP)) + 1):
        push = np.zeros(3)
        for g in held:
            for h in nearby(m, s.data, s.data.geom_xpos[g], exclude, m.geom_rbound[g] + EXTRACT_CLEAR):
                d = mujoco.mj_geomDistance(m, s.data, int(g), int(h), EXTRACT_CLEAR, fromto)
                if d >= EXTRACT_CLEAR:
                    continue
                away = fromto[:3] - fromto[3:]                    # from the neighbour to the held object
                norm = np.linalg.norm(away)
                if norm < 1e-9:
                    continue
                away = across @ (away / norm * (1.0 if d >= 0 else -1.0))
                if np.linalg.norm(away) < 0.5:                    # mostly along the motion (the support below)
                    continue
                closest = min(closest, d)
                push += away / np.linalg.norm(away) * (EXTRACT_CLEAR - d)
        if np.linalg.norm(push) > EXTRACT_SHIFT:
            push *= EXTRACT_SHIFT / np.linalg.norm(push)
        shift = shift + push
        if np.linalg.norm(shift) > EXTRACT_MAX_SHIFT:
            shift *= EXTRACT_MAX_SHIFT / np.linalg.norm(shift)
        travel = min(k * EXTRACT_STEP, distance)
        motion.move_tcp(s, side, start + direction * travel + shift, rotation, speed=speed, settle=0.0)
    motion.move_tcp(s, side, start + direction * distance + shift, rotation, speed=speed)
    _note(ctx, "extract", shift_mm=(shift * 1e3).round(2).tolist(),
          closest_mm=None if not np.isfinite(closest) else round(closest * 1e3, 2))


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


def _element_geoms(s, element):
    """tool(side) for the dock search: the pin or blade geoms, at the margin push frames are made with."""
    from .grasp import PUSHER_MARGIN
    gripper = lambda side: s.robot.profile["arms"][side]["gripper"]
    names = lambda side: [gripper(side)["press_pin"]["body"]] if element == "pin" else gripper(side)["pry_blade"]["geoms"]
    return lambda side: ([s.model.geom(s.robot.prefix + n).id for n in names(side)], PUSHER_MARGIN)


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


PUSH_STAGE = {"hinge": 0.6, "slide": 0.06}   # rad / m of the joint one push must reach without re-docking
MAX_PUSH_STAGES = 4


def push(ctx, interface, task, stage=0):
    """Drive a 1-DoF mechanism by pushing it with the side of the press pin along its own motion.

    A long motion (a lid opening through 90 degrees) goes in stages: each dock only has to follow
    the next stage of the motion. Between stages the other arm takes over from the same base while
    the first still holds the part (a lid that would fall back stays up); without a relay the tool
    backs off and the next push is synthesised and docked from where the part is now."""
    s, m = ctx.session, ctx.session.model
    joint = interface.mechanism.joint
    address = m.jnt_qposadr[joint]
    low, high = m.jnt_range[joint]
    now = s.data.qpos[address]
    far = high if high - now >= now - low else low
    target = now + (far - now) * DRIVE_FRACTION
    site = _site(s, interface)
    side0 = next(iter(s.robot.arms))
    frames, element, report = _push_synthesis(ctx, interface, side0, target)
    if not frames:
        raise Abort("no_push_pose", **report)
    part = set(part_geoms(m, s.data, interface, site, radius=0.015).tolist())
    s.allowed_contact = part
    dock = _approach(ctx, interface, frames, part, hand_state=_pin_hand(s, element), prepare=_pin_deploy(s, element),
                     path=operation_path(ctx, interface, limit=PUSH_STAGE.get(interface.mechanism.kind)),
                     tool=_element_geoms(s, element))
    dock.element = element
    _final_approach(ctx, dock)
    release_events(ctx, task, "grasped")          # the pin is at the part: its declared release (unlatch)
    done = lambda: s.monitor.evaluate(task.goal, task.key) is True
    s.monitor.evaluate(task.goal, task.key)
    begin = s.data.qpos[address]
    for relay in range(MAX_PUSH_STAGES):
        reached, handover = _drive(ctx, interface, task, dock, target, relay)
        if reached or done():
            break
        if handover is None:
            handover = _relay(ctx, interface, dock, target)
        if handover is None:
            break
        # The other arm is on the part: only now does this one let go.
        _retreat(ctx, dock)
        _stow_element(s, dock)
        dock = handover
    pushed = abs(s.data.qpos[address] - begin)
    if not done():
        s.wait(1.0, done)
    _retreat(ctx, dock)
    _stow_element(s, dock)
    s.wait(0.5)
    progress = abs(s.data.qpos[address] - begin)
    if pushed > 0.05 and progress < 0.5 * pushed:
        # The part goes back when let go (a lid below its balance point) and no relay was possible.
        raise Abort("falls_back", pushed=round(float(pushed), 3), kept=round(float(progress), 3), stage=stage)
    if not done() and stage + 1 < MAX_PUSH_STAGES and progress > 0.1 * PUSH_STAGE.get(interface.mechanism.kind, np.inf):
        push(ctx, interface, task, stage + 1)


RELAY_PRE = 0.03            # m: the other arm comes in short, beside the holding wrist
RELAY_SHIFTS = (0.0, 0.008, -0.008, 0.02, -0.02, 0.035, -0.035)   # m along the edge, clear of the first tool


def _push_synthesis(ctx, interface, side, target, relay=False):
    """Push frames for `side` at the part's current state: the pin first, then the pry blade.
    For a relay the second tool goes on the same edge but beside the first one."""
    from .grasp import push_frames
    s, m = ctx.session, ctx.session.model
    joint = interface.mechanism.joint
    now = s.data.qpos[m.jnt_qposadr[joint]]
    site = _site(s, interface)
    first = mechanism_poses(m, s.data, interface, joint, now + (target - now) * 0.02, site, np.eye(3), 1)[0][0]
    if np.linalg.norm(first - site) < 1e-6:
        raise Abort("no_push_pose", why="the site does not move with the joint")
    arm = s.robot.arms[side]
    report = {}
    for element in ("pin", "blade"):
        tip = arm.pin_tip if element == "pin" else arm.blade_tip
        if tip is None:
            continue
        # The blade goes only under an overhanging edge: try holding its front edge back from the site.
        depths = (0.0,) if element == "pin" else (0.001, 0.002, 0.004)
        extra = {"shifts": RELAY_SHIFTS, "part_radius": 0.06} if relay else {}
        frames, found = push_frames(m, s.data, interface, ctx.probe(side), s.robot.geoms, tip, first - site,
                                    element=element, depths=depths, **extra)
        _note(ctx, "push_synthesis", side=side, element=element, **found)
        report[element] = {k: v for k, v in found.items() if k != "blocking"}
        report.setdefault("blocking", {}).update(found["blocking"])
        if frames:
            return frames, element, report
    return [], None, report


RELAY_CHECKS = (0.5, 0.75)     # fractions of a stage at which the other arm tries to take over


def _drive(ctx, interface, task, dock, target, relay):
    """Closed-loop push with the docked tool through at most one stage of the motion.
    (goal reached, the other arm's dock if it took over part-way)."""
    s, m = ctx.session, ctx.session.model
    joint = interface.mechanism.joint
    address = m.jnt_qposadr[joint]
    arm = s.robot.arms[dock.side]
    tool_tip = arm.pin_tip if dock.element == "pin" else arm.blade_tip
    done = lambda: s.monitor.evaluate(task.goal, task.key) is True
    cartesian = s.cartesian[dock.side]
    cartesian.kp[:3] = PUSH_STIFFNESS
    cartesian.kd[:3] = 2 * np.sqrt(PUSH_STIFFNESS * 3.0)
    # Closed loop: from wherever the part is now, push the contact point a little further along the
    # part's own motion at that point, so the tool stays on the part even when it lags or speeds up.
    step = 0.05 if interface.mechanism.kind == "hinge" else 0.005
    reached, stalled, best = False, 0, s.data.qpos[address]
    begin = best
    # The tool's pose on the part at contact: every set-point is taken from where the part is now, so
    # the tool follows the part back if it slips and never runs off along the arc on its own.
    body = interface.body
    rot_b = s.data.xmat[body].reshape(3, 3)
    local_pos = rot_b.T @ (s.data.site_xpos[arm.tcp] - s.data.xpos[body])
    local_rot = rot_b.T @ s.data.site_xmat[arm.tcp].reshape(3, 3)
    stage_end = begin + np.sign(target - begin) * min(abs(target - begin), PUSH_STAGE.get(interface.mechanism.kind, np.inf))
    checks = list(RELAY_CHECKS) if abs(target - begin) > abs(stage_end - begin) + 1e-6 else []
    for _ in range(MAX_PUSH_STEPS):
        now = s.data.qpos[address]
        if (target - now) * np.sign(target - best + 1e-12) <= 0 or (stage_end - now) * np.sign(target - begin + 1e-12) <= 0:
            break
        # The tool keeps its pose relative to the part (it turns with a lid), pushing slightly ahead.
        rot_b = s.data.xmat[body].reshape(3, 3)
        tcp_pos, tcp_rot = s.data.xpos[body] + rot_b @ local_pos, rot_b @ local_rot
        tip = tcp_pos + tcp_rot @ tool_tip
        ahead = now + np.sign(target - now) * min(step, abs(target - now))
        (moved_tcp, moved_rot), = mechanism_poses(m, s.data, interface, joint, ahead, tcp_pos, tcp_rot, 1)
        moved_tip = moved_tcp + moved_rot @ tool_tip
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
        # A hand-over can fit at one arm posture and not at another: try it at a few points of the stage.
        covered = abs(s.data.qpos[address] - begin) / max(abs(stage_end - begin), 1e-9)
        while checks and covered >= checks[0]:
            checks.pop(0)
            handover = _relay(ctx, interface, dock, target)
            if handover is not None:
                _note(ctx, "push", side=dock.side, relay=relay, reached=False, detail=s.monitor.details.get(task.key))
                return False, handover
    _note(ctx, "push", side=dock.side, relay=relay, reached=reached, detail=s.monitor.details.get(task.key))
    return reached, None


def _relay(ctx, interface, dock, target):
    """The other arm docked on the part from where the robot stands, with the first arm still holding
    it; None when it cannot take over (then the first arm lets go as usual)."""
    from .dock import relay_dock
    s, m = ctx.session, ctx.session.model
    other = [o for o in s.robot.arms if o != dock.side and o != ctx.holding]
    if not other:
        return None                   # the other arm holds the part up: no hand to take over
    side = other[0]
    try:
        frames, element, report = _push_synthesis(ctx, interface, side, target, relay=True)
    except Abort:
        return None
    if not frames:
        _note(ctx, "relay", side=side, found=False, why="no_push_pose")
        return None
    part = set(part_geoms(m, s.data, interface, _site(s, interface), radius=0.06).tolist())
    relay, search = relay_dock(s, ctx.kinematics, frames, side, part, pre_distance=RELAY_PRE, hand_state=_pin_hand(s, element),
                               tool=_element_geoms(s, element),
                               path=operation_path(ctx, interface, limit=PUSH_STAGE.get(interface.mechanism.kind)))
    _note(ctx, "relay", side=side, found=relay is not None, **search)
    if relay is None:
        return None
    relay.element = element
    world = motion.arm_world(s, side)
    world.hold([g for g in s.robot.geoms if m.geom(g).name.startswith(s.robot.prefix + dock.side + "_")])
    try:
        motion.move_arm(s, side, relay.q_pre, world=world)
    except Abort as exc:
        _note(ctx, "relay", side=side, found=False, why=exc.reason)
        return None
    _pin_deploy(s, element)(relay)
    _final_approach(ctx, relay)
    return relay


def _stow_element(s, dock):
    arm = s.robot.arms[dock.side]
    s.data.ctrl[arm.pin_actuator if getattr(dock, "element", "pin") == "pin" else arm.blade_actuator] = 0.0


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


MOUNT_ANGLES = 12          # turns of the held tool about the mounting axis tried for a dock
MOUNT_DEPTH = 0.002        # m the tool's mount point is pushed past the part's orifice
MOUNT_SEATED = 0.004       # m largest mount-to-orifice distance that counts as seated
MOUNT_ABOVE = 0.03         # m above the orifice the mount point is aligned before going in
MOUNT_FIT_DEPTH = 0.008    # m of the tool end and of the part's open end compared for the fit
TOOL_SLIP_M = 0.10          # m the tool may shift in the hand before it counts as dropped (re-measured after)
MOUNT_LIFT_STEP = 0.01      # m per straight joint-space step when a mounted part is drawn out


def _axis_rotation(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.eye(3)[np.argmin(np.abs(a))]
        v = np.cross(a, p) / np.linalg.norm(np.cross(a, p))
        return 2 * np.outer(v, v) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _about(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _extent(m, d, bodies, origin, axis):
    """(least, most) reach of the bodies' collision geometry along `axis` from `origin` (box corners)."""
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    values = []
    for g in body_geoms(m, bodies):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        centre = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
        values.extend(((centre + (signs * m.geom_aabb[g, 3:]) @ rot.T) - origin) @ axis)
    return (min(values), max(values)) if values else (0.0, 0.0)


def _radial(m, d, bodies, origin, axis, lo, hi):
    """Radii from the line (origin, axis) of the bodies' collision box corners lying between `lo` and `hi`
    along the axis."""
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    radii = []
    for g in body_geoms(m, bodies):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        centre = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
        rel = (centre + (signs * m.geom_aabb[g, 3:]) @ rot.T) - origin
        along = rel @ axis
        keep = (along >= lo) & (along <= hi)
        radii.extend(np.linalg.norm(rel[keep] - np.outer(along[keep], axis), axis=1))
    return np.asarray(radii)


def mount_fit(m, d, mount, orifice):
    """How far a tool's mount end is from fitting a part's opening (m; 0 = it fits): the tool's outer
    radius over its leading length against the part's inner radius at its open end."""
    down = d.site_xmat[orifice.site].reshape(3, 3) @ orifice.approach
    down = down / np.linalg.norm(down)
    part = subtree_bodies(m, orifice.mechanism.body)
    near, _ = _extent(m, d, part, d.site_xpos[orifice.site], down)
    opening = d.site_xpos[orifice.site] + down * near
    bore = _radial(m, d, part, opening, down, 0.0, MOUNT_FIT_DEPTH)
    axis = d.site_xmat[mount.site].reshape(3, 3) @ mount.approach
    axis = axis / np.linalg.norm(axis)
    tool = subtree_bodies(m, mount.mechanism.body)
    _, lead = _extent(m, d, tool, d.site_xpos[mount.site], axis)
    nozzle = _radial(m, d, tool, d.site_xpos[mount.site], axis, max(lead - MOUNT_FIT_DEPTH, 0.0), lead)
    if not len(bore) or not len(nozzle):
        return float("inf")
    return abs(float(nozzle.max()) - float(bore.min()))


def _world_approach(s, interface):
    return s.data.site_xmat[interface.site].reshape(3, 3) @ interface.approach


def _joint_tcp(ctx, side, position, rotation, speed):
    """TCP to a pose by joint-space interpolation to its IK solution: the joint PD holds a payload to
    sub-millimetre, where the Cartesian impedance sags by centimetres at the end of a held tool."""
    from ..planning.ik import IKFailure, Target
    s, kin, arm = ctx.session, ctx.kinematics, ctx.session.robot.arms[side]
    base = s.robot.base_pose(s.data.qpos)
    q_base = kin.qpos(kin.robot.base_joint_values(base), float(s.data.qpos[s.robot.lift_qpos]), {})
    try:
        q = kin.ik[side].solve(q_base, Target(position, rotation, position_tolerance=5e-4),
                               seed=s.data.qpos[arm.qpos], restarts=1).q
    except IKFailure as exc:
        raise Abort("tool_ik", residual_mm=round(exc.best.position_error * 1e3, 2))
    start = s.data.qpos[arm.qpos].copy()
    distance = float(np.linalg.norm(s.data.site_xpos[arm.tcp] - position))
    steps = max(50, int(distance / speed / s.model.opt.timestep))
    for k in range(1, steps + 1):
        t = k / steps
        s.use_joint(side, start + (q - start) * t * t * (3 - 2 * t))
        s.step()
    s.use_joint(side, q)
    s.wait(0.5, lambda: np.max(np.abs(s.data.qvel[arm.dofs])) < 0.01)
    return float(np.linalg.norm(s.data.site_xpos[arm.tcp] - position))


def _free_root(m, body):
    """The body carrying the free joint that `body` moves with (the loose part itself), or `body`."""
    b = body
    while b > 0:
        j = int(m.body_jntadr[b])
        if j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return b
        b = int(m.body_parentid[b])
    return body


def _robot_contacts(s, extra_bodies=()):
    """Names of what the robot (and a held tool) touches now, for the log."""
    m, d = s.model, s.data
    mine = set(s.robot.geoms.tolist()) | set(body_geoms(m, extra_bodies).tolist())
    seen = set()
    for c in d.contact[:d.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if (a in mine) != (b in mine):
            inside, outside = (a, b) if a in mine else (b, a)
            seen.add((m.geom(inside).name.split("__")[-1][:24], m.geom(outside).name[:40]))
    return sorted(seen)[:8]


def tool_pick(ctx, interface, task, tool, tool_task, orifice):
    """Pick a part the fingers cannot hold (a pipette tip in its rack) with a tool: grasp the tool by
    its own grasp interface, carry it so its mount point enters the part's orifice along the declared
    axes, seat it and lift. The press fit is modelled as a weld once the mount point is seated, as the
    suction cup is; the part's declared release event lets it leave its holder."""
    from .grasp import Frame
    s, m = ctx.session, ctx.session.model
    mount = tool["mount"]
    held = pick(ctx, tool["grasp"], tool_task)
    side, arm = held.side, s.robot.arms[held.side]
    cup = arm.suction
    if cup is None:
        raise Abort("no_tool_attachment")
    tool_bodies = set(subtree_bodies(m, mount.mechanism.body))
    def still_held(step):
        rotation_now = s.data.site_xmat[arm.tcp].reshape(3, 3)
        moved = float(np.linalg.norm(rotation_now.T @ (_site(s, mount) - s.data.site_xpos[arm.tcp]) - offset))
        fingers = [g for g in s.robot.geoms if m.geom_bodyid[g] in arm.fingers]
        others = np.array(sorted(body_geoms(m, tool_bodies).tolist()))
        touch = s.monitor.contacts(np.array(fingers), others)
        _note(ctx, "tool_check", stage=step, moved_mm=round(moved * 1e3, 1), finger_contacts=len(touch),
              force_n=round(float(sum(r[2] for r in touch)), 1), mount=_site(s, mount).round(3).tolist())
        if moved > TOOL_SLIP_M:
            raise Abort("tool_dropped", step=step, moved_mm=round(moved * 1e3, 1))
    ctx.moved = (ctx.moved or set()) | tool_bodies           # the tool is moved on purpose
    # The planners treat the tool in the hand as part of the hand (not an obstacle); if it strikes
    # anything on the way the session's contact-force guard still aborts the task.
    s.carried = set(body_geoms(m, tool_bodies).tolist())
    offset = s.data.site_xmat[arm.tcp].reshape(3, 3).T @ (_site(s, mount) - s.data.site_xpos[arm.tcp])
    still_held("picked")
    s.carried_root = (_free_root(m, mount.mechanism.body), side)    # planned motions carry it along, checked
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    offset = rotation.T @ (_site(s, mount) - s.data.site_xpos[arm.tcp])
    local = rotation.T @ _world_approach(s, mount)            # mounting direction, tool frame
    target = _world_approach(s, orifice)
    down = target / np.linalg.norm(target)
    align = _axis_rotation(rotation @ local, target) @ rotation
    root = interface.mechanism.body
    # The tool enters the part at its far end against the approach (a tip's open top), on the orifice axis;
    # its own geometry leads its mount point by `lead` along the mounting direction.
    near, _ = _extent(m, s.data, subtree_bodies(m, root), _site(s, orifice), down)
    _, lead = _extent(m, s.data, tool_bodies, _site(s, mount), _world_approach(s, mount) / np.linalg.norm(_world_approach(s, mount)))
    opening = _site(s, orifice) + down * near
    entry = opening + down * MOUNT_DEPTH
    frames = []
    for k in range(MOUNT_ANGLES):
        r = _about(target, 2 * np.pi * k / MOUNT_ANGLES) @ align
        frames.append(Frame(entry - r @ offset, r, float("inf"), rank=0, variant=k, opening=None,
                            axis=tuple(local / np.linalg.norm(local))))
    part = set(body_geoms(m, subtree_bodies(m, root)).tolist())
    s.allowed_contact = part | set(body_geoms(m, tool_bodies).tolist())
    ctx.tool_check = still_held
    try:
        dock = _approach(ctx, orifice, frames, part, hand_state=_closed_hand(s), sides=[side])
    finally:
        ctx.tool_check = None
    still_held("approach")
    # The tool may have turned in the fingers on the way: re-measure it in the hand and steer its mount
    # point (not the TCP) above the orifice, then straight in along the orifice axis.
    _note(ctx, "tool_target", opening=opening.round(3).tolist(), orifice_now=_site(s, orifice).round(3).tolist(),
          tcp=s.data.site_xpos[arm.tcp].round(3).tolist(), mount=_site(s, mount).round(3).tolist(),
          near_mm=round(near * 1e3, 1), lead_mm=round(lead * 1e3, 1))
    for stand_off in (MOUNT_ABOVE + lead, MOUNT_ABOVE + lead, -MOUNT_DEPTH):
        rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
        held_offset = rotation.T @ (_site(s, mount) - s.data.site_xpos[arm.tcp])
        turn = _axis_rotation(_world_approach(s, mount), down) @ rotation
        goal = opening - down * stand_off
        tcp_error = _joint_tcp(ctx, side, goal - turn @ held_offset, turn, speed=0.01 if stand_off < 0 else 0.05)
        _note(ctx, "tool_align", stand_off_mm=round(stand_off * 1e3, 1), tcp_error_mm=round(tcp_error * 1e3, 1),
              error_mm=round(float(np.linalg.norm(_site(s, mount) - goal)) * 1e3, 2), contacts=_robot_contacts(s, tool_bodies))
    gap = float(np.linalg.norm(_site(s, mount) - entry))
    _note(ctx, "tool_seat", tool=mount.id, gap_mm=round(gap * 1e3, 2), opening_above_site_mm=round(-near * 1e3, 1),
          lead_mm=round(lead * 1e3, 1))
    if gap > MOUNT_SEATED + MOUNT_DEPTH:
        raise Abort("tool_not_seated", gap_mm=round(gap * 1e3, 2))
    # The fit: weld the part to the tool where it now sits (the free cup equality of this arm).
    e, own = cup["equality"], mount.mechanism.body
    m.eq_obj1id[e], m.eq_obj2id[e] = own, root
    turn = s.data.xmat[own].reshape(3, 3)
    m.eq_data[e, :3] = 0.0
    m.eq_data[e, 3:6] = turn.T @ (s.data.xpos[root] - s.data.xpos[own])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (turn.T @ s.data.xmat[root].reshape(3, 3)).flatten())
    m.eq_data[e, 6:10] = quat
    m.eq_data[e, 10] = 1.0
    s.data.eq_active[e] = 1
    release_events(ctx, task, "grasped")
    s.monitor.evaluate(task.goal, task.key)                   # anchors the lift height at the fit
    # Straight back out along the orifice axis on the joint PD: the part leaves its seat the way the tool
    # went in (steering sideways, as a loose part's extraction does, drags it against its neighbours).
    rotation = s.data.site_xmat[arm.tcp].reshape(3, 3).copy()
    start = s.data.site_xpos[arm.tcp].copy()
    lift = task.parameters.get("lift_m", 0.08)
    for k in range(1, int(np.ceil(lift / MOUNT_LIFT_STEP)) + 1):
        _joint_tcp(ctx, side, start - down * min(k * MOUNT_LIFT_STEP, lift), rotation, speed=0.02)
    _note(ctx, "tool_lift", rise_mm=round(float((s.data.site_xpos[arm.tcp] - start) @ -down) * 1e3, 1))
    s.wait(1.0, lambda: s.monitor.evaluate(task.goal, task.key) is True)
    return dock


HOLD_LIFT_M = 0.30         # m a loose part is lifted off the bench before it is worked on in the hand
# Where the held part is presented, tried in turn while the other arm cannot reach it: (m in front of the
# base centre, m towards the holding arm's side, turn of the hand about the vertical in rad, m higher).
PRESENT_SPOTS = ((0.55, 0.12, 0.0, 0.0), (0.50, 0.25, 0.0, 0.05), (0.55, 0.12, np.pi / 4, 0.05),
                 (0.55, 0.12, -np.pi / 4, 0.05), (0.65, 0.05, 0.0, 0.10))


def hold_up(ctx, holder, holder_task):
    """Pick a loose part by its own grasp interface and keep it up in that hand (a pipette whose knob,
    plunger or ejector cannot be worked while it lies on the bench); the other arm then works on it."""
    holder_task.parameters["lift_m"] = max(holder_task.parameters.get("lift_m", 0.08), HOLD_LIFT_M)
    dock = pick(ctx, holder, holder_task)
    ctx.holding = dock.side
    _present(ctx, dock.side, holder)
    _note(ctx, "hold_up", side=dock.side, by=holder.id)
    return dock


def present_again(ctx, holder, spot):
    """Move the part already held up to presentation spot `spot`; False when there are no more spots."""
    if spot >= len(PRESENT_SPOTS):
        return False
    _present(ctx, ctx.holding, holder, spot)
    return True


def _present(ctx, side, holder, spot=0):
    """Bring the held part in front of the robot, clear of the bench (spot 0: on the holding arm's side)."""
    from ..planning.ik import IKFailure, Target
    s, kin, arm = ctx.session, ctx.kinematics, ctx.session.robot.arms[side]
    m = s.model
    base = s.robot.base_pose(s.data.qpos)
    forward = np.array([np.cos(base[2]), np.sin(base[2]), 0.0])
    left = np.array([-forward[1], forward[0], 0.0])
    lateral = s.robot.profile["arms"][side]["lateral_m"]
    ahead, aside, turn, higher = PRESENT_SPOTS[spot]
    goal = np.array([base[0], base[1], 0.0]) + forward * ahead + left * np.sign(lateral) * aside
    goal[2] = s.data.site_xpos[arm.tcp][2] + higher
    c, si = np.cos(turn * np.sign(lateral)), np.sin(turn * np.sign(lateral))
    rotation = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]]) @ s.data.site_xmat[arm.tcp].reshape(3, 3)
    q_base = kin.qpos(kin.robot.base_joint_values(base), float(s.data.qpos[s.robot.lift_qpos]), {})
    root = holder.mechanism.body
    s.carried = set(body_geoms(m, subtree_bodies(m, root)).tolist())     # moves with the hand on the way
    s.carried_root = (_free_root(m, root), side)
    try:
        q = kin.ik[side].solve(q_base, Target(goal, rotation, position_tolerance=5e-3), seed=s.data.qpos[arm.qpos],
                               restarts=2).q
        motion.move_arm(s, side, q)
        _note(ctx, "present", spot=spot, at=goal.round(3).tolist())
    except (IKFailure, Abort) as exc:
        _note(ctx, "present", spot=spot, failed=getattr(exc, "reason", type(exc).__name__))   # work it where it is
    finally:
        s.carried, s.carried_root = set(), None


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
