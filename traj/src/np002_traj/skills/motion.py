"""Motion primitives on a Session: base travel, planned arm motion, Cartesian moves and the gripper.

Each primitive measures its own outcome and raises Abort with a structured reason on failure.
"""
import numpy as np

from ..executor.session import Abort
from ..geometry import quat_to_mat, mat_to_quat, rotation_error
from ..planning.base import OccupancyMap
from ..planning.collision import CollisionWorld
from ..planning.path import NoPath, rrt_connect, time_scale

ARM_VELOCITY, ARM_ACCELERATION = 1.0, 2.5          # rad/s, rad/s^2 (well inside FR3 limits)
BASE_SPEED, BASE_YAW_RATE = 0.4, 0.8


def travel_posture(robot, side):
    """Arms folded over the base for driving; taken from the profile."""
    return np.array(robot.profile["arms"][side].get("travel", robot.profile["arms"][side]["home"]), float)


# -- arm -------------------------------------------------------------------
def arm_world(session, side, margin=0.01):
    world = CollisionWorld(session.model, session.robot, session.robot.arms[side].qpos, margin=margin)
    world.sync(session.data)
    return world


def move_arm(session, side, goal, world=None, rng=None):
    """Collision-free joint motion to `goal`, time-scaled, tracked by joint PD."""
    arm = session.robot.arms[side]
    world = world or arm_world(session, side)
    start = session.data.qpos[arm.qpos].copy()
    limits = session.model.jnt_range[arm.joints]
    if not world.valid(goal):
        raise Abort("goal_in_collision", blocking=world.blocking(goal))
    try:
        path = rrt_connect(start, goal, limits, world.edge_valid, rng or np.random.default_rng(0))
    except NoPath as exc:
        raise Abort("no_arm_path", detail=str(exc))
    times, positions, velocities = time_scale(path, ARM_VELOCITY, ARM_ACCELERATION, rate=1 / session.model.opt.timestep)
    for i in range(0, len(positions), 25):                 # the smoothed curve leaves the checked edges slightly
        if not world.valid(positions[i]):
            positions, velocities = linear_profile(path, session.model.opt.timestep)
            break
    for q, qd in zip(positions, velocities):
        session.use_joint(side, q, qd)
        session.step()
    session.use_joint(side, goal)
    session.wait(1.0, lambda: np.max(np.abs(session.data.qvel[arm.dofs])) < 0.02)
    error = float(np.max(np.abs(session.data.qpos[arm.qpos] - goal)))
    if error > 0.02:
        raise Abort("arm_tracking", error_rad=error)


def linear_profile(path, dt):
    """Fallback timing that stays exactly on the checked straight edges (stops at each waypoint)."""
    positions, velocities = [], []
    for a, b in zip(path[:-1], path[1:]):
        t, p, v = time_scale([a, b], ARM_VELOCITY, ARM_ACCELERATION, rate=1 / dt)
        positions.extend(p)
        velocities.extend(v)
    return np.array(positions), np.array(velocities)


def slerp(r0, r1, t):
    q0, q1 = mat_to_quat(r0), mat_to_quat(r1)
    if q0 @ q1 < 0:
        q1 = -q1
    angle = np.arccos(np.clip(q0 @ q1, -1, 1))
    if angle < 1e-6:
        return r1
    q = (np.sin((1 - t) * angle) * q0 + np.sin(t * angle) * q1) / np.sin(angle)
    return quat_to_mat(q / np.linalg.norm(q))


def move_tcp(session, side, position, rotation, speed=0.05, settle=0.5, tolerance=0.003, until=None):
    """Straight-line Cartesian impedance move of the TCP. Returns the final position error (m).

    The set-point moves at `speed`; in contact the arm stops where the part resists, with the
    force bounded by the impedance. `until` ends the move early when it becomes true.
    """
    arm = session.robot.arms[side]
    session.use_cartesian(side)
    c = session.cartesian[side]
    p0, r0 = c.position.copy(), c.rotation.copy()
    distance = float(np.linalg.norm(position - p0))
    turn = float(np.linalg.norm(rotation_error(rotation, r0)))
    seconds = max(distance / speed, turn / 0.6, 0.05)
    steps = int(seconds / session.model.opt.timestep)
    for k in range(1, steps + 1):
        t = k / steps
        s = t * t * (3 - 2 * t)                              # smooth start and stop
        c.set(p0 + (position - p0) * s, slerp(r0, rotation, s))
        session.step()
        if until is not None and k % 10 == 0 and until():
            return float(np.linalg.norm(position - session.data.site_xpos[arm.tcp]))
    session.wait(settle, lambda: (until is not None and until()) or np.linalg.norm(c.error(session.data)[0]) < tolerance * 0.5)
    return float(np.linalg.norm(position - session.data.site_xpos[arm.tcp]))


def follow(session, side, poses, speed=0.05, until=None):
    """Track a sequence of TCP poses (e.g. computed from a mechanism) with the impedance controller."""
    for position, rotation in poses:
        move_tcp(session, side, position, rotation, speed=speed, settle=0.0, until=until)
        if until is not None and until():
            return True
    return until() if until is not None else True


def gripper(session, side, opening, force=40.0, seconds=1.5):
    """Command each finger to `opening` (m) with a force limit; returns the measured finger position."""
    arm, m, d = session.robot.arms[side], session.model, session.data
    m.actuator_forcerange[arm.gripper] = [-force, force]
    d.ctrl[arm.gripper] = opening
    finger = m.jnt_qposadr[m.actuator_trnid[arm.gripper, 0]]
    dof = m.jnt_dofadr[m.actuator_trnid[arm.gripper, 0]]
    session.wait(0.3)
    session.wait(seconds - 0.3, lambda: abs(d.qvel[dof]) < 1e-3)
    return float(d.qpos[finger])


# -- base ------------------------------------------------------------------
def footprint_radius(session):
    """Radius of the robot's floor-plane extent (base plus travel-posture arms) around the base centre."""
    m, d, robot = session.model, session.data, session.robot
    centre = robot.base_pose(d.qpos)[:2]
    from ..planning.base import world_aabbs
    low, high = world_aabbs(m, d, robot.geoms)
    corners = np.concatenate([low[:, :2], high[:, :2], np.column_stack([low[:, 0], high[:, 1]]),
                              np.column_stack([high[:, 0], low[:, 1]])])
    return float(np.max(np.linalg.norm(corners - centre, axis=1)))


def drive(session, goal, occupancy=None, radius=None, tolerance=(0.01, 0.02), timeout=60.0):
    """Drive the base to world (x, y, yaw) along an A* path, with the arms holding their posture."""
    robot, d = session.robot, session.data
    for side in robot.arms:
        session.use_joint(side)
    occupancy = occupancy or OccupancyMap(session.model, d, robot)
    radius = radius or footprint_radius(session) + 0.03
    start = robot.base_pose(d.qpos)
    route = occupancy.plan(start[:2], goal[:2], radius, goal_radius=radius + 0.3)
    if route is None:
        raise Abort("no_base_path", start=start.tolist(), goal=list(goal))
    end_time = d.time + timeout
    for k, waypoint in enumerate(route[1:], 1):
        final = k == len(route) - 1
        while d.time < end_time:
            pose = robot.base_pose(d.qpos)
            offset = np.asarray(waypoint) - pose[:2]
            distance = float(np.linalg.norm(offset))
            yaw_error = float(np.angle(np.exp(1j * (goal[2] - pose[2]))))
            if distance < (tolerance[0] if final else 0.05) and (not final or abs(yaw_error) < tolerance[1]):
                break
            speed = min(BASE_SPEED, 1.2 * distance) if final else BASE_SPEED
            velocity = offset / max(distance, 1e-9) * speed
            robot.base_command(d, velocity, np.clip(2.0 * yaw_error, -BASE_YAW_RATE, BASE_YAW_RATE))
            session.step(5)
        else:
            robot.base_command(d, [0, 0], 0)
            raise Abort("base_timeout", pose=robot.base_pose(d.qpos).tolist(), goal=list(goal))
    robot.base_command(d, [0, 0], 0)
    session.wait(0.5, lambda: np.max(np.abs(d.qvel[session.model.jnt_dofadr[robot.base_joints]])) < 1e-3)
    return robot.base_pose(d.qpos)


def set_lift(session, height, timeout=4.0):
    m, d, robot = session.model, session.data, session.robot
    low, high = m.actuator_ctrlrange[robot.lift_actuator]
    d.ctrl[robot.lift_actuator] = float(np.clip(height, low, high))
    session.wait(timeout, lambda: abs(d.qpos[robot.lift_qpos] - d.ctrl[robot.lift_actuator]) < 2e-3
                 and abs(d.qvel[m.jnt_dofadr[robot.lift_joint]]) < 1e-3)
