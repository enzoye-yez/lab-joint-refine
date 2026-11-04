"""Frame and orientation kernels (ported from the previous package; MuJoCo quaternions are wxyz)."""
import math

import mujoco
import numpy as np


def rotation_error(target, actual):
    """World-frame rotation vector taking `actual` to `target` (both 3x3)."""
    qa, qb, neg, delta, velocity = (np.zeros(n) for n in (4, 4, 4, 4, 3))
    mujoco.mju_mat2Quat(qa, np.asarray(target, float).reshape(9))
    mujoco.mju_mat2Quat(qb, np.asarray(actual, float).reshape(9))
    mujoco.mju_negQuat(neg, qb)
    mujoco.mju_mulQuat(delta, qa, neg)
    if delta[0] < 0:
        delta *= -1
    mujoco.mju_quat2Vel(velocity, delta, 1.0)
    return velocity


def axis_error(local_axis, desired, actual):
    """Rotation vector aligning the frame's `local_axis` with world `desired`; rotation about it is free."""
    axis = actual @ np.asarray(local_axis, float)
    desired = np.asarray(desired, float) / np.linalg.norm(desired)
    cross = np.cross(axis, desired)
    length, dot = np.linalg.norm(cross), float(np.clip(axis @ desired, -1, 1))
    if length > 1e-12:
        return cross * (math.atan2(length, dot) / length), axis
    if dot > 0:
        return np.zeros(3), axis
    tangent = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    return tangent * (math.pi / np.linalg.norm(tangent)), axis


def quat_to_mat(quat):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, float))
    return out.reshape(3, 3)


def mat_to_quat(mat):
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.asarray(mat, float).reshape(9))
    return out


def subtree_bodies(model, root):
    """Bodies in the kinematic subtree of `root` (body ids are topologically ordered)."""
    ids = {root}
    for body in range(root + 1, model.nbody):
        if model.body_parentid[body] in ids:
            ids.add(body)
    return ids


def body_geoms(model, bodies):
    return np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))
