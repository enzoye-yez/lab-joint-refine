"""Counterfactual scene edits on the compiled, settled scene.

A repair skill asks "would this task succeed if the scene were different in this one way?". The edits
are applied to a copy of the compiled MjModel (re-parsing the room's XML takes minutes), then the bodies
near the edit are settled again and the result becomes the copy's "assembled" keyframe, so every task
starts from a resting state exactly as in the benchmark.

Edit kinds (JSON-able dicts, names are model names):
  move_instance   {instance, dxyz=[dx,dy,dz] world m, dyaw=rad about the instance centre, carry=[instances]}
                  moves every top-level body of the instance, the `carry` instances (what the scene plan
                  stands on it) and loose bodies resting on it
  move_body       {body, dxyz}                       a part inside an instance (world-frame offset)
  disable_collision {geoms: [...]} or {bodies: [...]} the geoms stop colliding with anything
  scale_geom      {geom, factor=[fx,fy,fz]}          primitive geoms only (box, cylinder, capsule, sphere)
  joint           {joint, frictionloss|damping|stiffness|armature: {"scale": s} | {"value": v}}
  mass            {body, scale}                      mass and inertia of one body
  freeze          {bodies: [...]}                    the bodies' dofs get a huge damping (held in place)
"""
import copy

import mujoco
import numpy as np

from .locality import Locality, instance_of_bodies
from .settle import body_speeds

SETTLE_S = 1.0                # re-settling time after an edit (the scene was at rest before it)
SETTLE_RADIUS_M = 1.0         # bodies within this of an edit are simulated while it settles
RESTING_GAP_M = 0.01          # a loose body this close above an instance's top is carried with it


class EditError(ValueError):
    pass


def _yaw_quat(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def _rotate_z(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _free_qpos(model, body):
    for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None


def _top_bodies(model, instance):
    names = instance_of_bodies(model)
    return [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and names[b] == instance]


def _world_box(model, data, bodies):
    """World AABB of the geoms of `bodies` and their subtrees (from the local AABBs)."""
    roots = set(bodies)
    geoms = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] in roots]
    if not geoms:
        return None
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geoms:
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + centre
        world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        low, high = np.minimum(low, world.min(0)), np.maximum(high, world.max(0))
    return low, high


def _resting_on(model, data, instance, roots):
    """Loose top-level bodies of other instances that sit on the instance (tubes in its rack)."""
    box = _world_box(model, data, roots)
    if box is None:
        return []
    low, high = box
    names = instance_of_bodies(model)
    out = []
    for b in range(1, model.nbody):
        if model.body_parentid[b] != 0 or names[b] in (instance, "robot") or _free_qpos(model, b) is None:
            continue
        p = data.xpos[b]
        if low[0] <= p[0] <= high[0] and low[1] <= p[1] <= high[1] and low[2] <= p[2] <= high[2] + RESTING_GAP_M:
            out.append(b)
    return out


def _shift_top(model, key, body, offset, yaw, pivot, data):
    """Move a top-level body by `offset` and turn it by `yaw` about `pivot` (world)."""
    adr = _free_qpos(model, body)
    position = data.xpos[body].copy()
    target = pivot + _rotate_z(position - pivot, yaw) + offset
    turn = _yaw_quat(yaw)
    if adr is not None:                                    # its pose is its qpos
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, turn, model.key_qpos[key, adr + 3:adr + 7])
        model.key_qpos[key, adr:adr + 3] = target
        model.key_qpos[key, adr + 3:adr + 7] = q / np.linalg.norm(q)
    else:                                                  # welded to the world (or jointed): its frame
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, turn, model.body_quat[body])
        model.body_pos[body] = model.body_pos[body] + (target - position)
        model.body_quat[body] = q / np.linalg.norm(q)


def _move_instance(model, data, key, edit):
    roots = _top_bodies(model, edit["instance"])
    if not roots:
        raise EditError(f"no instance {edit['instance']!r}")
    offset, yaw = np.asarray(edit.get("dxyz", (0, 0, 0)), float), float(edit.get("dyaw", 0.0))
    carried = _resting_on(model, data, edit["instance"], roots)
    for other in edit.get("carry", ()):               # instances the scene plan stands on it (any depth)
        carried += [b for b in _top_bodies(model, other) if b not in carried]
    pivot = np.mean([data.xpos[b] for b in roots], axis=0)
    for b in roots + carried:
        _shift_top(model, key, b, offset, yaw, pivot, data)
    return {"bodies": len(roots), "carried": len(carried), "at": pivot.tolist()}


def _move_body(model, data, key, edit):
    body = model.body(edit["body"]).id
    offset = np.asarray(edit["dxyz"], float)
    if model.body_parentid[body] == 0:
        _shift_top(model, key, body, offset, 0.0, data.xpos[body], data)
    else:                                                  # offset in the parent's frame
        parent = data.xmat[model.body_parentid[body]].reshape(3, 3)
        model.body_pos[body] = model.body_pos[body] + parent.T @ offset
    return {"at": data.xpos[body].tolist()}


def _disable_collision(model, data, key, edit):
    geoms = [model.geom(n).id for n in edit.get("geoms", ())]
    for name in edit.get("bodies", ()):
        b = model.body(name).id
        geoms += [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
    model.geom_contype[geoms] = 0
    model.geom_conaffinity[geoms] = 0
    return {"geoms": len(geoms), "at": data.geom_xpos[geoms[0]].tolist() if geoms else None}


def _scale_geom(model, data, key, edit):
    g = model.geom(edit["geom"]).id
    f = np.broadcast_to(np.asarray(edit["factor"], float), 3)
    kind = int(model.geom_type[g])
    size = model.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        size[:] = size * f
        half, radius = size.copy(), float(np.linalg.norm(size))
    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        size[0], size[1] = size[0] * max(f[0], f[1]), size[1] * f[2]
        top = size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)
        half = np.array([size[0], size[0], top])
        radius = float(np.hypot(size[0], size[1])) if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else top
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        size[0] *= float(np.max(f))
        half, radius = np.full(3, size[0]), float(size[0])
    else:
        raise EditError(f"geom {edit['geom']!r} is not a primitive (type {kind})")
    model.geom_aabb[g, 3:] = half
    model.geom_rbound[g] = radius
    return {"at": data.geom_xpos[g].tolist(), "size": size.tolist()}


JOINT_FIELDS = {"frictionloss": "dof_frictionloss", "damping": "dof_damping", "armature": "dof_armature",
                "stiffness": "jnt_stiffness"}


def _joint(model, data, key, edit):
    j = model.joint(edit["joint"]).id
    dof, out = model.jnt_dofadr[j], {"at": data.xanchor[j].tolist()}
    n = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(int(model.jnt_type[j]), 1)
    for field, array in JOINT_FIELDS.items():
        if field not in edit:
            continue
        values = getattr(model, array)
        index = slice(j, j + 1) if array.startswith("jnt_") else slice(dof, dof + n)
        before = values[index].copy()
        change = edit[field]
        values[index] = before * change["scale"] if "scale" in change else change["value"]
        out[field] = {"before": before.tolist(), "after": values[index].tolist()}
    return out


def _mass(model, data, key, edit):
    b = model.body(edit["body"]).id
    s = float(edit["scale"])
    before = float(model.body_mass[b])
    model.body_mass[b] *= s
    model.body_inertia[b] *= s
    return {"at": data.xpos[b].tolist(), "mass": [before, float(model.body_mass[b])], "set_const": True}


FREEZE_DAMPING_PER_KG = 1e6


def _freeze(model, data, key, edit):
    bodies = [model.body(n).id for n in edit["bodies"]]
    dofs = [d for d in range(model.nv) if model.dof_bodyid[d] in bodies]
    mass = model.body_subtreemass[model.body_rootid[model.dof_bodyid[dofs]]] + 1.0 if dofs else []
    model.dof_damping[dofs] += FREEZE_DAMPING_PER_KG * mass
    return {"dofs": len(dofs), "at": data.xpos[bodies[0]].tolist() if bodies else None}


EDITS = {"move_instance": _move_instance, "move_body": _move_body, "disable_collision": _disable_collision,
         "scale_geom": _scale_geom, "joint": _joint, "mass": _mass, "freeze": _freeze}


def apply(model, edits, robot, key="assembled", settle_s=SETTLE_S):
    """A perturbed copy of `model` (settled, with its own "assembled" keyframe) and a report per edit."""
    model = copy.deepcopy(model)
    k = model.key(key).id
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, k)
    mujoco.mj_forward(model, data)
    reports, points, set_const = [], [], False
    for edit in edits:
        if edit.get("kind") not in EDITS:
            raise EditError(f"unknown edit kind {edit.get('kind')!r}")
        report = EDITS[edit["kind"]](model, data, k, edit)
        set_const |= bool(report.pop("set_const", False))
        if report.get("at") is not None:
            points.append(np.asarray(report["at"], float))
        reports.append({"edit": edit, **report})
    if set_const:
        mujoco.mj_setConst(model, data)
    settled = _settle_near(model, robot, k, points, settle_s) if points else {"seconds": 0.0, "moving": []}
    return model, {"edits": reports, "settle": settled}


def _settle_near(model, robot, key, points, seconds):
    """Simulate only what is near the edits for `seconds`, with the robot holding its pose, and store
    the result as the keyframe."""
    from ..control.joint import JointPD
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    locality = Locality(model, data)
    near = set().union(*(locality.near(p, radius=SETTLE_RADIUS_M) for p in points))
    saved, _ = locality.apply(points[0], keep=near)
    data = mujoco.MjData(model)                  # after the freeze: equality activity is read at reset
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    try:
        hold = [JointPD(model, arm) for arm in robot.arms.values()]
        for c in hold:
            c.hold(data)
        data.ctrl[robot.base_actuators] = 0.0
        for _ in range(int(seconds / model.opt.timestep)):
            for c in hold:
                c.apply(data)
            mujoco.mj_step(model, data)
    finally:
        locality.restore(saved)
    names = instance_of_bodies(model)
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0 and names[b] in near and names[b] != "robot"]
    moving = [model.body(b).name for b, (lin, ang) in body_speeds(model, data, roots).items() if lin > 1e-3 or ang > 0.01]
    model.key_qpos[key], model.key_qvel[key], model.key_ctrl[key] = data.qpos, 0.0, data.ctrl
    return {"seconds": float(data.time), "near_instances": len(near), "moving": moving}
