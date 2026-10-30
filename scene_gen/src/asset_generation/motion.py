"""Framework-owned motion checks for one asset: what moves, whether it can, and what holds.

The author declares `mechanisms` in interfaces.json, each tied to the requirement text it
implements and to real joints and locks. These checks never trust that declaration:
  mechanism_contract  every movable joint belongs to a declared mechanism with valid limits
  sampled_joint_sweep each mechanism swept across its range; new collider penetration fails
  mechanism_drive     a bounded PD drive moves each mechanism to both limits in simulation;
                      a declared lock must hold against the same drive and release when off
  numerical_stability gravity settle on a temporary ground at z=0: finite, no warnings,
                      resting free parts do not drift
  release_probe       a mechanically released connection really frees its bodies
Cross-asset motion (inserting a tube into another instrument, robot grasping) belongs to the
trajectory stage and is stated as out of scope in the report coverage.
"""
import math
from pathlib import Path
import numpy as np
import mujoco

HINGE, SLIDE, BALL, FREE = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE,
                            mujoco.mjtJoint.mjJNT_BALL, mujoco.mjtJoint.mjJNT_FREE)
ACTUATIONS = {"gripper", "button", "software", "gravity", "manual"}
SWEEP_SAMPLES = 9
DRIVE_SECONDS = 4.
# Operator-scale effort: a gripper or finger pushing a slide, or turning a lid/knob.
DRIVE_FORCE_N, DRIVE_TORQUE_NM = 5., 2.
REACH_FRACTION = .05
SETTLE_SECONDS = 2.
DRIFT_M, DRIFT_RAD = .002, math.radians(1)
LOCK_HOLD_FRACTION = .05


def movable(model):
    return [j for j in range(model.njnt) if model.jnt_type[j] in (HINGE, SLIDE, BALL)]


def declared(manifest):
    return manifest.get("mechanisms") if isinstance(manifest.get("mechanisms"), list) else []


def mechanism_contract(model, manifest):
    """Deterministic coverage of real joints by declared, requirement-bound mechanisms."""
    problems, mechanisms = [], declared(manifest)
    if "mechanisms" in manifest and not isinstance(manifest["mechanisms"], list):
        problems.append("mechanisms 必须是列表")
    names = {model.joint(j).name for j in movable(model)}
    equalities = {model.equality(e).name for e in range(model.neq)}
    covered, ids = set(), set()
    for m in mechanisms:
        mid = m.get("id") if isinstance(m, dict) else None
        if not isinstance(mid, str) or not mid or mid in ids:
            problems.append(f"机构编号缺失或重复：{mid!r}"); continue
        ids.add(mid)
        if not isinstance(m.get("requirement"), str) or len(m["requirement"].strip()) < 4:
            problems.append(f"{mid}: requirement 须引用资料中的运动要求原文")
        joints = m.get("joints")
        if not isinstance(joints, list) or not joints or not all(isinstance(j, str) for j in joints):
            problems.append(f"{mid}: joints 须为非空关节名列表"); continue
        for name in joints:
            if name not in names:
                problems.append(f"{mid}: 关节 {name} 不存在或不是 hinge/slide/ball"); continue
            covered.add(name)
            j = model.joint(name).id
            if model.jnt_type[j] in (HINGE, SLIDE):
                lo, hi = model.jnt_range[j]
                if not model.jnt_limited[j] and not m.get("continuous"):
                    problems.append(f"{mid}: {name} 无限位；连续旋转须声明 continuous=true")
                elif model.jnt_limited[j] and not hi-lo > 1e-6:
                    problems.append(f"{mid}: {name} 行程为零，固定连接不能用零行程关节冒充")
        lock = m.get("lock")
        if lock is not None and lock not in equalities:
            problems.append(f"{mid}: lock {lock!r} 不是已命名的 equality")
        if m.get("actuation") not in ACTUATIONS:
            problems.append(f"{mid}: actuation 须为 {sorted(ACTUATIONS)} 之一")
    missing = sorted(names-covered)
    if missing:
        problems.append("以下可动关节未归入任何机构：" + ", ".join(missing[:40]))
    return {"id": "mechanism_contract", "status": "fail" if problems else "pass",
            "detail": "可动关节全部归入引用资料要求的机构，关节存在且限位有效",
            "mechanism_count": len(mechanisms), "movable_joint_count": len(names), "problems": problems[:60]}


def _ranges(model, m):
    out = []
    for name in m.get("joints", []):
        try:
            j = model.joint(name).id
        except KeyError:
            continue
        kind = model.jnt_type[j]
        if kind in (HINGE, SLIDE):
            lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-math.pi, math.pi)
            out.append((j, float(lo), float(hi)))
    return out


def _welded_followers(model, data):
    """Free bodies held by an active weld: (follower, holder, relative pose) at the current state."""
    free = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == FREE}
    out = []
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_WELD or not data.eq_active[e]:
            continue
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        for follower, holder in ((a, b), (b, a)):
            if follower in free:
                ra, pa = data.xmat[holder].reshape(3, 3), data.xpos[holder]
                rb, pb = data.xmat[follower].reshape(3, 3), data.xpos[follower]
                out.append((follower, holder, ra.T@rb, ra.T@(pb-pa)))
                break
    return out


def _follow(model, data, followers):
    """Carry welded free bodies with their holders after a kinematic pose change."""
    for _ in range(3):
        mujoco.mj_kinematics(model, data)
        for follower, holder, rotation, offset in followers:
            j = next(j for j in range(model.body_jntadr[follower], model.body_jntadr[follower]+model.body_jntnum[follower]))
            q = model.jnt_qposadr[j]
            ra, pa = data.xmat[holder].reshape(3, 3), data.xpos[holder]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (ra@rotation).flatten())
            data.qpos[q:q+3], data.qpos[q+3:q+7] = pa+ra@offset, quat
    mujoco.mj_forward(model, data)


def _contacts(model, data, tol):
    return {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)): float(c.dist)
            for c in data.contact[:data.ncon] if c.dist < -tol}


def mechanism_sweep(model, manifest, qpos0, tol):
    """Kinematic sweep of each mechanism over its full range from the declared home pose."""
    data = mujoco.MjData(model)
    rows, poses = [], []
    for m in declared(manifest):
        ranges = _ranges(model, m)
        if not ranges:
            continue
        data.qpos[:] = qpos0; mujoco.mj_forward(model, data)
        followers = _welded_followers(model, data)
        baseline = _contacts(model, data, tol)
        hits = []
        for t in np.linspace(0, 1, SWEEP_SAMPLES):
            data.qpos[:] = qpos0
            for j, lo, hi in ranges:
                data.qpos[model.jnt_qposadr[j]] = lo+t*(hi-lo)
            _follow(model, data, followers)
            for key, depth in _contacts(model, data, tol).items():
                if depth < baseline.get(key, 0.)-tol:
                    hits.append({"geom1": model.geom(key[0]).name, "geom2": model.geom(key[1]).name,
                                 "depth_mm": round(-depth*1000, 2), "fraction": round(float(t), 3)})
            if t in (0., 1.):
                poses.append((f"mech_{m['id']}_{'low' if t == 0 else 'high'}", data.qpos.copy()))
        unique = {(h["geom1"], h["geom2"]): h for h in hits}
        rows.append({"id": f"sampled_joint_sweep:{m['id']}", "status": "fail" if unique else "pass",
                     "detail": f"机构全行程 {SWEEP_SAMPLES} 点运动学扫描；相对初始姿态新增的碰撞体穿透",
                     "joints": [model.joint(j).name for j, _, _ in ranges],
                     "collisions": list(unique.values())[:30], "collision_count": len(unique)})
    return rows, poses


def _diag_mass(model, data):
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full, data.qM)
    return np.maximum(np.diag(full), 1e-6)


def effort(model, j):
    return DRIVE_FORCE_N if model.jnt_type[j] == SLIDE else DRIVE_TORQUE_NM


def _drive(model, data, targets, mass, steps, ranges=None):
    """Saturated PD drive in generalized forces only; never writes joint positions.

    Gravity/bias and the joint's own spring are compensated, so only contacts, limits and
    equality constraints can stop the motion: exactly what the probe is meant to find.
    The PD effort saturates at an operator-scale force/torque; the gain is as stiff as
    the time step allows so small parts are not starved of force.
    """
    dt = model.opt.timestep
    gains, gains_span = {}, {}
    for j, _ in targets:
        v = model.jnt_dofadr[j]
        span = max(next((hi-lo for jj, lo, hi in (ranges or []) if jj == j), 1.), 1e-4)
        gains_span[j] = span
        w = min(.5/dt, max(2*math.pi*1.5, math.sqrt(effort(model, j)/(mass[v]*span))))
        gains[j] = (mass[v]*w*w, 2*mass[v]*w)
    for _ in range(steps):
        data.qfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        for j, target in targets:
            q, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
            kp, kd = gains[j]
            spring = float(model.jnt_stiffness[j])*(data.qpos[q]-model.qpos_spring[q])
            push = float(np.clip(kp*(target-data.qpos[q])-kd*data.qvel[v], -effort(model, j), effort(model, j)))
            data.qfrc_applied[v] = data.qfrc_bias[v]+spring+float(model.dof_damping[v])*data.qvel[v]+push
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            return False
        if all(abs(target-data.qpos[model.jnt_qposadr[j]]) <= .01*max(gains_span[j], 1e-4)
               and abs(data.qvel[model.jnt_dofadr[j]]) < 1e-3 for j, target in targets):
            break
    data.qfrc_applied[:] = 0
    return True


def _subtree(model, bodies):
    out = set(bodies)
    for b in range(model.nbody):
        p = b
        while p > 0 and p not in out:
            p = int(model.body_parentid[p])
        if p in out:
            out.add(b)
    return out


def _blocking(model, data, joints, limit=8):
    """Deepest contacts between the driven subtree and everything else, for the repair hint."""
    moving = _subtree(model, {int(model.jnt_bodyid[j]) for j in joints})
    rows = []
    for c in data.contact[:data.ncon]:
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 in moving) != (b2 in moving):
            rows.append({"geom1": model.geom(int(c.geom1)).name, "geom2": model.geom(int(c.geom2)).name,
                         "depth_mm": round(max(0., -float(c.dist))*1000, 2)})
    active = [model.equality(e).name for e in range(model.neq) if data.eq_active[e]
              and ({int(model.eq_obj1id[e]), int(model.eq_obj2id[e])} & moving)]
    rows.sort(key=lambda r: -r["depth_mm"])
    return {"contacts": rows[:limit], "active_equalities_on_subtree": active[:limit]}


def mechanism_drive(model, manifest, qpos0):
    """Dynamic: each mechanism reaches both limits under drive; declared locks hold, then release."""
    rows = []
    steps = int(DRIVE_SECONDS/model.opt.timestep)
    for m in declared(manifest):
        ranges = _ranges(model, m)
        if not ranges or m.get("actuation") == "software":
            continue
        data = mujoco.MjData(model)
        data.qpos[:] = qpos0
        mass = _diag_mass(model, data)
        lock = None
        if m.get("lock"):
            try:
                lock = model.equality(m["lock"]).id
            except KeyError:
                lock = None
        evidence, ok = {}, True
        if lock is not None:
            data.eq_active[lock] = 1
            start = {j: float(data.qpos[model.jnt_qposadr[j]]) for j, _, _ in ranges}
            # Same bounded effort as an unlocked drive; a lock must resist it.
            # The same operator-scale effort an unlocked drive uses; a lock must resist it.
            stable = _drive(model, data, [(j, hi) for j, lo, hi in ranges], mass, steps, ranges)
            moved = max(abs(float(data.qpos[model.jnt_qposadr[j]])-start[j])/max(hi-lo, 1e-9) for j, lo, hi in ranges)
            evidence["locked_motion_fraction"] = round(moved, 4)
            held = stable and moved <= LOCK_HOLD_FRACTION
            ok &= held
            evidence["lock_holds"] = bool(held)
            data.eq_active[lock] = 0
        reached = {}
        for side in ("high", "low"):
            targets = [(j, hi if side == "high" else lo) for j, lo, hi in ranges]
            stable = _drive(model, data, targets, mass, steps, ranges)
            errors = [abs(float(data.qpos[model.jnt_qposadr[j]])-t)/max(hi-lo, 1e-9)
                      for (j, t), (_, lo, hi) in zip(targets, ranges)]
            reached[side] = bool(stable and max(errors) <= REACH_FRACTION)
            evidence[f"{side}_error_fraction"] = round(float(max(errors)), 4)
            if not reached[side]:
                evidence[f"{side}_blocking_contacts"] = _blocking(model, data, [j for j, _, _ in ranges])
            ok &= reached[side]
        evidence["reached"] = reached
        rows.append({"id": f"mechanism_drive:{m['id']}", "status": "pass" if ok else "fail",
                     "detail": f"饱和 PD 广义力（滑动≤{DRIVE_FORCE_N:g} N、转动≤{DRIVE_TORQUE_NM:g} N·m，已补偿重力与弹簧，不直接写关节位置）驱动到两端限位；声明的锁须在相同力下保持并可解除",
                     "joints": [model.joint(j).name for j, _, _ in ranges], **evidence})
    return rows


def with_ground(xml_path):
    """The asset plus a temporary ground at its declared support plane z=0."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.worldbody.add_geom(name="harness_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, .1],
                            pos=[0, 0, 0], contype=0xFFFF, conaffinity=0xFFFF, rgba=[.5, .5, .5, 1])
    return spec.compile()


def settle(xml_path, qpos0=None):
    """Gravity settle on the temporary ground: finite, warning-free, resting free parts stay put."""
    model = with_ground(xml_path)
    data = mujoco.MjData(model)
    if qpos0 is not None and len(qpos0) == model.nq:
        data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    free = [model.jnt_bodyid[j] for j in range(model.njnt) if model.jnt_type[j] == FREE]
    start = {b: (data.xpos[b].copy(), data.xquat[b].copy()) for b in free}
    for _ in range(int(SETTLE_SECONDS/model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            break
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    warnings = [int(w.number) for w in data.warning if w.number]
    drift = []
    for b in free:
        p0, q0 = start[b]
        dp = float(np.linalg.norm(data.xpos[b]-p0)) if finite else math.inf
        dot = abs(float(np.dot(q0, data.xquat[b]))) if finite else 0.
        angle = 2*math.acos(min(1., dot))
        if dp > DRIFT_M or angle > DRIFT_RAD:
            drift.append({"body": model.body(b).name, "displacement_mm": round(dp*1000, 2),
                          "rotation_deg": round(math.degrees(angle), 2)})
    ok = finite and not warnings and not drift
    return {"id": "numerical_stability", "status": "pass" if ok else "fail",
            "detail": f"在 z=0 临时地面上重力静置 {SETTLE_SECONDS:g} 秒：数值有限、无引擎警告、自由部件位移<2 mm 且转角<1°",
            "finite": finite, "warnings": warnings, "drifting_bodies": drift[:30],
            "simulation_seconds": float(data.time)}


def release_probes(model, manifest, qpos0):
    """Mechanically released connections: bodies held while active, separable once released."""
    rows = []
    mechanical = {e["id"] for e in manifest.get("mechanical_events", []) if isinstance(e, dict)}
    for connection in manifest.get("connections", []):
        if connection.get("release_event") not in mechanical:
            continue
        try:
            e = model.equality(connection["equality"]).id
        except KeyError:
            rows.append({"id": "release_probe:"+connection["id"], "status": "fail", "detail": "连接引用的 equality 不存在"})
            continue
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            a, b = int(model.jnt_bodyid[a]), int(model.jnt_bodyid[b]) if b >= 0 else 0
        child = b if b > 0 and model.body_mass[b] <= model.body_mass[a] or a == 0 else a
        anchor = a if child == b else b
        data = mujoco.MjData(model); data.qpos[:] = qpos0; mujoco.mj_forward(model, data)
        away = data.xpos[child]-data.xpos[anchor]
        directions = [np.array([0, 0, 1.])]+([away/np.linalg.norm(away)] if np.linalg.norm(away) > 1e-9 else [])
        results = {1: 0., 0: 0., "all": 0.}
        blockers = set()
        for active in (1, 0, "all"):
            for direction in directions:
                data = mujoco.MjData(model)
                data.qpos[:] = qpos0
                if active == "all":
                    data.eq_active[:] = 0
                else:
                    data.eq_active[e] = active
                mujoco.mj_forward(model, data)
                start = data.xpos[child].copy()
                force = 3*float(model.body_subtreemass[child])*9.81
                for _ in range(int(.5/model.opt.timestep)):
                    data.xfrc_applied[:] = 0
                    data.xfrc_applied[child, :3] = force*direction
                    mujoco.mj_step(model, data)
                moved = float(np.linalg.norm(data.xpos[child]-start)) if np.isfinite(data.qpos).all() else math.inf
                if active == "all" and moved < .005:
                    for c in data.contact[:data.ncon]:
                        pair = (int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2]))
                        if child in pair and pair[0] != pair[1]:
                            blockers.add(pair[1] if pair[0] == child else pair[0])
                results[active] = max(results[active], moved)
        # Held while active; released means the equality no longer holds it (it moves clearly
        # further); removable means a removal path exists once every connection is released.
        # A path blocked only by other still-connected parts is a sequencing fact, not a defect.
        held = results[1] <= .002
        released = results[0] >= results[1]+.001
        # Blocked only by parts that themselves move on a joint (a closed lid, a door) is
        # sequencing: open that mechanism first. Blocked by fixed material means trapped.
        movable_bodies = _subtree(model, {int(model.jnt_bodyid[j]) for j in range(model.njnt)})
        # The holder itself never counts as sequencing: a part that cannot leave its own
        # holder once released is trapped by that holder's geometry.
        sequenced = bool(blockers) and anchor not in blockers and blockers <= movable_bodies
        removable = results["all"] >= .005 or sequenced
        rows.append({"id": "release_probe:"+connection["id"], "status": "pass" if held and released and removable else "fail",
                     "detail": "以 3 倍重力外力沿竖直向上及远离锚定件方向拉动：连接生效时≤2 mm；解除后明显移动；全部连接解除后可移出≥5 mm",
                     "moved_when_active_mm": round(results[1]*1000, 2), "moved_when_released_mm": round(results[0]*1000, 2),
                     "moved_when_all_released_mm": round(results["all"]*1000, 2),
                     "blocked_by_other_parts": bool(removable and results[0] < .005),
                     "requires_opening": sorted(model.body(b).name for b in blockers) if sequenced else [],
                     "trapped_by": [] if removable else sorted(model.body(b).name for b in blockers)[:12],
                     "body": model.body(child).name})
    return rows
