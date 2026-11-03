"""Framework-owned motion checks for one asset: what moves, whether it can, and what holds.

The author declares `mechanisms` in interfaces.json, each tied to the requirement text it
implements and to real joints and locks. These checks never trust that declaration:
  mechanism_contract  every movable joint belongs to a declared mechanism with valid limits
  sampled_joint_sweep each mechanism swept across its range; new collider penetration fails
  mechanism_retention a slide mechanism keeps its moving part engaged with the guide it
                      rests in at home, at both limits
  mechanism_drive     a bounded PD drive moves each mechanism to both limits in simulation;
                      a declared lock must hold against the same drive and be released by its
                      declared event (a trigger joint driven past its threshold, a button, software)
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
# A slide's moving part keeps its centre of mass over the guide it rests in at home; an
# extending drawer or tray declares extends=true and keeps this share of its home engagement.
EXTEND_FRACTION = .15
OVERRUN_M = .005
GUIDE_GAP_M = .002


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


def _interval(model, data, geoms, axis):
    """Span of the geoms' world bounding boxes projected on a unit axis."""
    lo, hi = math.inf, -math.inf
    for g in geoms:
        rotation = data.geom_xmat[g].reshape(3, 3)
        centre = axis@(data.geom_xpos[g]+rotation@model.geom_aabb[g, :3])
        radius = float(np.abs(rotation.T@axis)@model.geom_aabb[g, 3:])
        lo, hi = min(lo, float(centre)-radius), max(hi, float(centre)+radius)
    return lo, hi


def _overlap(a, b):
    return max(0., min(a[1], b[1])-max(a[0], b[0]))


def _guides(model, data, moving):
    """Fixed-side geoms within GUIDE_GAP_M of the moving geoms: what the part slides in or on."""
    fromto = np.zeros(6)
    others = [g for g in range(model.ngeom) if g not in moving and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    near = set()
    for g in moving:
        for h in others:
            if h in near or np.linalg.norm(data.geom_xpos[g]-data.geom_xpos[h]) > model.geom_rbound[g]+model.geom_rbound[h]+GUIDE_GAP_M:
                continue
            if mujoco.mj_geomDistance(model, data, g, h, GUIDE_GAP_M, fromto) < GUIDE_GAP_M:
                near.add(h)
    return sorted(near)


def _outside(model, data, part):
    """How far the moving subtree's centre of mass lies beyond its guide's span along the axis."""
    lo, hi = _interval(model, data, part["guides"], part["axis"])
    mass = model.body_mass[part["bodies"]]
    centre = float(part["axis"]@(mass@data.xipos[part["bodies"]])/max(mass.sum(), 1e-12))
    return max(0., lo-centre, centre-hi)


def mechanism_retention(model, manifest, qpos0):
    """Slides keep their moving part in its guide at both limits.

    The guide is the fixed geoms the moving subtree touches (within GUIDE_GAP_M) at home. At each
    limit the subtree's centre of mass may lie outside the guide's span along the slide axis by
    at most OVERRUN_M more than it does at home, so a door cannot run off its track while a
    button standing proud of its panel is fine; a mechanism declared extends=true (a drawer or
    tray pulled out) instead keeps EXTEND_FRACTION of its home overlap with the guide."""
    data = mujoco.MjData(model)
    rows = []
    for m in declared(manifest):
        ranges = [(j, lo, hi) for j, lo, hi in _ranges(model, m) if model.jnt_type[j] == SLIDE]
        if not ranges:
            continue
        extends = m.get("extends") is True
        data.qpos[:] = qpos0; mujoco.mj_forward(model, data)
        followers = _welded_followers(model, data)
        parts = []
        for j, lo, hi in ranges:
            bodies = _subtree(model, {int(model.jnt_bodyid[j])})
            moving = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
            axis = data.xaxis[j].copy()
            guides = _guides(model, data, moving)
            home = _overlap(_interval(model, data, moving, axis), _interval(model, data, guides, axis)) if guides else 0.
            part = {"joint": j, "bodies": sorted(bodies), "moving": moving, "axis": axis, "guides": guides, "home": home}
            part["outside"] = _outside(model, data, part) if guides else 0.
            parts.append(part)
        problems, kept = [], {}
        for side in ("low", "high"):
            data.qpos[:] = qpos0
            for j, lo, hi in ranges:
                data.qpos[model.jnt_qposadr[j]] = lo if side == "low" else hi
            _follow(model, data, followers)
            for part in parts:
                name = model.joint(part["joint"]).name
                if part["home"] <= 1e-6:
                    if side == "low":
                        problems.append(f"{name}: 初始姿态下活动件没有贴合任何导向材料（{GUIDE_GAP_M*1000:g} mm 内）")
                    continue
                guide = _interval(model, data, part["guides"], part["axis"])
                share = float(_overlap(_interval(model, data, part["moving"], part["axis"]), guide)/part["home"])
                overrun = _outside(model, data, part)-part["outside"]
                kept[f"{name}:{side}"] = {"engagement": round(share, 3), "overrun_mm": round(overrun*1000, 1),
                                          "guide_mm": [round(guide[0]*1000, 1), round(guide[1]*1000, 1)]}
                span = f"导向沿轴范围 {guide[0]*1000:.1f}–{guide[1]*1000:.1f} mm"
                if extends and share < EXTEND_FRACTION:
                    problems.append(f"{name} 在 {side} 端只保留 {share:.0%} 的导向啮合（拉出件需≥{EXTEND_FRACTION:.0%}），{span}")
                elif not extends and overrun > OVERRUN_M:
                    problems.append(f"{name} 在 {side} 端质心比初始位置多越出导向 {overrun*1000:.1f} mm（{span}，允许 {OVERRUN_M*1000:g} mm）；"
                                    "门、盖板不得滑出导轨；资料要求拉出的抽屉/托盘须声明 extends=true")
        rows.append({"id": f"mechanism_retention:{m['id']}", "status": "fail" if problems else "pass",
                     "detail": f"滑动件在两端限位质心越出导向的距离不得比初始位置多 {OVERRUN_M*1000:g} mm 以上（声明 extends=true 的抽屉/托盘须保留≥{EXTEND_FRACTION:.0%} 啮合）",
                     "engagement": kept, "problems": problems[:20],
                     "guides": {model.joint(p["joint"]).name: [model.geom(g).name for g in p["guides"]][:12] for p in parts}})
    return rows


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


def _release_event(manifest, m):
    """The event that releases a mechanism's lock: its own release_event, else the release
    event of the connection built on the same equality."""
    if m.get("release_event"):
        return m["release_event"]
    return next((c.get("release_event") for c in manifest.get("connections", []) if c.get("equality") == m["lock"]), None)


def _trigger(model, manifest, event):
    """(joint id, threshold) whose crossing is the event, 'software', or an error message."""
    software = {i["id"] for i in manifest.get("interfaces", []) if i.get("kind") == "software"}
    if event in software:
        return "software"
    button = next((b for b in manifest.get("buttons", []) if b.get("event") == event), None)
    if button:
        return model.joint(button["joint"]).id, float(button["pressed_at"])
    mechanical = next((e for e in manifest.get("mechanical_events", []) if e.get("id") == event), None)
    if mechanical and "joint" in mechanical:
        return model.joint(mechanical["joint"]).id, float(mechanical["at"])
    if mechanical:
        return ("manual", mechanical["interface"])
    return f"锁没有声明解除事件（mechanisms.release_event，或同一 equality 的 connection.release_event）"


def _unlock(model, data, manifest, m, lock, mass, steps):
    """Release a lock the way an operator would: drive its trigger joint past the threshold
    with the same bounded effort, and only then switch the equality off."""
    event = _release_event(manifest, m)
    trigger = _trigger(model, manifest, event)
    if isinstance(trigger, str) and trigger != "software":
        return False, {"unlock": trigger}
    if trigger == "software":
        data.eq_active[lock] = 0
        return True, {"unlock": "software"}
    if trigger[0] == "manual":
        # Without a trigger joint the event must be the operator working the locked part itself
        # (a snap or detent cap): a separate latch or lever has to really move.
        interface = next((i for i in manifest.get("interfaces", []) if i.get("id") == trigger[1]), {})
        moving = _subtree(model, {int(model.jnt_bodyid[j]) for j, _, _ in _ranges(model, m)})
        try:
            on_part = model.body(interface.get("body", "")).id in moving
        except KeyError:
            on_part = False
        if not on_part:
            return False, {"unlock": f"机械解锁事件 {event} 的操作接口不在被锁部件上；独立卡扣/拨杆须声明触发关节 joint 与阈值 at，框架据此真实驱动解锁"}
        data.eq_active[lock] = 0
        return True, {"unlock": {"manual_on_locked_part": trigger[1], "approximation": "直接操作被锁部件解除（按扣/止动近似），未模拟解除力阈值"}}
    crossed, evidence = _drive_past(model, data, *trigger, mass, steps)
    if crossed:
        data.eq_active[lock] = 0
    return crossed, {"unlock": evidence}


def _drive_past(model, data, j, at, mass, steps):
    """Drive one trigger joint towards and past its threshold with the bounded effort."""
    q = model.jnt_qposadr[j]
    lo, hi = (float(x) for x in model.jnt_range[j])
    target = hi if at >= data.qpos[q] else lo
    stable = _drive(model, data, [(j, target)], mass, steps, [(j, lo, hi)])
    crossed = bool(stable and (data.qpos[q] >= at if target == hi else data.qpos[q] <= at))
    evidence = {"joint": model.joint(j).name, "threshold": at, "reached": round(float(data.qpos[q]), 5), "crossed": crossed}
    if not crossed:
        evidence["blocking_contacts"] = _blocking(model, data, [j])
    return crossed, evidence


def mechanism_drive(model, manifest, qpos0, stops=None):
    """Dynamic: each mechanism reaches both limits under drive; declared locks hold, then release.

    stops, when given, collects (label, qpos, blocking geom ids) where a drive stopped short,
    so the repair renders show the real stuck pose with its blockers highlighted."""
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
            released, unlock = _unlock(model, data, manifest, m, lock, mass, steps)
            evidence.update(unlock)
            ok &= released
            if not released:
                evidence["reached"] = {}
                rows.append({"id": f"mechanism_drive:{m['id']}", "status": "fail",
                             "detail": "锁未能由其声明的解除事件通过真实动作解除，机构未继续驱动",
                             "joints": [model.joint(j).name for j, _, _ in ranges], **evidence})
                continue
        reached = {}
        for side in ("high", "low"):
            targets = [(j, hi if side == "high" else lo) for j, lo, hi in ranges]
            stable = _drive(model, data, targets, mass, steps, ranges)
            errors = [abs(float(data.qpos[model.jnt_qposadr[j]])-t)/max(hi-lo, 1e-9)
                      for (j, t), (_, lo, hi) in zip(targets, ranges)]
            reached[side] = bool(stable and max(errors) <= REACH_FRACTION)
            evidence[f"{side}_error_fraction"] = round(float(max(errors)), 4)
            if not reached[side]:
                blocking = _blocking(model, data, [j for j, _, _ in ranges])
                evidence[f"{side}_blocking_contacts"] = blocking
                if stops is not None:
                    stops.append((f"stop_{m['id']}_{side}", data.qpos.copy(),
                                  {model.geom(c[k]).id for c in blocking["contacts"] for k in ("geom1", "geom2")}))
            ok &= reached[side]
        evidence["reached"] = reached
        rows.append({"id": f"mechanism_drive:{m['id']}", "status": "pass" if ok else "fail",
                     "detail": f"饱和 PD 广义力（滑动≤{DRIVE_FORCE_N:g} N、转动≤{DRIVE_TORQUE_NM:g} N·m，已补偿重力与弹簧，不直接写关节位置）驱动到两端限位；声明的锁须在相同力下保持，并由其解除事件（触发关节越过阈值、按钮或软件）解除",
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


def _carried_mass(model, body, active, exclude=()):
    """What a pull on body has to lift: its subtree plus free parts welded or connected to it
    through active equalities, transitively (a tray carries the tips seated in it)."""
    group = _subtree(model, {body})
    free = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == FREE}
    grew = True
    while grew:
        grew = False
        for e in range(model.neq):
            if not active[e] or model.eq_type[e] not in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
                continue
            a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            for x, y in ((a, b), (b, a)):
                if x in group and y in free and y not in group and y not in exclude:
                    group |= _subtree(model, {y}); grew = True
    return float(sum(model.body_mass[b] for b in group))


def _pull(model, qpos0, active, pulls, anchors, stop=None, masses=None, com=True):
    """Pull bodies with 3x the weight they carry for up to 0.5 s; return each body's displacement
    in its anchor's frame (body 0 is the world), so a holder that yields or turns does not count.

    Displacement is the larger move of the body's origin and, with com, its centre of mass, so
    a lid turning about a hinge through its own origin still registers. A held check uses the
    origin alone: a weld's small angular give swings a long part's centre of mass by millimetres.
    pulls maps body -> unit direction; masses maps body -> carried kg (default: subtree mass).
    The run ends early once every body has either moved stop metres or come to rest (under
    0.05 mm in 0.1 s), since neither will change its verdict."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.eq_active[:] = active
    mujoco.mj_forward(model, data)
    def offset(b):
        a = anchors[b]
        points = [data.xpos[b], data.xipos[b]] if com else [data.xpos[b]]
        return (np.vstack(points)-data.xpos[a])@data.xmat[a].reshape(3, 3)
    start = {b: offset(b).copy() for b in pulls}
    masses = masses or {}
    force = {b: 3*masses.get(b, float(model.body_subtreemass[b]))*9.81*np.asarray(d) for b, d in pulls.items()}
    moved = lambda: {b: float(np.linalg.norm(offset(b)-start[b], axis=1).max()) for b in pulls}
    window = max(1, int(.1/model.opt.timestep))
    previous = moved()
    # MuJoCo resets a diverged simulation to its initial state; the warning is the only trace.
    diverged = lambda: not np.isfinite(data.qpos).all() or data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
    for step in range(1, int(.5/model.opt.timestep)+1):
        data.xfrc_applied[:] = 0
        for b, f in force.items():
            data.xfrc_applied[b, :3] = f
        mujoco.mj_step(model, data)
        if step % window == 0:
            if diverged():
                break
            now = moved()
            if all((stop is not None and now[b] >= stop) or abs(now[b]-previous[b]) < 5e-5 for b in pulls):
                break
            previous = now
    if diverged():
        return {b: math.inf for b in pulls}, data
    return moved(), data


def _mm(metres):
    """Millimetres for a report; a diverged run has no displacement to state."""
    return None if math.isinf(metres) else round(metres*1000, 2)


def release_probes(model, manifest, qpos0):
    """Mechanically released connections: bodies held while active, separable once released.

    Connections are first pulled together, one run per state and direction; those that do not
    pass are pulled again in ever smaller groups down to one, so a shared run can confirm a
    pass but never decides a failure."""
    rows, targets = [], []
    mechanical = {e["id"] for e in manifest.get("mechanical_events", []) if isinstance(e, dict)}
    data = mujoco.MjData(model); data.qpos[:] = qpos0; mujoco.mj_forward(model, data)
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
        away = data.xpos[child]-data.xpos[anchor]
        directions = [np.array([0, 0, 1.])]+([away/np.linalg.norm(away)] if np.linalg.norm(away) > 1e-9 else [])
        trigger = None
        event = next(x for x in manifest.get("mechanical_events", []) if x["id"] == connection["release_event"])
        if "joint" in event:
            probe = mujoco.MjData(model); probe.qpos[:] = qpos0
            crossed, trigger = _drive_past(model, probe, model.joint(event["joint"]).id, float(event["at"]),
                                           _diag_mass(model, probe), int(DRIVE_SECONDS/model.opt.timestep))
        targets.append({"connection": connection, "eq": e, "child": child, "anchor": anchor, "directions": directions,
                        "trigger": trigger})
    # Blocked only by parts that themselves move on a joint (a closed lid, a door) is
    # sequencing: open that mechanism first. Blocked by fixed material means trapped.
    # The holder itself never counts as sequencing: a part that cannot leave its own
    # holder once released is trapped by that holder's geometry.
    movable_bodies = _subtree(model, {int(model.jnt_bodyid[j]) for j in range(model.njnt)})

    def opened(blockers):
        """Home pose with every mechanism that moves a blocker set to its far limit."""
        qpos = qpos0.copy()
        names = []
        for m in declared(manifest):
            ranges = _ranges(model, m)
            if ranges and _subtree(model, {int(model.jnt_bodyid[j]) for j, _, _ in ranges}) & blockers:
                names.append(m["id"])
                for j, lo, hi in ranges:
                    q = model.jnt_qposadr[j]
                    qpos[q] = hi if abs(hi-qpos0[q]) >= abs(lo-qpos0[q]) else lo
        data = mujoco.MjData(model); data.qpos[:] = qpos0; mujoco.mj_forward(model, data)
        followers = _welded_followers(model, data)
        data.qpos[:] = qpos; _follow(model, data, followers)
        return data.qpos.copy(), names

    def measure(group):
        """Held: the group's equalities on; released: off; all: every equality off.

        A part blocked only by other moving parts is pulled again with those mechanisms opened;
        it counts as removable only if it then really comes out."""
        on, off = model.eq_active0.copy(), model.eq_active0.copy()
        for t in group:
            on[t["eq"]], off[t["eq"]] = 1, 0
            t.update(held=0., released=0., all=0., blockers=set(), diverged=False, opened=[], after_opening=None)
        carried = off.copy()
        masses = {t["child"]: _carried_mass(model, t["child"], carried, exclude={t["anchor"]}) for t in group}
        for k in range(2):
            pulls = {t["child"]: t["directions"][k] for t in group if len(t["directions"]) > k}
            if not pulls:
                continue
            anchors = {t["child"]: t["anchor"] for t in group}
            held, _ = _pull(model, qpos0, on, pulls, anchors, masses=masses, com=False)
            released, _ = _pull(model, qpos0, off, pulls, anchors, stop=.005, masses=masses)
            free, data = _pull(model, qpos0, np.zeros_like(on), pulls, anchors, stop=.005, masses=masses)
            for t in group:
                child = t["child"]
                if child not in pulls:
                    continue
                t["held"] = max(t["held"], held[child])
                t["released"] = max(t["released"], released[child])
                t["all"] = max(t["all"], free[child])
                t["diverged"] |= math.inf in (held[child], released[child], free[child])
                if free[child] < .005:
                    for c in data.contact[:data.ncon]:
                        pair = (int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2]))
                        if child in pair and pair[0] != pair[1]:
                            t["blockers"].add(pair[1] if pair[0] == child else pair[0])
        for t in group:
            t["sequenced"] = bool(t["blockers"]) and t["anchor"] not in t["blockers"] and t["blockers"] <= movable_bodies
        # Parts blocked only by moving parts are pulled again, together, with those mechanisms opened.
        waiting = [t for t in group if t["all"] < .005 and t["sequenced"]]
        if waiting:
            qpos, names = opened(set().union(*(t["blockers"] for t in waiting)))
            for k in range(2):
                pulls = {t["child"]: t["directions"][k] for t in waiting if len(t["directions"]) > k}
                if not pulls:
                    continue
                after, _ = _pull(model, qpos, np.zeros_like(on), pulls, {t["child"]: t["anchor"] for t in waiting},
                                 stop=.005, masses=masses)
                for t in waiting:
                    if t["child"] in pulls:
                        t["opened"] = names
                        t["after_opening"] = max(t["after_opening"] or 0., after[t["child"]])
                        t["diverged"] |= math.isinf(after[t["child"]])
        for t in group:
            # Held while active; released means the equality no longer holds it (it moves clearly
            # further); removable means a removal path exists once every connection is released.
            t["removable"] = t["all"] >= .005 or bool(t["after_opening"] and t["after_opening"] >= .005)
            t["pass"] = (not t["diverged"] and t["held"] <= .002 and t["released"] >= t["held"]+.001 and t["removable"]
                         and (t["trigger"] is None or t["trigger"]["crossed"]))

    # A part that holds other connected parts (a tray of tips) is pulled on its own: pulling
    # it in the shared run would drag its dependents' anchor and confound their verdicts.
    children, anchors = [t["child"] for t in targets], {t["anchor"] for t in targets}
    shared = [t for t in targets if children.count(t["child"]) == 1 and t["child"] not in anchors]
    def resolve(group):
        """Connections that do not pass together are split in halves until each stands alone."""
        measure(group)
        pending = [t for t in group if not t["pass"]]
        if len(group) == 1 or not pending:
            return
        if len(pending) <= 2:
            for t in pending:
                resolve([t])
            return
        resolve(pending[:len(pending)//2])
        resolve(pending[len(pending)//2:])

    if shared:
        resolve(shared)
    for t in targets:
        if t not in shared:
            measure([t])
    for t in targets:
        rows.append({"id": "release_probe:"+t["connection"]["id"], "status": "pass" if t["pass"] else "fail",
                     "detail": ("拉动探测时仿真发散；" if t["diverged"] else "")+("释放事件的触发关节未能越过阈值；" if t["trigger"] and not t["trigger"]["crossed"] else "")+"以 3 倍重力外力沿竖直向上及远离锚定件方向拉动至多 0.5 s，位移相对锚定件计：连接生效时≤2 mm；解除后明显移动；"
                               "全部连接解除后可移出≥5 mm；被可动部件挡住时先把这些机构打开到远端再拉，拉得出才算（移动达 5 mm 或静止即停止，记录值为下限）；拉力按其承载的焊接部件总质量计",
                     "moved_when_active_mm": _mm(t["held"]), "moved_when_released_mm": _mm(t["released"]),
                     "moved_when_all_released_mm": _mm(t["all"]), "diverged": t["diverged"],
                     "blocked_by_other_parts": bool(t["removable"] and t["released"] < .005),
                     "requires_opening": t["opened"], "moved_after_opening_mm": _mm(t["after_opening"]) if t["after_opening"] is not None else None,
                     "trapped_by": [] if t["removable"] else sorted(model.body(b).name for b in t["blockers"])[:12],
                     "trigger": t["trigger"], "body": model.body(t["child"]).name})
    return rows
