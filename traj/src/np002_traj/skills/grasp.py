"""Tool frames for an interface from its site, approach axis and local geometry.

Tool frame convention (profile TCP site): z points from the hand into the part, y is the finger
closing axis. The approach axis in interfaces.json is the outward surface normal in the site frame.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..geometry import subtree_bodies


@dataclass
class Frame:
    position: np.ndarray       # TCP position at the grasp / contact
    rotation: np.ndarray       # world 3x3, columns = tool x, y, z
    width: float               # part width across the fingers (m), inf if not measured
    rank: int = 0              # 0: declared approach, 1: from above or a side, 2: oblique
    variant: int = 0           # 0: centred full-depth grasp; >0 shifted along the part or shallower
    quality: float = 1.0       # antipodal alignment of the contacts with the closing axis (cos)
    opening: float = None      # finger opening (each, m) for the approach; None: fully open
    axis: tuple = (0.0, 0.0, 1.0)   # tool-frame direction the hand moves in along (the suction cup: -x)

    def offset(self, distance):
        """The same frame moved `distance` back along its approach axis (a pre-grasp)."""
        return Frame(self.position - self.rotation @ np.asarray(self.axis) * distance, self.rotation, self.width,
                     self.rank, self.variant, self.quality, self.opening, self.axis)


def _frame(z, y):
    z = z / np.linalg.norm(z)
    y = y - z * (y @ z)
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


SIDES = [np.array(v, float) for v in ([0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]
# 45 degrees between the top and each side: low parts near a table top are often only reachable this way.
SIDES += [(np.array([0, 0, 1.0]) + side) / np.sqrt(2) for side in SIDES[1:5]]


def outward_normals(model, data, interface, fallback=True):
    """The declared approach first; then from above and the four horizontal sides."""
    declared = []
    if interface.approach is not None and interface.site >= 0:
        declared = [data.site_xmat[interface.site].reshape(3, 3) @ interface.approach]
    if not fallback and declared:
        return declared
    rest = [n for n in SIDES if not declared or n @ declared[0] < 0.95]
    return declared + rest


def approach_rank(model, data, interface, normal):
    if interface.approach is not None and interface.site >= 0:
        declared = data.site_xmat[interface.site].reshape(3, 3) @ interface.approach
        if normal @ declared > 0.95:
            return 0
    return 2 if 0.3 < abs(normal[2]) < 0.95 else 1


GOAL_MARGIN = 0.002     # m, hand to neighbours at the grasp (as in dock)
PART_RADIUS = 0.03      # m, part geometry around the site the pads may hold
ANTIPODAL_COS = np.cos(np.radians(30))   # contact direction within 30 degrees of the closing axis (mu 0.9: 42)


def scratch(model, data):
    """A kinematics-only copy of the world for hand placement queries."""
    copy = mujoco.MjData(model)
    copy.qpos[:], copy.mocap_pos[:], copy.mocap_quat[:] = data.qpos, data.mocap_pos, data.mocap_quat
    mujoco.mj_kinematics(model, copy)
    return copy


def nearby(model, data, centre, exclude, radius):
    """Collidable non-plane geoms within `radius` of a point, minus `exclude` (the robot, the part)."""
    usable = ((model.geom_contype != 0) | (model.geom_conaffinity != 0)) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    near = usable & (np.linalg.norm(data.geom_xpos - centre, axis=1) - model.geom_rbound < radius)
    near[list(exclude)] = False
    return np.flatnonzero(near)


def part_geoms(model, data, interface, centre, radius=PART_RADIUS):
    """The geometry the pads may hold: the declared grasp geoms, else the part's geoms near the site."""
    declared = (interface.extra.get("operation") or {}).get("grasp_geoms") or []
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in declared]
    if ids and all(i >= 0 for i in ids):
        return np.array(ids, int)
    own = np.flatnonzero(np.isin(model.geom_bodyid, list(subtree_bodies(model, interface.body))))
    keep = []
    for g in own:
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        local = rot.T @ (centre - data.geom_xpos[g]) - model.geom_aabb[g, :3]
        if np.linalg.norm(np.maximum(np.abs(local) - model.geom_aabb[g, 3:], 0)) <= radius:
            keep.append(int(g))
    return np.array(keep, int)


def grasp_frames(model, data, interface, probe, robot_geoms, angles=12, depths=(0.0, -0.005), shifts=(0.0, 0.008, -0.008)):
    """Antipodal parallel grasps synthesised on the part's collision geometry with the real hand.

    For each approach normal and closing axis the open hand is placed at the site, the pad-to-part
    gaps are measured and the hand is re-centred so both pads close by the same amount. A grasp
    is kept when both contacts face the pads (antipodal), the fingers clear the part except at the
    pads, and the hand clears every neighbour open, closed and on the way in. Returns
    (frames, report) where the report counts why candidates were rejected.
    """
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    part = part_geoms(model, data, interface, centre)
    report = {"candidates": 0, "no_part_between": 0, "not_antipodal": 0, "open_collision": 0,
              "closed_collision": 0, "corridor": 0, "blocking": {}}
    if not len(part):
        report["no_part_geometry"] = 1
        return [], report
    world = scratch(model, data)
    obstacles = nearby(model, world, centre, set(robot_geoms) | set(part.tolist()), 0.25)
    pads, fingers = probe.is_pad, ~probe.is_pin
    others = fingers & ~probe.is_pad
    open_ = probe.open
    out = []

    def blocked(kind, hit):
        report[kind] += 1
        if hit[2] >= 0:
            name = model.geom(hit[2]).name
            report["blocking"][name] = report["blocking"].get(name, 0) + 1

    for normal in outward_normals(model, data, interface):
        rank = approach_rank(model, data, interface, normal)
        z = -normal / np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(z))]
        u0 = np.cross(z, helper)
        u0 /= np.linalg.norm(u0)
        u1 = np.cross(z, u0)
        for k in range(angles):
            theta = np.pi * k / angles
            y = np.cos(theta) * u0 + np.sin(theta) * u1
            rotation = _frame(z, y)
            x = rotation[:, 0]
            for variant, (dz, dx) in enumerate((a, b) for b in shifts for a in depths):
                report["candidates"] += 1
                position = centre + z * dz + x * dx
                contact = None
                for centring in range(3):
                    contact, position, opening, side = _close(probe, world, part, position, rotation, y, open_)
                    if contact is None:
                        break
                    # Curved parts: slide along the pad width until the contacts sit on the closing axis.
                    across = np.mean([(g[3][3:] - position) @ x for g in contact])
                    if abs(across) < 0.001:
                        break
                    position = position + x * across
                if contact is None:
                    blocked("no_part_between", (0, 0, -1))
                    continue
                facing = []
                for g, sgn in zip(contact, side):
                    direction = g[3][3:] - g[3][:3]
                    norm = np.linalg.norm(direction)
                    facing.append(1.0 if norm < 1e-6 else float(-sgn * (direction / norm) @ y) * (1 if g[0] >= 0 else -1))
                if min(facing) < ANTIPODAL_COS:
                    blocked("not_antipodal", (0, 0, -1))
                    continue
                closing = open_ - opening
                width = 2 * opening
                # Pre-shape: approach with the fingers just wider than the part, not fully open.
                approach = min(open_, opening + PRESHAPE)
                probe.place(world, position, rotation, approach)
                # Open at the grasp: only the pads may come near the part; nothing touches a neighbour.
                hit = probe.distance(world, others, part, limit=0.01)
                if hit[0] < 0.0005:
                    blocked("open_collision", hit)
                    continue
                hit = probe.distance(world, fingers, obstacles, limit=0.01)
                if hit[0] < GOAL_MARGIN:
                    blocked("open_collision", hit)
                    continue
                # Closed onto the part: the fingers sweep inwards.
                probe.place(world, position, rotation, max(open_ - closing - 0.0005, 0.0))
                hit = probe.distance(world, others, part, limit=0.01)
                if hit[0] < -0.0005:
                    blocked("closed_collision", hit)
                    continue
                hit = probe.distance(world, fingers, obstacles, limit=0.01)
                if hit[0] < GOAL_MARGIN:
                    blocked("closed_collision", hit)
                    continue
                # On the way in along the tool axis.
                clear = True
                for back in (0.02, 0.05, 0.09):
                    probe.place(world, position - z * back, rotation, approach)
                    hit = probe.distance(world, fingers, np.concatenate([obstacles, part]), limit=0.01)
                    if hit[0] < GOAL_MARGIN:
                        blocked("corridor", hit)
                        clear = False
                        break
                if not clear:
                    continue
                quality = min(facing)
                for sign in (1, -1):
                    r = rotation if sign > 0 else np.column_stack([-rotation[:, 0], -rotation[:, 1], rotation[:, 2]])
                    out.append(Frame(position.copy(), r, width, rank, int(dz != 0) + int(dx != 0), quality, approach))
    report["blocking"] = dict(sorted(report["blocking"].items(), key=lambda kv: -kv[1])[:5])
    report["kept"] = len(out)
    out.sort(key=lambda f: (f.rank, f.variant, -f.quality))
    return out[:MAX_FRAMES], report


MAX_FRAMES = 96
PRESHAPE = 0.005        # m per finger beyond the part when approaching


def _close(probe, world, part, position, rotation, y, open_):
    """Close the coupled fingers step by step, re-centring the hand, until both pads touch the part.

    Returns (contacts, position, opening, sides) or (None, ...) if the pads close on nothing."""
    pads = probe.is_pad
    opening, side = open_, None
    for attempt in range(8):
        probe.place(world, position, rotation, opening)
        gaps = [probe.distance(world, pads & (np.arange(len(pads)) == i), part, limit=2 * open_ + 0.01)
                for i in np.flatnonzero(pads)]
        if any(not np.isfinite(g[0]) for g in gaps) or min(g[0] for g in gaps) < -0.001:
            return None, position, opening, side
        side = [np.sign((world.geom_xpos[g[1]] - position) @ y) for g in gaps]
        if max(g[0] for g in gaps) < 0.0008:
            return gaps, position, opening, side
        # Pad a closes along -y, pad b along +y: shift so both gaps are equal, close by that gap.
        shift = 0.5 * (gaps[0][0] * side[0] + gaps[1][0] * side[1])
        position = position - y * shift
        opening -= max(0.5 * (gaps[0][0] + gaps[1][0]) - 0.0003, 0.0)
        if opening <= 0.0005:
            break
    return None, position, opening, side
def _segment_hits(origins, direction, starts, length, centres, rotations, halves):
    """Which oriented boxes each segment origin + t * direction, t in [start, length], passes through (slab test)."""
    hit = np.zeros(len(centres), bool)
    for o, t0 in zip(origins, starts):
        local_o = np.einsum("nji,nj->ni", rotations, o - centres)       # R^T (o - c)
        local_d = np.einsum("nji,j->ni", rotations, direction)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / local_d
            t1, t2 = (-halves - local_o) * inv, (halves - local_o) * inv
        low = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
        high = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
        # A segment parallel to a slab is inside it only if its origin is.
        parallel = np.abs(local_d) < 1e-12
        outside = parallel & (np.abs(local_o) > halves)
        enter, leave = low.max(axis=1), high.min(axis=1)
        hit |= ~outside.any(axis=1) & (enter <= leave) & (leave >= t0) & (enter <= length)
    return hit


def enclosure(model, data, interface, own_bodies, reach=0.30, hand_radius=0.035, movable=None):
    """The first foreign collision geometry on every straight way out from the site, or None if one is free.

    Checks the declared approach and the fallback sides with a hand-wide corridor (the axis plus a
    ring of lines beyond the fingertips) against every nearby geometry's oriented bounding box,
    exactly, so thin lids and walls are found. A site behind a closed lid or door is enclosed and
    needs that part moved first.
    """
    centre = data.site_xpos[interface.site] if interface.site >= 0 else data.xpos[interface.body]
    usable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    near = np.flatnonzero(usable & ~np.isin(model.geom_bodyid, list(own_bodies)) &
                          (np.linalg.norm(data.geom_xpos - centre, axis=1) < model.geom_rbound + reach + 0.05)
                          & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE))
    rotations = data.geom_xmat[near].reshape(-1, 3, 3)
    centres = data.geom_xpos[near] + np.einsum("nij,nj->ni", rotations, model.geom_aabb[near, :3])
    halves = model.geom_aabb[near, 3:]
    first = None
    for normal in outward_normals(model, data, interface):
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        a = np.cross(normal, helper)
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        ring = [hand_radius * (np.cos(t) * a + np.sin(t) * b) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        origins = np.array([centre] + [centre + o for o in ring])
        starts = np.array([0.002] + [0.03] * len(ring))      # the ring starts behind the fingertips
        hits = _segment_hits(origins, normal, starts, reach, centres, rotations, halves)
        if not hits.any():
            return None
        # The blocker nearest to the site along this way out.
        along = (centres[hits] - centre) @ normal
        blocker = int(near[hits][np.argmin(along)])
        # Prefer a way out that is blocked only by something that can be moved (a lid, a door).
        if movable is not None and movable(blocker) and (first is None or not movable(first)):
            first = blocker
        first = blocker if first is None else first
    return first


def contact_frames(model, data, interface, probe, robot_geoms, pin_tip, angles=8):
    """Frames for pressing with the extended press pin along -normal; the hand must clear everything else.

    pin_tip: the pin tip relative to the TCP in the tool frame. Returns (frames, report).
    """
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    part = part_geoms(model, data, interface, centre, radius=0.01)
    world = scratch(model, data)
    obstacles = nearby(model, world, centre, set(robot_geoms) | set(part.tolist()), 0.25)
    everything = np.concatenate([obstacles, part])
    report = {"candidates": 0, "open_collision": 0, "corridor": 0, "blocking": {}}
    out = []
    for normal in outward_normals(model, data, interface, fallback=False):
        z = -normal / np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(z))]
        u0 = np.cross(z, helper)
        u0 /= np.linalg.norm(u0)
        u1 = np.cross(z, u0)
        for k in range(angles):
            theta = 2 * np.pi * k / angles
            rotation = _frame(z, np.cos(theta) * u0 + np.sin(theta) * u1)
            position = centre - rotation @ pin_tip           # the pin tip touches the site
            report["candidates"] += 1
            ok = True
            for back, which, against in ((0.0, ~probe.is_pin, everything), (0.0, probe.is_pin, obstacles),
                                         (0.03, np.ones(len(probe.geoms), bool), everything)):
                probe.place(world, position - z * back, rotation, 0.0, probe.pin_extended)
                hit = probe.distance(world, which, against, limit=0.01)
                if hit[0] < GOAL_MARGIN:
                    kind = "open_collision" if back == 0 else "corridor"
                    report[kind] += 1
                    name = model.geom(hit[2]).name
                    report["blocking"][name] = report["blocking"].get(name, 0) + 1
                    ok = False
                    break
            if ok:
                out.append(Frame(position, rotation, 0.0))
    report["blocking"] = dict(sorted(report["blocking"].items(), key=lambda kv: -kv[1])[:5])
    report["kept"] = len(out)
    return out, report


PUSHER_MARGIN = 0.0007  # m, the pin or blade itself to neighbours (position-controlled, FR3 repeats to 0.1 mm)


def push_frames(model, data, interface, probe, robot_geoms, pin_tip, direction, angles=8, shifts=(0.0, 0.008, -0.008),
                element="pin", depths=(0.0,), part_radius=0.015):
    """Frames for moving a part by pushing it with the side of the extended press pin along `direction`.

    For articulations the hand cannot grasp (a tab flush with a wall, a lid edge): the pin goes in
    beside the part across `direction` and its side is brought into contact; the part is then
    pushed along its own motion. Returns (frames, report); frame.position is the TCP.
    """
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    part = part_geoms(model, data, interface, centre, radius=part_radius)
    report = {"candidates": 0, "no_contact": 0, "open_collision": 0, "corridor": 0, "blocking": {}}
    if not len(part):
        return [], report
    world = scratch(model, data)
    obstacles = nearby(model, world, centre, set(robot_geoms) | set(part.tolist()), 0.25)
    everything = np.concatenate([obstacles, part])
    t = direction / np.linalg.norm(direction)
    pusher = probe.is_pin if element == "pin" else probe.is_blade
    extend = {"pin": probe.pin_extended} if element == "pin" else {"blade": probe.blade_extended}
    out = []
    for normal in outward_normals(model, data, interface):
        if abs(normal @ t) > 0.5:
            continue                                      # the pin goes in across the push direction
        z = -normal / np.linalg.norm(normal)
        z = z - t * (z @ t)
        z /= np.linalg.norm(z)
        helper = np.eye(3)[np.argmin(np.abs(z))]
        u0 = np.cross(z, helper)
        u0 /= np.linalg.norm(u0)
        u1 = np.cross(z, u0)
        across = np.cross(z, t)                           # along the part's edge, beside a latch or a catch
        for k, shift, depth in ((k, shift, depth) for depth in depths for shift in shifts for k in range(angles)):
            theta = 2 * np.pi * k / angles
            rotation = _frame(z, np.cos(theta) * u0 + np.sin(theta) * u1)
            report["candidates"] += 1
            # Tip a little behind the part along -t, then slide it along t until its side touches.
            # `depth` holds the tip back from the site along the way in (only an overhanging edge is free).
            tip = centre - t * 0.02 + across * shift - z * depth
            contact = None
            for attempt in range(5):
                position = tip - rotation @ pin_tip
                probe.place(world, position, rotation, 0.0, **extend)
                hit = probe.distance(world, pusher, part, limit=0.04)
                if not np.isfinite(hit[0]) or hit[0] < -0.0005:
                    break
                direction_ = hit[3][3:] - hit[3][:3]
                norm = np.linalg.norm(direction_)
                if norm > 1e-9 and (direction_ / norm) @ t < 0.7:
                    break
                if hit[0] < 0.001:
                    contact = position
                    break
                tip = tip + t * (hit[0] - 0.0005)
            if contact is None:
                report["no_contact"] += 1
                continue
            ok = True
            for back, which, against, margin in ((0.0, ~pusher, everything, GOAL_MARGIN), (0.0, pusher, obstacles, PUSHER_MARGIN),
                                                 (0.04, np.ones(len(probe.geoms), bool), everything, GOAL_MARGIN)):
                probe.place(world, contact - z * back - t * 0.0015, rotation, 0.0, **extend)
                hit = probe.distance(world, which, against, limit=0.01)
                if hit[0] < margin:
                    kind = "open_collision" if back == 0 else "corridor"
                    report[kind] += 1
                    name = model.geom(hit[2]).name
                    report["blocking"][name] = report["blocking"].get(name, 0) + 1
                    ok = False
                    break
            if ok:
                out.append(Frame(contact - t * 0.002, rotation, 0.0, approach_rank(model, data, interface, normal)))
    report["blocking"] = dict(sorted(report["blocking"].items(), key=lambda kv: -kv[1])[:5])
    report["kept"] = len(out)
    out.sort(key=lambda f: f.rank)
    return out, report


def _ray_part(model, data, geoms, point, direction):
    """Nearest hit of a ray on the given geoms: (distance, world normal, geom) or None."""
    best = None
    normal = np.zeros(3)
    for g in geoms:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            x = mujoco.mj_rayMesh(model, data, int(g), point, direction, normal)
        else:
            x = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], point, direction,
                                   int(model.geom_type[g]), normal)
        if x >= 0 and (best is None or x < best[0]):
            best = (x, normal.copy(), int(g))
    return best


def suction_frames(model, data, interface, probe, robot_geoms, cup, spacing=0.012, flat=0.0008, angles=8):
    """Frames that put the suction cup flat on a face of the part (a cover, a plate, a box lid).

    Faces are found by casting rays onto the part from each approach direction; a spot is kept when
    the centre and four rim points hit the part within `flat` of a plane facing the cup. The hand
    must clear everything there and on the way in. Spots near the centre of mass come first.
    """
    root = interface.mechanism.body if interface.mechanism.body >= 0 else interface.body
    bodies = subtree_bodies(model, root)
    part = np.array([g for g in np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))
                     if model.geom_contype[g] or model.geom_conaffinity[g]], int)
    report = {"candidates": 0, "not_flat": 0, "open_collision": 0, "corridor": 0, "blocking": {}}
    if not len(part):
        return [], report
    world = scratch(model, data)
    from .dock import FREE_MARGIN  # noqa: F401  (same margins as the dock search)
    mass_centre = sum(data.xipos[b] * model.body_mass[b] for b in bodies) / max(sum(model.body_mass[b] for b in bodies), 1e-9)
    low = np.min(data.geom_xpos[part] - model.geom_rbound[part, None], axis=0)
    high = np.max(data.geom_xpos[part] + model.geom_rbound[part, None], axis=0)
    obstacles = nearby(model, world, mass_centre, set(robot_geoms) | set(part.tolist()), float(np.linalg.norm(high - low)) / 2 + 0.2)
    everything = np.concatenate([obstacles, part])
    out = []
    for n in outward_normals(model, data, interface):
        a = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        span = np.abs(high - low)
        ra, rb = float(np.abs(a) @ span) / 2, float(np.abs(b) @ span) / 2
        centre = (low + high) / 2
        start = centre + n * (float(np.abs(n) @ span) / 2 + 0.05)
        for i in np.arange(-ra, ra + 1e-9, spacing):
            for j in np.arange(-rb, rb + 1e-9, spacing):
                origin = start + a * i + b * j
                hit = _ray_part(model, world, part, origin, -n)
                if hit is None or hit[1] @ n < 0.97:
                    continue
                report["candidates"] += 1
                spot = origin - n * hit[0]
                ok = True
                for o in (a, -a, b, -b):
                    rim = _ray_part(model, world, part, origin + o * cup["radius"], -n)
                    if rim is None or abs(rim[0] - hit[0]) > flat:
                        ok = False
                        break
                if not ok:
                    report["not_flat"] += 1
                    continue
                for k in range(angles):
                    theta = 2 * np.pi * k / angles
                    z = np.cos(theta) * a + np.sin(theta) * b
                    # Cup normal (tool -x) onto the face: tool x = n.
                    rotation = np.column_stack([n, np.cross(z, n), z])
                    position = spot + n * 0.0005 - rotation @ cup["face"]
                    good = True
                    for back, which, against in ((0.0, ~probe.is_cup, everything), (0.0, probe.is_cup, obstacles),
                                                 (0.04, np.ones(len(probe.geoms), bool), everything)):
                        probe.place(world, position + n * back, rotation, 0.0)
                        h = probe.distance(world, which, against, limit=0.01)
                        if h[0] < GOAL_MARGIN:
                            kind = "open_collision" if back == 0 else "corridor"
                            report[kind] += 1
                            name = model.geom(h[2]).name
                            report["blocking"][name] = report["blocking"].get(name, 0) + 1
                            good = False
                            break
                    if good:
                        rank = approach_rank(model, data, interface, n)
                        f = Frame(position, rotation, 0.0, rank, axis=tuple(float(v) for v in cup["normal"]))  # the cup moves along its normal
                        f.quality = -float(np.linalg.norm((spot - mass_centre) - n * ((spot - mass_centre) @ n)))
                        out.append(f)
    report["blocking"] = dict(sorted(report["blocking"].items(), key=lambda kv: -kv[1])[:5])
    report["kept"] = len(out)
    out.sort(key=lambda f: (f.rank, -f.quality))
    return out[:MAX_FRAMES], report
