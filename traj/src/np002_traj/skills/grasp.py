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

    def offset(self, distance):
        """The same frame moved `distance` back along the tool z axis (a pre-grasp)."""
        return Frame(self.position - self.rotation[:, 2] * distance, self.rotation, self.width, self.rank, self.variant)


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


def local_points(model, data, body, centre, half=0.04, spacing=0.004):
    """Points of a regular grid around the site that lie inside the part's collision geometry boxes."""
    own = np.flatnonzero(np.isin(model.geom_bodyid, list(subtree_bodies(model, body))))
    axis = np.arange(-half, half + 1e-9, spacing)
    grid = centre + np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(grid), bool)
    for g in own:
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        if np.any(np.abs(rot.T @ (centre - data.geom_xpos[g]) - c) - h > half):
            continue
        local = (grid - data.geom_xpos[g]) @ rot - c
        inside |= np.all(np.abs(local) <= h + spacing / 2, axis=1)
    return grid[inside]


def grasp_frames(model, data, interface, max_width, angles=12, depth=0.010, backoffs=(0.0, 0.004)):
    """Candidate grasps: for each outward normal, the closing axes along which the part fits.

    The width is measured on the part's own geometry between the pads: within `depth` of the
    site along the tool axis, so a wall behind a handle does not count.
    """
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    cloud = local_points(model, data, interface.body, centre)
    out = []
    for normal in outward_normals(model, data, interface):
        rank = approach_rank(model, data, interface, normal)
        z = -normal / np.linalg.norm(normal)
        points = cloud[np.abs((cloud - centre) @ z) <= depth] if len(cloud) else cloud
        if not len(points):
            continue
        helper = np.eye(3)[np.argmin(np.abs(z))]
        u0 = np.cross(z, helper)
        u0 /= np.linalg.norm(u0)
        u1 = np.cross(z, u0)
        widths = []
        for k in range(angles):
            theta = np.pi * k / angles
            y = np.cos(theta) * u0 + np.sin(theta) * u1
            along = (points - centre) @ y
            # Geometry reaching the edge of the sampled cube continues beyond it: no finger fits there.
            reaches_edge = np.max(np.abs(along)) >= 0.04 - 0.006
            widths.append((np.inf if reaches_edge else float(np.ptp(along)), y))
        for width, y in sorted(widths, key=lambda w: w[0])[:2]:
            if width <= max_width:
                x = np.cross(y, z)
                along = (points - centre) @ x
                # An elongated part (a handle, a rod) can be held anywhere along its length.
                reach = max(0.0, min(float(np.max(along)), -float(np.min(along))) - 0.006)
                shifts = (0.0,) + ((0.6 * reach, -0.6 * reach) if reach > 0.004 else ())
                for sign in (1, -1):
                    for back in backoffs:          # shallower grasps keep the fingertips off a support below
                        for shift in shifts:
                            out.append(Frame(centre - z * back + x * shift, _frame(z, sign * y), width, rank,
                                             int(back > 0) + int(shift != 0)))
    return out


def enclosure(model, data, interface, own_bodies, reach=0.30, step=0.01, hand_radius=0.035, movable=None):
    """The first foreign collision geometry on every straight way out from the site, or None if one is free.

    Checks the declared approach and the fallback sides; a site behind a closed lid or door is
    enclosed and needs that part moved first.
    """
    centre = data.site_xpos[interface.site] if interface.site >= 0 else data.xpos[interface.body]
    usable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    near = np.flatnonzero(usable & ~np.isin(model.geom_bodyid, list(own_bodies)) &
                          (np.linalg.norm(data.geom_xpos - centre, axis=1) < model.geom_rbound + reach + 0.05)
                          & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE))
    first = None
    for normal in outward_normals(model, data, interface):
        # A corridor as wide as the hand: the axis plus a ring of points around it, beyond the fingertips.
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        a = np.cross(normal, helper)
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        ring = [np.zeros(3)] + [hand_radius * (np.cos(t) * a + np.sin(t) * b) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        points = np.array([centre + normal * t + o for t in np.arange(step, reach + 1e-9, step) for o in ring
                           if t > 0.03 or not o.any()])
        blocker = None
        for g in near:
            rot = data.geom_xmat[g].reshape(3, 3)
            local = (points - data.geom_xpos[g]) @ rot - model.geom_aabb[g, :3]
            if np.any(np.all(np.abs(local) <= model.geom_aabb[g, 3:], axis=1)):
                blocker = g
                break
        if blocker is None:
            return None
        # Prefer a way out that is blocked only by something that can be moved (a lid, a door).
        if movable is not None and movable(blocker) and (first is None or not movable(first)):
            first = blocker
        first = blocker if first is None else first
    return first


def contact_frames(model, data, interface, finger_reach):
    """Frames for pushing with the closed fingertips along -normal (press, nudge)."""
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    out = []
    for normal in outward_normals(model, data, interface, fallback=False):
        z = -normal / np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(z))]
        for y in (np.cross(z, helper), -np.cross(z, helper), np.cross(z, np.cross(z, helper))):
            # TCP sits finger_reach behind the fingertip, which touches the surface.
            out.append(Frame(centre - z * finger_reach, _frame(z, y), 0.0))
    return out
