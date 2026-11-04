"""Tool frames for an interface from its site, approach axis and local geometry.

Tool frame convention (profile TCP site): z points from the hand into the part, y is the finger
closing axis. The approach axis in interfaces.json is the outward surface normal in the site frame.
"""
from dataclasses import dataclass

import numpy as np

from ..geometry import subtree_bodies


@dataclass
class Frame:
    position: np.ndarray       # TCP position at the grasp / contact
    rotation: np.ndarray       # world 3x3, columns = tool x, y, z
    width: float               # part width across the fingers (m), inf if not measured

    def offset(self, distance):
        """The same frame moved `distance` back along the tool z axis (a pre-grasp)."""
        return Frame(self.position - self.rotation[:, 2] * distance, self.rotation, self.width)


def _frame(z, y):
    z = z / np.linalg.norm(z)
    y = y - z * (y @ z)
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


SIDES = [np.array(v, float) for v in ([0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]


def outward_normals(model, data, interface, fallback=True):
    """The declared approach first; then from above and the four horizontal sides."""
    declared = []
    if interface.approach is not None and interface.site >= 0:
        declared = [data.site_xmat[interface.site].reshape(3, 3) @ interface.approach]
    if not fallback and declared:
        return declared
    rest = [n for n in SIDES if not declared or n @ declared[0] < 0.95]
    return declared + rest


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


def grasp_frames(model, data, interface, max_width, angles=12, depth=0.010):
    """Candidate grasps: for each outward normal, the closing axes along which the part fits.

    The width is measured on the part's own geometry between the pads: within `depth` of the
    site along the tool axis, so a wall behind a handle does not count.
    """
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    cloud = local_points(model, data, interface.body, centre)
    out = []
    for normal in outward_normals(model, data, interface):
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
        for width, y in sorted(widths, key=lambda w: w[0])[:3]:
            if width <= max_width:
                for sign in (1, -1):
                    out.append(Frame(centre, _frame(z, sign * y), width))
    return out


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
