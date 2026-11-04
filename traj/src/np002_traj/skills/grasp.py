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


def outward_normals(model, data, interface):
    if interface.approach is not None and interface.site >= 0:
        return [data.site_xmat[interface.site].reshape(3, 3) @ interface.approach]
    # No declared approach: try from above and from the four horizontal sides.
    return [np.array(v, float) for v in ([0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]


def local_points(model, data, body, centre, radius=0.01):
    """Box corners of the part's collision geoms that pass within `radius` of the interface site.

    Only geometry at the site counts: a handle on a large door is measured, not the whole door.
    """
    own = np.flatnonzero(np.isin(model.geom_bodyid, list(subtree_bodies(model, body))))
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    points = []
    for g in own:
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        local = rot.T @ (centre - data.geom_xpos[g]) - c
        if np.linalg.norm(np.maximum(np.abs(local) - h, 0)) <= radius:
            points.append(data.geom_xpos[g] + (c + corners * h) @ rot.T)
    return np.concatenate(points) if points else np.zeros((0, 3))


def grasp_frames(model, data, interface, max_width, angles=12):
    """Candidate grasps: for each outward normal, the closing axes along which the part fits."""
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    points = local_points(model, data, interface.body, centre)
    out = []
    for normal in outward_normals(model, data, interface):
        z = -normal / np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(z))]
        u0 = np.cross(z, helper)
        u0 /= np.linalg.norm(u0)
        u1 = np.cross(z, u0)
        widths = []
        for k in range(angles):
            theta = np.pi * k / angles
            y = np.cos(theta) * u0 + np.sin(theta) * u1
            width = float(np.ptp(points @ y)) if len(points) else np.inf
            widths.append((width, y))
        for width, y in sorted(widths, key=lambda w: w[0])[:3]:
            if width <= max_width:
                for sign in (1, -1):
                    out.append(Frame(centre, _frame(z, sign * y), width))
    return out


def contact_frames(model, data, interface, finger_reach):
    """Frames for pushing with the closed fingertips along -normal (press, nudge)."""
    centre = data.site_xpos[interface.site].copy() if interface.site >= 0 else data.xpos[interface.body].copy()
    out = []
    for normal in outward_normals(model, data, interface):
        z = -normal / np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(z))]
        for y in (np.cross(z, helper), -np.cross(z, helper), np.cross(z, np.cross(z, helper))):
            # TCP sits finger_reach behind the fingertip, which touches the surface.
            out.append(Frame(centre - z * finger_reach, _frame(z, y), 0.0))
    return out
