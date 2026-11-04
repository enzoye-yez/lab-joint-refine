"""Rendering that never takes a run down: a failed frame is recorded as an error for that frame only."""
import math
import os

import mujoco
import numpy as np

GL_ENVIRONMENT = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "__EGL_VENDOR_LIBRARY_FILENAMES", "DISPLAY",
                  "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")


def worker_environment(extra=None):
    """Environment for a child process: keeps the GL settings, drops everything else (no credentials)."""
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": os.environ.get("HOME", "/tmp"),
           "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    env.update({k: os.environ[k] for k in GL_ENVIRONMENT if k in os.environ})
    env.update(extra or {})
    return env


def frame(model, points, azimuth, elevation, aspect=1.0, padding=1.15, minimum=0.18):
    """A free camera at the given angles that keeps all `points` in view (ported)."""
    points = np.asarray(points, float)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    yaw, pitch = np.radians([azimuth, elevation])
    forward = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    up = np.cross(right, forward)
    offsets = points - centre
    depth = offsets @ forward
    tangent = math.tan(math.radians(model.vis.global_.fovy) / 2)
    distance = max(np.max(np.abs(offsets @ up) * padding / tangent - depth),
                   np.max(np.abs(offsets @ right) * padding / (aspect * tangent) - depth), minimum)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = centre, float(distance), float(azimuth), float(elevation)
    return camera


def close_up(model, data, sites_or_points, azimuth, elevation, radius=0.12, aspect=1.0):
    """Frame a few points of interest (e.g. a TCP and a handle) with `radius` of context around them."""
    points = np.asarray(sites_or_points, float).reshape(-1, 3)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * radius
    return frame(model, (points[:, None, :] + corners).reshape(-1, 3), azimuth, elevation, aspect)


class Camera:
    def __init__(self, model, width=640, height=480):
        self.model, self.width, self.height = model, width, height
        self.renderer, self.error = None, None
        try:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self.renderer = mujoco.Renderer(model, height, width, max_geom=max(10000, model.ngeom + 2000))
        except Exception as exc:  # no GL here: every frame reports this error, the run continues
            self.error = f"{type(exc).__name__}: {exc}"

    def render(self, data, camera):
        """(image or None, error or None)."""
        if self.renderer is None:
            return None, self.error
        try:
            self.renderer.update_scene(data, camera)
            return self.renderer.render().copy(), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
