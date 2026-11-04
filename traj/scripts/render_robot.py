"""Render the assembled robot at home from three views into runs/robot_views.png (CPU EGL)."""
import mujoco
import numpy as np
from PIL import Image

from np002_traj.robot import ROOT

spec = mujoco.MjSpec.from_file(str(ROOT / "data/robot/model.xml"))
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.55, 0.57, 0.6, 1])
spec.worldbody.add_light(pos=[1.5, 1.0, 3.0], dir=[-0.4, -0.3, -1], diffuse=[0.8, 0.8, 0.8])
model = spec.compile()
model.vis.global_.offwidth, model.vis.global_.offheight = 640, 640
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 640, 640)
views = []
for azimuth, elevation in ((150, -20), (90, -10), (180, -60)):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.05, 0, 0.75]
    camera.distance, camera.azimuth, camera.elevation = 2.6, azimuth, elevation
    renderer.update_scene(data, camera)
    views.append(renderer.render())
renderer.close()
(ROOT / "runs").mkdir(exist_ok=True)
Image.fromarray(np.concatenate(views, axis=1)).save(ROOT / "runs/robot_views.png")
print(ROOT / "runs/robot_views.png")
