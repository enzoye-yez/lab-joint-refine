"""Render the least-colliding dock candidate of interfaces whose dock search failed.

  source scripts/env.sh && $PYTHON scripts/debug_dock.py runs/bench_08 [N]
"""
import json
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.argv += [] if len(sys.argv) > 2 else ["6"]
import run_benchmark as rb
from np002_traj.executor.session import Abort, Session
from np002_traj.record.camera import Camera, close_up
from np002_traj.skills.library import Context, SKILLS
from np002_traj.tasks.benchmark import task_for

rows = [json.loads(line) for line in open(f"{sys.argv[1]}/results.jsonl")]
tasks = [r["task"] for r in rows if r["reason"] == "no_dock"][: int(sys.argv[2])]
sys.argv = sys.argv[:1]
rb.init()
S = rb.STATE
m, robot = S["model"], S["robot"]
tiles = []
for name in tasks:
    interface = S["interfaces"][name]
    task = task_for(interface, m, 3.0)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("assembled").id)
    mujoco.mj_forward(m, d)
    ctx = Context(Session(m, d, robot), S["kinematics"], S["occupancy"], [], keep_closest=True)
    try:
        SKILLS[task.skill](ctx, interface, task)
        continue
    except Abort as exc:
        detail = exc.detail
    closest = detail.get("reasons", {}).get("_closest")
    if not closest:
        print(name, "no IK-feasible candidate at all", detail.get("reasons"))
        continue
    d.qpos[robot.base_qpos] = robot.base_joint_values(np.array(closest["base"]))
    d.qpos[robot.lift_qpos] = closest["lift"]
    d.qpos[robot.arms[closest["side"]].qpos] = closest["q"]
    for o in robot.arms:
        if o != closest["side"]:
            d.qpos[robot.arms[o].qpos] = robot.profile["arms"][o]["travel"]
    mujoco.mj_forward(m, d)
    print(name, json.dumps({k: v for k, v in closest.items() if k not in ("q",)}))
    site = d.site_xpos[interface.site] if interface.site >= 0 else d.xpos[interface.body]
    camera = Camera(m, 420, 320)
    heading = np.degrees(closest["base"][2])
    for azimuth in (heading + 140, heading - 140):
        image, _ = camera.render(d, close_up(m, d, [site, d.site_xpos[robot.arms[closest["side"]].tcp]], azimuth, -25, 0.2, 420 / 320))
        tile = Image.fromarray(image)
        ImageDraw.Draw(tile).text((6, 6), f"{name[:44]} | {closest['stage']} {closest['distance_m']*1e3:.0f} mm", fill=(255, 255, 0))
        tiles.append(np.asarray(tile))
    camera.close()
rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
Image.fromarray(np.concatenate(rows, 0)).save(rb.ROOT / "runs/debug_dock.png")
