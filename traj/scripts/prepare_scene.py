"""Assemble the NP-002 room with our robot, settle it and cache scene.mjb with a `settled` keyframe.

Run once per scene/robot change (the source XML takes about two minutes to parse):
  source scripts/env.sh && $PYTHON scripts/prepare_scene.py
"""
import json
import time

import mujoco
import numpy as np

from np002_traj.robot import ROOT, Robot
from np002_traj.scene.assemble import assemble
from np002_traj.scene.settle import settle

config = json.loads((ROOT / "config/np002.json").read_text())
started = time.time()
out = assemble(config["scene_source"], prefix=config["robot_prefix"])
print(f"assembled {out} in {time.time() - started:.0f} s", flush=True)
model = mujoco.MjModel.from_binary_path(str(out / "model.mjb"))
robot = Robot(model, prefix=config["robot_prefix"])
started = time.time()
data, report = settle(model, robot, **{k: v for k, v in config["settle"].items() if k == "max_seconds"},
                      linear=config["settle"]["linear_mps"], angular=config["settle"]["angular_radps"])
report["wall_seconds"] = time.time() - started
key = model.key("assembled").id
model.key_qpos[key], model.key_qvel[key], model.key_ctrl[key] = data.qpos, 0.0, data.ctrl
mujoco.mj_saveModel(model, str(out / "scene.mjb"))
(out / "settle.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"settled {report['seconds']:.2f} s sim in {report['wall_seconds']:.0f} s; "
      f"{len(report['unsettled'])} of {report['trees']} trees still moving; wrote {out / 'scene.mjb'}")
for row in report["unsettled"][:15]:
    print(f"  {row['body']}: {row['linear_mps']*1e3:.1f} mm/s, {row['angular_radps']:.3f} rad/s")
