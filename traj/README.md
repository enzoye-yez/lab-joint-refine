# np002_traj

NP-002 trajectory package: holonomic base + lift column + two Franka FR3 arms in MuJoCo (CPU only).
The plan and the review of the previous package are in `/home/coder/np002_traj_review/REVIEW_zh.md`.

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_robot.py          # data/robot/model.xml + profile.json
source scripts/env.sh && PYTHONPATH=src:tests $PYTHON -m unittest discover -s tests -v
source scripts/env.sh && $PYTHON scripts/render_robot.py   # runs/robot_views.png
```

`data/robot/profile.json` is the only source of robot names and limits. FR3 comes from
mujoco_menagerie 0059d43 (`data/vendor/fr3`, Apache-2.0).
