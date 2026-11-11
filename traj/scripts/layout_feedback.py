"""Scene feedback: operable parts the robot cannot reach from any free floor position.

Contract for scene generation: every operable interface (one with a single-skill task) must have
its site within arm reach of at least one collision-free base pose. This script checks that with
the same footprint and reach model as the dock search, before any IK, and lists the violations
with what would fix them (how much closer to a free floor area the part has to be).

  source scripts/env.sh && $PYTHON scripts/layout_feedback.py [runs/full_11]
Writes layout_feedback.json and layout_feedback.md (into the run directory if given, else runs/).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rb                                    # noqa: E402
from np002_traj.skills.dock import SHOULDER_FORWARD, nearest_free   # noqa: E402

BEARINGS = 36


def check(interface, data, robot, kinematics, occupancy):
    half = robot.profile["base"]["footprint_half_m"]
    reach = robot.capability["arms"]["left"]["reach_m"]
    lift_range = robot.profile["lift"]["range_m"]
    shoulder0 = kinematics.shoulder_height(0.0)
    target = data.site_xpos[interface.site] if interface.site >= 0 else data.xpos[interface.body]
    best = None
    for k in range(BEARINGS):
        phi = 2 * np.pi * k / BEARINGS
        u = np.array([np.cos(phi), np.sin(phi)])
        normal = np.array([-u[1], u[0]])
        for side in robot.arms:
            lateral = robot.profile["arms"][side]["lateral_m"]
            r = nearest_free(occupancy, target[:2], u, normal, lateral, phi, half, stop=2.0)
            if r is None:
                continue
            lift = float(np.clip(target[2] - shoulder0, *lift_range))
            shoulder = np.array([*(target[:2] - (r - SHOULDER_FORWARD) * u), shoulder0 + lift])
            distance = float(np.linalg.norm(target - shoulder))
            if best is None or distance < best["shoulder_distance_m"]:
                best = {"shoulder_distance_m": round(distance, 3), "bearing_deg": round(np.degrees(phi)),
                        "base_distance_m": round(float(r), 3), "side": side}
    reachable = best is not None and best["shoulder_distance_m"] <= reach
    return reachable, best, target


def main(directory=None):
    rb.init()
    model, robot = rb.STATE["model"], rb.STATE["robot"]
    payload = robot.capability["arms"]["left"]["payload_kg"]
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("assembled").id)
    mujoco.mj_forward(model, data)
    reach = robot.capability["arms"]["left"]["reach_m"]
    rows = []
    for interface in rb.STATE["interfaces"].values():
        task = rb.make_task(interface, model, payload)
        if task.excluded or not task.skill:
            continue
        ok, best, target = check(interface, data, robot, rb.STATE["kinematics"], rb.STATE["occupancy"])
        if not ok:
            rows.append({"interface": interface.id, "instance": interface.instance, "skill": task.skill,
                         "site_m": [round(float(v), 3) for v in target],
                         "closest": best, "move_closer_m": None if best is None else round(best["shoulder_distance_m"] - reach, 3)})
    out = Path(directory) if directory else rb.ROOT / "runs"
    (out / "layout_feedback.json").write_text(json.dumps({"reach_m": reach, "contract": __doc__.split("\n\n")[1],
                                                          "violations": rows}, indent=1), encoding="utf-8")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["instance"]].append(r)
    lines = ["# Layout feedback for scene generation", "", __doc__.split("\n\n")[1], "",
             f"{len(rows)} operable interfaces are out of reach (arm reach {reach} m).", "",
             "| instance | interfaces | worst gap (m) | example | nearest free base |", "|---|---|---|---|---|"]
    for instance, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
        worst = max(group, key=lambda r: r["move_closer_m"] or 9)
        lines.append(f"| {instance} | {len(group)} | {worst['move_closer_m']} | {worst['interface']} | {worst['closest']} |")
    (out / "layout_feedback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
