"""Harness requirements -> the scene generator's planning_feedback message (deterministic, no LLM).

Each verified (or located) task type contributes one requirement. Rows with the same target instance and
kind are merged into one failure so the generator sees one request per instance and cause; every row keeps
its evidence (task type, failing task, the edit that made it succeed in simulation, verified or located).
Verified rows come first, then located ones, each ordered by how many tasks they unblock.
"""
import collections
from pathlib import Path

from ..common import read, write
from ..feedback import KINDS, MAX_FAILURES, plan_instances, render

VERIFIED_NOTE = ("Verified in simulation: applying exactly this change to the current scene made the robot "
                 "complete the task with the same trajectory code.")
LOCATED_NOTE = ("Located in simulation: removing the named geometry's collision made the robot complete the task; "
                "clear it by moving or reshaping that geometry.")


def build(requirements, scene_artifact, scene_sha256):
    instances = set(plan_instances(scene_artifact))
    groups = collections.OrderedDict()
    skipped = []
    order = sorted(requirements, key=lambda r: (not r["evidence"]["verified"], -r["evidence"]["tasks_of_type"]))
    for r in order:
        inst = r["instances"][0]
        if inst not in instances or r["kind"] not in KINDS:
            skipped.append({"type": r["evidence"]["type"], "why": f"instance {inst!r} not in the plan" if inst not in instances
                            else f"kind {r['kind']!r}"})
            continue
        groups.setdefault((inst, r["kind"]), []).append(r)
    failures = []
    for (inst, kind), rows in groups.items():
        verified = all(r["evidence"]["verified"] for r in rows)
        detail = " ".join(dict.fromkeys(r["detail"] for r in rows))
        failures.append({"kind": kind, "instances": [inst],
                         "detail": f"{detail} {VERIFIED_NOTE if verified else LOCATED_NOTE}",
                         "evidence": {"rows": [r["evidence"] for r in rows],
                                      "tasks_unblocked": sum(r["evidence"]["tasks_of_type"] for r in rows)}})
    sent, cut = failures[:MAX_FAILURES], failures[MAX_FAILURES:]
    return {"scene_sha256": scene_sha256, "failures": sent}, skipped + [{"instance": f["instances"][0], "why": "over the cap"} for f in cut]


def write_feedback(eval_dir, scene_artifact, scene_sha256):
    eval_dir = Path(eval_dir)
    requirements = read(eval_dir / "harness/requirements.json", [])
    payload, skipped = build(requirements, scene_artifact, scene_sha256)
    out = eval_dir / "feedback"
    write(out / "feedback.json", payload)
    write(out / "selection.json", {"source": "harness", "sent": [r["evidence"]["type"] for f in payload["failures"]
                                                                  for r in f["evidence"]["rows"]], "skipped": skipped})
    (out / "feedback.md").write_text(render(payload, []), encoding="utf-8")
    return payload
