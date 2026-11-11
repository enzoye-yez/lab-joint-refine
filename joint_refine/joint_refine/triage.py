"""Triage of failing task types: who works on each one.

  trajectory  the robot side can fix it (the part is reachable and graspable; the motion, the skill or its
              sequencing fails) -- fixed in np002_traj, never sent to the scene generator;
  scene       the scene must change (no grasp the fingers fit, no free floor to stand on, a part enclosed
              by its own instrument, wrong metadata) -- sent to the scene repair;
  deferred    set aside with the user (config/triage_deferred.json) -- neither side works on it.

The decision is made per task type (np002_traj.tasks.types) from the most common failure reason of its
tasks; a type blocked by a prerequisite or an unlock takes the class of the blocking type.
"""
import collections
import json
import re
import sys
from pathlib import Path

from .common import ROOT, write

# The same definition as np002_traj.tasks.types (T6); kept here so the triage also reads older runs.
GRID = re.compile(r"(?<![a-z])[A-H]\d{1,2}(?![a-z])")
NUMBER = re.compile(r"\d+")


def task_type(interface_id, instance, skill):
    local = interface_id.split("__", 1)[1] if "__" in interface_id else interface_id
    return f"{NUMBER.sub('#', instance)}|{skill}|{NUMBER.sub('#', GRID.sub('#', local))}"

DEFERRED = ROOT / "config/triage_deferred.json"
TRAJECTORY_REASONS = {"goal_not_met", "disturbed", "contact_force_limit", "grasp_missed", "no_arm_path",
                      "goal_in_collision", "regrasp_failed", "regrasp_missed", "replan_at_dock", "task_timeout",
                      "arm_tracking", "tool_ik", "tool_not_seated", "tool_dropped", "no_dock_held", "no_press_pose"}
SCENE_REASONS = {"no_grasp", "enclosed", "no_interface", "no_push_pose", "no_frames"}
TOOL_REASON = "no_grasp_tool"        # a part the fingers cannot hold but a declared tool can (a pipette tip)


def _template(type_key):
    return type_key.split("|", 2)[2]


def _reason(result):
    reason = result.get("reason") or ("disturbed" if result.get("disturbed") else "goal_not_met")
    detail = result.get("detail") or {}
    if reason == "no_dock":
        reasons = detail.get("reasons") or {}
        # Docks that reach the part but cannot follow its motion are the robot's; no floor or no reach is layout.
        return "no_dock_path" if reasons.get("path") else "no_dock_layout"
    if reason == "no_grasp" and result.get("tool"):
        return TOOL_REASON
    return reason


def classify(results, deferred=None):
    """{type: {"class", "reason", "tasks", "solved", "blocked_by"}} for every type in `results`."""
    deferred = deferred if deferred is not None else {d["template"]: d["why"] for d in json.loads(DEFERRED.read_text())["deferred"]}
    groups = collections.defaultdict(list)
    by_task = {}
    for r in results:
        key = task_type(r["task"], r.get("instance", ""), r.get("skill", ""))
        groups[key].append(r)
        by_task[r["task"]] = key
    out = {}
    for key, rows in groups.items():
        solved = sum(bool(r.get("success")) for r in rows)
        entry = {"tasks": len(rows), "solved": solved}
        if solved:
            entry.update({"class": "solved", "reason": None})
        elif _template(key) in deferred:
            entry.update({"class": "deferred", "reason": deferred[_template(key)]})
        else:
            reasons = collections.Counter(_reason(r) for r in rows)
            reason = reasons.most_common(1)[0][0]
            entry["reason"] = reason
            if reason in ("prerequisite_failed", "unlock_failed"):
                steps = [((r.get("detail") or {}).get("step") or {}).get("interface") for r in rows]
                blockers = sorted({by_task.get(s) or task_type(s, s.split("__")[0], "") for s in steps if s})
                entry["blocked_by"] = blockers
            elif reason in TRAJECTORY_REASONS or reason in ("no_dock_path", TOOL_REASON):
                entry["class"] = "trajectory"
            elif reason in SCENE_REASONS or reason == "no_dock_layout":
                entry["class"] = "scene"
            else:
                entry["class"] = "scene"            # metadata faults and anything unrecognised
        out[key] = entry
    # A blocked type follows its blocker (once, then by majority of its blockers' classes).
    for key, entry in out.items():
        if "class" in entry:
            continue
        classes = [out[b]["class"] for b in entry.get("blocked_by", []) if b in out and "class" in out[b]]
        known = [c for c in classes if c != "solved"]
        entry["class"] = collections.Counter(known).most_common(1)[0][0] if known else "scene"
    return out


def write_triage(eval_dir, tool_results=()):
    """tool_results: result files of a run whose code tries a declared tool when the fingers cannot hold a
    part (T6); types it tried a tool on are trajectory work even where this evaluation predates the tool."""
    eval_dir = Path(eval_dir)
    results = [json.loads(l) for l in (eval_dir / "trajectory/results.jsonl").open()]
    tooled = {task_type(r["task"], r.get("instance", ""), r.get("skill", ""))
              for f in tool_results for r in map(json.loads, Path(f).open()) if r.get("tool")}
    for r in results:
        if not r.get("success") and r.get("reason") == "no_grasp" and \
                task_type(r["task"], r.get("instance", ""), r.get("skill", "")) in tooled:
            r["tool"] = True
    table = classify(results)
    counts = collections.Counter(e["class"] for e in table.values())
    write(eval_dir / "triage.json", {"counts": dict(counts), "types": table})
    return table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("evals", nargs="+")
    parser.add_argument("--tool-results", nargs="*", default=[])
    args = parser.parse_args()
    for path in args.evals:
        table = write_triage(path, args.tool_results)
        print(path, dict(collections.Counter(e["class"] for e in table.values())))
