"""Classify every failure of a benchmark run into a problem taxonomy; write issues.jsonl and issues.md.

Each failure gets one primary category (where in the pipeline it stopped and why) and the
evidence behind it, so fixes can be made per category across all interfaces at once.

  .venv/bin/python scripts/classify_failures.py runs/full_01
"""
import collections
import json
import sys
from pathlib import Path

CATEGORIES = {
    "P1_enclosed_by_movable": "Precondition: the part is behind a lid/door/cover that must be opened first",
    "P2_enclosed_by_fixed": "Precondition: the part is enclosed by fixed geometry (no straight way in for the hand)",
    "R1_out_of_workspace": "Reach: no base pose gives an IK solution for the pre-grasp",
    "R2_no_base_pose": "Reach: the base cannot park anywhere near the part",
    "C1_arm_collides_pre": "Clearance: the arm/hand collides at the pre-grasp pose",
    "C2_fingers_collide_at_grasp": "Clearance: fingers or hand collide with neighbours at the grasp pose",
    "C3_robot_collides_own_instrument": "Clearance: the robot collides with another part of the target instrument",
    "G1_grasp_missed_nothing": "Grasp: fingers closed on nothing (part smaller than the position error, or wrong site)",
    "G2_grasp_missed_other": "Grasp: fingers touched something else, not the part",
    "G3_grasp_missed_part_not_pads": "Grasp: fingers touched the part but not with the pads",
    "G4_replan_failed": "Grasp: no IK solution from the measured parking pose",
    "F1_force_on_target": "Force: contact force limit on the target part",
    "F2_force_on_other": "Force: contact force limit against another object",
    "F3_latched": "Force: the part is held by a latch/lock that must be released first (mechanical event)",
    "M1_mechanism_partial": "Mechanism: moved but stopped short of the goal",
    "M2_mechanism_wrong_way": "Mechanism: moved the wrong way",
    "M3_mechanism_no_motion": "Mechanism: grasped but the joint did not move",
    "M4_rotation_beyond_wrist": "Mechanism: required rotation exceeds one wrist turn (needs regrasping)",
    "K1_pick_not_lifted": "Pick: grasped but the object did not rise",
    "K2_pick_partial": "Pick: lifted less than required (slipped or tethered)",
    "B1_press_short": "Press: pressed but not deep enough",
    "B2_press_no_motion": "Press: the button did not move",
    "D1_disturbed": "Side effect: another instrument moved more than 5 mm",
    "E1_arm_path_or_tracking": "Execution: no arm path, tracking error or base timeout",
    "P3_prerequisite_failed": "Precondition: opening the blocking lid/door with its own interface failed",
    "G5_regrasp_failed": "Mechanism: a regrasp between wrist turns failed",
    "L1_layout_beyond_reach": "Layout: the part is beyond arm reach from every free floor position (scene feedback)",
    "T1_tool_does_not_fit": "Tool: no antipodal grasp where the fingers clear the neighbours (tool design feedback)",
    "T2_pin_does_not_fit": "Tool: the press pin cannot reach the button without the hand touching something",
    "G6_no_antipodal_grasp": "Grasp: the part geometry offers no antipodal pair within the opening",
    "P4_unlock_failed": "Precondition: the interface that releases a declared lock failed",
    "X1_code_error": "Code error (not a task failure)",
    "X3_worker_memory": "Run guard: the worker was above its memory cap and skipped the task",
    "X2_other": "Unclassified",
}


def instrument(name):
    return name.split("__")[0] if name else ""


def classify(r):
    reason, detail = r["reason"], r.get("detail") or {}
    goal = r.get("goal_detail") or {}
    if r["success"]:
        return None, {}
    if reason == "worker_memory":
        return "X3_worker_memory", detail
    if reason == "code_error":
        return "X1_code_error", {"error": detail.get("error", "")[:200]}
    if reason == "enclosed":
        movable = detail.get("movable") in ("slide", "hinge", "free")
        return ("P1_enclosed_by_movable" if movable else "P2_enclosed_by_fixed"), {"by": detail.get("geom")}
    if reason == "no_dock":
        reasons = detail.get("reasons", {})
        closest = detail.get("closest")
        tried = detail.get("tried", 0)
        if not tried:
            # Every free base pose leaves the part beyond the arm: a placement problem for the scene.
            return ("L1_layout_beyond_reach" if reasons.get("reach") or reasons.get("footprint") else "R1_out_of_workspace"), {"reasons": reasons}
        if not closest:
            return "R1_out_of_workspace", {"reasons": reasons}
        robot_geom, other = (closest.get("geoms") or ["self", ""])[:2]
        evidence = {"robot": robot_geom.replace("robot__", ""), "other": other, "distance_mm": round(closest["distance_m"] * 1e3, 1),
                    "stage": closest["stage"]}
        if instrument(other) == r["instance"]:
            return "C3_robot_collides_own_instrument", evidence
        if closest["stage"] == "goal" or any(k in robot_geom for k in ("finger", "pad", "hand")):
            return "C2_fingers_collide_at_grasp" if closest["stage"] == "goal" else "C1_arm_collides_pre", evidence
        return "C1_arm_collides_pre", evidence
    if reason == "no_grasp":
        fit = sum(detail.get(k, 0) for k in ("open_collision", "closed_collision", "corridor"))
        geometry = sum(detail.get(k, 0) for k in ("no_part_between", "not_antipodal"))
        evidence = {"blocking": detail.get("blocking", {}), "fit": fit, "geometry": geometry}
        return ("T1_tool_does_not_fit" if fit >= geometry else "G6_no_antipodal_grasp"), evidence
    if reason == "no_press_pose":
        return "T2_pin_does_not_fit", {"blocking": detail.get("blocking", {})}
    if reason == "unlock_failed":
        return "P4_unlock_failed", detail.get("step", {})
    if reason == "prerequisite_failed":
        return "P3_prerequisite_failed", detail.get("step", {})
    if reason in ("regrasp_failed", "regrasp_missed"):
        return "G5_regrasp_failed", {"reason": reason}
    if reason == "replan_at_dock":
        return "G4_replan_failed", detail
    if reason == "grasp_missed":
        fingers = next((x for x in r.get("log", []) if x["step"] == "fingers"), {})
        touching = fingers.get("touching", [])
        if not touching:
            return "G1_grasp_missed_nothing", {"opening_mm": fingers.get("opening_mm")}
        own = [t for t in touching if instrument(t) == r["instance"]]
        return ("G3_grasp_missed_part_not_pads" if own else "G2_grasp_missed_other"), {"touching": touching[:3]}
    if reason == "contact_force_limit":
        own = instrument(detail.get("geom", "")) == r["instance"]
        if own and any(k in detail.get("geom", "") for k in ("latch", "lock", "catch", "clip")):
            return "F3_latched", {"geom": detail.get("geom"), "force_n": round(detail.get("force_n", 0))}
        return ("F1_force_on_target" if own else "F2_force_on_other"), {"geom": detail.get("geom"), "force_n": round(detail.get("force_n", 0))}
    if reason in ("no_arm_path", "arm_tracking", "base_timeout", "no_base_path", "goal_in_collision"):
        return "E1_arm_path_or_tracking", {"reason": reason, **{k: v for k, v in detail.items() if k != "trace"}}
    if r["disturbed"]:
        return "D1_disturbed", {"moved": dict(list(r["disturbed"].items())[:3])}
    if r["skill"] in ("actuate", "rotate"):
        moved, needed = goal.get("moved", 0), goal.get("needed", 1)
        if r["skill"] == "rotate" and needed > 5.5:
            return "M4_rotation_beyond_wrist", {"moved": round(moved, 3), "needed": round(needed, 3)}
        if moved < -0.05 * needed:
            return "M2_mechanism_wrong_way", {"moved": round(moved, 4), "needed": round(needed, 4)}
        if moved < 0.05 * needed:
            return "M3_mechanism_no_motion", {"moved": round(moved, 4), "needed": round(needed, 4)}
        return "M1_mechanism_partial", {"fraction": round(moved / needed, 2)}
    if r["skill"] == "pick":
        rise = goal.get("rise_m", 0)
        return ("K1_pick_not_lifted" if rise < 0.005 else "K2_pick_partial"), {"rise_mm": round(rise * 1e3, 1)}
    if r["skill"] == "press":
        moved, needed = goal.get("moved", 0), goal.get("needed", 1)
        return ("B2_press_no_motion" if moved < 0.1 * needed else "B1_press_short"), {"fraction": round(moved / needed, 2)}
    return "X2_other", {"reason": reason}


def main(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "results.jsonl").open()]
    issues = []
    for r in rows:
        category, evidence = classify(r)
        if category:
            issues.append({"task": r["task"], "skill": r["skill"], "mechanism": r["mechanism"], "instance": r["instance"],
                           "category": category, "evidence": evidence})
    with (directory / "issues.jsonl").open("w") as out:
        for issue in issues:
            out.write(json.dumps(issue) + "\n")
    ok = sum(r["success"] for r in rows)
    by = collections.Counter(i["category"] for i in issues)
    lines = [f"# Failure taxonomy: {directory.name}", "",
             f"{ok}/{len(rows)} tasks succeeded; {len(issues)} failures in {len(by)} categories.", "",
             "| category | count | meaning | skills | top evidence |", "|---|---|---|---|---|"]
    for category, count in by.most_common():
        group = [i for i in issues if i["category"] == category]
        skills = collections.Counter(i["skill"] for i in group)
        evidence = collections.Counter()
        for i in group:
            e = i["evidence"]
            if isinstance(e.get("blocking"), dict) and e["blocking"]:
                evidence[instrument(next(iter(e["blocking"])))] += 1
                continue
            key = e.get("robot", "") + " vs " + instrument(e.get("other", "")) if "robot" in e else \
                instrument(e.get("by", "") or e.get("geom", "") or "") or ",".join(map(instrument, e.get("touching", []))) or ""
            if key:
                evidence[key] += 1
        lines.append(f"| {category} | {count} | {CATEGORIES[category]} | {dict(skills)} | {dict(evidence.most_common(4))} |")
    success = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        success[f"{r['skill']}/{r['mechanism']}"][0] += r["success"]
        success[f"{r['skill']}/{r['mechanism']}"][1] += 1
    lines += ["", "| skill/mechanism | success |", "|---|---|"] + [f"| {k} | {a}/{b} |" for k, (a, b) in sorted(success.items())]
    (directory / "issues.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
