"""Diagnoser: trajectory failures -> attributed, measured scene findings (deterministic, no LLM).

Every failed or excluded task is attributed to one of
  asset     the target asset's own geometry/mechanism stops the robot (clearance, grasp surfaces,
            hinge that does not hold, latch),
  layout    placement: out of reach from every free floor pose, or a neighbouring instance blocks,
  metadata  interface/joint declarations (a movable blocker without an opening interface, an
            interface type that does not match its mechanism, a latch with no release interface),
  method    the trajectory side's own limits (planning, tracking, grasp execution) - never sent.
A finding groups tasks with the same root cause on the same instance. Prerequisite failures are
charged to the blocker that could not be opened, so its finding carries every task it blocks.
Measurements come from the assembled MuJoCo model only; nothing here names an asset.
"""
import collections
import json
import re
from pathlib import Path

import mujoco
import numpy as np

# Requirements derived from the robot profile (data/robot/profile.json in np002_traj):
FINGER_SIDE_GAP_MM = 9.0        # 5 mm fingertip + 2 mm clearance each side
GRIP_HEIGHT_MM = 5.0            # minimum graspable height for the 20 mm pads
GRIP_OPENING_MM = (5.0, 45.0)   # parallel faces must be 5-45 mm apart (50 mm stroke)
PRESS_CLEAR_DIAMETER_MM, PRESS_CLEAR_HEIGHT_MM = 70.0, 35.0
REACH_M = 0.80                  # interface to shoulder from a dockable floor pose
HOLD_OPEN_RAD = np.radians(45)

CATEGORY = {
    # category prefix -> (attribution, feedback kind)
    "L1": ("layout", "unreachable"), "R1": ("layout", "unreachable"), "R2": ("layout", "unreachable"),
    "C1": ("asset", "collision"), "C2": ("asset", "collision"), "C3": ("asset", "collision"),
    "T1": ("asset", "grasp"), "G6": ("asset", "grasp"), "T2": ("asset", "fit"),
    "M4_falls_back": ("asset", "mechanism"), "F3": ("asset", "mechanism"), "P4": ("asset", "mechanism"),
    "M2": ("asset", "mechanism"), "M3": ("asset", "mechanism"),
    "P2": ("asset", "collision"),
}


def attribution(category):
    if category in CATEGORY:
        return CATEGORY[category]
    return CATEGORY.get(category.split("_")[0], ("method", None))


def instance_of(name):
    return name.split("__")[0] if "__" in name else None


class Measure:
    """Geometry and joint measurements on the assembled scene (robot excluded)."""

    def __init__(self, cache, source, robot_prefix="robot__"):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "np002_traj/src"))
        from np002_traj.scene.interfaces import load
        self.model = mujoco.MjModel.from_binary_path(str(Path(cache) / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("assembled").id)
        mujoco.mj_forward(self.model, self.data)
        self.interfaces = {i.id: i for i in load(self.model, source, self.data)[0]}
        self.prefix = robot_prefix
        m = self.model
        self.collidable = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                           and not (m.geom(g).name or m.body(m.geom_bodyid[g]).name).startswith(robot_prefix)]

    def geom_name(self, g):
        m = self.model
        return m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}#{g}"

    def subtree_geoms(self, body):
        m = self.model
        bodies = {body}
        for b in range(body + 1, m.nbody):
            if m.body_parentid[b] in bodies:
                bodies.add(b)
        return [g for g in self.collidable if m.geom_bodyid[g] in bodies], bodies

    def neighbours(self, body, within_m=0.03, top=6):
        """Closest collidable geoms outside the part's subtree: name, instance, gap in mm."""
        own, bodies = self.subtree_geoms(body)
        if not own:
            return []
        m, d = self.model, self.data
        centre = d.xpos[body]
        candidates = [g for g in self.collidable if m.geom_bodyid[g] not in bodies
                      and np.linalg.norm(d.geom_xpos[g] - centre) < m.geom_rbound[g] + 0.25]
        fromto = np.zeros(6)
        best = {}
        for g in candidates:
            gap = min(mujoco.mj_geomDistance(m, d, a, g, within_m, fromto) for a in own)
            if gap < within_m:
                best[g] = gap
        rows = sorted(best.items(), key=lambda kv: kv[1])[:top]
        return [{"geom": self.geom_name(g), "instance": instance_of(self.geom_name(g)) or instance_of(m.body(m.geom_bodyid[g]).name),
                 "gap_mm": round(gap * 1e3, 1)} for g, gap in rows]

    def part_size(self, body):
        """Axis-aligned extent of the part's own collidable geoms (mm), as a size hint."""
        own, _ = self.subtree_geoms(body)
        if not own:
            return None
        m, d = self.model, self.data
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in own:
            r = m.geom_rbound[g] if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX else np.linalg.norm(m.geom_size[g])
            half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else np.full(3, r)
            lo, hi = np.minimum(lo, d.geom_xpos[g] - half), np.maximum(hi, d.geom_xpos[g] + half)
        return [round(float(v) * 1e3, 1) for v in hi - lo]

    def joint(self, joint):
        """Joint parameters and, for a hinge, whether friction holds it open at 45 degrees."""
        m = self.model
        if joint < 0:
            return None
        kind = {2: "slide", 3: "hinge"}.get(int(m.jnt_type[joint]), "free")
        dof = m.jnt_dofadr[joint]
        row = {"joint": m.joint(joint).name, "type": kind,
               "range": [round(float(v), 3) for v in m.jnt_range[joint]] if m.jnt_limited[joint] else None,
               "damping": round(float(m.dof_damping[dof]), 4), "frictionloss": round(float(m.dof_frictionloss[dof]), 4),
               "stiffness": round(float(m.jnt_stiffness[joint]), 4), "armature": round(float(m.dof_armature[dof]), 5)}
        if kind == "hinge":
            d = mujoco.MjData(m)
            d.qpos[:] = self.data.qpos
            lo = m.jnt_range[joint][0] if m.jnt_limited[joint] else 0.0
            torques = {}
            for label, angle in (("45deg", lo + HOLD_OPEN_RAD), ("90deg", lo + np.pi / 2)):
                if m.jnt_limited[joint] and angle > m.jnt_range[joint][1]:
                    continue
                d.qpos[m.jnt_qposadr[joint]] = angle
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                torques[label] = round(float(abs(d.qfrc_bias[dof] + d.qfrc_passive[dof])), 4)
            row["gravity_torque_nm"] = torques
            row["holds_open_at_45deg"] = bool(torques and row["frictionloss"] >= torques.get("45deg", 0))
        return row


def _task_interface(task):
    return task  # task ids are interface ids in the single-skill benchmark


def diagnose(round_dir, measure=None):
    """Read trajectory/ results of one round and write diagnosis/findings.json."""
    round_dir = Path(round_dir)
    traj = round_dir / "trajectory"
    results = {r["task"]: r for r in map(json.loads, (traj / "results.jsonl").open())}
    issues = {r["task"]: r for r in map(json.loads, (traj / "issues.jsonl").open())}
    config = json.loads((traj / "config.json").read_text())
    layout = json.loads((traj / "layout_feedback.json").read_text()) if (traj / "layout_feedback.json").exists() else {"violations": []}
    reach = {v["interface"]: v for v in layout["violations"]}

    # Charge prerequisite failures to the blocker interface that could not be opened.
    charged = collections.defaultdict(list)
    roots = {}
    for task, issue in issues.items():
        category = issue["category"]
        if category.startswith("P3") or category.startswith("P1"):
            step = issue.get("evidence") or {}
            blocker = step.get("interface")
            if blocker:
                charged[blocker].append(task)
                continue
            roots[task] = ("metadata", "mechanism", "P5_blocker_without_interface", step)
            continue
        roots[task] = (*attribution(category), category, issue.get("evidence", {}))

    groups = collections.defaultdict(lambda: {"tasks": [], "interfaces": set(), "evidence": []})
    for task, (attr, kind, category, evidence) in roots.items():
        instance = results[task]["instance"]
        # A neighbour that blocks the hand turns an asset clearance finding into a layout one.
        blockers = (evidence or {}).get("blocking") or {}
        other = evidence.get("other") if isinstance(evidence, dict) else None
        neighbour_names = [n for n in list(blockers)[:3] + ([other] if other else []) if n]
        foreign = [n for n in neighbour_names if instance_of(n) and instance_of(n) not in (instance, "environment")]
        if attr == "asset" and kind in ("collision", "grasp", "fit") and foreign and len(foreign) == len(neighbour_names):
            attr, kind = "layout", "collision"
        key = (attr, kind, category.split("_")[0], instance)
        g = groups[key]
        g["tasks"].append(task)
        g["interfaces"].add(task)
        g["evidence"].append({"task": task, "category": category, **({"evidence": evidence} if evidence else {})})
        g["tasks"] += charged.get(task, [])

    # Blockers that are not runnable tasks themselves (excluded, or succeeded in isolation but not
    # as a prerequisite): their blocked tasks still need a home.
    for blocker, tasks in charged.items():
        if blocker in roots:
            continue
        state = "succeeds_alone" if results.get(blocker, {}).get("success") else ("not_runnable" if blocker not in results else "failed")
        attr, kind, prefix = ("method", None, "P6") if state == "succeeds_alone" else ("metadata", "mechanism", "P5")
        g = groups[(attr, kind, prefix, instance_of(blocker) or "scene")]
        g["tasks"] += tasks
        g["interfaces"].add(blocker)
        g["evidence"].append({"task": blocker, "category": f"{prefix}_blocker_{state}", "blocks": len(tasks)})

    # Tasks excluded before running: interface metadata that does not match the mechanism.
    excluded = config.get("excluded", {})
    for reason, count in excluded.items():
        if reason.startswith(("insert", "mount", "cut", "software", "held by")):
            continue
        match = re.search(r"latch (\S+) with no release", reason)
        instance = instance_of(match.group(1)) if match else None
        g = groups[("metadata", "mechanism", "X0", instance or "scene")]
        g["evidence"].append({"excluded": reason, "count": count})

    findings = []
    for (attr, kind, prefix, instance), g in sorted(groups.items(), key=lambda kv: -len(kv[1]["tasks"])):
        interfaces = sorted(g["interfaces"])
        finding = {"id": f"{attr}:{prefix}:{instance}", "attribution": attr, "kind": kind, "category": prefix,
                   "instance": instance, "interfaces": interfaces, "blocked_tasks": sorted(set(g["tasks"])),
                   "categories": dict(collections.Counter(e.get("category", "excluded") for e in g["evidence"])),
                   "evidence": g["evidence"][:12], "measured": {}, "requirement": {}}
        if measure is not None and attr != "method":
            _measure(finding, measure, reach)
        findings.append(finding)
    out = round_dir / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    summary = collections.Counter()
    for f in findings:
        summary[f["attribution"]] += len(f["blocked_tasks"])
    (out / "findings.json").write_text(json.dumps({"findings": findings, "blocked_by_attribution": dict(summary),
                                                   "tasks": len(results), "success": sum(r["success"] for r in results.values())},
                                                  indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return findings


def _measure(finding, measure, reach):
    rows = []
    for interface_id in finding["interfaces"][:6]:
        interface = measure.interfaces.get(interface_id)
        if interface is None:
            continue
        row = {"interface": interface_id, "kind": interface.kind, "mechanism": interface.mechanism.kind,
               "part_size_mm": measure.part_size(interface.body), "neighbours": measure.neighbours(interface.body)}
        if interface.mechanism.joint >= 0 and finding["kind"] == "mechanism":
            row["joint"] = measure.joint(interface.mechanism.joint)
        if interface_id in reach:
            v = reach[interface_id]
            row["reach"] = {"shoulder_distance_m": (v.get("closest") or {}).get("shoulder_distance_m"),
                            "move_closer_m": v.get("move_closer_m"), "site_m": v.get("site_m")}
        rows.append(row)
    finding["measured"] = {"interfaces": rows}
    kind = finding["kind"]
    if kind == "unreachable":
        finding["requirement"] = {"max_shoulder_distance_m": REACH_M, "dock_footprint_m": [0.6, 0.5]}
    elif kind in ("grasp", "collision"):
        finding["requirement"] = {"finger_side_gap_mm": FINGER_SIDE_GAP_MM, "grip_height_mm": GRIP_HEIGHT_MM,
                                  "parallel_face_spacing_mm": list(GRIP_OPENING_MM)}
    elif kind == "fit":
        finding["requirement"] = {"press_clear_cylinder_mm": {"diameter": PRESS_CLEAR_DIAMETER_MM, "height": PRESS_CLEAR_HEIGHT_MM}}
    elif kind == "mechanism":
        finding["requirement"] = {"hinge_holds_open_from_deg": 45, "latch_needs_release_interface": True,
                                  "movable_blocker_needs_opening_interface": True}
