"""Repair skills: one per family of trajectory failures.

A skill turns one failed task into a small set of counterfactual scene edits (np002_traj
scene/perturb.py kinds) and, for the edit that makes the task succeed, a repair requirement for the
scene generator. Every variant carries
  label        short id
  edits        perturb edits (applied together)
  realisable   True when the edit is itself a scene change the generator can make (move, turn, joint
               parameters); False for a locating probe (a neighbour's collision switched off) whose
               success says *what* is in the way, and the requirement says how to clear it
  target       {"kind": layout|asset|metadata, "instance", "job"?} who has to change
  kind         planning_feedback kind (collision, unreachable, grasp, mechanism, fit, support)
  requirement  English sentence with the verified numbers, sent to the generator
Skills never name an asset: everything comes from the result, the plan and the model.
"""
import math

from ..triage import _reason

MAX_VARIANTS = 8


class Skill:
    name = ""
    reasons = ()
    doc = ""

    def match(self, result):
        return _reason(result) in self.reasons

    def variants(self, scene, result):
        raise NotImplementedError

    @staticmethod
    def _variant(label, edits, realisable, target, kind, requirement, **meta):
        return {"label": label, "edits": edits, "realisable": realisable, "target": target, "kind": kind,
                "requirement": requirement, **meta}


def _foreign_blockers(scene, result, own_instance, top=3):
    """Non-robot geoms of other instances that the failure detail names, most frequent first."""
    detail = result.get("detail") or {}
    counts = {}
    for name, count in (detail.get("blocking") or {}).items():
        if isinstance(count, list):
            count = count[0]
        if not isinstance(count, (int, float)) or scene.is_robot(name):
            continue
        counts[name] = counts.get(name, 0) + count
    for name in ((detail.get("closest") or {}).get("geoms") or []):
        if not scene.is_robot(name):
            counts[name] = counts.get(name, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(n, c) for n, c in rows if scene.geom_instance(n) not in (None, own_instance)][:top], \
           [(n, c) for n, c in rows if scene.geom_instance(n) == own_instance][:top]


def _move(scene, instance, offset_xy, dyaw=0.0):
    base = scene.base_of(instance)
    return {"kind": "move_instance", "instance": base, "carry": scene.carried_by(base),
            "dxyz": [float(offset_xy[0]), float(offset_xy[1]), 0.0], "dyaw": float(dyaw)}


def _geom_exists(scene, name):
    try:
        scene.model.geom(name)
        return True
    except KeyError:
        return False


class DockSpace(Skill):
    name = "dock_space"
    reasons = ("no_dock_layout", "no_dock_path", "no_arm_path", "goal_in_collision", "replan_at_dock")
    doc = "The robot cannot stand where it reaches the part, or its arm collides on the way: move or turn the " \
          "instance (with what stands on it) toward the aisle, or clear the neighbour that is in the way."

    def variants(self, scene, result):
        inst = result["instance"]
        base = scene.base_of(inst)
        front, along = scene.front(inst), scene.along(inst)
        target = {"kind": "layout", "instance": base}
        out = []
        for d in (0.05, 0.10):
            out.append(self._variant(f"front_{int(d*100)}cm", [_move(scene, inst, front * d)], True, target, "unreachable",
                                     f"Move {base} (with everything standing on it) {d*100:.0f} cm toward the robot aisle: "
                                     f"in simulation the robot then completes {result['task']}."))
        for sign in (1, -1):
            out.append(self._variant(f"along_{'+' if sign > 0 else '-'}20cm", [_move(scene, inst, along * 0.2 * sign)], True,
                                     target, "unreachable",
                                     f"Shift {base} 20 cm along its bench (direction {[round(x, 2) for x in along * sign]}) to free "
                                     f"floor in front of it: in simulation the robot then completes {result['task']}."))
        for yaw in (math.pi / 2, -math.pi / 2, math.pi):
            out.append(self._variant(f"turn_{round(math.degrees(yaw))}", [_move(scene, inst, (0, 0), yaw)], True, target,
                                     "unreachable",
                                     f"Turn {base} by {math.degrees(yaw):.0f} deg about vertical so the part faces the aisle: "
                                     f"in simulation the robot then completes {result['task']}."))
        foreign, _ = _foreign_blockers(scene, result, inst)
        if foreign:
            names = [n for n, _ in foreign if _geom_exists(scene, n)]
            if names:
                others = sorted({scene.geom_instance(n) for n in names})
                out.append(self._variant("clear_neighbours", [{"kind": "disable_collision", "geoms": names}], False,
                                         {"kind": "layout", "instance": others[0]}, "collision",
                                         f"Neighbour geometry {names} of {others} blocks the robot at {inst}: keep it out of the "
                                         f"robot's approach to {result['task']} (move the neighbour or leave a gap).",
                                         blockers=names))
        return out[:MAX_VARIANTS]


class GraspGeometry(Skill):
    name = "grasp_geometry"
    reasons = ("no_grasp", "no_grasp_tool", "grasp_missed")
    doc = "No finger pose closes on the part: find whether neighbours or the part's own housing block the fingers."

    def variants(self, scene, result):
        inst = result["instance"]
        iface = scene.interface(result["task"])
        foreign, own = _foreign_blockers(scene, result, inst)
        out = []
        if iface is not None:
            m = scene.measure
            exposed = m.exposed_height(iface.body) or {}
            measured = {"exposed": exposed, "neighbours": m.neighbours(iface.body)}
        else:
            measured = {}
        f_names = [n for n, _ in foreign if _geom_exists(scene, n)]
        o_names = [n for n, _ in own if _geom_exists(scene, n) and iface is not None
                   and int(scene.model.geom(n).bodyid[0]) not in scene.measure.subtree_geoms(iface.body)[1]]
        if f_names:
            others = sorted({scene.geom_instance(n) for n in f_names})
            out.append(self._variant("clear_foreign", [{"kind": "disable_collision", "geoms": f_names}], False,
                                     {"kind": "layout", "instance": others[0]}, "grasp",
                                     f"The fingers closing on {result['task']} hit {f_names} of {others}: leave at least "
                                     f"9 mm free beside the grasped part (fingertip 5 mm + 2 mm each side).",
                                     measured=measured, blockers=f_names))
        if o_names:
            out.append(self._variant("clear_own", [{"kind": "disable_collision", "geoms": o_names}], False,
                                     {"kind": "asset", "instance": inst}, "grasp",
                                     f"The asset's own geometry {o_names} leaves no room for the fingers at {result['task']}: "
                                     f"the grasped part must stand at least 12 mm clear of it with 9 mm side gaps.",
                                     measured=measured, blockers=o_names))
        if f_names and o_names:
            out.append(self._variant("clear_both", [{"kind": "disable_collision", "geoms": f_names + o_names}], False,
                                     {"kind": "asset", "instance": inst}, "grasp",
                                     f"Both {o_names} (own) and {f_names} (neighbours) block the fingers at {result['task']}: "
                                     f"the part needs 12 mm exposed height and 9 mm side gaps.",
                                     measured=measured, blockers=f_names + o_names))
        if iface is not None and scene.base_of(inst) != inst:
            pass                                    # a part in a holder: the holder's geometry is reported via clear_*
        return out[:MAX_VARIANTS]


class MechanismEffort(Skill):
    name = "mechanism_effort"
    reasons = ("contact_force_limit", "goal_not_met", "regrasp_failed", "regrasp_missed")
    doc = "The robot holds the part but cannot move the mechanism far enough within its force limits: relax the " \
          "joint's friction, damping, spring or the moving mass."

    def match(self, result):
        return super().match(result) and result.get("skill") in ("actuate", "rotate", "press", "push")

    def variants(self, scene, result):
        joint = scene.mechanism_joint(result["task"])
        if joint is None:
            return []
        m = scene.model
        j = m.joint(joint).id
        dof = m.jnt_dofadr[j]
        now = {"frictionloss": float(m.dof_frictionloss[dof]), "damping": float(m.dof_damping[dof]),
               "stiffness": float(m.jnt_stiffness[j])}
        inst = result["instance"]
        target = {"kind": "asset", "instance": inst}
        out = []
        for field in ("frictionloss", "damping", "stiffness"):
            if now[field] <= 0:
                continue
            for scale in (0.3, 0.1):
                out.append(self._variant(f"{field}_x{scale}", [{"kind": "joint", "joint": joint, field: {"scale": scale}}],
                                         True, target, "mechanism",
                                         f"Joint {joint} of {inst}: {field} {now[field]:.4g} -> at most {now[field] * scale:.4g} "
                                         f"(x{scale}); with it the robot completes {result['task']} within its force limits "
                                         f"(contact 80 N, Cartesian 60 N / 6 N*m).", joint=joint, before=now))
        if len([f for f in now if now[f] > 0]) > 1:
            out.append(self._variant("all_x0.1", [{"kind": "joint", "joint": joint,
                                                   **{f: {"scale": 0.1} for f in now if now[f] > 0}}], True, target,
                                     "mechanism", f"Joint {joint} of {inst}: frictionloss, damping and stiffness each at most "
                                     f"x0.1 of {now}; with it the robot completes {result['task']}.", joint=joint, before=now))
        return out[:MAX_VARIANTS]


class BlockerChain(Skill):
    name = "blocker_chain"
    reasons = ("prerequisite_failed", "unlock_failed", "enclosed")
    doc = "Something must be opened or released first: locate the blocker; when the blocker's own task fails, " \
          "the fix belongs to the blocker's type."

    def variants(self, scene, result):
        detail = result.get("detail") or {}
        inst = result["instance"]
        out = []
        geom = detail.get("geom") or (detail.get("step") or {}).get("for")
        body = detail.get("body")
        if body:
            out.append(self._variant("remove_blocker", [{"kind": "disable_collision", "bodies": [body]}], False,
                                     {"kind": "asset", "instance": scene.geom_instance(body) or inst}, "collision",
                                     f"{body} encloses the target of {result['task']}"
                                     + (f" and its opener {(detail.get('step') or {}).get('interface')} fails "
                                        f"({(detail.get('step') or {}).get('failed')})" if detail.get("step") else
                                        " and no interface opens it") +
                                     ": give it an operable opening interface the robot can drive, or do not enclose the part.",
                                     blockers=[body]))
        elif geom and _geom_exists(scene, geom):
            out.append(self._variant("remove_blocker", [{"kind": "disable_collision", "geoms": [geom]}], False,
                                     {"kind": "asset", "instance": scene.geom_instance(geom) or inst}, "collision",
                                     f"{geom} blocks {result['task']} and its opener "
                                     f"{(detail.get('step') or {}).get('interface')} fails "
                                     f"({(detail.get('step') or {}).get('failed')}): the opener must work (see its own "
                                     f"finding) or the part must not be enclosed.", blockers=[geom]))
        return out


class PartRetention(Skill):
    name = "part_retention"
    reasons = ("disturbed",)
    doc = "The task moved a part it should not move: check whether holding that part in place makes it succeed."

    def variants(self, scene, result):
        moved = sorted((result.get("disturbed") or {}).items(), key=lambda kv: -kv[1])[:4]
        if not moved:
            return []
        bodies = [b for b, _ in moved]
        owner = scene.geom_instance(bodies[0]) or result["instance"]
        return [self._variant("hold_disturbed", [{"kind": "freeze", "bodies": bodies}], False,
                              {"kind": "asset", "instance": owner}, "support",
                              f"Operating {result['task']} moves {', '.join(f'{b} by {d*1000:.0f} mm' for b, d in moved)}: "
                              f"these parts must be retained (seated, latched or constrained) so they stay within 5 mm.",
                              blockers=bodies)]


class RobotLimit(Skill):
    name = "robot_limit"
    reasons = ("task_timeout", "code_error", "arm_tracking")
    doc = "Robot-side limits (self-collision, planning time): try turning the instance; otherwise the trajectory side."

    def variants(self, scene, result):
        inst = result["instance"]
        base = scene.base_of(inst)
        return [self._variant(f"turn_{round(math.degrees(y))}", [_move(scene, inst, (0, 0), y)], True,
                              {"kind": "layout", "instance": base}, "unreachable",
                              f"Turn {base} by {math.degrees(y):.0f} deg: in simulation the robot then completes {result['task']}.")
                for y in (math.pi / 2, -math.pi / 2)]


SKILLS = [BlockerChain(), DockSpace(), GraspGeometry(), MechanismEffort(), PartRetention(), RobotLimit()]


def skill_for(result):
    return next((s for s in SKILLS if s.match(result)), None)
