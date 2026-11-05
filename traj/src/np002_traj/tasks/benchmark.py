"""Single-skill tasks generated for every operable interface; success follows from the mechanism.

The same rules apply to every interface: nothing here knows which instrument it is.
"""
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    interface: str
    skill: str                       # actuate, pick, press, rotate, grasp_hold
    goal: dict                       # condition evaluated by evidence.predicates.Monitor
    invariants: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    excluded: str | None = None      # why the interface is not in the single-skill benchmark
    momentary: bool = False          # success if the goal held at some point (a button springs back)
    releases: list = field(default_factory=list)   # mechanical events this skill fires itself (see semantics)
    unlock: list = field(default_factory=list)     # interfaces whose task releases a lock first
    key: str = "goal"                # monitor key of the goal (prerequisite tasks get their own)


def semantic_rule(task, interface, semantics, active):
    """Apply what the asset declares about connections holding the part; may exclude the task."""
    if semantics is None or task.excluded or not task.skill:
        return task
    for c in semantics.holding(semantics.moving_bodies(interface), active):
        by_kind = semantics.kinds.get(c.by)
        if c.by is None:
            task.excluded = f"locked by {c.kind} {c.id} with no release interface"
        elif by_kind == "cut":
            task.excluded = f"held by {c.id}: needs a cut first"
        elif c.by == interface.id:
            when = "driven" if c.joint is not None else "grasped"
            task.releases.append({"id": c.release, "equality": c.equality, "when": when, "joint": c.joint, "at": c.at})
        else:
            task.unlock.append({"interface": c.by, "connection": c.id, "equality": c.equality, "event": c.release})
    return task


def task_for(interface, model, payload_kg, travel=0.8, semantics=None, active=None):
    task = _task_for(interface, model, payload_kg, travel, semantics)
    return semantic_rule(task, interface, semantics, model.eq_active0 if active is None else active)


def _task_for(interface, model, payload_kg, travel=0.8, semantics=None):
    m, mech, name = model, interface.mechanism, interface.id
    joint = m.joint(mech.joint).name if mech.joint >= 0 else None
    if interface.kind == "grasp" and mech.kind in ("slide", "hinge") and mech.range:
        # Drive the articulation from where it is towards its far end, `travel` of the way.
        return Task(name, name, "actuate", {"kind": "joint_travel", "joint": joint, "travel": travel, "for_s": 0.3},
                    parameters={"joint": joint, "travel": travel})
    if interface.kind == "grasp" and mech.kind == "free" and mech.mass_kg <= payload_kg:
        body = m.body(mech.body).name
        return Task(name, name, "pick", {"kind": "lifted", "body": body, "height_m": 0.05, "for_s": 0.5},
                    parameters={"body": body, "lift_m": 0.08})
    if interface.kind == "press" and mech.kind in ("slide", "hinge") and mech.range:
        pressed = (interface.button or {}).get("pressed_at")
        return Task(name, name, "press", {"kind": "joint_travel", "joint": joint, "travel": 0.6, "absolute": pressed},
                    parameters={"joint": joint}, momentary=True)
    if interface.kind == "rotate" and mech.kind == "hinge" and mech.range:
        return Task(name, name, "rotate", {"kind": "joint_travel", "joint": joint, "travel": 0.5, "for_s": 0.3},
                    parameters={"joint": joint, "travel": 0.5})
    if interface.kind in ("insert", "mount", "cut", "software"):
        return Task(name, name, "", {}, excluded=f"{interface.kind} needs a held object or another skill first")
    if interface.kind == "grasp" and mech.kind == "free":
        return Task(name, name, "", {}, excluded=f"loose object of {mech.mass_kg:.1f} kg exceeds the payload")
    if mech.kind == "attached" and interface.kind == "grasp" and mech.mass_kg <= payload_kg and semantics is not None:
        # Held by a connection: a pick if a declared release frees it (checked by semantic_rule).
        body = m.body(mech.body).name
        return Task(name, name, "pick", {"kind": "lifted", "body": body, "height_m": 0.05, "for_s": 0.5},
                    parameters={"body": body, "lift_m": 0.08})
    if mech.kind == "attached":
        return Task(name, name, "", {}, excluded="object held by an equality needs its release event first")
    return Task(name, name, "", {}, excluded=f"{interface.kind} on a {mech.kind} mechanism has no single-skill rule")
