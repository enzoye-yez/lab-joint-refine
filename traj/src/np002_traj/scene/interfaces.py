"""Interface catalogue and mechanism classification, read from the scene metadata and the model.

An interface (grasp, press, rotate, insert, mount, cut) names a body and usually a site and an
approach axis. Its mechanism is the first movable joint on the kinematic chain from that body to
the world: a slide or hinge makes it an articulation (door, drawer, lid, knob, button); a free
joint makes it a loose object, liftable only if its subtree is within the robot payload.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ..geometry import subtree_bodies

FREE, BALL, SLIDE, HINGE = 0, 1, 2, 3


@dataclass
class Mechanism:
    kind: str                      # "slide", "hinge", "free", "fixed"
    joint: int = -1
    body: int = -1                 # body carrying the joint (or the free root)
    range: tuple | None = None
    mass_kg: float = 0.0


@dataclass
class Interface:
    id: str
    kind: str
    body: int
    site: int
    approach: np.ndarray | None
    instance: str
    mechanism: Mechanism
    button: dict | None = None
    extra: dict = field(default_factory=dict)


def attached(model, bodies):
    """Whether an active weld/connect equality ties these bodies to something outside them."""
    for e in range(model.neq):
        if model.eq_type[e] not in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT) or not model.eq_active0[e]:
            continue
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if (a in bodies) != (b in bodies):
            return True
    return False


def mechanism(model, body):
    b = body
    while b > 0:
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            kind = int(model.jnt_type[j])
            if kind == FREE:
                bodies = subtree_bodies(model, b)
                mass = float(sum(model.body_mass[x] for x in bodies))
                # A loose object held in place by an equality (a clip, a mount) needs a release event first.
                return Mechanism("attached" if attached(model, bodies) else "free", j, b, None, mass)
            if kind in (SLIDE, HINGE):
                rng = tuple(model.jnt_range[j]) if model.jnt_limited[j] else None
                return Mechanism("slide" if kind == SLIDE else "hinge", j, b, rng)
        b = model.body_parentid[b]
    return Mechanism("fixed")


def load(model, source):
    meta = json.loads((Path(source) / "interfaces.json").read_text(encoding="utf-8"))
    buttons = {b["joint"]: b for b in meta.get("buttons", [])}
    result, missing = [], []
    for row in meta["interfaces"]:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, row["body"])
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, row.get("site", ""))
        if body < 0:
            missing.append(row["id"])            # e.g. interfaces of the removed robot tool
            continue
        mech = mechanism(model, body)
        button = buttons.get(model.joint(mech.joint).name) if mech.joint >= 0 else None
        approach = np.asarray(row["approach_axis"], float) if "approach_axis" in row else None
        result.append(Interface(row["id"], row["kind"], body, site, approach, row["body"].split("__")[0], mech, button,
                                {k: v for k, v in row.items() if k not in {"id", "kind", "body", "site", "approach_axis"}}))
    return result, missing
