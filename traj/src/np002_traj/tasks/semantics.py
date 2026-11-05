"""Asset semantics for task definition: which connections hold a part and what releases them.

Read from the scene metadata (interfaces.json): every connection names its equality and its
release event, every mechanical event names the interface that triggers it. A part held by an
active connection is operable only once that connection is released:

  released by this interface   -> the skill fires the event once the part is in hand (a clip, a seat)
  released by another interface -> that interface's task is a prerequisite (unlatch, then open)
  released by a cut              -> needs a cutting tool first (not a gripper task)
  no release declared            -> locked; reported to scene generation
"""
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco

from ..geometry import subtree_bodies


@dataclass
class Connection:
    id: str
    kind: str                  # detachable, latch
    equality: int
    irreversible: bool
    release: str | None        # release event id
    by: str | None             # interface that triggers the release
    joint: int | None = None   # release when this joint passes `at` (an unscrewed cap)
    at: float | None = None


class Semantics:
    def __init__(self, model, source):
        meta = json.loads((Path(source) / "interfaces.json").read_text(encoding="utf-8"))
        events = {e["id"]: e for e in meta.get("mechanical_events", [])}
        self.kinds = {r["id"]: r["kind"] for r in meta["interfaces"]}
        self.model = model
        self.connections = []
        for c in meta.get("connections", []):
            eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, c.get("equality", ""))
            if eq < 0:
                continue
            event = events.get(c.get("release_event"))
            joint = None
            if event and event.get("joint"):
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, event["joint"])
                joint = joint if joint >= 0 else None
            self.connections.append(Connection(c["id"], c.get("kind", ""), eq, bool(c.get("irreversible")),
                                               c.get("release_event"), event and event.get("interface"), joint,
                                               event and event.get("at")))

    def _bodies(self, e):
        m = self.model
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            # A joint equality without a second joint locks the first joint; one with two couples them (no lock).
            if m.eq_obj2id[e] >= 0:
                return -1, -1
            return int(m.jnt_bodyid[m.eq_obj1id[e]]), 0
        ids = []
        for obj, kind in ((m.eq_obj1id[e], m.eq_objtype[e]), (m.eq_obj2id[e], m.eq_objtype[e])):
            if obj < 0:
                ids.append(0)                        # unset second object: the world
            else:
                ids.append(int(m.site_bodyid[obj]) if kind == mujoco.mjtObj.mjOBJ_SITE else int(obj))
        return ids

    def holding(self, bodies, active):
        """Active connections tying `bodies` to something outside them."""
        out = []
        for c in self.connections:
            if not active[c.equality]:
                continue
            a, b = self._bodies(c.equality)
            if a >= 0 and (a in bodies) != (b in bodies):
                out.append(c)
        return out

    def moving_bodies(self, interface):
        mech = interface.mechanism
        root = mech.body if mech.body >= 0 else interface.body
        return subtree_bodies(self.model, root)
