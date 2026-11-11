"""Condition evaluation from simulation state.

A condition is a dict with a `kind`. Names refer to model entities directly; task files bind
roles to names before they reach this module, and unknown names raise immediately.
Results are True, False or None (cannot be observed now). Combinators use Kleene logic.
"""
import mujoco
import numpy as np

from ..geometry import rotation_error, subtree_bodies, body_geoms


def all_of(values):
    values = list(values)
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def any_of(values):
    values = list(values)
    if any(v is True for v in values):
        return True
    return None if any(v is None for v in values) else False


class Monitor:
    """Evaluates conditions every tick; keeps the state of stateful conditions between ticks."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.anchors, self.timers, self.origins, self.details = {}, {}, {}, {}
        self.reached = set()          # keys that have been true at least once
        self._geoms = {}
        self._force = np.zeros(6)

    # -- measurement helpers -------------------------------------------------
    def geoms(self, name):
        """Geoms of a named geom, or of a named body's subtree."""
        if name not in self._geoms:
            m = self.model
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g >= 0:
                self._geoms[name] = np.array([g])
            else:
                self._geoms[name] = body_geoms(m, subtree_bodies(m, m.body(name).id))
        return self._geoms[name]

    def contacts(self, a, b):
        """Contacts between geom sets a and b: list of (geom_a, geom_b, normal force N, depth m)."""
        d = self.data
        n = d.ncon
        if not n:
            return []
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        forward = np.isin(g1, a) & np.isin(g2, b)
        backward = np.isin(g2, a) & np.isin(g1, b)
        rows = []
        for i in np.flatnonzero(forward | backward):
            mujoco.mj_contactForce(self.model, d, int(i), self._force)
            pair = (int(g1[i]), int(g2[i])) if forward[i] else (int(g2[i]), int(g1[i]))
            rows.append((*pair, float(self._force[0]), float(-d.contact.dist[i])))
        return rows

    def gap(self, a, b, limit=0.05):
        best = limit
        for g in a:
            for h in b:
                best = min(best, mujoco.mj_geomDistance(self.model, self.data, int(g), int(h), limit, None))
        return float(best)

    def pose(self, body):
        b = self.model.body(body).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, rule, key=None):
        key = key or rule.get("id") or repr(sorted(rule.items()))
        value, detail = getattr(self, "_" + rule["kind"])(rule, key)
        value = None if value is None else bool(value)   # numpy booleans would defeat `is True`
        if "for_s" in rule and value is True:
            start = self.timers.setdefault(key, self.data.time)
            value = (self.data.time - start) >= rule["for_s"]
            detail = {**detail, "held_s": self.data.time - start}
        elif "for_s" in rule:
            self.timers.pop(key, None)
        self.details[key] = {"value": value, **detail}
        if value:
            self.reached.add(key)
        return value

    def _items(self, rule, key):
        return [self.evaluate(item, f"{key}/{i}") for i, item in enumerate(rule["items"])]

    def _all(self, rule, key):
        return all_of(self._items(rule, key)), {}

    def _any(self, rule, key):
        return any_of(self._items(rule, key)), {}

    def _not(self, rule, key):
        value = self.evaluate(rule["item"], key + "/not")
        return (None if value is None else not value), {}

    def _joint_range(self, rule, key):
        j = self.model.joint(rule["joint"])
        value = float(self.data.qpos[j.qposadr[0]])
        low, high = rule["range"]
        return low <= value <= high, {"value": value}

    def _joint_fraction(self, rule, key):
        """Fraction of the joint's travel, e.g. a door opened to at least 90 % of its guide."""
        j = self.model.joint(rule["joint"])
        low, high = self.model.jnt_range[j.id]
        fraction = (float(self.data.qpos[j.qposadr[0]]) - low) / (high - low)
        if rule.get("from") == "high":
            fraction = 1 - fraction
        return rule.get("min", 0.0) <= fraction <= rule.get("max", 1.0), {"fraction": fraction}

    def _joint_travel(self, rule, key):
        """Progress of a joint from where it was first evaluated towards its farther range end.

        `absolute` (m or rad), when given, is the displacement that counts as done (e.g. a button's
        actuation point); otherwise `travel` is the required fraction of the remaining range.
        """
        j = self.model.joint(rule["joint"])
        value = float(self.data.qpos[j.qposadr[0]])
        start = self.origins.setdefault(key, value)
        low, high = self.model.jnt_range[j.id]
        far = high if high - start >= start - low else low
        moved = (value - start) * (1 if far >= start else -1)
        needed = rule["absolute"] if rule.get("absolute") else rule.get("travel", 0.8) * abs(far - start)
        return moved >= needed, {"moved": moved, "needed": needed, "far_end": float(far)}

    def _lifted(self, rule, key):
        """The body is at least height_m above where it was first evaluated."""
        position, _ = self.pose(rule["body"])
        start = self.origins.setdefault(key, position)
        rise = float(position[2] - start[2])
        return rise >= rule["height_m"], {"rise_m": rise}

    def _contact(self, rule, key):
        rows = self.contacts(self.geoms(rule["a"]), self.geoms(rule["b"]))
        force = sum(r[2] for r in rows)
        return force >= rule.get("min_force_n", 0.0) and bool(rows), {"force_n": force}

    def _gap(self, rule, key):
        value = self.gap(self.geoms(rule["a"]), self.geoms(rule["b"]), rule["min_m"] * 2 + 0.01)
        return value >= rule["min_m"], {"gap_m": value}

    def _grasped(self, rule, key):
        """Both pads press the object and its pose in the hand has not drifted since the grasp began.

        The anchor is set once when the grasp is established and cleared only when the grasp is lost;
        a failed sample never moves it, so slow slip accumulates and is detected.
        """
        forces = [sum(r[2] for r in self.contacts(self.geoms(pad), self.geoms(rule["object"]))) for pad in rule["pads"]]
        pressed = all(f >= rule.get("min_force_n", 2.0) for f in forces)
        site = self.model.site(rule["tcp"]).id
        frame = self.data.site_xmat[site].reshape(3, 3)
        position, rotation = self.pose(rule["object"])
        relative = frame.T @ (position - self.data.site_xpos[site]), frame.T @ rotation
        if not pressed:
            self.anchors.pop(key, None)
            return False, {"pad_force_n": forces}
        anchor = self.anchors.setdefault(key, relative)
        drift = float(np.linalg.norm(relative[0] - anchor[0]))
        turn = float(np.linalg.norm(rotation_error(anchor[1], relative[1])))
        ok = drift <= rule.get("max_drift_m", 3e-3) and turn <= rule.get("max_turn_rad", 0.05)
        return ok, {"pad_force_n": forces, "drift_m": drift, "turn_rad": turn}

    def _displaced(self, rule, key):
        """Invariant: a body stays within max_m / max_rad of where it was when first evaluated."""
        position, rotation = self.pose(rule["body"])
        origin = self.origins.setdefault(key, (position, rotation))
        move = float(np.linalg.norm(position - origin[0]))
        turn = float(np.linalg.norm(rotation_error(origin[1], rotation)))
        return move <= rule.get("max_m", 2e-3) and turn <= rule.get("max_rad", 0.02), {"moved_m": move, "turned_rad": turn}

    def _force_limit(self, rule, key):
        """Invariant: total normal force between a and b stays below the limit."""
        rows = self.contacts(self.geoms(rule["a"]), self.geoms(rule["b"]))
        force = sum(r[2] for r in rows)
        return force <= rule["max_n"], {"force_n": force}

    def _at_rest(self, rule, key):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.model.body(rule["body"]).id, velocity, 0)
        angular, linear = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
        return linear <= rule.get("max_mps", 1e-3) and angular <= rule.get("max_radps", 0.01), {"linear_mps": linear, "angular_radps": angular}

    def _inside(self, rule, key):
        """The body origin lies in an axis-aligned box given in the frame of a reference body."""
        position, _ = self.pose(rule["body"])
        origin, rotation = self.pose(rule["frame"]) if "frame" in rule else (np.zeros(3), np.eye(3))
        local = rotation.T @ (position - origin)
        low, high = np.asarray(rule["low"]), np.asarray(rule["high"])
        return bool(np.all(local >= low) and np.all(local <= high)), {"local_m": local.tolist()}

    def _on(self, rule, key):
        """The object touches the support and is at rest."""
        touching = bool(self.contacts(self.geoms(rule["object"]), self.geoms(rule["support"])))
        rest, detail = self._at_rest({"body": rule["object"]}, key)
        return touching and rest, {"touching": touching, **detail}
