"""Signed-distance collision queries for the robot against its own links and nearby obstacles.

Runs on its own MjData. Edges are subdivided until no robot geometry moves more than `step`
metres between samples, so thin plates and handles cannot be tunnelled through.
"""
import mujoco
import numpy as np

from ..geometry import rotation_error

PLANES = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))


def collides(model, g, h):
    return bool(model.geom_contype[g] & model.geom_conaffinity[h] or model.geom_contype[h] & model.geom_conaffinity[g])


def excluded(model, g, h, signatures):
    a, b = model.body_weldid[model.geom_bodyid[g]], model.body_weldid[model.geom_bodyid[h]]
    parent = model.body_weldid[model.body_parentid]
    if a == b or parent[a] == b or parent[b] == a:   # MuJoCo filters parent-child weld groups
        return True
    signature = (min(a, b) << 16) + max(a, b)
    return signature in signatures


class CollisionWorld:
    def __init__(self, model, robot, joints, margin=0.01, self_margin=0.004, radius=2.0, ignore=()):
        """joints: qpos addresses varied by queries. ignore: geoms never checked (e.g. a grasped handle)."""
        self.model, self.robot, self.joints = model, robot, np.asarray(joints)
        self.margin, self.self_margin, self.radius = margin, self_margin, radius
        self.data = mujoco.MjData(model)
        self.moving = np.array([g for g in robot.geoms if model.geom_contype[g] or model.geom_conaffinity[g]])
        self.ignore = set(ignore)
        signatures = set(model.exclude_signature.tolist())
        pairs = [(g, h) for i, g in enumerate(self.moving) for h in self.moving[i + 1:]
                 if collides(model, g, h) and not excluded(model, g, h, signatures)]
        self.self_pairs = np.array(pairs, int).reshape(-1, 2)
        self.obstacle_pairs = np.zeros((0, 2), int)

    def sync(self, data):
        """Copy the world state; obstacle pairs are re-selected around the robot by `select`."""
        d = self.data
        d.qpos[:], d.mocap_pos[:], d.mocap_quat[:], d.eq_active[:] = data.qpos, data.mocap_pos, data.mocap_quat, data.eq_active
        mujoco.mj_kinematics(self.model, d)
        self.select()

    def select(self, centre=None):
        """Obstacle pairs within `radius` of the robot root (or `centre`), vectorised."""
        m, d = self.model, self.data
        if not hasattr(self, "_others"):
            robot = np.zeros(m.ngeom, bool)
            robot[self.robot.geoms] = True
            usable = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
            self._others = np.flatnonzero(~robot & usable)
        others = np.array([g for g in self._others if g not in self.ignore], int) if self.ignore else self._others
        centre = d.xpos[self.robot.root] if centre is None else np.asarray(centre)
        near = (np.linalg.norm(d.geom_xpos[others] - centre, axis=1) - m.geom_rbound[others] < self.radius) \
            | np.isin(m.geom_type[others], PLANES)
        others = others[near]
        a = np.repeat(self.moving, len(others))
        b = np.tile(others, len(self.moving))
        ok = ((m.geom_contype[a] & m.geom_conaffinity[b]) | (m.geom_contype[b] & m.geom_conaffinity[a])) != 0
        self.obstacle_pairs = np.stack([a[ok], b[ok]], axis=1)

    def set_joints(self, addresses, values):
        """Fix other joints of the query world (e.g. a candidate base pose and lift height)."""
        self.data.qpos[np.asarray(addresses)] = values
        mujoco.mj_kinematics(self.model, self.data)

    def _pose(self, q):
        self.data.qpos[self.joints] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _closest(self, pairs, limit):
        """Smallest signed distance among pairs closer than `limit` (inf if none), with the pair."""
        if not len(pairs):
            return np.inf, None
        m, d = self.model, self.data
        a, b = pairs[:, 0], pairs[:, 1]
        gap = np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
        plane = np.isin(m.geom_type[a], PLANES) | np.isin(m.geom_type[b], PLANES)
        best, pair = np.inf, None
        for i in np.flatnonzero((gap < limit) | plane):
            value = mujoco.mj_geomDistance(m, d, int(a[i]), int(b[i]), limit, None)
            if value < best:
                best, pair = value, (int(a[i]), int(b[i]))
        return best, pair

    def allow(self, geoms=()):
        """Geoms the robot may touch in the next queries (the part being grasped or pressed)."""
        self.allowed = np.asarray(sorted(set(int(g) for g in geoms)), int)

    def clearance(self, q):
        """(distance to obstacles, distance between robot links, closest obstacle pair) at posture q."""
        self._pose(q)
        pairs = self.obstacle_pairs
        if len(getattr(self, "allowed", ())):
            pairs = pairs[~np.isin(pairs[:, 1], self.allowed)]
        obstacle, pair = self._closest(pairs, self.margin * 4)
        own, _ = self._closest(self.self_pairs, self.self_margin * 4)
        return obstacle, own, pair

    def valid(self, q):
        obstacle, own, _ = self.clearance(q)
        return obstacle >= self.margin and own >= self.self_margin

    def blocking(self, q):
        """Names of the closest obstacle pair if q is invalid, else None (for failure summaries)."""
        obstacle, own, pair = self.clearance(q)
        if obstacle >= self.margin and own >= self.self_margin:
            return None
        if pair is None:
            return {"kind": "self", "distance_m": float(own)}
        return {"kind": "obstacle", "geoms": [self.model.geom(g).name for g in pair], "distance_m": float(obstacle)}

    def _motion(self, qa, qb):
        """Upper bound of how far any robot geometry surface moves between two postures."""
        self._pose(qa)
        pa, ra = self.data.geom_xpos[self.moving].copy(), self.data.geom_xmat[self.moving].copy()
        self._pose(qb)
        pb, rb = self.data.geom_xpos[self.moving], self.data.geom_xmat[self.moving]
        turn = np.array([np.linalg.norm(rotation_error(x, y)) for x, y in zip(ra, rb)])
        return float(np.max(np.linalg.norm(pb - pa, axis=1) + self.model.geom_rbound[self.moving] * turn))

    def edge_valid(self, qa, qb, step=0.005):
        """Check the straight joint-space edge with samples no more than `step` metres apart."""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        if not self.valid(qb):
            return False
        pieces = max(1, int(np.ceil(self._motion(qa, qb) / step)))
        for t in np.linspace(0, 1, pieces + 1)[1:-1]:
            if not self.valid(qa + (qb - qa) * t):
                return False
        return True
