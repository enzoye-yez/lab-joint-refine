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


ATTACHED_GEOMS = 8          # geoms of a carried part checked (the largest; the rest lie within them)


class CollisionWorld:
    def __init__(self, model, robot, joints, margin=0.01, self_margin=0.004, radius=2.0, ignore=(), driven_only=False):
        """joints: qpos addresses varied by queries. ignore: geoms never checked (e.g. a grasped handle).
        driven_only: only robot geoms moved by `joints` are checked against the world; the rest of the robot
        (the other arm holding a lid, a hand resting on the bench) stays where it is during the motion, so a
        contact it already has neither starts nor ends there. Robot-robot pairs are always checked."""
        self.model, self.robot, self.joints = model, robot, np.asarray(joints)
        self.margin, self.self_margin, self.radius = margin, self_margin, radius
        self.data = mujoco.MjData(model)
        self.moving = np.array([g for g in robot.geoms if model.geom_contype[g] or model.geom_conaffinity[g]])
        self.ignore = set(ignore)
        self.driven = self._driven_geoms() if driven_only else None
        signatures = set(model.exclude_signature.tolist())
        pairs = [(g, h) for i, g in enumerate(self.moving) for h in self.moving[i + 1:]
                 if collides(model, g, h) and not excluded(model, g, h, signatures)]
        self.self_pairs = np.array(pairs, int).reshape(-1, 2)
        self.obstacle_pairs = np.zeros((0, 2), int)

    def _driven_geoms(self):
        m = self.model
        addresses = set(int(a) for a in self.joints.tolist())
        roots = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_qposadr[j]) in addresses}
        driven = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            a = b
            while a > 0 and not driven[b]:
                driven[b] = a in roots
                a = int(m.body_parentid[a])
        return self.moving[driven[m.geom_bodyid[self.moving]]]

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
        moving = self.moving if self.driven is None else self.driven
        a = np.repeat(moving, len(others))
        b = np.tile(others, len(moving))
        ok = ((m.geom_contype[a] & m.geom_conaffinity[b]) | (m.geom_contype[b] & m.geom_conaffinity[a])) != 0
        # The base rides on its planar joints: its own geometry is not checked against the floor.
        ok &= ~((m.geom_bodyid[a] == self.robot.root) & np.isin(m.geom_type[b], PLANES))
        self.obstacle_pairs = np.stack([a[ok], b[ok]], axis=1)

    def attach(self, root, site):
        """A free part held in the hand (a carried tool) rides with `site` in the next queries and its
        geometry is checked against the world and the rest of the robot, as the hand's own is."""
        m, d = self.model, self.data
        joint = int(m.body_jntadr[root])
        if joint < 0 or m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            return
        rotation = d.site_xmat[site].reshape(3, 3)
        self.attached = (int(site), int(m.jnt_qposadr[joint]),
                         rotation.T @ (d.xpos[root] - d.site_xpos[site]), rotation.T @ d.xmat[root].reshape(3, 3))
        bodies = [b for b in range(m.nbody) if self._within(b, root)]
        geoms = np.flatnonzero(np.isin(m.geom_bodyid, bodies) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))
        geoms = geoms[np.argsort(-m.geom_rbound[geoms])][:ATTACHED_GEOMS]     # its largest pieces bound it
        # The hand holding it (the site's body and below) touches it by design; the rest of the robot may not.
        hand = int(m.site_bodyid[site])
        robot = np.asarray([g for g in self.robot.geoms if not self._within(int(m.geom_bodyid[g]), hand)
                            and (m.geom_contype[g] or m.geom_conaffinity[g])], int)
        others = np.concatenate([self.obstacle_pairs[:, 1], robot]) if len(self.obstacle_pairs) else robot
        others = np.unique(others[~np.isin(others, geoms)])
        a, b = np.repeat(geoms, len(others)), np.tile(others, len(geoms))
        self.obstacle_pairs = np.concatenate([self.obstacle_pairs, np.stack([a, b], axis=1)])

    def _within(self, body, root):
        while body > 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return body == root

    def set_joints(self, addresses, values):
        """Fix other joints of the query world (e.g. a candidate base pose and lift height)."""
        self.data.qpos[np.asarray(addresses)] = values
        mujoco.mj_kinematics(self.model, self.data)

    def _pose(self, q):
        self.data.qpos[self.joints] = q
        mujoco.mj_kinematics(self.model, self.data)
        if getattr(self, "attached", None) is not None:
            site, address, offset, turn = self.attached
            d = self.data
            rotation = d.site_xmat[site].reshape(3, 3)
            d.qpos[address:address + 3] = d.site_xpos[site] + rotation @ offset
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (rotation @ turn).flatten())
            d.qpos[address + 3:address + 7] = quat
            mujoco.mj_kinematics(self.model, d)

    def hold(self, geoms=()):
        """Robot geoms that stay where they are, holding the part, during the next queries: only their
        clearance to the moving arm (self pairs) is checked, not their contact with the world."""
        self.held = np.asarray(sorted(int(g) for g in geoms), int)

    def tool(self, geoms=(), margin=None):
        """Robot geoms held to their own (smaller) margin in the next queries: the tool element that
        makes the contact (a pry blade, a press pin), checked to the margin it was synthesised with."""
        self.tight = {int(g): float(margin) for g in geoms} if margin is not None else {}

    def _closest(self, pairs, limit, credit=False):
        """Smallest signed distance among pairs closer than `limit` (inf if none), with the pair.
        A tool geom's distance is credited by how much smaller its own margin is than the world's."""
        if not len(pairs):
            return np.inf, None
        m, d = self.model, self.data
        a, b = pairs[:, 0], pairs[:, 1]
        gap = np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
        plane = np.isin(m.geom_type[a], PLANES) | np.isin(m.geom_type[b], PLANES)
        best, pair = np.inf, None
        for i in np.flatnonzero((gap < limit) | plane):
            value = mujoco.mj_geomDistance(m, d, int(a[i]), int(b[i]), limit, None)
            tight = getattr(self, "tight", None) if credit else None
            if tight and int(a[i]) in tight:
                value += max(self.margin - tight[int(a[i])], 0.0)
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
        if len(getattr(self, "held", ())):
            pairs = pairs[~np.isin(pairs[:, 0], self.held)]
        obstacle, pair = self._closest(pairs, self.margin * 4, credit=True)
        own, self_pair = self._closest(self.self_pairs, self.self_margin * 4)
        self.last_self_pair = self_pair
        return obstacle, own, pair

    def close_pairs(self, q, limit):
        """Every robot-obstacle and robot-robot pair closer than `limit` at posture q:
        (signed distance, robot geom, other geom, point on robot, unit direction robot -> other, self pair)."""
        self._pose(q)
        m, d = self.model, self.data
        out = []
        pairs = self.obstacle_pairs
        if len(getattr(self, "allowed", ())):
            pairs = pairs[~np.isin(pairs[:, 1], self.allowed)]
        fromto = np.zeros(6)
        for group, own in ((pairs, False), (self.self_pairs, True)):
            if not len(group):
                continue
            a, b = group[:, 0], group[:, 1]
            gap = np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
            plane = np.isin(m.geom_type[a], PLANES) | np.isin(m.geom_type[b], PLANES)
            for i in np.flatnonzero((gap < limit) | plane):
                value = mujoco.mj_geomDistance(m, d, int(a[i]), int(b[i]), limit, fromto)
                if value >= limit:
                    continue
                direction = fromto[3:] - fromto[:3]
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    direction, norm = d.geom_xpos[b[i]] - d.geom_xpos[a[i]], 1.0
                    norm = np.linalg.norm(direction) or 1.0
                direction = direction / norm * (1.0 if value >= 0 else -1.0)
                out.append((float(value), int(a[i]), int(b[i]), fromto[:3].copy(), direction, own))
        return out

    def valid(self, q):
        obstacle, own, _ = self.clearance(q)
        return obstacle >= self.margin and own >= self.self_margin

    def blocking(self, q):
        """Names of the closest obstacle pair if q is invalid, else None (for failure summaries)."""
        obstacle, own, pair = self.clearance(q)
        if obstacle >= self.margin and own >= self.self_margin:
            return None
        if own < self.self_margin and (obstacle >= self.margin or own - self.self_margin < obstacle - self.margin):
            names = [self.model.geom(g).name for g in self.last_self_pair] if self.last_self_pair else ["self", "self"]
            return {"kind": "self", "geoms": names, "distance_m": float(own)}
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
