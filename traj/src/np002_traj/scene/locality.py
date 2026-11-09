"""Task-local physics: only the instances near the task are simulated.

A generated lab has hundreds of instances (86k geoms, 5k dofs); simulating all of them for every
single-skill task makes each step cost ~0.5 s, almost all of it in the constraint solver for parts
that are metres away and cannot be touched. For one task, every instance farther than RADIUS_M from
the task site (horizontally, to the nearest geom bound) is frozen: its geoms stop colliding with each
other (they still touch simulated geoms, so a simulated part resting on a frozen one stays put), its dofs
get a damping large enough that gravity moves it < 1 mm per simulated minute, and equalities that
only involve frozen bodies are switched off. The robot, the task's own instance and the instances
the task must unlock stay fully simulated. The robot's reach from any dock (dock distance <= 1.3 m
plus 0.8 m arm) is inside RADIUS_M, so nothing it can touch is frozen.
"""
import mujoco
import numpy as np

from .settle import SLEEP

RADIUS_M = 2.5
DAMPING_PER_KG = 1e6        # v = g / (damping / mass) ~ 1e-5 m/s
AWAKE_RADIUS_M = 0.3        # instances this close to the task site never sleep (openers of nested prerequisites)
FROZEN_BIT = 1 << 30        # collision bit shared only by frozen geoms (affinity) and simulated ones (type)


def instance_of_bodies(model):
    return np.array([model.body(model.body_rootid[b]).name.split("__")[0] for b in range(model.nbody)])


def _equality_bodies(model, i):
    """Bodies an equality constrains (0 = world). Compared as ints: `np.int32 in (enum, ...)` is False."""
    kind, objects = int(model.eq_type[i]), [int(o) for o in (model.eq_obj1id[i], model.eq_obj2id[i]) if o >= 0]
    if kind in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
        if int(model.eq_objtype[i]) == int(mujoco.mjtObj.mjOBJ_SITE):
            return [int(model.site_bodyid[o]) for o in objects] or [0]
        return objects or [0]
    if kind == int(mujoco.mjtEq.mjEQ_JOINT):
        return [int(model.jnt_bodyid[o]) for o in objects]
    if kind == int(mujoco.mjtEq.mjEQ_TENDON):
        return [int(model.jnt_bodyid[model.wrap_objid[j]]) for o in objects
                for j in range(model.tendon_adr[o], model.tendon_adr[o] + model.tendon_num[o])
                if model.wrap_type[j] == mujoco.mjtWrap.mjWRAP_JOINT]
    return [-1]                                   # unknown kind: never switched off


class Locality:
    def __init__(self, model, data, robot_instance="robot"):
        """`data` is the settled start state (geom positions decide what is near)."""
        self.model = model
        self.instance = instance_of_bodies(model)
        self.robot = robot_instance
        self.geom_xy = data.geom_xpos[:, :2].copy()
        self.geom_instance = self.instance[model.geom_bodyid]
        self.equalities = [_equality_bodies(model, i) for i in range(model.neq)]
        self.tendon = np.array([model.eq_type[i] == mujoco.mjtEq.mjEQ_TENDON for i in range(model.neq)], bool)
        self.dof_mass = model.body_subtreemass[model.body_rootid[model.dof_bodyid]] + 1.0

    def near(self, point, keep=(), radius=RADIUS_M):
        distance = np.linalg.norm(self.geom_xy - np.asarray(point)[:2], axis=1) - self.model.geom_rbound
        return set(self.geom_instance[distance < radius]) | set(keep) | {self.robot}

    def apply(self, point, keep=()):
        """Freeze everything not near `point`; returns what restore() needs. Sleep is enabled when
        no active tendon equality remains (MuJoCo cannot sleep with those)."""
        m = self.model
        near = self.near(point, keep)
        far_body = np.array([i not in near for i in self.instance])
        far_body[0] = False
        saved = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.dof_damping.copy(), m.eq_active0.copy(),
                 int(m.opt.enableflags), m.tree_sleep_policy.copy())
        far_geom = far_body[m.geom_bodyid]
        solid = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        m.geom_contype[far_geom] = 0
        m.geom_conaffinity[far_geom & solid] = FROZEN_BIT
        m.geom_contype[~far_geom & solid] |= FROZEN_BIT
        far_dof = far_body[m.dof_bodyid]
        m.dof_damping[far_dof] += DAMPING_PER_KG * self.dof_mass[far_dof]
        off = [i for i, bodies in enumerate(self.equalities)
               if bodies and -1 not in bodies and any(b > 0 for b in bodies) and all(b == 0 or far_body[b] for b in bodies)]
        m.eq_active0[off] = 0
        if not np.any(m.eq_active0.astype(bool) & self.tendon):
            m.opt.enableflags |= SLEEP
            # The parts the task acts on never sleep: a pushed hood fell asleep mid-stroke and stopped.
            awake = self.near(point, keep, AWAKE_RADIUS_M)
            task_trees = np.unique(m.body_treeid[[b for b in range(1, m.nbody) if self.instance[b] in awake
                                                  and m.body_treeid[b] >= 0]])
            m.tree_sleep_policy[task_trees] = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
        return saved, {"near_instances": len(near), "frozen_instances": len(set(self.instance) - near),
                       "frozen_geoms": int(far_geom.sum()), "sleep": bool(m.opt.enableflags & SLEEP)}

    def restore(self, saved):
        m = self.model
        m.geom_contype[:], m.geom_conaffinity[:], m.dof_damping[:], m.eq_active0[:], m.opt.enableflags, policy = saved
        m.tree_sleep_policy[:] = policy
