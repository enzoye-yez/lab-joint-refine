"""The execution world. Controllers write actuator commands; one mj_step per tick; invariants checked as it runs.

Failures never make the robot go limp: `hold` freezes every arm at its measured posture and keeps
the grippers' commands, so a recovery starts from a held state.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..control.cartesian import CartesianImpedance
from ..control.joint import JointPD
from ..evidence.predicates import Monitor


class Abort(Exception):
    """An invariant was violated or a skill gave up; carries a structured reason."""

    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass
class Session:
    model: mujoco.MjModel
    data: mujoco.MjData
    robot: object
    recorder: object = None
    invariants: list = field(default_factory=list)
    check_every: int = 10
    max_robot_force: float = None         # default: capability contact_force_n
    max_grip_force: float = None          # on the part being grasped (capability grasp_reaction_n): the squeeze is intended

    def __post_init__(self):
        cap = self.robot.capability
        self.max_robot_force = cap["contact_force_n"] if self.max_robot_force is None else self.max_robot_force
        self.max_grip_force = cap["grasp_reaction_n"] if self.max_grip_force is None else self.max_grip_force
        self.joint = {s: JointPD(self.model, a) for s, a in self.robot.arms.items()}
        self.cartesian = {s: CartesianImpedance(self.model, a, max_force=cap["cartesian_force_n"], max_torque=cap["cartesian_torque_nm"]) for s, a in self.robot.arms.items()}
        self.mode = {s: "joint" for s in self.robot.arms}
        for c in self.joint.values():
            c.hold(self.data)
        self.monitor = Monitor(self.model, self.data)
        self.allowed_contact = set()          # geoms the robot may press on (the target part)
        self.ticks = 0
        mask = np.zeros(self.model.ngeom, bool)
        mask[self.robot.geoms] = True
        self.robot_mask = mask
        self.peak_force = 0.0
        self.base_target = None

    @property
    def time(self):
        return float(self.data.time)

    def use_joint(self, side, q=None, qd=None):
        self.mode[side] = "joint"
        if q is None:
            self.joint[side].hold(self.data)
        else:
            self.joint[side].set(q, qd)

    def use_cartesian(self, side, position=None, rotation=None, posture=None):
        c = self.cartesian[side]
        if self.mode[side] != "cartesian" or position is None:
            c.hold(self.data)
        if position is not None:
            c.set(position, rotation, posture)
        self.mode[side] = "cartesian"

    def hold(self):
        for side in self.robot.arms:
            if self.mode[side] == "cartesian":
                self.cartesian[side].hold(self.data)
            else:
                self.joint[side].hold(self.data)
        self.data.ctrl[self.robot.base_actuators] = 0.0

    def hold_base(self, pose=None):
        """Keep the base at `pose` (default: where it is) with a position loop on the velocity drives."""
        self.base_target = None if pose is False else (self.robot.base_pose(self.data.qpos) if pose is None else np.asarray(pose))

    def _base_loop(self):
        pose = self.robot.base_pose(self.data.qpos)
        error = self.base_target - pose
        error[2] = np.angle(np.exp(1j * error[2]))
        self.robot.base_command(self.data, 4.0 * error[:2], 4.0 * error[2])

    def step(self, ticks=1):
        for _ in range(ticks):
            if self.base_target is not None:
                self._base_loop()
            for side in self.robot.arms:
                (self.cartesian if self.mode[side] == "cartesian" else self.joint)[side].apply(self.data)
            mujoco.mj_step(self.model, self.data)
            if self.recorder:
                self.recorder.tick()
            self.ticks += 1
            if self.ticks % self.check_every == 0:
                self._guard()

    def _guard(self):
        d, n = self.data, self.data.ncon
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            mine = self.robot_mask[g1] ^ self.robot_mask[g2]           # robot against the world
            force = np.zeros(6)
            for i in np.flatnonzero(mine):
                mujoco.mj_contactForce(self.model, d, int(i), force)
                self.peak_force = max(self.peak_force, float(force[0]))
                other = int(g2[i] if self.robot_mask[g1[i]] else g1[i])
                limit = self.max_grip_force if other in self.allowed_contact else self.max_robot_force
                if force[0] > limit:
                    self.hold()
                    raise Abort("contact_force_limit", force_n=float(force[0]), geom=self.model.geom(other).name)
        for rule in self.invariants:
            if self.monitor.evaluate(rule) is False:
                self.hold()
                raise Abort("invariant_violated", rule=rule.get("id"), detail=self.monitor.details.get(rule.get("id")))

    def wait(self, seconds, until=None):
        """Advance up to `seconds`; stop early when `until()` is true. Returns whether it became true."""
        end = self.data.time + seconds
        while self.data.time < end:
            self.step(5)
            if until is not None and until():
                return True
        return until is None
