"""The hand alone, posed at a tool frame without the arm, for exact distance queries against the scene.

Grasp and press synthesis use it before any arm or base is chosen: the queries run on the same
convex geometry the simulator collides (mj_geomDistance), so a grasp that passes here only fails
later for reasons of reach or arm clearance.
"""
import mujoco
import numpy as np

from ..geometry import subtree_bodies


class HandProbe:
    def __init__(self, scene_model, robot, kinematics, side):
        """robot: bound to the scene model (prefixed names); kinematics: the robot-only model."""
        self.model, self.side = scene_model, side
        km, kr = kinematics.model, kinematics.robot
        arm = kr.arms[side]
        hand = km.body_parentid[km.body(robot.profile["arms"][side]["gripper"]["finger_bodies"][0]).id]
        bodies = subtree_bodies(km, int(hand))
        geoms = [g for g in range(km.ngeom) if km.geom_bodyid[g] in bodies and (km.geom_contype[g] or km.geom_conaffinity[g])]
        self.names = [km.geom(g).name for g in geoms]
        self.geoms = np.array([scene_model.geom(robot.prefix + n).id for n in self.names])
        pads = set(robot.profile["arms"][side]["gripper"]["pads"].values())
        self.is_pad = np.array([n in pads for n in self.names])
        pin = robot.profile["arms"][side]["gripper"].get("press_pin")
        self.is_pin = np.array([bool(pin) and n == pin["body"] for n in self.names])
        blade = robot.profile["arms"][side]["gripper"].get("pry_blade")
        self.is_blade = np.array([bool(blade) and n in blade["geoms"] for n in self.names])
        cup = robot.profile["arms"][side]["gripper"].get("suction")
        self.is_cup = np.array([bool(cup) and n == cup["geom"] for n in self.names])
        self.tools = ~(self.is_pin | self.is_blade)          # the fingers, body and cup (always there)
        self.open = arm.gripper_open
        self.pin_extended, self.blade_extended = arm.pin_extended, arm.blade_extended
        fingers = [km.jnt_qposadr[km.joint(j).id] for j in robot.profile["arms"][side]["gripper"]["joints"]]

        def relative(opening, pin_value, blade_value=0.0):
            d = mujoco.MjData(km)
            mujoco.mj_resetDataKeyframe(km, d, 0)
            d.qpos[fingers] = opening
            if arm.pin_qpos >= 0:
                d.qpos[arm.pin_qpos] = pin_value
            if arm.blade_qpos >= 0:
                d.qpos[arm.blade_qpos] = blade_value
            mujoco.mj_kinematics(km, d)
            rot = d.site_xmat[arm.tcp].reshape(3, 3)
            pos = d.site_xpos[arm.tcp]
            return np.array([rot.T @ (d.geom_xpos[g] - pos) for g in geoms]), \
                np.array([rot.T @ d.geom_xmat[g].reshape(3, 3) for g in geoms])

        self.pos_closed, self.mat = relative(0.0, 0.0)
        pos_open, _ = relative(self.open, 0.0)
        pos_pin, _ = relative(0.0, self.pin_extended)
        self.finger_motion = (pos_open - self.pos_closed) / self.open          # per metre of finger opening
        self.pin_motion = (pos_pin - self.pos_closed) / max(self.pin_extended, 1e-9)
        pos_blade, _ = relative(0.0, 0.0, self.blade_extended)
        self.blade_motion = (pos_blade - self.pos_closed) / max(self.blade_extended, 1e-9)
        self.rbound = scene_model.geom_rbound[self.geoms]

    def place(self, data, position, rotation, opening, pin=0.0, blade=0.0):
        """Write the hand geometry into `data` (a scratch copy) at TCP pose (position, rotation)."""
        rel = self.pos_closed + self.finger_motion * opening + self.pin_motion * pin + self.blade_motion * blade
        data.geom_xpos[self.geoms] = position + rel @ rotation.T
        data.geom_xmat[self.geoms] = np.einsum("ij,njk->nik", rotation, self.mat).reshape(-1, 9)

    def distance(self, data, which, others, limit=0.05):
        """(smallest signed distance, hand geom, other geom, fromto) between selected hand geoms and others."""
        best = (np.inf, -1, -1, None)
        if not len(others):
            return best
        m = self.model
        fromto = np.zeros(6)
        for i in np.flatnonzero(which):
            g = int(self.geoms[i])
            gap = np.linalg.norm(data.geom_xpos[others] - data.geom_xpos[g], axis=1) - m.geom_rbound[others] - self.rbound[i]
            for h in np.asarray(others)[gap < limit]:
                value = mujoco.mj_geomDistance(m, data, g, int(h), limit, fromto)
                if value < best[0]:
                    best = (value, g, int(h), fromto.copy())
        return best
