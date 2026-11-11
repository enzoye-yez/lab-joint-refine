"""Whole-body posture repair: keep the TCP on its target and push every link out of collision.

Sampling IK finds one solution and the collision check either accepts or rejects it. When it is
rejected, this solver moves the active arm in the null space of its task, and (optionally) the
other arm and the lift, along the signed-distance gradient of every close pair until all pairs
clear their margin. The task is the primary objective, clearance the secondary one.
"""
import mujoco
import numpy as np

from ..geometry import rotation_error


class WholeBody:
    def __init__(self, kinematics, world, robot, side, rotation_weight=0.2, damping=1e-4):
        """kinematics: robot-only model (Jacobians); world: CollisionWorld on the scene varying this arm;
        robot: bound to the scene model."""
        self.kin, self.world, self.robot, self.side = kinematics, world, robot, side
        km, kr = kinematics.model, kinematics.robot
        self.other = [o for o in robot.arms if o != side][0]
        self.arm_k = kr.arms[side]
        self.columns = {"arm": kr.arms[side].dofs, "other": kr.arms[self.other].dofs,
                        "lift": np.array([km.jnt_dofadr[kr.lift_joint]])}
        # scene body -> robot-only body, by name
        self.body = {}
        for b in robot.bodies:
            name = world.model.body(b).name[len(robot.prefix):]
            k = mujoco.mj_name2id(km, mujoco.mjtObj.mjOBJ_BODY, name)
            if k >= 0:
                self.body[b] = k
        self.limits = {"arm": km.jnt_range[kr.arms[side].joints], "other": km.jnt_range[kr.arms[self.other].joints],
                       "lift": np.array([robot.profile["lift"]["range_m"]])}
        self.w, self.damping = rotation_weight, damping
        self.data = mujoco.MjData(km)
        self.jp, self.jr = np.zeros((3, km.nv)), np.zeros((3, km.nv))

    def _scene(self, base, lift, other):
        r = self.robot
        addresses = np.concatenate([r.base_qpos, [r.lift_qpos], r.arms[self.other].qpos])
        self.world.set_joints(addresses, np.concatenate([r.base_joint_values(base), [lift], other]))

    def repair(self, base, lift, q, other, target, free=("arm", "other", "lift"), margin=None, self_margin=None,
               iterations=60, step=0.05, buffer=0.015):
        """Returns (q, other, lift) clear of collision with the TCP on `target`, or None."""
        km, kd, kr = self.kin.model, self.data, self.kin.robot
        world = self.world
        margin = world.margin if margin is None else margin
        self_margin = world.self_margin if self_margin is None else self_margin
        q, other, lift = np.array(q, float), np.array(other, float), float(lift)
        names = [n for n in ("arm", "other", "lift") if n in free]
        base_values = kr.base_joint_values(base)
        for _ in range(iterations):
            self._scene(base, lift, other)
            pairs = world.close_pairs(q, max(margin, self_margin) + buffer)
            # Task residual and Jacobian in the robot-only model.
            kd.qpos[:] = self.kin.qpos(base_values, lift, {self.side: q, self.other: other})
            mujoco.mj_kinematics(km, kd)
            mujoco.mj_comPos(km, kd)
            tcp = self.arm_k.tcp
            dp = target.position - kd.site_xpos[tcp]
            dr = rotation_error(target.rotation, kd.site_xmat[tcp].reshape(3, 3)) if target.rotation is not None else np.zeros(3)
            violated = [p for p in pairs if p[0] < (self_margin if p[5] else margin)]
            if not violated and np.linalg.norm(dp) <= target.position_tolerance and np.linalg.norm(dr) <= target.rotation_tolerance:
                return q, other, lift
            cols = np.concatenate([self.columns[n] for n in names])
            mujoco.mj_jacSite(km, kd, self.jp, self.jr, tcp)
            jt = np.vstack([self.jp[:, cols], self.w * self.jr[:, cols]])
            et = np.concatenate([dp, self.w * dr])
            rows, rhs = [], []
            jac = np.zeros((3, km.nv))
            for distance, g, h, point, direction, own in pairs:
                want = (self_margin if own else margin) + buffer
                bodies = [(int(world.model.geom_bodyid[g]), point, 1.0)]
                if own:
                    bodies.append((int(world.model.geom_bodyid[h]), point + direction * max(distance, 0.0), -1.0))
                row = np.zeros(len(cols))
                for b, at, sign in bodies:
                    if b not in self.body:
                        continue
                    mujoco.mj_jac(km, kd, jac, None, at, self.body[b])
                    row += sign * (direction @ jac[:, cols])
                rows.append(-row)                         # d(distance)/dq
                rhs.append(want - distance)
            inverse = jt.T @ np.linalg.inv(jt @ jt.T + self.damping * np.eye(6))
            dq = inverse @ et
            if rows:
                jc, ec = np.array(rows), np.array(rhs)
                null = np.eye(len(cols)) - np.linalg.pinv(jt, rcond=1e-4) @ jt
                jn = jc @ null
                dq = dq + null @ (jn.T @ np.linalg.solve(jn @ jn.T + 1e-4 * np.eye(len(rows)), ec - jc @ dq))
            dq = np.clip(dq, -step, step)
            offset = 0
            for n in names:
                size = len(self.columns[n])
                part = dq[offset:offset + size]
                offset += size
                if n == "arm":
                    q = np.clip(q + part, *self.limits["arm"].T)
                elif n == "other":
                    other = np.clip(other + part, *self.limits["other"].T)
                else:
                    lift = float(np.clip(lift + part[0], *self.limits["lift"][0]))
        return None
