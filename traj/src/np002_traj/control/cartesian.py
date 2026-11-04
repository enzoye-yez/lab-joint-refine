"""Cartesian impedance for one arm: a spring-damper at the TCP plus null-space posture, with gravity compensation.

Used for every phase in contact (approach, grasp, mechanism following, pressing): the force on the
environment is bounded by stiffness x error and by `max_force`, instead of an integrator winding up.
"""
import mujoco
import numpy as np

from ..geometry import rotation_error


class CartesianImpedance:
    def __init__(self, model, arm, stiffness=(1200.0, 40.0), damping_ratio=1.0, effective_mass=(3.0, 0.08),
                 null_gain=10.0, max_force=60.0, max_torque=6.0):
        self.model, self.arm = model, arm
        self.kp = np.array([stiffness[0]] * 3 + [stiffness[1]] * 3)
        self.kd = 2 * damping_ratio * np.sqrt(self.kp * np.array([effective_mass[0]] * 3 + [effective_mass[1]] * 3))
        self.null_gain, self.max_force, self.max_torque = null_gain, max_force, max_torque
        self.limit = model.actuator_ctrlrange[arm.actuators, 1]
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self.position = self.rotation = self.posture = None
        self.wrench = np.zeros(6)

    def set(self, position, rotation, posture=None):
        self.position, self.rotation = np.asarray(position, float), np.asarray(rotation, float)
        if posture is not None:
            self.posture = np.asarray(posture, float)

    def hold(self, data):
        a = self.arm
        self.set(data.site_xpos[a.tcp].copy(), data.site_xmat[a.tcp].reshape(3, 3).copy(), data.qpos[a.qpos].copy())

    def error(self, data):
        a = self.arm
        return (self.position - data.site_xpos[a.tcp],
                rotation_error(self.rotation, data.site_xmat[a.tcp].reshape(3, 3)))

    def apply(self, data):
        a, m = self.arm, self.model
        mujoco.mj_jacSite(m, data, self.jp, self.jr, a.tcp)
        jac = np.vstack([self.jp[:, a.dofs], self.jr[:, a.dofs]])
        dp, dr = self.error(data)
        velocity = jac @ data.qvel[a.dofs]
        wrench = self.kp * np.concatenate([dp, dr]) - self.kd * velocity
        force = np.linalg.norm(wrench[:3])
        if force > self.max_force:
            wrench[:3] *= self.max_force / force
        torque = np.linalg.norm(wrench[3:])
        if torque > self.max_torque:
            wrench[3:] *= self.max_torque / torque
        self.wrench = wrench
        posture = self.posture if self.posture is not None else data.qpos[a.qpos]
        null = np.eye(len(a.dofs)) - np.linalg.pinv(jac, rcond=1e-3) @ jac
        tau_null = null @ (self.null_gain * (posture - data.qpos[a.qpos]) - 2.0 * data.qvel[a.dofs])
        tau = jac.T @ wrench + tau_null + data.qfrc_bias[a.dofs]
        data.ctrl[a.actuators] = np.clip(tau, -self.limit, self.limit)
