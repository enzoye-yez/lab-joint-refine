"""Joint-space PD with inverse-dynamics gravity and Coriolis compensation for torque-driven arms."""
import numpy as np


class JointPD:
    def __init__(self, model, arm, ki=None, integral_band=0.02, integral_limit=3.0):
        self.model, self.arm = model, arm
        self.limit = model.actuator_ctrlrange[arm.actuators, 1]
        self.target = None
        self.target_velocity = np.zeros(len(arm.joints))
        # Integral action removes the static error left by joint friction (FR3 frictionloss ~1 Nm).
        self.ki = np.asarray(ki if ki is not None else np.zeros_like(arm.kp), float)
        self.band, self.integral_limit = integral_band, integral_limit
        self.integral = np.zeros(len(arm.joints))

    def hold(self, data):
        """Hold the current posture; used after a failure instead of going limp."""
        self.set(data.qpos[self.arm.qpos])

    def set(self, q, qd=None):
        if self.target is None or np.max(np.abs(np.asarray(q) - self.target)) > self.band:
            self.integral[:] = 0.0                     # a new move, not a correction
        self.target = np.array(q, float)
        self.target_velocity = np.zeros_like(self.target) if qd is None else np.array(qd, float)

    def apply(self, data):
        """Write torques for the current state; call once per tick before mj_step."""
        a = self.arm
        error = self.target - data.qpos[a.qpos]
        near = np.abs(error) < self.band
        self.integral = np.where(near, self.integral + error * self.model.opt.timestep, 0.0)
        torque_i = np.clip(self.ki * self.integral, -self.integral_limit, self.integral_limit)
        self.integral = np.where(self.ki > 0, torque_i / np.maximum(self.ki, 1e-9), 0.0)   # anti-windup
        tau = a.kp * error + a.kd * (self.target_velocity - data.qvel[a.dofs]) + torque_i + data.qfrc_bias[a.dofs]
        data.ctrl[a.actuators] = np.clip(tau, -self.limit, self.limit)
        return tau
