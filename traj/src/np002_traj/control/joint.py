"""Joint-space PD with inverse-dynamics gravity and Coriolis compensation for torque-driven arms."""
import numpy as np


class JointPD:
    def __init__(self, model, arm):
        self.model, self.arm = model, arm
        self.limit = model.actuator_ctrlrange[arm.actuators, 1]
        self.target = None
        self.target_velocity = np.zeros(len(arm.joints))

    def hold(self, data):
        """Hold the current posture; used after a failure instead of going limp."""
        self.set(data.qpos[self.arm.qpos])

    def set(self, q, qd=None):
        self.target = np.array(q, float)
        self.target_velocity = np.zeros_like(self.target) if qd is None else np.array(qd, float)

    def apply(self, data):
        """Write torques for the current state; call once per tick before mj_step."""
        a = self.arm
        error = self.target - data.qpos[a.qpos]
        tau = a.kp * error + a.kd * (self.target_velocity - data.qvel[a.dofs]) + data.qfrc_bias[a.dofs]
        data.ctrl[a.actuators] = np.clip(tau, -self.limit, self.limit)
        return tau
