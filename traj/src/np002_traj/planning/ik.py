"""Weighted damped least-squares IK for one arm with null-space posture control.

Continuity first: the solver starts from the current posture and prefers the solution nearest
to it, so a moving target never jumps to another elbow branch. Random restarts are a fallback.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..geometry import axis_error, rotation_error


@dataclass
class Target:
    position: np.ndarray                 # world, m
    rotation: np.ndarray | None = None   # world 3x3; or
    axis: tuple | None = None            # (local axis, desired world axis): pointing only
    position_tolerance: float = 5e-4     # m
    rotation_tolerance: float = 5e-3     # rad


@dataclass
class Solution:
    q: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int
    restarts: int = 0


class IKFailure(Exception):
    def __init__(self, best):
        super().__init__(f"no IK solution; best residual {best.position_error*1e3:.1f} mm, {best.rotation_error:.3f} rad")
        self.best = best


@dataclass
class IK:
    model: mujoco.MjModel
    arm: object                          # robot.Arm
    rotation_weight: float = 0.2         # m per rad: 1 rad of orientation error weighs like 0.2 m
    damping: float = 1e-3
    posture_gain: float = 0.05
    max_step: float = 0.1                # rad per iteration
    iterations: int = 300
    data: mujoco.MjData = field(init=False)

    def __post_init__(self):
        self.data = mujoco.MjData(self.model)
        self.limits = self.model.jnt_range[self.arm.joints].copy()
        self.span = self.limits[:, 1] - self.limits[:, 0]
        self.jp, self.jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))

    def _residual(self, q, target):
        a, d = self.arm, self.data
        d.qpos[a.qpos] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        dp = target.position - d.site_xpos[a.tcp]
        mujoco.mj_jacSite(self.model, d, self.jp, self.jr, a.tcp)
        jp, jr = self.jp[:, a.dofs], self.jr[:, a.dofs].copy()
        actual = d.site_xmat[a.tcp].reshape(3, 3)
        if target.rotation is not None:
            dr = rotation_error(target.rotation, actual)
        elif target.axis is not None:
            dr, axis = axis_error(*target.axis, actual)
            jr -= np.outer(axis, axis @ jr)          # rotation about the pointing axis is free
        else:
            dr, jr = np.zeros(3), np.zeros_like(jr)
        return dp, dr, jp, jr

    def solve(self, base_qpos, target, seed=None, posture=None, valid=None, restarts=6, rng=None):
        """base_qpos: full qpos giving the base, lift and other arm; seed: this arm's start posture."""
        a = self.arm
        self.data.qpos[:] = base_qpos
        seed = np.array(base_qpos[a.qpos] if seed is None else seed, float)
        posture = seed if posture is None else np.asarray(posture, float)
        rng = rng or np.random.default_rng(0)
        best, found = None, []
        for restart in range(restarts + 1):
            if restart == 0:
                q = seed.copy()
            else:  # widen the search gradually around the seed before sampling the whole range
                scale = min(1.0, 0.15 * restart)
                q = np.clip(seed + rng.normal(0, scale, len(seed)) * self.span, *self.limits.T)
            for step in range(self.iterations):
                dp, dr, jp, jr = self._residual(q, target)
                pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
                if best is None or pe + self.rotation_weight * re < best.position_error + self.rotation_weight * best.rotation_error:
                    best = Solution(q.copy(), pe, re, step, restart)
                if pe <= target.position_tolerance and re <= target.rotation_tolerance:
                    if valid is None or valid(q):
                        found.append(Solution(q.copy(), pe, re, step, restart))
                    break
                w = self.rotation_weight
                jac = np.vstack([jp, w * jr])
                err = np.concatenate([dp, w * dr])
                inverse = jac.T @ np.linalg.inv(jac @ jac.T + self.damping * np.eye(6))
                # Null space: stay near the preferred posture and away from the joint limits.
                middle = self.limits.sum(axis=1) / 2
                pull = self.posture_gain * ((posture - q) + 0.1 * (middle - q))
                # The exact pseudo-inverse projector keeps the posture pull out of the task space.
                null = np.eye(len(q)) - np.linalg.pinv(jac, rcond=1e-4) @ jac
                dq = inverse @ err + null @ pull
                q = np.clip(q + np.clip(dq, -self.max_step, self.max_step), *self.limits.T)
            if found and found[0].restarts == 0:
                break                              # the continuous solution from the seed
            if len(found) >= 2:
                break
        if not found:
            raise IKFailure(best)
        return min(found, key=lambda s: float(np.sum(((s.q - seed) / self.span) ** 2)))
