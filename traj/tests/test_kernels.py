"""Orientation kernels, IK and collision queries (plan §0, §4)."""
import unittest

import mujoco
import numpy as np

from np002_traj.geometry import axis_error, quat_to_mat, rotation_error
from np002_traj.planning.collision import CollisionWorld
from np002_traj.planning.ik import IK, IKFailure, Target
from support import world


def rotation(axis, angle):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, float), angle)
    return quat_to_mat(quat)


class OrientationTests(unittest.TestCase):
    def test_rotation_error_recovers_axis_angle(self):
        a = rotation([0, 0, 1], 0.3)
        b = rotation([1, 0, 0], 0.2) @ a
        np.testing.assert_allclose(rotation_error(b, a), [0.2, 0, 0], atol=1e-9)
        np.testing.assert_allclose(rotation_error(a, a), 0, atol=1e-12)

    def test_axis_error_ignores_rotation_about_the_axis(self):
        frame = rotation([0, 0, 1], 1.0)                    # spin about z leaves z pointing up
        error, axis = axis_error([0, 0, 1], [0, 0, 1], frame)
        np.testing.assert_allclose(error, 0, atol=1e-12)
        error, _ = axis_error([0, 0, 1], [1, 0, 0], np.eye(3))
        np.testing.assert_allclose(error, [0, np.pi / 2, 0], atol=1e-9)


class IKTests(unittest.TestCase):
    def setUp(self):
        self.model, self.data, self.robot = world()
        self.arm = self.robot.arms["left"]
        self.ik = IK(self.model, self.arm)

    def fk(self, q):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.qpos[self.arm.qpos] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.arm.tcp].copy(), d.site_xmat[self.arm.tcp].reshape(3, 3).copy()

    def test_reachable_poses_are_solved_from_home(self):
        rng = np.random.default_rng(1)
        home = self.data.qpos[self.arm.qpos].copy()
        low, high = self.model.jnt_range[self.arm.joints].T
        for _ in range(20):
            q = np.clip(home + rng.normal(0, 0.35, 7), low, high)
            position, frame = self.fk(q)
            solution = self.ik.solve(self.data.qpos, Target(position, frame))
            self.assertLess(solution.position_error, 5e-4)
            self.assertLess(solution.rotation_error, 5e-3)

    def test_small_target_motion_gives_small_joint_motion(self):
        position, frame = self.fk(self.data.qpos[self.arm.qpos])
        previous = self.data.qpos[self.arm.qpos].copy()
        for step in range(20):                           # a 10 cm sweep in 5 mm increments
            solution = self.ik.solve(self.data.qpos, Target(position + [0, 0.005 * step, 0], frame), seed=previous)
            self.assertLess(np.max(np.abs(solution.q - previous)), 0.1)
            previous = solution.q

    def test_pointing_target_leaves_roll_free(self):
        position, _ = self.fk(self.data.qpos[self.arm.qpos])
        solution = self.ik.solve(self.data.qpos, Target(position + [0.05, 0, -0.1], axis=([0, 0, 1], [0, 0, -1])))
        _, frame = self.fk(solution.q)
        self.assertGreater(frame[:, 2] @ [0, 0, -1], 0.999)

    def test_unreachable_target_reports_best_residual(self):
        with self.assertRaises(IKFailure) as failure:
            self.ik.solve(self.data.qpos, Target(np.array([3.0, 0, 1.0])), restarts=1)
        self.assertGreater(failure.exception.best.position_error, 1.0)


class CollisionTests(unittest.TestCase):
    def test_edge_through_a_3mm_plate_is_rejected(self):
        def plate(spec):
            body = spec.worldbody.add_body(name="plate", pos=[0.55, 0.22, 0.9])
            body.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0015, 0.15, 0.15])
        model, data, robot = world(extra=plate)
        arm = robot.arms["left"]
        ik = IK(model, arm)
        check = CollisionWorld(model, robot, arm.qpos, margin=0.005)
        check.sync(data)
        down = ([0, 0, 1], [0, 0, -1])                       # gripper pointing down, sweeping along +x
        before = ik.solve(data.qpos, Target(np.array([0.42, 0.22, 0.9]), axis=down), valid=check.valid).q
        after = ik.solve(data.qpos, Target(np.array([0.68, 0.22, 0.9]), axis=down), seed=before, valid=check.valid).q
        self.assertTrue(check.valid(before) and check.valid(after))
        self.assertFalse(check.edge_valid(before, after))
        self.assertEqual(check.blocking((before + after) / 2)["kind"], "obstacle")

    def test_home_posture_has_clearance_and_floor_is_seen(self):
        model, data, robot = world()
        arm = robot.arms["left"]
        check = CollisionWorld(model, robot, arm.qpos)
        check.sync(data)
        self.assertTrue(check.valid(data.qpos[arm.qpos]))
        reach_floor = data.qpos[arm.qpos].copy()
        reach_floor[1], reach_floor[3] = 1.6, -0.2          # shoulder pitched down, elbow straight
        obstacle, _, pair = check.clearance(reach_floor)
        self.assertIn("floor", [model.geom(g).name for g in pair])


if __name__ == "__main__":
    unittest.main()
