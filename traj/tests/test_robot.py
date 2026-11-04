"""Stand-alone acceptance of the assembled robot on an empty floor (plan §1)."""
import unittest

import mujoco
import numpy as np

from support import controllers, run, world


class RobotTests(unittest.TestCase):
    def test_home_has_no_contact_and_tcps_are_forward(self):
        model, data, robot = world()
        self.assertEqual(data.ncon, 0)
        for arm in robot.arms.values():
            self.assertGreater(data.site_xpos[arm.tcp][0], 0.30)  # ahead of the base front face

    def test_joint_steps_settle_without_oscillation(self):
        model, data, robot = world()
        control = controllers(model, data, robot)
        arm = robot.arms["left"]
        for i in range(7):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            start = data.qpos[arm.qpos].copy()
            target = start.copy()
            target[i] += 0.1
            control["left"].set(target)
            control["right"].hold(data)
            trace = run(model, data, control, 2.0, lambda d: d.qpos[arm.qpos[i]])
            with self.subTest(joint=i):
                self.assertLess(trace.max() - target[i], 0.015)              # overshoot < 15 %
                self.assertLess(abs(trace[-1] - target[i]), 2e-3)
                self.assertLess(trace[-250:].std(), 2e-4)                  # no residual oscillation

    def test_home_is_held_under_gravity(self):
        model, data, robot = world()
        control = controllers(model, data, robot)
        start = {s: data.qpos[a.qpos].copy() for s, a in robot.arms.items()}
        run(model, data, control, 3.0)
        for side, arm in robot.arms.items():
            self.assertLess(np.abs(data.qpos[arm.qpos] - start[side]).max(), 1e-3)

    def test_base_tracks_velocity_commands(self):
        model, data, robot = world()
        control = controllers(model, data, robot)
        x, y, yaw = robot.base_actuators
        data.ctrl[x] = 0.3
        run(model, data, control, 2.0)
        self.assertAlmostEqual(data.qpos[model.jnt_qposadr[robot.base_joints[0]]], 0.6, delta=0.03)
        data.ctrl[x], data.ctrl[yaw] = 0.0, 0.5
        run(model, data, control, 2.0)
        self.assertAlmostEqual(data.qpos[model.jnt_qposadr[robot.base_joints[2]]], 1.0, delta=0.05)

    def test_lift_travels_and_holds_the_arms(self):
        model, data, robot = world()
        control = controllers(model, data, robot)
        address = model.jnt_qposadr[robot.lift_joint]
        for target in (0.6, 0.0):
            data.ctrl[robot.lift_actuator] = target
            run(model, data, control, 3.0)
            self.assertAlmostEqual(data.qpos[address], target, delta=2e-3)

    def test_workspace_covers_floor_to_shelf_heights(self):
        model, data, robot = world()
        arm = robot.arms["left"]
        low, high = model.jnt_range[arm.joints].T
        rng = np.random.default_rng(0)
        heights, bench = [], []
        lift = model.jnt_qposadr[robot.lift_joint]
        for lift_value in (0.0, 0.6):
            for _ in range(3000):
                mujoco.mj_resetDataKeyframe(model, data, 0)
                data.qpos[lift] = lift_value
                data.qpos[arm.qpos] = rng.uniform(low, high)
                mujoco.mj_forward(model, data)
                if data.ncon:
                    continue
                x, _, z = data.site_xpos[arm.tcp]
                heights.append(z)
                if 0.75 <= z <= 0.95:
                    bench.append(x)
        self.assertLessEqual(min(heights), 0.40)
        self.assertGreaterEqual(max(heights), 1.60)
        self.assertGreaterEqual(max(bench) - 0.30, 0.55)   # depth past the base front at bench height

    def test_gripper_holds_200g_without_slip(self):
        model, data, robot = world()
        tcp = data.site_xpos[robot.arms["left"].tcp].copy()
        model, data, robot = world({"pos": tcp.tolist(), "half": 0.02, "mass": 0.2})
        arm = robot.arms["left"]
        control = controllers(model, data, robot)
        data.ctrl[arm.gripper] = 0.0
        run(model, data, control, 1.0)
        data.eq_active[model.equality("box_fixture").id] = 0  # test fixture only: hand the box to the gripper
        box = model.body("box").id
        relative = lambda d: d.site_xmat[arm.tcp].reshape(3, 3).T @ (d.xpos[box] - d.site_xpos[arm.tcp])
        start = relative(data)
        trace = run(model, data, control, 2.0, relative)
        self.assertLess(np.linalg.norm(trace - start, axis=1).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
