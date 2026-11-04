"""Condition evaluation: Kleene logic, slow-slip detection, joint fraction, invariants (plan §5)."""
import unittest

import mujoco
import numpy as np

from np002_traj.evidence.predicates import Monitor, all_of, any_of

PADS = """
<mujoco>
  <option gravity="0 0 0" cone="elliptic"/>
  <worldbody>
    <site name="tcp" pos="0 0 0"/>
    <geom name="pad_a" type="box" size="0.01 0.002 0.02" pos="0 0.0215 0"/>
    <geom name="pad_b" type="box" size="0.01 0.002 0.02" pos="0 -0.0215 0"/>
    <body name="part" pos="0 0 0">
      <freejoint/>
      <geom name="part" type="box" size="0.03 0.02 0.01" mass="0.1"/>
    </body>
    <body name="door">
      <joint name="slide" type="slide" axis="0 1 0" range="0 0.1"/>
      <geom type="box" size="0.01 0.01 0.01" pos="1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


class LogicTests(unittest.TestCase):
    def test_kleene_combinators(self):
        self.assertIs(any_of([True, None]), True)
        self.assertIs(any_of([False, None]), None)
        self.assertIs(all_of([False, None]), False)
        self.assertIs(all_of([True, None]), None)
        self.assertIs(all_of([True, True]), True)


class PredicateTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(PADS)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.monitor = Monitor(self.model, self.data)
        self.rule = {"id": "grip", "kind": "grasped", "object": "part", "pads": ["pad_a", "pad_b"], "tcp": "tcp",
                     "min_force_n": 0.1, "max_drift_m": 3e-3}

    def test_slow_slip_of_15mm_is_detected_and_stays_failed(self):
        self.assertTrue(self.monitor.evaluate(self.rule))
        values = []
        for step in range(1, 31):                     # 0.5 mm per sample, 15 mm in total
            self.data.qpos[0] = 0.0005 * step
            mujoco.mj_forward(self.model, self.data)
            values.append(self.monitor.evaluate(self.rule))
        self.assertTrue(all(values[:6]))              # within 3 mm
        self.assertFalse(any(values[7:]))             # never re-anchored after the first failure
        self.assertAlmostEqual(self.monitor.details["grip"]["drift_m"], 0.015, places=4)

    def test_lost_contact_is_not_a_grasp(self):
        self.data.qpos[2] = 0.05
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(self.monitor.evaluate(self.rule))

    def test_joint_fraction_and_duration(self):
        rule = {"id": "open", "kind": "joint_fraction", "joint": "slide", "min": 0.9, "for_s": 0.2}
        self.data.qpos[7] = 0.095
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(self.monitor.evaluate(rule))  # true now, but not yet for 0.2 s
        self.data.time += 0.25
        self.assertTrue(self.monitor.evaluate(rule))
        self.assertAlmostEqual(self.monitor.details["open"]["fraction"], 0.95)

    def test_displacement_invariant(self):
        rule = {"id": "still", "kind": "displaced", "body": "part", "max_m": 2e-3}
        self.assertTrue(self.monitor.evaluate(rule))
        self.data.qpos[2] = 0.003
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(self.monitor.evaluate(rule))

    def test_unknown_name_raises_instead_of_passing(self):
        with self.assertRaises(KeyError):
            self.monitor.evaluate({"kind": "joint_range", "joint": "missing", "range": [0, 1]})


if __name__ == "__main__":
    unittest.main()
