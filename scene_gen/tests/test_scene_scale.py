"""Generalisation tests for large-scene checking and repair convergence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from asset_generation.checks import measure_model, scale_limits, load_model
from asset_prep.errors import ValidationError
from scene_generation.geometry import collision_scan


def grid_xml(count, spacing=.01, size=".004"):
    geoms = ''.join(f'<geom name="g{i}" pos="{i*spacing} 0 0" size="{size}"/>' for i in range(count))
    return f'<mujoco><compiler angle="radian"/><worldbody>{geoms}</worldbody></mujoco>'


class ScaleBudgetTests(unittest.TestCase):
    """A large scene must yield measurable, partial evidence instead of a bare refusal."""

    def test_measure_model_accepts_a_scene_past_the_asset_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"model.xml").write_text(grid_xml(10001), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_model(root)
            model, scale = measure_model(root, scope="scene")
            self.assertEqual(scale["ngeom"], 10001)
            self.assertGreater(model.ngeom, 10000)

    def test_budgets_shrink_with_scale_and_never_reach_zero(self):
        small = mujoco.MjModel.from_xml_string(grid_xml(10))
        base = scale_limits(small, "scene")
        self.assertFalse(base["degraded"])
        # Same function, a model whose size is past the base limit: budgets shrink but
        # stay positive, so a large scene is still measured rather than refused.
        class Fake:
            ngeom = 200000
        big = scale_limits(Fake(), "scene")
        self.assertTrue(big["degraded"])
        self.assertGreater(big["pair_budget"], 0)
        self.assertLessEqual(big["pair_budget"], base["pair_budget"])
        self.assertGreaterEqual(big["joint_samples"], 56)
        self.assertLessEqual(big["joint_samples"], base["joint_samples"])

    def test_measure_model_reports_body_and_dof_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"model.xml").write_text(grid_xml(5), encoding="utf-8")
            model, scale = measure_model(root, scope="scene")
            self.assertEqual((scale["nbody"], scale["nv"]), (model.nbody, model.nv))


class BroadPhaseTests(unittest.TestCase):
    """Sweep-and-prune must find exactly the pairs the former full scan found."""

    def scan(self, xml, **kwargs):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, collision_scan(model, data, .0005, max_pairs=200000, **kwargs)

    def test_close_boxes_on_opposing_sides_are_found(self):
        xml = ('<mujoco><compiler angle="radian"/><worldbody>'
               '<body name="a"><geom name="a__g" type="box" size=".05 .05 .05" pos="0 0 .05"/></body>'
               '<body name="b"><geom name="b__g" type="box" size=".05 .05 .05" pos=".09 0 .05"/></body>'
               '</worldbody></mujoco>')
        _, result = self.scan(xml)
        self.assertTrue(any({h["geom1"], h["geom2"]} == {"a__g", "b__g"} for h in result["collisions"]),
                        result)

    def test_distant_boxes_are_not_reported(self):
        xml = ('<mujoco><compiler angle="radian"/><worldbody>'
               '<body name="a"><geom name="a__g" type="box" size=".01 .01 .01" pos="0 0 .01"/></body>'
               '<body name="b"><geom name="b__g" type="box" size=".01 .01 .01" pos="5 0 .01"/></body>'
               '</worldbody></mujoco>')
        _, result = self.scan(xml)
        self.assertEqual(result["collisions"], [])

    def test_ordering_does_not_hide_a_pair_that_appears_earlier_in_x(self):
        # b is listed first but sits at larger x; the sweep must still compare them once.
        xml = ('<mujoco><compiler angle="radian"/><worldbody>'
               '<body name="far"><geom name="far__g" type="box" size=".05 .05 .05" pos=".2 0 .05"/></body>'
               '<body name="near"><geom name="near__g" type="box" size=".05 .05 .05" pos=".12 0 .05"/></body>'
               '</worldbody></mujoco>')
        _, result = self.scan(xml)
        pairs = [{h["geom1"], h["geom2"]} for h in result["collisions"]]
        self.assertIn({"near__g", "far__g"}, pairs, result)

    def test_only_owners_restricts_work_but_keeps_moving_contacts(self):
        xml = ('<mujoco><compiler angle="radian"/><worldbody>'
               '<body name="a"><geom name="a__g" type="box" size=".05 .05 .05" pos="0 0 .05"/></body>'
               '<body name="b"><geom name="b__g" type="box" size=".05 .05 .05" pos=".09 0 .05"/></body>'
               '<body name="c"><geom name="c__g" type="box" size=".05 .05 .05" pos=".18 0 .05"/></body>'
               '</worldbody></mujoco>')
        _, focused = self.scan(xml, only_owners={"a"})
        pairs = [{h["geom1"], h["geom2"]} for h in focused["collisions"]]
        self.assertIn({"a__g", "b__g"}, pairs)
        self.assertNotIn({"b__g", "c__g"}, pairs)


if __name__ == "__main__":
    unittest.main()
