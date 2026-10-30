import unittest

from asset_prep.errors import ValidationError
from scene_generation.contracts import SceneConfig, validate_plan, repair_action, numerical_contract
from scene_generation.feedback import planning_feedback
from scene_fixtures import plan


class SceneContractsTests(unittest.TestCase):
    def test_complete_plan_and_dependency_order(self):
        _, order = validate_plan(plan(),["A01"],SceneConfig())
        self.assertEqual(order,["instrument","tool"])

    def test_missing_asset_and_dependency_cycle_rejected(self):
        with self.assertRaises(ValidationError):
            validate_plan(plan(),["A01","A02"],SceneConfig())
        value = plan()
        value["jobs"][0]["depends_on"] = ["tool"]
        with self.assertRaises(ValidationError):
            validate_plan(value,["A01"],SceneConfig())

    def test_repair_cannot_change_robot_or_unrelated_assets(self):
        value = {"jobs":[],"placements":[{"id":"gripper","position":[1,0,0],"yaw":0}],"reason":"bad"}
        with self.assertRaises(ValidationError):
            repair_action(value,plan(),[],["gripper"],SceneConfig())
        value = {"jobs":["tool"],"placements":[],"reason":"bad"}
        with self.assertRaises(ValidationError):
            repair_action(value,plan(),["instrument"],[],SceneConfig())

    def test_planner_feedback_binds_version_and_known_instances(self):
        value = {"scene_sha256":"hash","failures":[{"kind":"collision","instances":["station"],"detail":"hit","evidence":"planner test"}]}
        self.assertEqual(planning_feedback(value,"hash",plan())["status"],"fail")
        with self.assertRaises(ValidationError):
            planning_feedback(value,"new",plan())
        value["failures"][0]["instances"] = ["unknown"]
        with self.assertRaises(ValidationError):
            planning_feedback(value,"hash",plan())
