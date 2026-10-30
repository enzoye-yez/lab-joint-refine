import tempfile
import unittest
from pathlib import Path

import mujoco
from asset_generation.checks import load_model
from asset_generation.contracts import read_json
from scene_generation.assembly import assemble
from scene_generation.checks import scene_checks, reach_target
from scene_generation.contracts import SceneConfig,numerical_contract
from scene_fixtures import environment,write_component,plan


class SceneAssemblyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = environment(self.root/"robot")
        self.artifacts = {"instrument":write_component(self.root/"instrument"),"tool":write_component(self.root/"tool",True)}
        self.plan = plan()
        self.contract = numerical_contract({"targets":[{"id":"press","site":"station__operation","tool_site":"gripper__tcp",
            "offset":[0,0,0],"quaternion":[1,0,0,0],"position_tolerance_m":.002,"angle_tolerance_rad":.05,"allowed_contacts":[]}]})

    def build(self):
        return assemble(self.plan,self.env,self.artifacts,self.root/"robot",self.root/"scene")

    def test_assembly_namespaces_and_true_mujoco_checks(self):
        output = self.build()
        model = load_model(output)
        self.assertGreaterEqual(model.site("gripper__tcp").id,0)
        self.assertGreaterEqual(model.body("station__base").id,0)
        report = scene_checks(output,SceneConfig(),self.contract)
        self.assertEqual(report["status"],"pass",report)
        reached = next(c for c in report["checks"] if c["id"] == "reach:press")
        self.assertEqual(reached["status"],"pass")

    def test_floating_asset_detected(self):
        self.plan["instances"][0]["position"][2] = 1
        report = scene_checks(self.build(),SceneConfig(),self.contract)
        self.assertEqual(next(c for c in report["checks"] if c["id"] == "support:station")["status"],"fail")

    def test_static_inter_instance_overlap_detected(self):
        self.plan["instances"].append({"id":"other","job":"instrument","placement":"table","position":[.51,0,.8],"yaw":0})
        report = scene_checks(self.build(),SceneConfig(),self.contract)
        hit = next(c for c in report["checks"] if c["id"] == "scene_collisions")
        self.assertEqual(hit["status"],"fail")
        self.assertIn("other",hit["instances"])

    def test_unreachable_and_payload_report_failure(self):
        self.plan["instances"][0]["position"][0] = 3
        self.contract["payload_mass_kg"] = 3
        report = scene_checks(self.build(),SceneConfig(ik_iterations=20,ik_restarts=1),self.contract)
        self.assertEqual(next(c for c in report["checks"] if c["id"] == "payload")["status"],"fail")
        self.assertEqual(next(c for c in report["checks"] if c["id"] == "reach:press")["status"],"fail")

    def test_scene_renders_real_geometry(self):
        output = self.build()
        report = scene_checks(output,SceneConfig(),self.contract,self.root/"renders")
        self.assertGreaterEqual(len(report["renders"]),3)
        self.assertTrue(all((self.root/"renders"/r).stat().st_size > 1000 for r in report["renders"]))

    def test_runtime_rejects_changed_handoff_and_sets_home(self):
        from scene_generation.runtime import load_scene
        from asset_prep.errors import ValidationError
        output = self.build()
        model,data,runtime,meta = load_scene(output)
        self.assertEqual(len(runtime.actor_geoms),1)
        self.assertAlmostEqual(float(data.qpos[model.joint("robot__x").qposadr[0]]),0)
        with self.assertRaises(ValidationError):
            load_scene(output,"wrong-version")

    def test_measurement_marker_must_touch_real_collision_surface(self):
        from scene_generation.geometry import site_surface_distance
        path = self.artifacts["instrument"]/"model.xml"
        text = path.read_text(encoding="utf-8")
        text = text.replace('<site name="operation"', '<site name="surface" pos=".05 0 .05"/><site name="operation"')
        path.write_text(text,encoding="utf-8")
        model = load_model(self.build())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        self.assertLess(site_surface_distance(model,data,"station__surface"),1e-6)
        self.assertGreater(site_surface_distance(model,data,"station__operation"),.2)

    def test_explicit_support_instance_keeps_object_movable_and_checks_contact(self):
        self.plan["instances"].append({"id":"upper","job":"instrument","placement":"table",
            "position":[.5,0,.9],"yaw":0,"support_instance":"station"})
        report = scene_checks(self.build(),SceneConfig(),self.contract)
        row = next(c for c in report["checks"] if c["id"] == "support:upper")
        self.assertEqual(row["status"],"pass",row)
