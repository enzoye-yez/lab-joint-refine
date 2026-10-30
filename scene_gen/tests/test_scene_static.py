import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from asset_generation.checks import validate_artifact
from asset_generation.contracts import GenerationConfig, check_profile
from asset_generation.pipeline import snapshot
from asset_prep.config import Config
from asset_prep.errors import StaleEvidence, ValidationError
from asset_prep.storage import atomic_json, digest
from generation_fixtures import FakeTransport, PROVENANCE, write_artifact
from scene_fixtures import environment, plan, write_component, SceneSandbox
from scene_generation.assembly import assemble
from scene_generation.checks import scene_checks
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.inputs import catalog
from scene_generation.pipeline import SceneRun
from scene_generation.preview import export_preview, resolve_artifact, show_static


class StaticSceneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = environment(self.root/"robot")
        self.plan = plan()

    def scene(self):
        return assemble(self.plan,self.env,{"instrument":write_component(self.root/"instrument"),
            "tool":write_component(self.root/"tool",True)},self.root/"robot",self.root/"scene")

    def test_static_scene_mode_still_runs_asset_motion_and_checks_dimensions(self):
        # Scene static mode never skips the asset's own motion, settling or probe checks.
        write_artifact(self.root/"asset")
        profile = check_profile({"validation_mode":"static", "dimensions":[{"id":"width", "geoms":["base_geom"],
            "axis":0, "expected_m":.2, "tolerance_m":.001}]})
        report = validate_artifact(self.root/"asset",["A01"],profile,self.root/"asset_views")
        self.assertEqual(report["status"],"fail")
        self.assertEqual(report["validation_mode"],"static")
        rows = {row["id"]:row for row in report["checks"]}
        self.assertEqual(rows["width"]["status"],"fail")
        for key in ("sampled_joint_sweep:button","mechanism_drive:button","numerical_stability","interaction_probes"):
            self.assertEqual(rows[key]["status"],"pass",rows[key])
        # Initial three views plus both ends of the declared button mechanism.
        self.assertEqual(len(report["renders"]),5)

    def test_static_scene_never_steps_or_solves_robot_targets(self):
        artifact = self.scene()
        with patch("mujoco.mj_step",side_effect=AssertionError("No stepping")), \
             patch("scene_generation.checks.reach_target",side_effect=AssertionError("No IK")):
            report = scene_checks(artifact,SceneConfig(validation_mode="static"),numerical_contract({}),self.root/"views")
        self.assertEqual(report["status"],"pass",report)
        self.assertEqual(len(report["renders"]),5)
        self.assertEqual(len([name for name in report["renders"] if name.startswith("detail_")]),2)
        self.assertEqual(next(r for r in report["checks"] if r["id"] == "robot_reachability")["status"],"skipped")
        self.assertFalse(any(r["id"].startswith("reach:") for r in report["checks"]))

    def test_static_scene_still_rejects_floating_and_overlap(self):
        self.plan["instances"][0]["position"][2] = 1
        self.plan["instances"].append({"id":"other","job":"instrument","placement":"table","position":[.51,0,1],"yaw":0})
        report = scene_checks(self.scene(),SceneConfig(validation_mode="static"),numerical_contract({}))
        rows = {r["id"]:r for r in report["checks"]}
        self.assertEqual(rows["support:station"]["status"],"fail")
        self.assertEqual(rows["scene_collisions"]["status"],"fail")

    def test_missing_tool_metadata_reports_validation_error(self):
        artifact=self.scene()
        path=artifact/"scene.json"
        meta=json.loads(path.read_text(encoding="utf-8"))
        meta["plan"]["instances"]=[i for i in meta["plan"]["instances"] if i["placement"] != "robot"]
        atomic_json(path,meta)
        with self.assertRaisesRegex(ValidationError,"恰有一个"):
            scene_checks(artifact,SceneConfig(validation_mode="static"),numerical_contract({}))

    def test_gallery_uses_home_pose_without_step_or_artifact_mutation(self):
        artifact = self.scene()
        before = snapshot(artifact)
        with patch("mujoco.mj_step",side_effect=AssertionError("No stepping")):
            page = export_preview(artifact,self.root/"preview",{"status":"static_candidate_ready", "provenance":PROVENANCE})
        self.assertTrue(page.is_file())
        self.assertEqual(len(list(page.parent.glob("*.png"))),3)
        from PIL import Image
        with Image.open(page.parent/"scene_45.png") as picture:
            self.assertEqual(picture.size,(1280,960))
        self.assertEqual(before,snapshot(artifact))

    def test_preview_rejects_tampered_scene(self):
        artifact = self.scene()
        run = self.root/"runs/example"
        run.mkdir(parents=True)
        # Put a genuine assembled artifact under a result directory.
        import shutil
        shutil.copytree(artifact,run/"artifact")
        atomic_json(run/"result.json",{"scene_artifact":"artifact","scene_sha256":digest(snapshot(run/"artifact"))})
        self.assertEqual(resolve_artifact(self.root,"example")[1],run/"artifact")
        (run/"artifact/model.xml").write_text("changed",encoding="utf-8")
        with self.assertRaises(ValidationError):
            resolve_artifact(self.root,"example")

    def test_interactive_view_keeps_pose_and_time_frozen(self):
        artifact = self.scene()
        import mujoco.viewer
        viewer = MagicMock()
        viewer.is_running.side_effect = [True,True,False]
        def launch(model,data):
            home = data.qpos.copy()
            def sync():
                self.assertEqual(data.time,0)
                self.assertTrue((data.qpos == home).all())
                data.time = 1
                data.qpos[:] += .01
            viewer.sync.side_effect = sync
            context = MagicMock()
            context.__enter__.return_value = viewer
            return context
        with patch.object(mujoco.viewer,"launch_passive",side_effect=launch), \
             patch("mujoco.mj_step",side_effect=AssertionError("No stepping")), \
             patch("scene_generation.preview.time.sleep"):
            show_static(artifact)
        self.assertEqual(viewer.sync.call_count,2)

    def test_static_pipeline_scope_persisted_and_full_resume_rejected(self):
        source = self.root/"package.md"
        source.write_text("### A01｜合成\n测试尺寸。",encoding="utf-8")
        project = Path(__file__).resolve().parents[1]
        provider = Config()
        generation = GenerationConfig()
        inventory = catalog(source,generation)
        review = {"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[]}
        transport = FakeTransport([self.plan,{"model_py":"def build(output_dir):\n    pass\n"},
            {"model_py":"def build(output_dir):\n    # TOOL_FIXTURE\n    pass\n"},review,review,review])
        sandbox = SceneSandbox()
        profiles = []
        build = sandbox.build_and_check
        def capture(script,destination,asset_ids,profile):
            profiles.append(profile)
            return build(script,destination,asset_ids,profile)
        sandbox.build_and_check = capture
        args = [inventory,self.env,numerical_contract({}),provider,generation,SceneConfig(validation_mode="static"),
            {"HARNESS_MODEL_API_KEY":"fixture"},self.root/"run",transport,sandbox,self.root/"robot",project]
        run = SceneRun(*args,provenance=PROVENANCE)
        result = run.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertTrue(all(p["validation_mode"] == "static" for p in profiles))
        handoff = json.loads((self.root/"run/planning_handoff.json").read_text(encoding="utf-8"))
        self.assertEqual(handoff["validation_mode"],"static")
        with self.assertRaises(ValidationError):
            run.run({"failures":[]})
        args[5] = SceneConfig()
        with self.assertRaises(StaleEvidence):
            SceneRun(*args,resume=True,provenance=PROVENANCE)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValidationError):
            SceneConfig(validation_mode="pretend_pass")

