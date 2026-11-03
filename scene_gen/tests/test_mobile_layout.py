"""Developer synthetic scenes only; never formal model experiment evidence."""
import copy
import math
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from asset_generation.checks import load_model
from asset_prep.errors import ValidationError
from asset_generation.contracts import read_json
from scene_fixtures import environment, plan, write_component
from scene_generation.assembly import assemble
from scene_generation.contracts import validate_plan, repair_action, SceneConfig, numerical_contract
from scene_generation.layout import validate_layout, revision, table_geometry
from scene_generation.checks import scene_checks
from scene_generation.geometry import bounds
from asset_generation.contracts import GenerationConfig
from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded
from scene_generation.inputs import catalog
from scene_generation.pipeline import SceneRun
from generation_fixtures import FakeTransport, PROVENANCE
from scene_fixtures import SceneSandbox


def mobile_fixture(root):
    env=environment(root/"robot")
    env["robot"]["mobile"]={"base_size":[.8,.58,.294],"mount_height":.85,"mass_kg":80,
                           "source":"developer_synthetic","approximations":["开发者合成移动底盘"]}
    p=plan()
    p["instances"][0]["table_id"]="bench"
    p["layout"]={"room":{"size":[8,8],"reason":"开发者合成房间"},
        "tables":[{"id":"bench","size":[2,1.6,.06],"height":.8,"position":[0,0],"yaw":0,"reason":"开发者样例"}],
        "robot_home":[0,-2,0],"docks":[{"id":"dock","instance":"station","pose":[.5,-1.5,0],"reason":"开发者样例"}]}
    return env,p


class MobileLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.env,self.plan=mobile_fixture(self.root)
        self.limits=SceneConfig(validation_mode="static")

    def build(self,p=None):
        return assemble(p or self.plan,self.env,
            {"instrument":write_component(self.root/"instrument"),"tool":write_component(self.root/"tool",True)},
            self.root/"robot",self.root/"scene")

    def test_explicit_room_revision_requires_reason_and_all_instances(self):
        changed=copy.deepcopy(self.plan["layout"])
        changed["room"]["size"]=[7,7]
        action={"jobs":[],"placements":[],"layout":changed,"reason":"developer fixture",
                "room_revision_reason":"reuse measured workspace"}
        result=repair_action(action,self.plan,[],["station"],self.limits,self.env)
        self.assertEqual(result["layout"]["room"]["size"],[7,7])
        with self.assertRaises(ValidationError):repair_action(action,self.plan,[],[],self.limits,self.env)
        action.pop("room_revision_reason")
        with self.assertRaises(ValidationError):repair_action(action,self.plan,[],["station"],self.limits,self.env)

    def test_operation_targets_ignore_mount_sites_and_do_not_claim_verification(self):
        from scene_generation.mobile import assess
        from asset_generation.static_state import geometry_state
        root=self.build();model=load_model(root,scope="scene");data=mujoco.MjData(model)
        data.qpos[:]=read_json(root/"scene.json")["home_qpos"];geometry_state(model,data)
        geometry={g:bounds(model,data,g) for g in range(model.ngeom) if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE}
        manifest=read_json(root/"interfaces.json")
        manifest["interfaces"].append({"id":"station__internal","site":"station__operation","kind":"mount"})
        report=assess(model,data,self.plan,self.env,geometry,manifest)
        row=report["instances"][0]
        self.assertEqual([t["target"] for t in row["targets"]],["station__operation"])
        self.assertFalse(row["operation_verified"])
        fallback=assess(model,data,self.plan,self.env,geometry,{"interfaces":[]})
        self.assertEqual(fallback["instances"][0]["target_source"],"geometry_center_approximation")

    def test_deferred_author_preserves_latest_source_and_returns_without_inner_retry(self):
        run,transport=self.run_fixture([self.plan,{"model_py":"def build(output_dir):\n pass"},
            {"deferred":{"reason":"requires missing interface", "missing":["capture site"]}}])
        run.initialize();job=run.state["plan"]["jobs"][0]
        self.assertTrue(run.generate_job(job,draft=True))
        source=run.state["jobs"][job["id"]]["source"]
        self.assertFalse(run.generate_job(job))
        self.assertEqual(run.state["jobs"][job["id"]]["source"],source)
        self.assertEqual(len(transport.requests),3)
        self.assertTrue((run.root/source).is_file())

    def test_mobile_assembly_real_joints_home_and_artifacts(self):
        root=self.build()
        model=load_model(root,scope="scene")
        meta=read_json(root/"scene.json")
        data=mujoco.MjData(model);data.qpos[:]=meta["home_qpos"];mujoco.mj_forward(model,data)
        self.assertAlmostEqual(data.qpos[model.joint("robot__base_y").qposadr[0]],-2)
        old=data.site_xpos[model.site("robot__attachment_site").id].copy()
        data.qpos[model.joint("robot__base_x").qposadr[0]]+=1
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.site_xpos[model.site("robot__attachment_site").id]-old,[1,0,0],atol=1e-8)
        self.assertEqual(meta["plan"],self.plan)

    def test_table_rotation_carries_support_tree_without_scaling(self):
        child=copy.deepcopy(self.plan["instances"][0]);child.update(id="child",position=[.6,0,.9],support_instance="station")
        self.plan["instances"].append(child)
        before=copy.deepcopy(self.plan)
        changed=copy.deepcopy(self.plan["layout"])
        changed["tables"][0].update(position=[1,2],yaw=math.pi/2,height=1.0)
        result=repair_action({"jobs":[],"placements":[],"layout":changed,"reason":"旋转工作台"},
            self.plan,[],["station","child"],self.limits,self.env)
        np.testing.assert_allclose(result["instances"][0]["position"],[1,2.5,1])
        np.testing.assert_allclose(result["instances"][2]["position"],[1,2.6,1.1])
        self.assertEqual(self.plan,before)
        self.assertNotEqual(revision(result),revision(before))

    def test_permissions_room_and_missing_table_checked_atomically(self):
        changed=copy.deepcopy(self.plan["layout"])
        changed["tables"][0]["height"]=1
        with self.assertRaises(ValidationError):
            repair_action({"jobs":[],"placements":[],"layout":changed,"reason":"x"},self.plan,[],[],self.limits,self.env)
        changed["room"]["size"]=[10,10]
        with self.assertRaises(ValidationError):
            repair_action({"jobs":[],"placements":[],"layout":changed,"reason":"x"},self.plan,[],["station"],self.limits,self.env)
        self.plan["instances"][0]["table_id"]="missing"
        with self.assertRaises(ValidationError):
            validate_plan(self.plan,["A01"],self.limits)

    def test_support_is_specific_to_assigned_table(self):
        self.plan["layout"]["tables"].append({"id":"other","size":[2,1.6,.06],"height":.8,
                                               "position":[3,0],"yaw":0,"reason":"独立工作台"})
        self.plan["instances"][0]["table_id"]="other"
        report=scene_checks(self.build(),self.limits,numerical_contract({}))
        support=next(r for r in report["checks"] if r["id"]=="support:station")
        self.assertEqual(support["status"],"fail")
        self.assertIn("layout_assessment",report)

    def test_partial_scene_without_tool_keeps_quality_unknown(self):
        self.plan["jobs"]=self.plan["jobs"][:1]
        self.plan["instances"]=self.plan["instances"][:1]
        report=scene_checks(self.build(),self.limits,numerical_contract({}))
        self.assertEqual(next(r for r in report["checks"] if r["id"]=="payload")["status"],"partial")

    def test_rotated_geometry_uses_original_shape_not_world_box(self):
        self.plan["instances"][0]["yaw"]=math.pi/4
        self.plan["layout"]["tables"][0]["yaw"]=math.pi/4
        root=self.build();model=load_model(root,scope="scene");data=mujoco.MjData(model)
        data.qpos[:]=read_json(root/"scene.json")["home_qpos"];mujoco.mj_forward(model,data)
        g=model.geom("station__base_geom").id
        local=table_geometry(model,data,{g:bounds(model,data,g)},self.plan["layout"]["tables"][0])
        # MuJoCo XML serialization rounds quaternion components.
        np.testing.assert_allclose(local[g][1]-local[g][0],[.1,.1,.1],atol=2e-7)

    def test_empty_environment_is_loadable_and_advice_not_operation_pass(self):
        self.plan["instances"]=[];self.plan["jobs"]=[];self.plan["layout"]["docks"]=[]
        report=scene_checks(self.build(),self.limits,numerical_contract({}))
        self.assertEqual(report["layout_assessment"]["instances"],[])
        self.assertIn("cannot certify",report["layout_assessment"]["qualification"])

    def test_table_cut_uses_its_local_frame_and_no_central_robot(self):
        self.plan["table_modifications"]=[{"id":"pocket","instance":"station","center":[0,0],
                                          "size":[.03,.03],"depth":.02,"reason":"开发者切槽"}]
        self.build()

    def run_fixture(self, replies, calls=30, resume=False):
        source=self.root/"asset_package.md"
        source.write_text("# 合成资料\n### A01｜合成仪器\n宽100毫米。\n",encoding="utf-8")
        generation=GenerationConfig()
        project=Path(__file__).resolve().parents[1]
        transport=FakeTransport(replies)
        run=SceneRun(catalog(source,generation),self.env,numerical_contract({}),
            Config(),generation,
            SceneConfig(validation_mode="static",max_model_calls=calls),
            {"HARNESS_MODEL_API_KEY":"developer-fixture"},self.root/"run",transport,SceneSandbox(),
            self.root/"robot",project,resume=resume,provenance=PROVENANCE)
        return run,transport

    def test_mobile_pipeline_refines_once_and_delivers_document(self):
        review={"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":["开发者样例"]}
        run,transport=self.run_fixture([self.plan,{"model_py":"def build(output_dir):\n pass"},
            {"keep_layout":"已核对合成几何"},{"model_py":"def build(output_dir):\n # TOOL_FIXTURE\n pass"},
            review,review])
        result=run.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        # Instrument and scene reviews; the tool's review is deferred to the operation stage.
        self.assertEqual(result["budget"]["model_calls"],6)
        import json
        tool_request=next(p for p in transport.requests
            if json.loads(p["messages"][1]["content"][0]["text"]).get("job",{}).get("role")=="tool"
            and "current_model_py" in p["messages"][1]["content"][0]["text"])
        tool_context=json.loads(tool_request["messages"][1]["content"][0]["text"])
        self.assertEqual(tool_context["table_edit_policy"]["robot_protected_half_size_m"],0)
        self.assertNotIn("The tabletop as a whole and the robot-arm mounting zone are fixed",tool_request["messages"][0]["content"])
        self.assertIn("local coordinates of the worktable",tool_request["messages"][0]["content"])
        self.assertTrue((self.root/"run/scene_document.md").is_file())
        self.assertTrue(run.state["layout_refinement"]["complete"])
        self.assertGreaterEqual(sum(s["kind"]=="scene_candidate" for s in run.state["stages"]),3)
        resumed,transport=self.run_fixture([],resume=True)
        resumed.run()
        self.assertEqual(len(transport.requests),0)

    def test_budget_interruption_keeps_partial_scene_and_document(self):
        run,_=self.run_fixture([self.plan,{"model_py":"def build(output_dir):\n pass"}],calls=2)
        with self.assertRaises(BudgetExceeded):
            run.run()
        result=read_json(self.root/"run/result.json")
        self.assertTrue(result["assembly"]["available"])
        self.assertFalse(result["assembly"]["complete"])
        self.assertEqual(result["budget"]["model_calls"],2)
        self.assertIn("tool",result["assembly"]["missing_jobs"])
        self.assertIn("not assembled",(self.root/"run/scene_document.md").read_text(encoding="utf-8"))

    def test_asset_author_can_receive_more_than_one_image_page(self):
        run,transport=self.run_fixture([self.plan,{"context_notes":[]},
            {"model_py":"def build(output_dir):\n pass"}])
        run.initialize()
        # Synthetic decoded input injection; no external provider and no image files.
        run.catalogue["bundles"]["A01"]["images"]=[
            {"path":str(i),"sha256":str(i),"data_url":"data:fixture"+str(i)} for i in range(25)]
        self.assertTrue(run.generate_job(run.state["plan"]["jobs"][0],draft=True))
        self.assertEqual(len(transport.requests),3)
        pages=[p["messages"][1]["content"] for p in transport.requests[1:]]
        self.assertEqual([sum(v["type"]=="image_url" for v in page) for page in pages],[24,1])

    def test_room_is_arranged_without_a_model_call_before_the_tool(self):
        review={"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[]}
        run,_=self.run_fixture([self.plan,{"model_py":"def build(output_dir):\n pass"},
            {"model_py":"def build(output_dir):\n # TOOL_FIXTURE\n pass"},review,review,review])
        result=run.run()
        self.assertTrue(result["assembly"]["complete"])
        self.assertEqual(run.state["layout_refinement"]["method"],"deterministic_rows")
        self.assertTrue(run.state["plan"]["layout"]["docks"])

    def test_plan_failure_still_delivers_description(self):
        run,_=self.run_fixture([{}]*5)
        with self.assertRaises(ValidationError):
            run.run()
        self.assertIn("No loadable model yet",(self.root/"run/scene_document.md").read_text(encoding="utf-8"))

    def test_cross_table_reassignment_carries_child(self):
        child=copy.deepcopy(self.plan["instances"][0])
        child.update(id="child",position=[.5,0,.9],support_instance="station")
        self.plan["instances"].append(child)
        new=copy.deepcopy(self.plan["layout"])
        new["tables"].append({"id":"other","size":[2,1,.08],"height":1,
                              "position":[3,0],"yaw":0,"reason":"分开操作区域"})
        updated=repair_action({"jobs":[],"layout":new,"placements":[
            {"id":"station","position":[3,0,1],"yaw":0,"table_id":"other"}],"reason":"迁移支撑链"},
            self.plan,[],["station","child"],self.limits,self.env)
        self.assertEqual(updated["instances"][2]["table_id"],"other")
        np.testing.assert_allclose(updated["instances"][2]["position"],[3,0,1.1])
        bad={"jobs":[],"placements":[{"id":"station","position":[0,0,.8],"yaw":0,"scale":.5}],"reason":"非法缩放"}
        with self.assertRaises(ValidationError):
            repair_action(bad,self.plan,[],["station"],self.limits,self.env)

    def test_bad_layout_identifiers_and_support_cycles(self):
        self.plan["layout"]["tables"][0]["id"]="../outside"
        with self.assertRaises(ValidationError):
            validate_layout(self.plan,self.env)
        self.plan["layout"]["tables"][0]["id"]="bench"
        with self.assertRaises(ValidationError):
            repair_action({"jobs":[],"placements":[{"id":"station","position":[.4,0,.8],
                "yaw":0,"support_instance":"station"}],"reason":"非法支撑"},
                self.plan,[],["station"],self.limits,self.env)

    def test_removed_table_can_reassign_instrument_to_floor(self):
        layout=copy.deepcopy(self.plan["layout"])
        layout["tables"][0]["id"]="replacement"
        updated=repair_action({"jobs":[],"layout":layout,"placements":[
            {"id":"station","position":[3,0,0],"yaw":0,"placement":"floor","table_id":None}],
            "reason":"保留落地仪器并移除原工作台"},self.plan,[],["station"],self.limits,self.env)
        self.assertEqual(updated["instances"][0]["placement"],"floor")
        self.assertNotIn("table_id",updated["instances"][0])

    def test_overhanging_table_asset_blocks_base_home(self):
        from scene_generation.mobile import assess
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="station__body">'
            '<geom name="station__box" type="box" pos="0 -2 1" size=".3 .3 .3"/>'
            '</body></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        advice=assess(model,data,self.plan,self.env,{0:bounds(model,data,0)})
        self.assertFalse(advice["home_clear"])

    def test_blocked_corridor_is_advice_not_operation_pass(self):
        self.plan["layout"]["tables"][0]["size"]=[.5,8,.06]
        self.plan["layout"]["robot_home"]=[-2,0,0]
        self.plan["instances"][0]["position"]=[2,0,0]
        self.plan["instances"][0]["placement"]="floor"
        self.plan["instances"][0].pop("table_id")
        report=scene_checks(self.build(),self.limits,numerical_contract({}))
        advice=report["layout_assessment"]
        self.assertTrue(advice["home_clear"])
        self.assertEqual(advice["instances"][0]["status"],"needs_layout_review")
        self.assertFalse(any(r["id"].startswith("reach:") for r in report["checks"]))

    def test_joint_probe_observes_motion_without_dynamics(self):
        from scene_generation.mobile import articulation
        model=mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="station__lid">'
            '<joint name="station__hinge" axis="0 0 1" range="0 90"/>'
            '<geom type="box" pos=".1 0 0" size=".1 .01 .01"/></body></worldbody></mujoco>')
        rows=articulation(model,mujoco.MjData(model))
        self.assertTrue(rows[0]["kinematic_motion"])
        self.assertIn("not certified",rows[0]["qualification"])


if __name__=="__main__":
    unittest.main()
