"""Pinned manufacturer resource + developer synthetic objects; no model experiments."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from asset_generation.checks import load_model
from asset_generation.contracts import read_json
from asset_prep.errors import ValidationError
from scene_generation.assembly import assemble
from scene_generation.contracts import environment, validate_plan, repair_action, SceneConfig, numerical_contract
from scene_generation.robot import prepare_environment
from scene_generation.robot_contract import initialize, synchronize_grippers, joint_catalogue, validate_model_posture
from scene_generation.layout import revision
from scene_generation.checks import scene_checks
from scene_generation.inspection import mount as inspect_mount
from scene_generation.prompts import PLAN, for_environment
from scene_fixtures import write_component
from test_mobile_layout import mobile_fixture


class FfwTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project=Path(__file__).resolve().parents[1]
        cls.robot=cls.project/"data/robots/ffw_sg2"
        if not cls.robot.is_dir():
            raise unittest.SkipTest("Official FFW resources absent; tests never download")
        cls.env=environment(read_json(cls.project/"config/scene_environment_ffw.json"))
        cls.catalogue=joint_catalogue(cls.robot,cls.env)

    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        _,p=mobile_fixture(self.root)
        self.plan=p
        p["jobs"]=p["jobs"][:1];p["instances"]=p["instances"][:1]
        p["end_effectors"]={a:{"mode":"stock","instance":None,"reason":"开发合成测试：原厂夹爪满足需求"} for a in ("left","right")}
        self.limits=SceneConfig(validation_mode="static")

    def custom(self,side,mode):
        jid="tool_"+side
        self.plan["jobs"].append({"id":jid,"asset_ids":["A01"],"role":"tool","depends_on":["instrument"],"requirements":"开发测试专用总成"})
        self.plan["instances"].append({"id":jid,"job":jid,"placement":"robot","position":[0,0,0],"yaw":0})
        self.plan["end_effectors"][side]={"mode":mode,"instance":jid,"reason":"开发测试安装变体"}

    def build(self):
        artifacts={j["id"]:write_component(self.root/j["id"],j["role"]=="tool") for j in self.plan["jobs"]}
        return assemble(self.plan,self.env,artifacts,self.robot,self.root/"scene")

    def test_source_adapter_poses_limits_and_inertias_match(self):
        self.assertEqual(prepare_environment(self.project,self.env),self.robot)
        src=mujoco.MjModel.from_xml_path(str(self.robot/"source.xml"))
        dst=mujoco.MjModel.from_xml_path(str(self.robot/"model.xml"))
        self.assertAlmostEqual(src.body_mass.sum(),dst.body_mass.sum())
        for lift in (-.5,-.25,0):
            a,b=mujoco.MjData(src),mujoco.MjData(dst)
            initialize(dst,b,self.env,prefix="",posture={"lift_joint":lift,"gripper_l_joint1":.5,"gripper_r_joint1":.7})
            for j in range(src.njnt):
                if src.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
                other=dst.joint(src.joint(j).name)
                np.testing.assert_allclose(src.jnt_range[j],other.range)
                a.qpos[src.jnt_qposadr[j]]=b.qpos[other.qposadr[0]]
            mujoco.mj_forward(src,a);mujoco.mj_forward(dst,b)
            for body in range(1,src.nbody):
                other=dst.body(src.body(body).name).id
                np.testing.assert_allclose(a.xpos[body],b.xpos[other],atol=1e-9)
                np.testing.assert_allclose(a.xmat[body],b.xmat[other],atol=1e-9)

    def test_joint_motion_lift_and_gripper_followers_velocity_zero(self):
        m=load_model(self.robot); d=mujoco.MjData(m)
        initialize(m,d,self.env,prefix="",posture={"left_wheel_drive_joint":1.2,"gripper_l_joint1":.6})
        self.assertAlmostEqual(d.qpos[m.joint("gripper_l_joint2").qposadr[0]],-.6)
        for a in range(m.nu):
            if m.actuator_biasprm[a,1]==0:self.assertEqual(d.ctrl[a],0)
        mujoco.mj_forward(m,d)
        before=d.site_xpos.copy()
        d.qpos[m.joint("lift_joint").qposadr[0]]+=.1
        mujoco.mj_forward(m,d)
        for side in ("left","right"):
            sid=m.site(side+"_wrist").id
            np.testing.assert_allclose(d.site_xpos[sid]-before[sid],[0,0,.1],atol=1e-8)
        before=d.site_xpos.copy()
        d.qpos[m.joint("arm_l_joint1").qposadr[0]]+=.3
        mujoco.mj_forward(m,d)
        self.assertGreater(np.linalg.norm(d.site_xpos[m.site("left_wrist").id]-before[m.site("left_wrist").id]),.01)
        np.testing.assert_allclose(d.site_xpos[m.site("right_wrist").id],before[m.site("right_wrist").id])

    def test_four_mount_configurations_and_other_hand_preserved(self):
        for modes in ((),(("left","adapter"),),(("left","replace"),),(("right","adapter"),),(("right","replace"),),(("left","replace"),("right","replace"))):
            with self.subTest(modes=modes):
                old=copy.deepcopy(self.plan)
                for side,mode in modes:self.custom(side,mode)
                validate_plan(self.plan,["A01"],self.limits,self.env)
                output=self.build()
                m=load_model(output,scope="scene");meta=read_json(output/"scene.json")
                d=mujoco.MjData(m);d.qpos[:]=meta["home_qpos"];mujoco.mj_forward(m,d)
                from scene_generation.runtime import actor_groups
                groups=actor_groups(m,meta)
                self.assertTrue(groups["left"] and groups["right"])
                self.assertFalse(set(groups["left"]) & set(groups["right"]))
                removed={side for side,mode in modes if mode=="replace"}
                self.assertEqual(m.neq,6-3*len(removed))
                for side,s in (("left","l"),("right","r")):
                    present=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot__gripper_{s}_joint1")>=0
                    self.assertEqual(present,side not in removed)
                for mount in meta["mounts"]:
                    site=m.site(mount["attachment_site"]).id
                    tool=m.body(mount["instance"]+"__base").id
                    np.testing.assert_allclose(d.xpos[tool],d.site_xpos[site],atol=1e-8)
                    self.assertTrue(mount["attachment_site"].startswith("robot__"+mount["arm"]))
                self.plan=old

    def test_assembled_wheels_touch_floor_without_changing_source(self):
        from scene_generation.geometry import bounds
        output=self.build();m=load_model(output,scope="scene")
        meta=read_json(output/"scene.json")
        d=mujoco.MjData(m);d.qpos[:]=meta["home_qpos"];mujoco.mj_forward(m,d)
        wheels=[i for i in range(m.ngeom) if "wheel" in m.body(m.geom_bodyid[i]).name and m.geom_contype[i]]
        self.assertTrue(wheels)
        bottom=min(bounds(m,d,i)[0][2] for i in wheels)
        self.assertAlmostEqual(bottom,0,places=5)
        self.assertAlmostEqual(meta["robot_ground_offset_m"],-.1465)

    def test_duplicate_unbound_wrong_arm_and_offset_are_rejected(self):
        self.custom("left","adapter")
        for change in ("duplicate","unknown_arm","unbound","offset"):
            p=copy.deepcopy(self.plan)
            if change=="duplicate":p["end_effectors"]["right"]=copy.deepcopy(p["end_effectors"]["left"])
            if change=="unknown_arm":p["end_effectors"]["wrong"]=p["end_effectors"].pop("left")
            if change=="unbound":p["end_effectors"]["left"]={"mode":"stock","instance":None,"reason":"x"}
            if change=="offset":p["instances"][-1]["position"]=[0,0,.4]
            with self.subTest(change=change),self.assertRaises(ValidationError):
                validate_plan(p,["A01"],self.limits,self.env)

    def test_missing_replacement_publishes_without_silent_stock_restore(self):
        self.custom("left","replace")
        self.plan["jobs"]=self.plan["jobs"][:1]
        self.plan["instances"]=self.plan["instances"][:1]
        output=self.build();m=load_model(output,scope="scene")
        self.assertEqual(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot__gripper_l_joint1"),-1)
        self.assertGreaterEqual(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"robot__gripper_r_joint1"),0)
        report=scene_checks(output,self.limits,numerical_contract({}))
        rows={r["id"]:r for r in report["checks"]}
        self.assertEqual(rows["payload:left"]["status"],"partial")
        self.assertEqual(rows["payload:right"]["status"],"pass")

    def test_per_arm_payload_and_missing_arm_assignment(self):
        output=self.build()
        report=scene_checks(output,self.limits,numerical_contract({"arm_payloads":{"left":2.6,"right":.1}}))
        rows={r["id"]:r for r in report["checks"]}
        self.assertEqual(rows["payload:left"]["status"],"fail")
        self.assertEqual(rows["payload:right"]["status"],"pass")
        report=scene_checks(output,self.limits,numerical_contract({"payload_mass_kg":1}))
        self.assertTrue(all(r["status"]=="partial" for r in report["checks"] if r["id"].startswith("payload:")))

    def test_whole_robot_envelope_and_per_arm_lift_advice(self):
        from scene_generation.mobile import assess
        from scene_generation.geometry import bounds
        output=self.build();m=load_model(output,scope="scene")
        d=mujoco.MjData(m);d.qpos[:]=read_json(output/"scene.json")["home_qpos"];mujoco.mj_forward(m,d)
        g={i:bounds(m,d,i) for i in range(m.ngeom) if m.geom_type[i]!=mujoco.mjtGeom.mjGEOM_PLANE}
        result=assess(m,d,self.plan,self.env,g)
        self.assertGreater(result["base_radius_m"],np.linalg.norm(self.env["robot"]["mobile"]["base_size"][:2])/2)
        distances=result["instances"][0]["declared_docks"][0]["arm_distances"]
        self.assertEqual({r["arm"] for r in distances},{"left","right"})
        self.assertEqual({tuple(r["robot_posture"].values()) for r in distances},{(),(-.5,),(0.,)})
        for r in result["instances"][0]["targets"]:
            if r["suggested_dock"]:self.assertIn(r["arm"],("left","right"))

    def test_tool_addition_contract_and_revision_invalidation(self):
        self.custom("left","replace")
        newjob=self.plan["jobs"].pop();newitem=self.plan["instances"].pop()
        selected=copy.deepcopy(self.plan["end_effectors"])
        self.plan["end_effectors"]["left"]={"mode":"stock","instance":None,"reason":"初始原厂"}
        action={"jobs":[newjob["id"]],"placements":[],"reason":"审查发现需要定制",
                "tool_additions":[{"job":newjob,"instance":newitem}],"end_effectors":selected}
        updated=repair_action(action,self.plan,[],["station"],self.limits,self.env)
        self.assertNotEqual(revision(updated),revision(self.plan))
        self.assertEqual(validate_plan(updated,["A01"],self.limits,self.env)[1],["instrument","tool_left"])
        for malformed in ([],None,"bad"):
            with self.assertRaises(ValidationError):
                repair_action({**action,"end_effectors":malformed},self.plan,[],[],self.limits,self.env)
        moved=copy.deepcopy(self.plan);moved["robot_posture"]={"lift_joint":-.1}
        self.assertNotEqual(revision(moved),revision(self.plan))

    def test_posture_limits_and_passive_joint_feedback(self):
        for values in ({"lift_joint":2},{"gripper_l_joint2":-.4},{"unknown":1}):
            with self.assertRaises(ValidationError):
                validate_model_posture({"robot_posture":values},self.catalogue)
        validate_model_posture({"robot_posture":{"lift_joint":-.2,"arm_l_joint4":-.8}},self.catalogue)
        self.assertGreater(len(self.env["robot"]["joints"]),16)
        prompt=for_environment(PLAN,self.env)
        self.assertNotIn("恰好一个tool任务",prompt)
        self.assertNotIn("唯一tool实例",prompt)
        self.assertIn("end_effectors",prompt)

    def run_fixture(self,replies,calls=30,resume=False):
        from asset_generation.contracts import GenerationConfig
        from asset_prep.config import Config
        from scene_generation.inputs import catalog
        from scene_generation.pipeline import SceneRun
        from scene_fixtures import SceneSandbox
        from generation_fixtures import FakeTransport, PROVENANCE
        source=self.root/"asset_package.md"
        source.write_text("# 合成资料\n### A01｜合成仪器\n宽100毫米。\n",encoding="utf-8")
        generation=GenerationConfig()
        transport=FakeTransport(replies)
        run=SceneRun(catalog(source,generation),self.env,numerical_contract({}),
            Config(),generation,
            SceneConfig(validation_mode="static",max_model_calls=calls),
            {"HARNESS_MODEL_API_KEY":"developer-fixture"},self.root/"run",transport,SceneSandbox(),
            self.robot,self.project,resume=resume,provenance=PROVENANCE)
        return run,transport

    def test_stock_pipeline_budget_resume_reuses_original_grippers(self):
        from asset_prep.errors import BudgetExceeded
        run,_=self.run_fixture([self.plan,{"model_py":"def build(output_dir):\n pass"}],calls=2)
        with self.assertRaises(BudgetExceeded):run.run()
        result=read_json(run.root/"result.json")
        self.assertTrue(result["assembly"]["available"])
        self.assertEqual(result["budget"]["model_calls"],2)
        review={"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":["开发合成测试，不证明质量"]}
        resumed,transport=self.run_fixture([{"keep_layout":"合成尺寸已核对"},review,review],resume=True)
        result=resumed.run()
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertEqual(set(resumed.state["jobs"]),{"instrument"})
        self.assertEqual(result["budget"]["model_calls"],2+len(transport.requests))
        self.assertIn("原厂保留",(run.root/"scene_document.md").read_text(encoding="utf-8"))

    def test_new_tool_scheduler_and_rollback_restore_plan_without_refunding(self):
        from scene_generation.scene_first import collect
        self.custom("left","replace")
        job=self.plan["jobs"].pop();item=self.plan["instances"].pop()
        selected=copy.deepcopy(self.plan["end_effectors"])
        self.plan["end_effectors"]["left"]={"mode":"stock","instance":None,"reason":"初始原厂"}
        original=copy.deepcopy(self.plan)
        action={"jobs":[job["id"]],"placements":[],"reason":"确定需要专用安装",
                "tool_additions":[{"job":job,"instance":item}],"end_effectors":selected}
        run,transport=self.run_fixture([self.plan,{"model_py":"def build(output_dir):\n pass"},
            {"keep_layout":"合成布局"},action,{"model_py":"def build(output_dir):\n # TOOL_FIXTURE\n pass"}])
        run.initialize();self.assertEqual(collect(run),[])
        baseline=copy.deepcopy(run.state["published_scene"])
        feedback={"status":"fail","checks":[{"id":"support:station","status":"fail","instances":["station"]}]}
        run.state.update(feedback=feedback,latest={**baseline,"assessed":True})
        self.assertTrue(run.route_repair())
        self.assertEqual(run.state["order"],["instrument","tool_left"])
        self.assertEqual(collect(run),[])
        self.assertIn("tool_left",run.state["jobs"])
        before=copy.deepcopy(run.store.state["calls"])
        run.state["repair_history"][-1].update(status="evaluated",comparison={"adopt":False})
        self.assertTrue(run.select_repair_candidate())
        self.assertEqual(run.state["plan"],original)
        self.assertEqual(run.state["order"],["instrument"])
        self.assertNotIn("tool_left",run.state["jobs"])
        self.assertEqual(run.store.state["calls"],before)
        self.assertTrue(run.state["candidate_attempts"]["tool_left"])

    def test_floating_accessory_is_not_validated_by_a_rigid_parent(self):
        from scene_generation.inspection import mount_contact
        from asset_generation.distance import DistanceQuery
        self.custom("left","adapter")
        artifact=write_component(self.root/"tool_left",True)
        path=artifact/"model.xml"
        path.write_text(path.read_text(encoding="utf-8").replace('name="base"','name="base" pos="0 0 2"'),encoding="utf-8")
        output=assemble(self.plan,self.env,{"instrument":write_component(self.root/"instrument"),
                        "tool_left":artifact},self.robot,self.root/"scene")
        m=load_model(output,scope="scene");meta=read_json(output/"scene.json")
        d=mujoco.MjData(m);d.qpos[:]=meta["home_qpos"];mujoco.mj_forward(m,d)
        row=mount_contact(m,d,meta,"tool_left",.003,.0005,DistanceQuery(m,d))
        self.assertEqual(row["status"],"fail")
        self.assertEqual(row["arm"],"left")

    def test_old_supplementary_evidence_rejected_after_mount_or_posture_change(self):
        from types import SimpleNamespace
        from scene_generation.evidence_state import session, merge_checks
        output=self.build()
        run=SimpleNamespace(root=self.root,generation=None,store=SimpleNamespace(state={"identity":{"implementation":"engineering"}}))
        state,_=session(run,output,"scene_review","scene",{"plan":self.plan},self.root)
        changed=copy.deepcopy(self.plan);changed["robot_posture"]={"lift_joint":-.1}
        new,_=session(run,output,"scene_review","scene",{"plan":changed},self.root)
        self.assertNotEqual(state["input_key"],new["input_key"])
        with self.assertRaises(ValidationError):
            merge_checks({"checks":[]},{"input_key":state["input_key"],"checks":[]},scope="scene",input_key=new["input_key"])


if __name__=="__main__":
    unittest.main()
