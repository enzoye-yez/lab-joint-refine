"""离线行为测试，全部使用开发者合成场景。"""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import mujoco
import numpy as np
from asset_prep.storage import digest,atomic_json
from robot_operation.common import Settings,OperationError,identity,read
from robot_operation.scene import prepare,load
from robot_operation.planning import rrt_connect,astar,wheel_commands,Kinematics
from robot_operation.contracts import validate_graph
from robot_operation.runtime import Session
from robot_operation.workflow import run,GraphRunner
from robot_operation.recording import replay
from operation_fixtures import fixture,task_graph


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.source=fixture(self.root/"source")
        self.settings=Settings(settle_seconds=.5)
        self.prepared=prepare(self.source,self.root/"dynamic",identity(self.source),self.settings)
        self.scene=self.root/"dynamic"; self.sha=self.prepared["scene_sha256"]
        self.task,self.graph=task_graph(self.sha)

    def tearDown(self): self.temp.cleanup()

    def session(self,task=None):
        m,d,meta,manifest=load(self.scene,self.sha)
        return Session(m,d,meta,manifest,task or self.task,self.settings)

    def test_conversion_preserves_source_and_other_joints(self):
        self.assertEqual(identity(self.source),read(self.scene/"scene.json")["operation_conversion"]["source_sha256"])
        m,d,meta,_=load(self.scene,self.sha)
        self.assertEqual(m.joint("robot__free_base").type[0],mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(len(meta["home_qpos"]),m.nq)
        self.assertEqual(self.prepared["admission"]["status"],"pass",self.prepared)

    def test_stale_scene_rejected(self):
        with self.assertRaises(OperationError): load(self.scene,"bad")

    def test_state_restore_includes_rng_device_and_graph(self):
        s=self.session(); runner=GraphRunner(s,self.graph); checkpoint=runner.checkpoint()
        random=s.rng.random(); s.advance(.02); runner.visits["one"]=2
        runner.restore(checkpoint)
        self.assertEqual(float(s.data.time),0)
        self.assertEqual(s.rng.random(),random)
        self.assertEqual(runner.visits,{})

    def test_observed_joint_motion_and_command_replay(self):
        goal={"kind":"joint_range","joint":"robot__left_x","range":[.07,.09]}
        self.task["stages"][0]["goal"]=goal; self.task["final"]=goal
        self.graph["task_sha256"]=digest(self.task)
        node=self.graph["nodes"][0]
        node.update(operation="joint_move",parameters={"arms":["left"],"positions":{"robot__left_x":.08}},completion=goal,timeout=4)
        result=run(self.scene,self.sha,self.task,self.graph,self.root/"run",self.settings,images=False)
        self.assertEqual(result["status"],"success",result)
        report=replay(self.root/"run")
        self.assertEqual(report["status"],"pass")
        self.assertGreater(report["steps"],1)
        self.assertFalse(read(self.root/"run"/"recording.json")["dataset_complete"])

    def test_failed_condition_cannot_be_declared_success(self):
        self.graph["nodes"][0]["completion"]={"kind":"joint_range","joint":"robot__left_x","range":[.3,.4]}
        self.graph["nodes"][0]["timeout"]=.05
        result=run(self.scene,self.sha,self.task,self.graph,self.root/"failed",self.settings,images=False)
        self.assertEqual(result["status"],"failed")
        request=read(self.root/"failed"/"feedback_request.json")
        self.assertFalse(request["scene_repair_authorized"])
        self.assertEqual(request["network_calls"],0)

    def test_gripper_cannot_control_arm(self):
        s=self.session()
        with self.assertRaises(OperationError): s.gripper("left",{"robot__left_x_motor":.1})

    def test_empty_grasp_fails(self):
        s=self.session(); s.deadline=.05
        with self.assertRaises(OperationError):
            s.grasp({"arm":"left","body":"object__body","controls":{"robot__left_grip1":.02,"robot__left_grip2":.02}})
        self.assertEqual(s.holding,{})

    def test_device_start_requires_observation(self):
        task={**self.task,"devices":[{"id":"device","interface":"start","duration_seconds":1,
            "start_condition":{"kind":"joint_range","joint":"robot__left_x","range":[.3,.4]},
            "running_condition":{"kind":"joint_range","joint":"robot__left_x","range":[-.3,.5]}}]}
        m,d,meta,manifest=load(self.scene,self.sha)
        manifest["interfaces"].append({"id":"start","kind":"software"})
        s=Session(m,d,meta,manifest,task,self.settings)
        with self.assertRaises(OperationError): s.evidence.start("device")
        self.assertEqual(s.evidence.devices["device"]["state"],"idle")

    def test_graph_rejects_invented_operation_and_unbounded_loop(self):
        self.graph["nodes"][0]["operation"]="python"
        with self.assertRaises(OperationError): validate_graph(self.graph,self.task,self.sha,self.settings)
        self.graph["nodes"][0]["operation"]="wait"; self.graph["nodes"][0]["max_visits"]=100
        with self.assertRaises(OperationError): validate_graph(self.graph,self.task,self.sha,self.settings)

    def test_ik_operates_on_copy(self):
        s=self.session(); original=s.data.qpos.copy(); joints=s.joints_for(["left"])
        _,_,valid=s.query(joints)
        k=Kinematics(s.model,s.data,joints,valid,s.settings,s.rng)
        sid=s.model.site(s.tcps["left"]).id
        goal=k.solve([{"site":s.tcps["left"],"position":s.data.site_xpos[sid]+[.08,0,0],"rotation":s.data.site_xmat[sid].reshape(3,3)}])
        self.assertAlmostEqual(goal[0],.08,delta=.003)
        np.testing.assert_array_equal(s.data.qpos,original)

    def test_real_wheel_drive_moves_free_base(self):
        s=self.session(); q=int(s.base.qposadr[0]); initial=float(s.data.qpos[q])
        for wheel,(_,speed) in zip(s.wheels,wheel_commands(.05,0,0,s.wheels)):
            s.data.ctrl[wheel["drive"]]=speed
        s.advance(.5)
        self.assertGreater(float(s.data.qpos[q])-initial,.005)

    def test_navigation_stops_and_images_are_recorded(self):
        node=self.graph["nodes"][0]
        node.update(operation="navigate",timeout=8,parameters={"goal":[.04,0,0],"bounds":[[-.5,-.5],[.5,.5]],
            "transport":{"arms":["left","right"],"positions":{"robot__left_x":0.,"robot__right_x":0.}}},
            completion={"kind":"base_stopped","duration":.1})
        result=run(self.scene,self.sha,self.task,self.graph,self.root/"navigation",self.settings,images=True)
        self.assertEqual(result["status"],"success",result)
        recording=read(self.root/"navigation"/"recording.json")
        self.assertIsNone(recording["image_error"],recording)
        self.assertGreater(recording["frames"],0)
        self.assertTrue((self.root/"navigation"/"images"/"0000000_overview.png").exists())
        self.assertEqual(replay(self.root/"navigation")["status"],"pass")

    def test_dual_arm_cartesian_tracks_both(self):
        s=self.session(); s.deadline=4
        rows=[]
        for arm in ("left","right"):
            sid=s.model.site(s.tcps[arm]).id
            rows.append({"arm":arm,"position":(s.data.site_xpos[sid]+[.015,0,0]).tolist(),"quaternion":[1,0,0,0]})
        s.cartesian({"targets":rows},True)
        for arm in ("left","right"):
            self.assertGreater(s.data.qpos[s.model.joint("robot__"+arm+"_x").qposadr[0]],.01)

    def test_contact_grasp_lower_release_and_stable_support(self):
        source=fixture(self.root/"grasp_source",grasp_ready=True)
        prepared=prepare(source,self.root/"grasp_dynamic",identity(source),self.settings)
        m,d,meta,manifest=load(self.root/"grasp_dynamic",prepared["scene_sha256"])
        s=Session(m,d,meta,manifest,self.task,self.settings); s.deadline=6
        s.allowed=[["robot__left_finger_body1","object__body"],["robot__left_finger_body2","object__body"],["object__body","support"]]
        s.grasp({"arm":"left","body":"object__body","controls":{"robot__left_grip1":.016,"robot__left_grip2":.016}})
        self.assertEqual(s.holding["left"],"object__body")
        self.assertEqual(m.neq,0,"接触抓取不创建焊接约束")
        s.joint_move({"arms":["left"],"positions":{"robot__left_z":-.05}})
        s.place({"arm":"left","body":"object__body","support":"support","controls":{"robot__left_grip1":0.,"robot__left_grip2":0.}})
        self.assertEqual(s.holding,{})
        self.assertTrue(s.evidence.evaluate({"kind":"supported","body":"object__body","support":"support"}))

    def test_mechanical_event_requires_actual_actor_contact(self):
        task={**self.task,"mechanical_detectors":[{"event":"unlock","source":"developer fixture",
              "condition":{**self.task["initial"],"duration":.004}}]}
        m,d,meta,manifest=load(self.scene,self.sha)
        manifest["mechanical_events"]=[{"id":"unlock","interface":"object__grasp","action":"release"}]
        s=Session(m,d,meta,manifest,task,self.settings)
        with self.assertRaises(OperationError) as raised: s.advance(.02)
        self.assertEqual(raised.exception.code,"mechanical_evidence_missing")

    def test_dock_uses_interface_and_checks_reach_before_driving(self):
        s=self.session(); s.deadline=8
        s.dock({"arm":"left","interface":"object__grasp","radius":.4,"bounds":[[-.5,-.5],[1.,1.]],
            "transport":{"arms":["left","right"],"positions":{"robot__left_x":0.,"robot__right_x":0.}},
            "reach_targets":[{"arm":"left","position":[0,.2,.47],"quaternion":[1,0,0,0],"frame":"object__site"}]})
        self.assertAlmostEqual(float(s.data.qpos[s.base.qposadr[0]]),.2,delta=.01)


class PinnedRobotTests(unittest.TestCase):
    def test_official_robot_conversion_preserves_full_pose(self):
        from scene_generation.assembly import assemble
        from scene_fixtures import write_component
        from test_mobile_layout import mobile_fixture
        project=Path(__file__).resolve().parents[1]
        env=read(project/"config/scene_environment_ffw.json")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,plan=mobile_fixture(root)
            plan["jobs"]=plan["jobs"][:1]; plan["instances"]=plan["instances"][:1]
            plan["end_effectors"]={a:{"mode":"stock","instance":None,"reason":"developer fixture"} for a in ("left","right")}
            source=assemble(plan,env,{"instrument":write_component(root/"asset")},project/"data/robots/ffw_sg2",root/"scene")
            result=prepare(source,root/"dynamic",identity(source),Settings(settle_seconds=.05))
            self.assertEqual(read(root/"dynamic"/"scene.json")["operation_conversion"]["max_body_position_error_m"],0)
            self.assertIn(result["admission"]["status"],{"pass","blocked"})
            vendor=mujoco.MjModel.from_xml_path(str(project/"data/robots/ffw_sg2/model.xml"))
            assembled=mujoco.MjModel.from_xml_path(str(source/"model.xml"))
            self.assertEqual(assembled.opt.cone,vendor.opt.cone)
            self.assertEqual(assembled.opt.impratio,vendor.opt.impratio)


class SearchTests(unittest.TestCase):
    def test_rrt_detour_and_blocked(self):
        valid=lambda q: not (.4<q[0]<.6 and q[1]<.7)
        route=rrt_connect([0,0],[1,0],[[0,1],[0,1]],valid,np.random.default_rng(4),3000,.025)
        self.assertTrue(any(q[1]>=.7 for q in route))
        with self.assertRaises(OperationError):
            rrt_connect([0,0],[1,0],[[0,1],[0,1]],lambda q:q[0]<.4,np.random.default_rng(0))

    def test_astar_does_not_cut_wall(self):
        route=astar([0,0],[1,0],[[-.1,-.1],[1.1,1]],lambda p:not(.4<p[0]<.6 and p[1]<.6),.05)
        self.assertTrue(any(p[1]>=.6 for p in route))

    def test_swerve_reverse_preserves_steering_range(self):
        angle,speed=wheel_commands(-1,0,0,[{"xy":[0,0],"radius":.1}])[0]
        self.assertAlmostEqual(angle,0); self.assertLess(speed,0)
