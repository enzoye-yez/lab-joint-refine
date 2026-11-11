"""Developer-authored offline contracts and physical failure probes; no model success claims."""
import json,tempfile,unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,tostring
import mujoco,numpy as np
from asset_prep.storage import Store,atomic_json,digest,file_hash
from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded,UncertainRequest,StaleEvidence
from asset_generation.construction import Builder,attach
from asset_generation.contracts import GenerationConfig
from asset_generation.operation_interface import validate_operation
from robot_operation.common import identity,read,Settings,OperationError
from robot_operation.requests import Exchange,AwaitingResponse,import_exchange,read_page
from robot_operation.joint_state import Budget,compare,duplicate,mechanism
from robot_operation.improve import open_run
from robot_operation.scene import prepare
from robot_operation.diagnosis import inspect_scene,diagnose_run
from operation_fixtures import fixture,task_graph

PROJECT=Path(__file__).resolve().parents[1]

class ConstructionTests(unittest.TestCase):
    def model(self,modify=False):
        xml=Element("mujoco");SubElement(xml,"compiler",angle="radian")
        world=SubElement(xml,"worldbody");base=SubElement(world,"body",name="base")
        b=Builder();b.container(base,"cup",[.2,.2,.15],.01)
        b.bracket(base,"bracket",[.12,.12,.01],[.05,.05]).set("pos",".4 0 0")
        b.hinge(base,"lid",[0,0,.15 if not modify else .2],[0,1,0],[0,1.4])
        b.slide(base,"slide",[.7,0,0],[0,0,1],[0,.1],rail_length=.3)
        return mujoco.MjModel.from_xml_string(tostring(xml,encoding="unicode")),b
    def test_helpers_compile_and_joint_limits_preserved(self):
        model,b=self.model();self.assertEqual(model.njnt,2)
        np.testing.assert_allclose(model.joint("lid_joint").range,[0,1.4])
        self.assertEqual(len(b.records),4);self.assertTrue(all(not r["certified"] for r in b.records))
    def test_parameter_edit_keeps_pin_and_axis_aligned(self):
        model,_=self.model(True);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.xanchor[model.joint("lid_joint").id],data.geom_xpos[model.geom("lid_pin").id])
    def test_hollow_opening_is_not_solid_collision_box(self):
        model,_=self.model();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        geom=np.array([-1],np.int32)
        distance=mujoco.mj_ray(model,data,np.array([.04,0.,.3]),np.array([0.,0.,-1.]),None,1,-1,geom)
        self.assertAlmostEqual(distance,.29,places=5)
    def test_invalid_wall_and_frame_rejected(self):
        with self.assertRaises(ValueError):Builder().container(Element("body"),"bad",[.1,.1,.1],.1)
        with self.assertRaises(ValueError):attach([0,0,0],np.zeros((3,3)),[1,0,0],np.eye(3))
    def test_feature_alignment(self):
        p,r=attach([.1,0,0],np.eye(3),[.5,.2,.3],np.eye(3));np.testing.assert_allclose(p+r@np.array([.1,0,0]),[.5,.2,.3])

class ExchangeTests(unittest.TestCase):
    def test_offline_ledger_idempotent_and_unknown_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root,{"x":1},Config(max_model_calls=5));exchange=Exchange(store,GenerationConfig())
            with self.assertRaises(AwaitingResponse):exchange.ask("instruction",{"source":"input"},[],"phase")
            request=read(Path(store.state["pending_exchange"]["path"]));key=request["request_id"]
            reply={"request_id":key,"delivered_batches":[0],"provenance":"model_generated","model_provenance":{"provider":"gpugeek","requested_model":"GPT-6-astra","response_model":"GPT-6-astra"},"value":{"ok":True}}
            atomic_json(root/"reply.json",reply);import_exchange(root,root/"reply.json")
            for _ in range(2):self.assertEqual(exchange.ask("instruction",{"source":"input"},[],"phase")[0],{"ok":True})
            self.assertEqual(store.totals()["model_calls"],1)
            store.reserve("unknown","model",123)
            with self.assertRaises(UncertainRequest):exchange.ask("other",{},[],"other")
            self.assertEqual(store.totals()["model_calls"],2)
    def test_paging_recovers_full_source_and_rejects_foreign_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root,{"x":1},Config());exchange=Exchange(store,replace(GenerationConfig(),review_context_bytes=4096))
            with self.assertRaises(AwaitingResponse):exchange.ask("instruction",{"source":"a"*40000},[],"phase")
            path=Path(store.state["pending_exchange"]["path"]);request=read(path)
            page=read_page(path,{"reference":request["source_ref"],"path":["source"],"offset":39000,"limit":2000})
            self.assertEqual(page["data"],"a"*1000)
            request["context_root"]=str(root);atomic_json(path,request)
            with self.assertRaises(OperationError):read_page(path,{"reference":request["source_ref"]})
    def test_budget_resumption_never_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root,{"x":1},Config());budget=Budget(store)
            for i in range(80):budget.round(str(i),"operation")
            restored=Budget(Store(root,{"x":1},Config(),resume=True));restored.round("0","operation")
            with self.assertRaises(BudgetExceeded):restored.round("81","operation")
            for i in range(60):restored.asset("cup",str(i))
            with self.assertRaises(BudgetExceeded):restored.asset("cup","61")
    def test_progress_and_new_defect_are_exploration(self):
        old={"result":{"status":"failed","completed_stages":[]},"diagnosis":{"rows":[],"coverage_gaps":[]}}
        new=deepcopy(old);new["result"]["completed_stages"]=["grasp"]
        self.assertEqual(compare(old,new),"improved_partial")
        new["diagnosis"]["rows"]=[{"repair_authorized":True,"kind":"collision","geoms":["a","b"]}]
        self.assertEqual(compare(old,new),"exploration")

class JointTests(unittest.TestCase):
    def inputs(self,root):
        source=fixture(root/"source");taskroot=root/"task";taskroot.mkdir()
        (taskroot/"protocol.txt").write_text("Developer fixture: maintain left joint at zero",encoding="utf-8")
        atomic_json(taskroot/"input_manifest.json",{"task_id":"fixture","files":[{"path":"protocol.txt","sha256":file_hash(taskroot/"protocol.txt")}]})
        profile=root/"profile.json";atomic_json(profile,{"version":1,"robot":{"model":"developer fixture"},"workspace":{"description":"developer fixture"}})
        return source,taskroot,profile
    def create(self,root,source,taskroot,profile,resume=False):
        return open_run(root/"joint",source,identity(source),taskroot,profile,PROJECT,developer=True,resume=resume,settings=Settings(settle_seconds=.1))
    def reply(self,run,value):
        request=read(Path(run.store.state["pending_exchange"]["path"]));path=run.root/"reply.json"
        atomic_json(path,{"request_id":request["request_id"],"delivered_batches":list(range(len(request["batches"]))),"provenance":"developer_fixture","value":value})
        import_exchange(run.root,path)
    def test_separate_frozen_task_and_complete_trial_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,taskroot,profile=self.inputs(root);run=self.create(root,source,taskroot,profile)
            self.assertEqual(run.advance()["status"],"awaiting_response")
            request=read(Path(run.store.state["pending_exchange"]["path"]))
            self.assertEqual(request["phase"],"task_compile")
            task,graph=task_graph(run.state["current"]["scene_sha256"])
            task.update(input_sha256=run.bundle["input_sha256"],public_profile_sha256=run.bundle["public_profile_sha256"])
            self.reply(run,{"task":task,"unverified":[]});self.assertEqual(run.advance()["status"],"awaiting_response")
            self.assertEqual(run.state["phase"],"graph")
            graph["task_sha256"]=digest(task)
            self.reply(run,{"graph":graph})
            result=run.advance();self.assertEqual(result["status"],"verified_task",result)
            self.assertEqual(result["consumption"]["model_calls"],0)
            self.assertTrue((run.root/"final/0/steps.jsonl.gz").exists())
            restored=self.create(root,source,taskroot,profile,True)
            self.assertEqual(restored.advance()["rounds"],1)
    def test_pause_and_frozen_task_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,taskroot,profile=self.inputs(root);run=self.create(root,source,taskroot,profile)
            atomic_json(run.root/"pause.request",{"user":True})
            self.assertEqual(run.advance()["status"],"paused_user")
            self.assertTrue((run.root/"pause.request").exists())
    def test_readonly_diagnostic_search_failure_not_asset_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/"source");result=prepare(source,root/"dynamic",identity(source),Settings(settle_seconds=.1))
            before=identity(root/"dynamic");report=inspect_scene(root/"dynamic",before)
            self.assertEqual(identity(root/"dynamic"),before)
            self.assertFalse(any(r.get("repair_authorized") for r in report["rows"]))
            report=inspect_scene(root/"dynamic",before,queries=[{"kind":"mechanism","interface":"object__grasp"}])
            self.assertEqual(report["rows"][-1]["status"],"unknown")
