"""输入隔离、修订预算、设备门控及记录完整性；零模型请求。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from asset_prep.storage import atomic_json,digest,file_hash
from robot_operation.common import OperationError,Settings,identity,read
from robot_operation.scene import prepare,load
from robot_operation.workflow import request_task,run,repeat,GraphRunner
from robot_operation.feedback import import_response,import_revision,verify_source_collisions
from robot_operation.recording import Recorder,replay
from robot_operation.runtime import Session
from operation_fixtures import fixture,task_graph


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        source=fixture(self.root/"source")
        self.settings=Settings(settle_seconds=.1)
        self.prepared=prepare(source,self.root/"dynamic",identity(source),self.settings)
        # 使用实际工程静置长度，不放宽任何物理阈值。
        if self.prepared["admission"]["status"]!="pass":
            self.prepared=prepare(source,self.root/"settled",identity(source),Settings(settle_seconds=.5))
        self.scene=Path(self.prepared["scene"]); self.sha=self.prepared["scene_sha256"]
        self.task,self.graph=task_graph(self.sha)

    def tearDown(self): self.temp.cleanup()

    def input_files(self):
        taskroot=self.root/"inputs"; taskroot.mkdir()
        for name,text in (("asset_package.md","developer task"),("source.xml","<root>not needed</root>"),("rubric.json","SECRET_RUBRIC")):
            (taskroot/name).write_text(text,encoding="utf-8")
        atomic_json(taskroot/"input_manifest.json",{"task_id":"developer","files":[
            {"path":name,"sha256":file_hash(taskroot/name)} for name in ("asset_package.md","source.xml")]})
        profile=self.root/"profile.json"
        atomic_json(profile,{"version":1,"robot":{"model":"developer mobile dual arm"},"workspace":{"description":"developer fixture"}})
        return taskroot,profile

    def test_request_reads_only_allowlist_and_skips_xml(self):
        taskroot,profile=self.input_files()
        with patch("urllib.request.urlopen",side_effect=AssertionError("no network")):
            result=request_task(taskroot,self.scene,self.sha,profile,self.root/"request")
        request=read(result["request"])
        self.assertEqual([v["relative_path"] for v in request["bundle"]["files"]],["asset_package.md"])
        self.assertNotIn("SECRET_RUBRIC",json.dumps(request))
        self.assertEqual(result["network_calls"],0)

    def test_allowlist_rejects_rubric_even_if_listed(self):
        taskroot,profile=self.input_files()
        listing=read(taskroot/"input_manifest.json")
        listing["files"].append({"path":"rubric.json","sha256":file_hash(taskroot/"rubric.json")})
        atomic_json(taskroot/"input_manifest.json",listing)
        with self.assertRaises(OperationError): request_task(taskroot,self.scene,self.sha,profile,self.root/"request")

    def test_import_response_binds_task_and_scene(self):
        taskroot,profile=self.input_files()
        request_task(taskroot,self.scene,self.sha,profile,self.root/"request")
        request=read(self.root/"request"/"request.json")
        self.task.update(input_sha256=request["bundle"]["input_sha256"],public_profile_sha256=request["bundle"]["public_profile_sha256"])
        self.graph["task_sha256"]=digest(self.task)
        response=self.root/"response.json"; atomic_json(response,{"task":self.task,"graph":self.graph})
        result=import_response(self.root/"request"/"request.json",response,self.root/"accepted")
        self.assertEqual(result["status"],"accepted")
        self.graph["scene_sha256"]="stale"; atomic_json(response,{"task":self.task,"graph":self.graph})
        with self.assertRaises(OperationError): import_response(self.root/"request"/"request.json",response,self.root/"bad")

    def test_revision_budget_cannot_be_reset_by_new_graph(self):
        self.graph["nodes"][0]["completion"]={"kind":"joint_range","joint":"robot__left_x","range":[.3,.4]}
        self.graph["nodes"][0]["timeout"]=.05
        previous=self.root/"failed0"
        run(self.scene,self.sha,self.task,self.graph,previous,self.settings,images=False)
        for index in (1,2):
            candidate=self.root/f"candidate{index}.json"; atomic_json(candidate,self.graph)
            accepted=import_revision(previous,candidate,self.root/f"revision{index}")
            previous=self.root/f"failed{index}"
            result=run(self.scene,self.sha,self.task,read(accepted["graph"]),previous,self.settings,images=False)
            self.assertEqual(result["status"],"failed")
        with self.assertRaises(OperationError): import_revision(previous,candidate,self.root/"revision3")

    def test_replay_rejects_modified_control_trace(self):
        result=run(self.scene,self.sha,self.task,self.graph,self.root/"run",self.settings,images=False)
        self.assertEqual(result["status"],"success",result)
        trace=self.root/"run"/"steps.jsonl.gz"
        trace.write_bytes(trace.read_bytes()+b"tampered")
        with self.assertRaises(OperationError): replay(self.root/"run")

    def test_final_recording_forbids_checkpoint_restore(self):
        m,d,meta,manifest=load(self.scene,self.sha)
        session=Session(m,d,meta,manifest,self.task,self.settings)
        output=self.root/"record"; output.mkdir(); record=Recorder(output,mode="final",images=False)
        session.recorder=record; record.start(session,self.sha,self.task,self.graph,self.scene)
        try:
            with self.assertRaises(OperationError): session.restore(session.snapshot())
        finally: record.close({"status":"failed"})

    def test_device_logical_wait_and_replay(self):
        source=fixture(self.root/"device_source")
        manifest=read(source/"interfaces.json")
        manifest["interfaces"].append({"id":"device_start","kind":"software","body":"object__body"})
        atomic_json(source/"interfaces.json",manifest)
        prepared=prepare(source,self.root/"device_dynamic",identity(source),Settings(settle_seconds=.5))
        task,graph=task_graph(prepared["scene_sha256"])
        task["devices"]=[{"id":"device","interface":"device_start","duration_seconds":.1,
            "start_condition":task["initial"],"running_condition":task["initial"]}]
        task["allow_logical_time"]=True
        task["final"]={"kind":"device_state","device":"device","state":"complete"}
        task["stages"][0]["goal"]=task["final"]
        node=graph["nodes"][0]
        wait=deepcopy(node); wait.update(id="wait",parameters={"seconds":.02,"logical_seconds":.2},completion=task["final"],
            precondition={"kind":"device_state","device":"device","state":"running"})
        node.update(operation="device",parameters={"device":"device"},completion=wait["precondition"],on_success="wait")
        graph["nodes"].append(wait); graph["task_sha256"]=digest(task)
        result=run(prepared["scene"],prepared["scene_sha256"],task,graph,self.root/"device_run",Settings(),images=False)
        self.assertEqual(result["status"],"success",result)
        self.assertGreater(result["logical_time"],result["sim_time"])
        self.assertEqual(replay(self.root/"device_run")["status"],"pass")

    def test_repeat_schedule_retains_failures_and_fixed_settings(self):
        # 仅测试调度，不运行14次正式或合成试验。
        with patch("robot_operation.workflow.run",return_value={"status":"failed"}) as execute:
            result=repeat(self.scene,self.sha,self.task,self.graph,self.root/"repeat",self.settings,images=False)
        self.assertEqual(execute.call_count,14); self.assertEqual(result["success_rate"],0)
        cases=read(self.root/"repeat"/"schedule.json")["cases"]
        self.assertEqual(sum(delta is None for _,delta in cases),4)
        self.assertEqual(len(read(self.root/"repeat"/"summary.json")["runs"]),14)

    def test_diagnosis_does_not_turn_search_failure_into_scene_defect(self):
        result=verify_source_collisions(self.scene,self.sha,self.root/"diagnosis",self.settings)
        self.assertFalse(result["scene_repair_authorized"])
        self.assertFalse((self.root/"diagnosis"/"scene_feedback.json").exists())

    def test_observe_records_state_without_advancing_physics(self):
        self.graph["nodes"][0].update(operation="observe",parameters={})
        result=run(self.scene,self.sha,self.task,self.graph,self.root/"observation",self.settings,images=False)
        self.assertEqual(result["status"],"success",result)
        events=[json.loads(line) for line in (self.root/"observation"/"events.jsonl").read_text(encoding="utf-8").splitlines()]
        observations=[event for event in events if event["type"]=="observation"]
        self.assertEqual(len(observations),1)
        self.assertEqual(observations[0]["time"],0.)
        self.assertIn("body_position",observations[0])
