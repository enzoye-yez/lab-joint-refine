import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_generation.contracts import GenerationConfig
from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded, StaleEvidence, ValidationError, PrepError, ProviderError, UncertainRequest
from scene_generation.contracts import SceneConfig,numerical_contract
from scene_generation.inputs import catalog
from scene_generation.pipeline import SceneRun
from generation_fixtures import FakeTransport,PROVENANCE
from scene_fixtures import environment,plan,SceneSandbox

ASSET_SOURCE = 'def build(output_dir):\n    pass\n'
TOOL_SOURCE = 'def build(output_dir):\n    # TOOL_FIXTURE\n    pass\n'
REVIEW = {"verdict":"accept","findings":[],"unverified":["合成响应，不是正式模型结果"]}


class ScenePipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root/"asset_package.md"
        self.source.write_text("# 合成资料\n### A01｜合成仪器\n宽100毫米。\n",encoding="utf-8")
        self.generation = GenerationConfig()
        self.catalog = catalog(self.source,self.generation)
        self.env = environment(self.root/"robot")
        self.project = Path(__file__).resolve().parents[1]
        self.base_replies = [plan(),{"model_py":ASSET_SOURCE},REVIEW,{"model_py":TOOL_SOURCE},REVIEW]

    def run_loop(self,replies,*,resume=False,limits=None,sandbox=None,feedback=None,retry_uncertain=False):
        if limits and limits.validation_mode=="static" and len(replies)>=4 and isinstance(replies[2],dict) and "verdict" in replies[2] and isinstance(replies[3],dict) and "model_py" in replies[3]:
            replies=[*replies[:2],replies[3],replies[2],*replies[4:]]
        transport = FakeTransport(replies)
        with patch("asset_prep.pipeline.execute",side_effect=AssertionError("No extraction pipeline")), \
             patch("asset_prep.extraction.extract_inventory",side_effect=AssertionError("No extraction")), \
             patch("asset_prep.search.Search",side_effect=AssertionError("No search")):
            run = SceneRun(self.catalog,self.env,numerical_contract({}),Config(),
                self.generation,limits or SceneConfig(),{"HARNESS_MODEL_API_KEY":"fixture-only"},self.root/"run",transport,
                sandbox or SceneSandbox(),self.root/"robot",self.project,resume=resume,
                retry_uncertain=retry_uncertain,provenance=PROVENANCE)
            result = run.run(feedback)
        return result,transport

    def test_complete_scene_no_extraction_and_gpugeek_model(self):
        result,transport = self.run_loop(self.base_replies+[REVIEW])
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(result["budget"]["model_calls"],6)
        self.assertTrue(all(x["model"] == "GPT-6-astra" for x in transport.requests))
        self.assertEqual(set(transport.urls),{"https://api.gpugeek.com/v1/chat/completions"})
        self.assertTrue((self.root/"run"/result["scene_artifact"]/"model.xml").is_file())
        self.assertEqual(len(result["scene_sha256"]),64)
        tool_context = json.loads(transport.requests[3]["messages"][1]["content"][0]["text"])
        self.assertEqual(tool_context["instance_contract"]["delivery_role"],"tool")
        self.assertEqual(tool_context["instance_contract"]["manifest_asset_ids"],["A01"])
        self.assertEqual(tool_context["instance_contract"]["reference_only_jobs"],["instrument"])
        self.assertIn("首个实际渲染",tool_context["reference_policy"])
        self.assertEqual(len(tool_context["input"]["images"]),1)
        self.assertIn("renders",tool_context["input"]["images"][0]["path"])

    def test_layout_repair_reuses_all_asset_versions(self):
        repair = {"jobs":[],"placements":[{"id":"station","position":[.6,0,.8],"yaw":0}],"reason":"合成碰撞"}
        sandbox = SceneSandbox(fail_scene_once=True)
        result,transport = self.run_loop(self.base_replies+[repair,REVIEW,REVIEW],sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,2)
        self.assertEqual(sandbox.scene_calls,2)
        context=json.loads(transport.requests[6]['messages'][1]['content'][0]['text'])
        self.assertIsNotNone(context['layout_revalidation'])
        self.assertEqual(context['dependencies']['instrument']['instances'][0]['position'],[.6,0,.8])

    def test_measurement_failure_is_not_sent_to_geometry_author(self):
        class MeasurementBackend(SceneSandbox):
            def assemble_and_check(inner,*args):
                report,renders=super().assemble_and_check(*args)
                report.update(status='error',error_kind='measurement_error',
                    checks=[{'id':'scene_collisions','status':'error','detail':'synthetic bad witness','instances':[]}])
                return report,renders
        result,transport=self.run_loop(self.base_replies,sandbox=MeasurementBackend())
        self.assertEqual(result['status'],'needs_check_support')
        self.assertEqual(result['check_failure']['kind'],'measurement_error')
        self.assertEqual(len(transport.requests),5)

    def test_empty_repair_replans_with_evidence_then_succeeds(self):
        empty={"jobs":[],"placements":[],"reason":"缺少信息"}
        repair={"jobs":[],"placements":[{"id":"station","position":[.6,0,.8],"yaw":0}],"reason":"补证后调整"}
        sandbox=SceneSandbox(fail_scene_once=True)
        result,transport=self.run_loop(self.base_replies+[empty,repair,REVIEW,REVIEW],sandbox=sandbox)
        self.assertEqual(result['status'],'candidate_ready')
        first=json.loads(transport.requests[5]['messages'][1]['content'][0]['text'])
        second=json.loads(transport.requests[6]['messages'][1]['content'][0]['text'])
        self.assertEqual(first['validation_feedback'],[])
        self.assertEqual(second['validation_feedback']['latest']['response'],empty)
        self.assertEqual(second['current_assets']['instrument']['model_py'],ASSET_SOURCE)
        self.assertEqual(sandbox.build_calls,2)

    def test_empty_repair_is_bounded_and_resume_does_not_repeat_same_requests(self):
        empty={"jobs":[],"placements":[],"reason":"仍然没有方案"}
        limits=SceneConfig(max_repair_plan_attempts=2)
        result,_=self.run_loop(self.base_replies+[empty]*4,sandbox=SceneSandbox(fail_scene_once=True),limits=limits)
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(result['stop_reason']['reason'],'repair_tasks_deferred')
        self.assertEqual(result['stop_reason']['attempts'],4)
        result,transport=self.run_loop([],resume=True,limits=limits)
        self.assertEqual(transport.requests,[])
        self.assertIsNotNone(result['scene_artifact'])

    def test_table_only_repair_reassembles_without_regenerating_assets(self):
        repair={"jobs":[],"placements":[],"reason":"局部安装槽",
                "table_modifications":[{"id":"slot","instance":"station","center":[.5,0],"size":[.04,.04],"depth":.006,"reason":"合成局部凹槽"}]}
        sandbox=SceneSandbox(fail_scene_once=True)
        result,_=self.run_loop(self.base_replies+[repair,REVIEW],sandbox=sandbox)
        self.assertEqual(result['status'],'candidate_ready')
        self.assertEqual(sandbox.build_calls,2)
        self.assertEqual(sandbox.scene_calls,2)

    def test_geometry_repair_revalidates_dependent_tool_before_editing(self):
        repair = {"jobs":["instrument"],"placements":[],"reason":"合成形状错误"}
        patch_reply = {"edits":[{"old":"    pass","new":"    return None"}]}
        sandbox = SceneSandbox(fail_scene_once=True)
        result,transport = self.run_loop(self.base_replies+[repair,patch_reply,REVIEW,REVIEW,REVIEW],sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,3)
        for index in (6,7,8):
            context = json.loads(transport.requests[index]["messages"][1]["content"][0]["text"])
            self.assertEqual(context["repair_directive"],repair)
        context=json.loads(transport.requests[8]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['current_model_py'],TOOL_SOURCE)
        self.assertIsNotNone(context['layout_revalidation'])

    def test_repeated_asset_generation_and_review_share_explicit_instance_contract(self):
        p = plan()
        p["instances"].append({"id":"second","job":"instrument","placement":"table","position":[.7,0,.8],"yaw":0})
        replies = [p,self.base_replies[1],{**REVIEW,"prototype_units":1},*self.base_replies[3:],REVIEW]
        result,transport = self.run_loop(replies)
        self.assertEqual(result["status"],"candidate_ready")
        for index in (1,2):
            context = json.loads(transport.requests[index]["messages"][1]["content"][0]["text"])
            self.assertEqual(context["instance_contract"]["expected_model_units"],1)
            self.assertEqual(len(context["instance_contract"]["instances"]),2)

    def test_shared_budget_pause_and_resume_cached_response(self):
        with self.assertRaises(BudgetExceeded):
            self.run_loop(self.base_replies[:2],limits=SceneConfig(max_model_calls=2))
        result,transport = self.run_loop([REVIEW,{"model_py":TOOL_SOURCE},REVIEW,REVIEW],resume=True)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(len(transport.requests),4)
        self.assertEqual(result["budget"]["model_calls"],6)

    def test_finished_resume_does_not_call_model(self):
        self.run_loop(self.base_replies+[REVIEW])
        result,transport = self.run_loop([],resume=True)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(transport.requests,[])

    def test_review_timeout_recovers_without_rebuilding_and_resume_is_clean(self):
        sandbox = SceneSandbox()
        timeout = ProviderError("synthetic timeout", uncertain=True,
                                diagnostics={"failure":"timeout","phase":"waiting_headers"})
        with self.assertRaises(UncertainRequest):
            self.run_loop(self.base_replies[:2]+[timeout],sandbox=sandbox)
        with self.assertRaises(UncertainRequest):self.run_loop([],resume=True,sandbox=sandbox)
        result,transport = self.run_loop(self.base_replies[2:]+[REVIEW],resume=True,retry_uncertain=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(result["budget"]["model_calls"],7)
        self.assertEqual(sandbox.build_calls,2)
        state=json.loads((self.root/'run/state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['calls'][2]['request_sha256'],state['calls'][3]['request_sha256'])
        result,transport = self.run_loop([],resume=True)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(transport.requests,[])

    def test_modified_asset_evidence_stops_resume(self):
        result,_ = self.run_loop(self.base_replies+[REVIEW])
        (self.root/"run"/result["scene_artifact"]/"model.xml").write_text("modified",encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            self.run_loop([],resume=True)

    def test_changed_backend_cannot_resume_existing_run(self):
        self.run_loop(self.base_replies+[REVIEW])
        sandbox = SceneSandbox()
        with patch.object(sandbox,"preflight",return_value={"backend":"different"}), self.assertRaises(StaleEvidence):
            self.run_loop([],resume=True,sandbox=sandbox)

    def test_planner_feedback_triggers_bounded_repair(self):
        result,_ = self.run_loop(self.base_replies+[REVIEW])
        feedback = {"scene_sha256":result["scene_sha256"],"failures":[{"kind":"collision","instances":["station"],
                    "detail":"合成规划失败","evidence":"开发者指定碰撞对"}]}
        repair = {"jobs":[],"placements":[{"id":"station","position":[.6,0,.8],"yaw":0}],"reason":"规划失败证据"}
        result,transport = self.run_loop([repair,REVIEW,REVIEW],resume=True,feedback=feedback)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(len(transport.requests),3)
        result,transport = self.run_loop([],resume=True,feedback=feedback)
        self.assertEqual(transport.requests,[])

    def test_visual_uncertainty_is_not_success(self):
        result,_ = self.run_loop(self.base_replies+[{"verdict":"uncertain","findings":[],"unverified":["看不清"],"blocking":["当前视角无法辨认部件"]}])
        self.assertEqual(result["status"],"needs_review")

    def test_asset_review_format_retry_does_not_modify_geometry(self):
        sandbox = SceneSandbox()
        invalid = {"verdict":"accept","findings":["未见缺件"],"unverified":[]}
        replies = self.base_replies[:2]+[invalid]+self.base_replies[2:]+[REVIEW]
        result,transport = self.run_loop(replies,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,2)
        self.assertEqual(result["budget"]["model_calls"],7)
        self.assertIn("review_format_correction",json.loads(transport.requests[3]["messages"][1]["content"][0]["text"]))

    def test_invalid_scene_review_retries_without_reassembly(self):
        sandbox = SceneSandbox()
        invalid = {"verdict":"accept","findings":["station: 外形一致"],"unverified":[]}
        result,_ = self.run_loop(self.base_replies+[invalid,REVIEW],sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.scene_calls,1)

    def test_resume_failed_asset_restores_feedback_and_previews_and_reuses_passed_job(self):
        class FailedTool(SceneSandbox):
            def build_and_check(self,script,destination,asset_ids,profile):
                report,renders = super().build_and_check(script,destination,asset_ids,profile)
                if "TOOL_FIXTURE" in script.read_text(encoding="utf-8"):
                    report.update(status="fail",checks=[{"id":"initial_contacts","status":"fail",
                        "detail":"fixture lug penetration","contacts":[{"distance_m":-.0015}]}])
                return report,renders
        patch_one = {"edits":[{"old":"    pass","new":"    return None"}]}
        patch_two = {"edits":[{"old":"    return None","new":"    pass"}]}
        result,_ = self.run_loop(self.base_replies[:4]+[patch_one,patch_two],sandbox=FailedTool())
        self.assertEqual(result["status"],"asset_repair_budget_exhausted")
        sandbox = SceneSandbox()
        exhausted,none=self.run_loop([],resume=True,sandbox=sandbox)
        self.assertEqual(exhausted['status'],'asset_repair_budget_exhausted')
        self.assertEqual(none.requests,[])
        from dataclasses import replace
        self.generation=replace(self.generation,max_iterations=4)
        result,transport = self.run_loop([patch_one,REVIEW,REVIEW],resume=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,1)
        context = json.loads(transport.requests[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(context["feedback"]["numerical"]["checks"][0]["detail"],"fixture lug penetration")
        self.assertEqual(context["current_model_py"],TOOL_SOURCE)
        image_parts = [x for x in transport.requests[0]["messages"][1]["content"] if x["type"] == "image_url"]
        self.assertEqual(len(image_parts),1)  # 合成样例两张图像素完全相同，传一次并保留两个标签。
        aliases = [path for paths in context["image_aliases"].values() for path in paths]
        self.assertEqual(len(aliases), 2)
        self.assertEqual(len(set(aliases)), 2)

    def test_deferred_asset_reaches_assembly_and_scene_must_resolve_it(self):
        deferred = {"verdict":"defer","findings":[],"unverified":[],
                    "deferred":[{"stage":"scene","detail":"组合装配后的夹具间隙"},
                                {"stage":"operation","detail":"真实抓取保持力"}]}
        resolved = {**REVIEW,"deferred_results":[{"id":"tool:00007_asset_review:1","status":"resolved",
                      "evidence":"合成场景视角及夹具间隙检查证据"}]}
        result,transport = self.run_loop(self.base_replies[:4]+[deferred,resolved],limits=SceneConfig(validation_mode="static"))
        self.assertEqual(result["status"],"static_candidate_ready")
        self.assertEqual(result["jobs"]["tool"],"assembly_ready")
        self.assertEqual([x["status"] for x in result["deferred_checks"]],["resolved","pending"])
        scene_context = json.loads(transport.requests[-1]["messages"][1]["content"][0]["text"])
        self.assertEqual(len(scene_context["deferred_checks"]),2)
        self.assertTrue(any("真实抓取保持力" in x for x in result["unverified"]))

    def test_missing_deferred_resolution_cannot_pass_scene(self):
        deferred = {"verdict":"defer","findings":[],"unverified":[],
                    "deferred":[{"stage":"scene","detail":"装配间隙"}]}
        result,_ = self.run_loop(self.base_replies[:4]+[deferred,REVIEW,REVIEW,REVIEW])
        self.assertEqual(result["status"],"needs_review")
        self.assertEqual(result["deferred_checks"][0]["status"],"unresolved")
        self.assertIn("框架记录",result["deferred_checks"][0]["evidence"])
        self.assertIsNotNone(result["scene_artifact"])

    def test_static_preview_keeps_other_pose_checks_pending_not_resolved(self):
        deferred = {"verdict":"defer","findings":[],"unverified":[],
                    "deferred":[{"stage":"scene","detail":"工具抓取时的配合姿态"}]}
        review = {**REVIEW,"deferred_results":[{"id":"tool:00007_asset_review:1","status":"out_of_scope",
                 "target_stage":"static_fit","evidence":"当前悬停，只验证初始几何布局"}]}
        result,_ = self.run_loop(self.base_replies[:4]+[deferred,review],limits=SceneConfig(validation_mode="static"))
        self.assertEqual(result["status"],"static_preview_ready")
        self.assertEqual(result["deferred_checks"][0]["status"],"pending")
        self.assertEqual(result["deferred_checks"][0]["stage"],"static_fit")
        self.assertEqual(result["deferred_checks"][0]["deferred_from"],"scene")
        self.assertTrue(any("工具抓取" in text for text in result["unverified"]))
        self.assertEqual(result["verification_scope"],"initial_static_layout")

    def test_deferred_instrument_can_feed_tool_without_false_acceptance(self):
        deferred = {"verdict":"defer","findings":[],"unverified":[],
                    "deferred":[{"stage":"scene","detail":"夹具装配配合"}]}
        resolved = {**REVIEW,"deferred_results":[{"id":"instrument:00002_asset:1","status":"resolved",
                    "evidence":"合成组合场景配合检查"}]}
        result,transport = self.run_loop(self.base_replies[:2]+[deferred]+self.base_replies[3:]+[resolved])
        self.assertEqual(result["status"],"candidate_ready")
        context = json.loads(transport.requests[3]["messages"][1]["content"][0]["text"])
        self.assertEqual(context["dependencies"]["instrument"]["status"],"assembly_ready")
        self.assertEqual(len(context["dependencies"]["instrument"]["deferred_checks"]),1)

    def test_resume_uncertain_reviews_existing_geometry_only(self):
        uncertain = {"verdict":"uncertain","findings":[],"unverified":[],"blocking":["部件图像不清楚"]}
        result,_ = self.run_loop(self.base_replies[:4]+[uncertain])
        self.assertEqual(result["status"],"needs_review")
        sandbox = SceneSandbox()
        result,transport = self.run_loop([REVIEW,REVIEW],resume=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,0)
        self.assertEqual(len(transport.requests),2)
        self.assertTrue((self.root/"run/stages/00004_asset_review/review.json").is_file())

    def test_hard_asset_failure_does_not_reach_review_or_assembly(self):
        class Failing(SceneSandbox):
            def build_and_check(self,script,destination,asset_ids,profile):
                report,renders = super().build_and_check(script,destination,asset_ids,profile)
                report.update(status="fail",checks=[{"id":"initial_contacts","status":"fail","detail":"穿模"}])
                return report,renders
        sandbox = Failing()
        a = {"edits":[{"old":"    pass","new":"    return None"}]}
        b = {"edits":[{"old":"    return None","new":"    pass"}]}
        result,transport = self.run_loop([plan(),{"model_py":ASSET_SOURCE},a,b],sandbox=sandbox)
        self.assertEqual(result["status"],"asset_repair_budget_exhausted")
        self.assertEqual(sandbox.scene_calls,0)
        self.assertEqual(len(transport.requests),4)

    def test_device_review_has_original_pixels_and_dependency_context(self):
        from PIL import Image
        Image.new("RGB",(20,20),"red").save(self.root/"reference.png")
        self.source.write_text("### A01｜合成仪器\n![照片](reference.png)",encoding="utf-8")
        self.catalog = catalog(self.source,self.generation)
        _,transport = self.run_loop(self.base_replies+[REVIEW])
        request = transport.requests[4]["messages"][1]["content"]
        context = json.loads(request[0]["text"])
        self.assertIn("instrument",context["dependencies"])
        self.assertEqual(len(context["evidence"]["original_reference_images"]),1)
        self.assertGreaterEqual(sum(x["type"] == "image_url" for x in request),2)
        for request in transport.requests[1:]:
            self.assertIn("不需要人工批准模型自己的设计值变更",request["messages"][0]["content"])
            context = json.loads(request["messages"][1]["content"][0]["text"])
            self.assertIn("fixed_contract",context)

    def test_pause_at_asset_boundary_preserves_completed_call(self):
        root = self.root
        class Pausing(SceneSandbox):
            def build_and_check(self,*args):
                result = super().build_and_check(*args)
                (root/"run/pause.request").write_text("pause",encoding="utf-8")
                return result
        with self.assertRaisesRegex(PrepError,"已请求暂停"):
            self.run_loop(self.base_replies[:2],sandbox=Pausing())
        state = json.loads((root/"run/state.json").read_text(encoding="utf-8"))
        self.assertEqual(len(state["calls"]),2)
        self.assertTrue(all(c["status"] == "done" for c in state["calls"]))
        self.assertEqual(state["scene"]["status"],"paused")
