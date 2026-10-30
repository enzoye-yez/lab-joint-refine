"""开发者合成故障注入，不访问模型服务、不使用实验产物。"""

import json
import unittest
from unittest.mock import patch

from asset_prep.errors import ProviderError, UncertainRequest, StaleEvidence, ValidationError
from asset_prep.storage import file_hash
from scene_generation.contracts import SceneConfig
from scene_generation.review import scoped_review
import test_scene_pipeline as pipeline_tests
from test_scene_pipeline import REVIEW, TOOL_SOURCE
from scene_fixtures import SceneSandbox, plan


class SceneRecoveryTests(unittest.TestCase):
    setUp = pipeline_tests.ScenePipelineTests.setUp
    run_loop = pipeline_tests.ScenePipelineTests.run_loop

    def test_long_transport_outage_does_not_consume_geometry_iterations(self):
        from dataclasses import replace
        self.generation = replace(self.generation, max_iterations=1, max_transport_retries=30)
        outage = ProviderError("synthetic unavailable", diagnostics={"http_status":503})
        sandbox = SceneSandbox()
        with patch("asset_generation.model.time.sleep"):
            result, _ = self.run_loop(self.base_replies[:3] + [outage]*8 + self.base_replies[3:] + [REVIEW],
                                     sandbox=sandbox, limits=SceneConfig(max_iterations=1))
        self.assertEqual(result["status"], "candidate_ready")
        self.assertEqual(sandbox.build_calls, 2)
        self.assertEqual(sandbox.scene_calls, 1)
        self.assertEqual(result["requests"]["retry_attempts"], 8)
        self.assertEqual(result["requests"]["successful_responses"], 6)

    def test_provider_failure_is_reported_and_cleared_after_resume(self):
        error = ProviderError("synthetic", diagnostics={"http_status":400,
                             "provider_error":{"code":"invalid_request", "message":"synthetic rejection"}})
        with self.assertRaises(ProviderError):
            self.run_loop([error])
        result = json.loads((self.root / "run/result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["request_failure"]["diagnostics"]["http_status"], 400)
        self.assertEqual(result["request_failure"]["phase"], "scene_plan")
        result, _ = self.run_loop(self.base_replies + [REVIEW], resume=True)
        self.assertEqual(result["status"], "candidate_ready")
        self.assertNotIn("request_failure", result)

    def test_exhausted_asset_review_resumes_without_regenerating_asset(self):
        timeout = ProviderError("synthetic timeout", uncertain=True, diagnostics={"failure":"timeout"})
        with patch("asset_generation.model.time.sleep"), self.assertRaises(UncertainRequest):
            self.run_loop(self.base_replies[:2]+[timeout])
        report = json.loads((self.root/"run/result.json").read_text(encoding="utf-8"))
        self.assertEqual(report["jobs"]["instrument"],"review_pending")
        self.assertEqual(report['budget']['model_calls'],3)
        source = self.root/"run/stages/00002_asset/model.py"
        before = file_hash(source)
        sandbox = SceneSandbox()
        result,transport = self.run_loop([REVIEW,{"model_py":TOOL_SOURCE},REVIEW,REVIEW],
                                        resume=True,retry_uncertain=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,1)  # 仅新工具，已有资产不重建。
        self.assertEqual(file_hash(source),before)
        self.assertEqual(len(transport.requests),4)
        state=json.loads((self.root/"run/state.json").read_text(encoding="utf-8"))
        self.assertFalse(any(c["status"] == "pending" for c in state["calls"]))
        self.assertEqual(self.run_loop([],resume=True)[1].requests,[])

    def test_scene_review_interrupt_resumes_without_reassembly(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.base_replies+[KeyboardInterrupt()])
        sandbox=SceneSandbox()
        result,transport=self.run_loop([REVIEW],resume=True,retry_uncertain=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.scene_calls,0)
        self.assertEqual(sandbox.build_calls,0)
        self.assertEqual(len(transport.requests),1)
        self.assertTrue((self.root/"run"/result["scene_artifact"]/"model.xml").is_file())
        self.assertEqual(self.run_loop([],resume=True)[1].requests,[])

    def test_checkpoint_tampering_is_rejected_before_any_new_call(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.base_replies[:2]+[KeyboardInterrupt()])
        (self.root/"run/stages/00002_asset/model.py").write_text("# tampered",encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            self.run_loop([],resume=True,retry_uncertain=True)

    def test_repeat_review_interrupt_preserves_same_asset(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.base_replies[:2]+[KeyboardInterrupt()])
        sandbox=SceneSandbox()
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([KeyboardInterrupt()],resume=True,retry_uncertain=True,sandbox=sandbox)
        self.assertEqual(sandbox.build_calls,0)
        result,_=self.run_loop([REVIEW,{"model_py":TOOL_SOURCE},REVIEW,REVIEW],
                               resume=True,retry_uncertain=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.build_calls,1)

    def test_response_saved_before_budget_pause_is_reused(self):
        from asset_prep.errors import BudgetExceeded
        with self.assertRaises(BudgetExceeded):
            self.run_loop(self.base_replies[:3],limits=SceneConfig(max_model_calls=3))
        saved=json.loads((self.root/'run/result.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['jobs']['instrument'],'candidate_ready')
        self.assertEqual(saved['budget']['model_calls'],3)
        result,transport=self.run_loop([{"model_py":TOOL_SOURCE},REVIEW,REVIEW],resume=True)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(len(transport.requests),3)  # 已保存的审查不再次付费。
        self.assertEqual(result['budget']['model_calls'],6)

    def test_secondary_report_failure_preserves_keyboard_interrupt(self):
        from scene_generation import pipeline
        real_write=pipeline.atomic_json
        def broken_report(path,value):
            if path.name == "result.json":
                raise OSError("synthetic disk failure")
            return real_write(path,value)
        with patch.object(pipeline,"atomic_json",side_effect=broken_report):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_loop([KeyboardInterrupt()])
        self.assertIn("OSError"," ".join(caught.exception.__notes__))

    def test_planning_attempts_are_independent_of_asset_budget(self):
        with self.assertRaisesRegex(ValidationError,"场景计划格式修复次数耗尽"):
            self.run_loop([{},{}],limits=SceneConfig(max_plan_attempts=2))
        state=json.loads((self.root/"run/state.json").read_text(encoding="utf-8"))
        self.assertEqual(len(state["calls"]),2)

    def test_structured_targets_do_not_match_other_instance_suffixes(self):
        p=plan()
        for instance in p["instances"]:
            instance["id"]="micro_rack" if instance["placement"] == "robot" else "rack"
        defect={"verdict":"repair","findings":["micro_rack: synthetic defect"],"unverified":[],
                "affected_instances":["micro_rack"]}
        action={"jobs":["tool"],"placements":[],"reason":"synthetic tool repair"}
        edit={"edits":[{"old":"    pass","new":"    return None"}]}
        result,transport=self.run_loop([p,*self.base_replies[1:],defect,action,edit,REVIEW,REVIEW])
        self.assertEqual(result["status"],"candidate_ready")
        context=json.loads(transport.requests[6]["messages"][1]["content"][0]["text"])
        self.assertEqual(context["allowed_instances"],["micro_rack"])
        self.assertEqual(context["allowed_jobs"],["tool"])

    def test_unknown_or_missing_repair_targets_require_format_correction(self):
        value={"verdict":"repair","findings":["rack has a defect"],"unverified":[]}
        for affected in (None,[],["nonexistent"],["rack","rack"]):
            candidate=dict(value)
            if affected is not None:
                candidate["affected_instances"]=affected
            with self.subTest(affected=affected),self.assertRaises(ValidationError):
                scoped_review(candidate,"scene",instance_ids={"rack"})

    def test_real_local_static_workflow_and_review_recovery(self):
        from asset_generation.local import LocalBackend
        from scene_fixtures import ASSET, TOOL, manifest
        def source(xml, tool):
            return ("def build(output_dir):\n"
                    f"    (output_dir/'model.xml').write_text({xml!r},encoding='utf-8')\n"
                    f"    (output_dir/'interfaces.json').write_text({json.dumps(manifest(tool))!r},encoding='utf-8')\n")
        backend=LocalBackend(self.generation,self.project)
        limits=SceneConfig(validation_mode="static")
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([plan(),{"model_py":source(ASSET,False)},REVIEW,
                           {"model_py":source(TOOL,True)},REVIEW,KeyboardInterrupt()],
                          sandbox=backend,limits=limits)
        # 真正完成本地构建、检查、渲染、装配后，恢复只允许审查。
        with patch.object(backend,"build_and_check",side_effect=AssertionError("must reuse assets")), \
             patch.object(backend,"assemble_and_check",side_effect=AssertionError("must reuse scene")):
            result,_=self.run_loop([REVIEW],resume=True,retry_uncertain=True,sandbox=backend,limits=limits)
        self.assertEqual(result["status"],"static_candidate_ready")
        from asset_generation.checks import load_model
        artifact=self.root/"run"/result["scene_artifact"]
        model=load_model(artifact,scope="scene")
        self.assertGreater(model.ngeom,0)
        self.assertFalse(list((self.root/"run").rglob("local-worker-*")))
