import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded, ProviderError, StaleEvidence, UncertainRequest
from asset_generation.contracts import GenerationConfig
from asset_generation.inputs import load_package
from asset_generation.pipeline import execute
from generation_fixtures import SOURCE, PROVENANCE, FakeTransport, FixtureSandbox, package
from test_generation_checks import dimension_profile

REVIEW = {"verdict": "accept", "motion_coverage": [], "findings": [], "unverified": ["机器人执行未验证"]}


class GenerationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.generation = GenerationConfig()
        self.package = package(self.root)
        self.bundle = load_package(self.package, ["A01"], self.generation)
        self.run = self.root/"run"
        self.project = Path(__file__).resolve().parents[1]

    def run_loop(self, replies, *, sandbox=None, resume=False, generation=None, retry_uncertain=False):
        transport = FakeTransport(replies)
        # 基准资料包路径绝不能触发第一模块；全部闭环测试都带这个隔离断言。
        with patch("asset_prep.pipeline.execute", side_effect=AssertionError("Benchmark must bypass asset preparation")), \
             patch("asset_prep.extraction.extract_inventory", side_effect=AssertionError("Benchmark must bypass extraction")), \
             patch("asset_prep.search.Search", side_effect=AssertionError("Benchmark must bypass web search")):
            result = execute(self.bundle, dimension_profile(), Config(), generation or self.generation,
                             {"HARNESS_MODEL_API_KEY": "fixture-only-not-real"}, self.run, transport,
                             sandbox or FixtureSandbox(), self.project, resume=resume,
                             retry_uncertain=retry_uncertain, provenance=PROVENANCE)
        return result, transport

    def test_complete_loop_uses_pixels_and_records_models(self):
        result, transport = self.run_loop([{"model_py": SOURCE}, REVIEW])
        self.assertEqual(result["status"], "candidate_ready")

        self.assertEqual(result["provenance"], PROVENANCE)
        self.assertEqual(result["budget"]["model_calls"], 2)
        self.assertEqual(transport.urls, [Config().base_url + "/chat/completions"] * 2)
        self.assertEqual([request["model"] for request in transport.requests], [Config().model] * 2)
        self.assertTrue(any(p["type"] == "image_url" for p in transport.requests[0]["messages"][1]["content"]))
        ledger = json.loads((self.run/"state.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["generation"]["revisions"][0]["author"]["response_model"], "developer-fixture-model")
        self.assertNotIn("fixture-only-not-real", (self.run/"state.json").read_text(encoding="utf-8"))

    def test_review_interruption_reuses_checked_geometry(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([{"model_py":SOURCE},KeyboardInterrupt()])
        sandbox=FixtureSandbox()
        result,transport=self.run_loop([REVIEW],resume=True,retry_uncertain=True,sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.calls,0)
        self.assertEqual(len(transport.requests),1)
        self.assertEqual(self.run_loop([],resume=True)[1].requests,[])

    def test_bad_dimension_causes_scoped_patch(self):
        result, transport = self.run_loop([{"model_py": SOURCE},
            {"edits": [{"old": "    pass", "new": "    return None"}]}, REVIEW], sandbox=FixtureSandbox(fail_first=True))
        self.assertEqual(result["status"], "candidate_ready")
        self.assertEqual(result["iterations"], 2)
        context = json.loads(transport.requests[1]["messages"][1]["content"][0]["text"])
        self.assertIn("actual_m", json.dumps(context["feedback"]))
        self.assertEqual(context["current_model_py"], SOURCE)

    def test_resume_finished_run_makes_no_calls(self):
        self.run_loop([{"model_py": SOURCE}, REVIEW])
        result, transport = self.run_loop([], resume=True)
        self.assertEqual(result["status"], "candidate_ready")
        self.assertEqual(transport.requests, [])

    def test_tampered_version_is_rejected(self):
        self.run_loop([{"model_py": SOURCE}, REVIEW])
        (self.run/"revisions/0001/model.py").write_text("changed")
        with self.assertRaises(StaleEvidence):
            self.run_loop([], resume=True)

    def test_tampered_check_snapshot_is_rejected(self):
        self.run_loop([{"model_py": SOURCE}, REVIEW])
        (self.run/"checks.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            self.run_loop([], resume=True)

    def test_budget_pause_keeps_cached_generation(self):
        with self.assertRaises(BudgetExceeded):
            self.run_loop([{"model_py": SOURCE}], generation=replace(self.generation, max_model_calls=1))
        result, transport = self.run_loop([REVIEW], resume=True)
        self.assertEqual(result["status"], "candidate_ready")
        self.assertEqual(len(transport.requests), 1)

    def test_uncertain_request_requires_explicit_retry(self):
        with self.assertRaises(UncertainRequest):
            self.run_loop([ProviderError("synthetic connection loss", uncertain=True)])
        with self.assertRaises(UncertainRequest):
            self.run_loop([], resume=True)
        result, _ = self.run_loop([{"model_py": SOURCE}, REVIEW], resume=True, retry_uncertain=True)
        self.assertEqual(result["budget"]["model_calls"], 3)

    def test_review_uncertainty_is_not_success(self):
        result, _ = self.run_loop([{"model_py": SOURCE}, {"verdict": "uncertain", "findings": [], "unverified": ["缺少视角"]}])
        self.assertEqual(result["status"], "needs_review")

    def test_accept_with_findings_retries_review_without_rebuilding(self):
        sandbox = FixtureSandbox()
        invalid = {"verdict":"accept", "motion_coverage": [],"findings":["外形一致"],"unverified":[]}
        result, transport = self.run_loop([{"model_py":SOURCE},invalid,REVIEW],sandbox=sandbox)
        self.assertEqual(result["status"],"candidate_ready")
        self.assertEqual(sandbox.calls,1)
        self.assertEqual(result["budget"]["model_calls"],3)
        correction = json.loads(transport.requests[2]["messages"][1]["content"][0]["text"])
        self.assertEqual(correction["review_format_correction"]["previous_response"],invalid)

    def test_repeated_invalid_review_preserves_asset_and_requires_review(self):
        sandbox = FixtureSandbox()
        invalid = {"verdict":"accept", "motion_coverage": [],"findings":["存在真实缺陷"],"unverified":[]}
        result,_ = self.run_loop([{"model_py":SOURCE},invalid,invalid,invalid],sandbox=sandbox)
        self.assertEqual(result["status"],"needs_review")
        self.assertEqual(sandbox.calls,1)
        self.assertEqual(len(list((self.run/"revisions/0001").glob("review_format_*.json"))),3)

    def test_invalid_code_never_reaches_builder(self):
        sandbox = FixtureSandbox()
        result, _ = self.run_loop([{"model_py": "def ???"}], sandbox=sandbox,
                                   generation=replace(self.generation, max_iterations=1))
        self.assertEqual(result["status"], "repair_budget_exhausted")
        self.assertEqual(sandbox.calls, 0)

    def test_changed_input_cannot_reuse_old_evidence(self):
        self.run_loop([{"model_py": SOURCE}, REVIEW])
        self.package.write_text(self.package.read_text(encoding="utf-8")+"新要求", encoding="utf-8")
        self.bundle = load_package(self.package, ["A01"], self.generation)
        with self.assertRaises(StaleEvidence):
            self.run_loop([], resume=True)


if __name__ == "__main__":
    unittest.main()
