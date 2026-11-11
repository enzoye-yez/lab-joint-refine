"""整个工程链的合成测试；不证明正式模型质量或物理可执行性。"""

import json
from dataclasses import replace

from asset_prep.config import Config
from asset_prep.errors import BudgetExceeded, ProviderError, StaleEvidence, UncertainRequest, ValidationError
from asset_prep.pipeline import execute
from fixtures import PROVENANCE, FakeTransport, documents, inventory, selection
from test_boundaries import TempCase


class PipelineTests(TempCase):
    keys = {"HARNESS_MODEL_API_KEY": "fake-model-key", "TAVILY_API_KEY": "fake-search-key"}

    def run_pipeline(self, transport, config=None, **kwargs):
        # 保留旧 Tavily 路径的回归；默认模型联网路径由 test_search.py 覆盖。
        config = replace(config or Config(), search_provider="tavily")
        return execute(documents(), config, kwargs.pop("keys", self.keys), self.root,
                       transport, provenance=PROVENANCE, **kwargs)

    def test_end_to_end_and_resume_no_duplicate_requests(self):
        transport = FakeTransport()
        package = self.run_pipeline(transport)
        self.assertEqual(package["provenance"], PROVENANCE)
        self.assertEqual(package["assets"][0]["selection"]["mode"], "reference_adapted")
        self.assertTrue(package["assets"][0]["downloaded_images"])
        self.assertFalse(package["checks"]["geometry_verified"])
        self.assertFalse(package["checks"]["execution_verified"])
        self.assertEqual(transport.model_calls, 2)
        resumed = FakeTransport([])
        self.run_pipeline(resumed, resume=True)
        self.assertEqual((resumed.model_calls, resumed.search_calls, resumed.image_calls), (0, 0, 0))

    def test_search_missing_keeps_generation_requirements(self):
        package = self.run_pipeline(FakeTransport(), keys={"HARNESS_MODEL_API_KEY": "fake"})
        item = package["assets"][0]
        self.assertEqual(item["selection"]["mode"], "requirements_only")
        self.assertEqual(item["target"]["quantity"]["value"], 2)
        self.assertEqual(package["status"], "completed_with_gaps")

    def test_budget_stop_then_resume_with_larger_limit(self):
        with self.assertRaises(BudgetExceeded):
            self.run_pipeline(FakeTransport(), replace(Config(), max_model_calls=1))
        partial = json.loads((self.root / "asset_package.json").read_text(encoding="utf-8"))
        self.assertEqual(partial["status"], "paused")
        resumed = FakeTransport([selection()])
        package = self.run_pipeline(resumed, resume=True)
        self.assertEqual(package["budget"]["model_calls"], 2)
        self.assertEqual(resumed.model_calls, 1)

    def test_invalid_model_output_repaired_and_charged(self):
        transport = FakeTransport([{}, inventory(), selection()])
        result = self.run_pipeline(transport)
        self.assertEqual(result["budget"]["model_calls"], 3)

    def test_selection_failure_falls_back_not_fake_success(self):
        transport = FakeTransport([inventory(), {}, {}, {}])
        result = self.run_pipeline(transport)
        self.assertEqual(result["assets"][0]["selection"]["mode"], "requirements_only")

    def test_auth_failure_does_not_loop_forever(self):
        transport = FakeTransport([ProviderError("HTTP 401")])
        with self.assertRaises(ProviderError):
            self.run_pipeline(transport)
        self.assertEqual(transport.model_calls, 1)

    def test_timeout_is_uncertain_not_automatically_retried(self):
        transport = FakeTransport([ProviderError("timeout", uncertain=True)])
        with self.assertRaises(UncertainRequest):
            self.run_pipeline(transport)
        state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["calls"][0]["status"], "pending")
        retry = FakeTransport()
        with self.assertRaises(UncertainRequest):
            self.run_pipeline(retry, resume=True)
        self.assertEqual(retry.model_calls, 0)
        result = self.run_pipeline(retry, resume=True, retry_uncertain=True)
        self.assertEqual(result["budget"]["model_calls"], 3)

    def test_budget_before_reference_phase_keeps_all_targets(self):
        with self.assertRaises(BudgetExceeded):
            self.run_pipeline(FakeTransport(), replace(Config(), max_model_calls=1))
        partial = json.loads((self.root / "asset_package.json").read_text(encoding="utf-8"))
        self.assertEqual(len(partial["assets"]), 1)
        self.assertEqual(partial["assets"][0]["target"]["quantity"]["value"], 2)
        resumed = FakeTransport([selection()])
        result = self.run_pipeline(resumed, resume=True)
        self.assertEqual(resumed.search_calls, 1)
        self.assertEqual(result["status"], "completed_with_gaps")

    def test_edited_package_invalidates_resume(self):
        self.run_pipeline(FakeTransport())
        (self.root / "asset_package.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            self.run_pipeline(FakeTransport([]), resume=True)

    def test_changed_model_invalidates_resume(self):
        self.run_pipeline(FakeTransport())
        with self.assertRaises(StaleEvidence):
            self.run_pipeline(FakeTransport([]), replace(Config(), model="different"), resume=True)

    def test_tampered_image_invalidates_resume(self):
        result = self.run_pipeline(FakeTransport())
        image = result["assets"][0]["downloaded_images"][0]
        (self.root / image["local_path"]).write_bytes(b"changed")
        with self.assertRaises(StaleEvidence):
            self.run_pipeline(FakeTransport([]), resume=True)

    def test_image_failure_keeps_targets_and_sources(self):
        transport = FakeTransport()
        def fail_download(*args, **kwargs):
            raise ProviderError("HTTP 403")
        transport.request = fail_download
        result = self.run_pipeline(transport)
        item = result["assets"][0]
        self.assertEqual(item["target"]["quantity"]["value"], 2)
        self.assertTrue(item["references"]["images"])
        self.assertFalse(item["downloaded_images"])
        self.assertEqual(item["status"], "prepared_with_gaps")

    def test_cannot_claim_formal_model_source_for_fixtures(self):
        result = self.run_pipeline(FakeTransport())
        self.assertEqual(result["returned_models"], ["fixture-model-not-a-real-provider"])
        self.assertEqual(result["provenance"], "developer_authored_test_fixture")

    def test_input_limit_prevents_hidden_truncation_and_charges_nothing(self):
        with self.assertRaises(ValidationError):
            self.run_pipeline(FakeTransport(), replace(Config(), max_input_bytes=10))
        state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["calls"], [])
