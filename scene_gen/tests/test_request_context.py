"""Developer-authored context regression tests. No network or formal model calls."""
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from asset_prep.config import Config
from asset_prep.storage import Store, encoded, digest
from asset_prep.errors import ContextLimitError, ValidationError, StaleEvidence, UncertainRequest, BudgetExceeded
from asset_generation.contracts import GenerationConfig
from asset_generation.request_context import RequestContext, measure
from asset_generation.model import GenerationModel


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = GenerationConfig()
        self.identity = {"implementation": "developer_fixture", "model": "fixture"}

    def view(self, context, images=(), **kwargs):
        return RequestContext(self.root, self.identity, "developer fixture",
                              context, list(images), "scene_repair", replace(self.config, **kwargs))

    def large(self):
        return {"plan": {"instances": [{"id": "tube", "position": [1, 2, 3]}]},
                "fixed_contract": {"tolerance_m": 0.0005},
                "feedback": {"status": "fail", "checks": [
                    {"id": "support:tube", "status": "fail", "complete": False, "instances": ["tube"],
                     "component_scope": "whole_instance", "unexamined_pairs": 2,
                     "details": [{"gap": -0.008, "name": str(i)} for i in range(12000)]}]}}

    def test_discarded_projection_branches_do_not_create_orphan_files(self):
        view=self.view({'job':'fixture'})
        before=set(view.root.glob('*.json'))
        value=[{'detail':str(i)+'x'*1000} for i in range(200)]
        folded=view.fold(value,200)
        self.assertEqual(len(set(view.root.glob('*.json'))-before),1)
        page=view.read({'reference':folded['context_ref'],'path':[199,'detail'],'limit':1024})
        self.assertEqual(page['data'],value[199]['detail'])

    def test_historical_zip_reads_exact_bytes_and_rejects_tampering(self):
        import zipfile
        from asset_generation.request_context import reference_bytes
        view=self.view(self.large())
        source=view.root/(view.source_ref+'.json')
        raw=source.read_bytes()
        archive=view.root.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as packed:
            packed.writestr(source.name,raw)
        source.unlink()
        self.assertEqual(reference_bytes(view.root,view.source_ref),raw)
        self.assertEqual(view.read({'reference':view.source_ref,'path':['fixed_contract']})['data'][0]['value'],.0005)
        with zipfile.ZipFile(archive,'w') as packed:packed.writestr(source.name,b'{}')
        with self.assertRaises(StaleEvidence):view.read({'reference':view.source_ref})

    def test_realistic_repair_report_preserves_failures_and_exact_details(self):
        original = self.large()
        before = deepcopy(original)
        v = self.view(original)
        self.assertEqual(original, before)
        self.assertLess(v.after["required"], v.before["required"])
        self.assertEqual(v.context["plan"], original["plan"])
        self.assertEqual(v.context["fixed_contract"], original["fixed_contract"])
        row = v.context["_context"]["check_index"][0]
        for key in ("id", "status", "complete", "instances", "component_scope", "unexamined_pairs"):
            self.assertEqual(row[key], original["feedback"]["checks"][0][key])
        page = v.read({"reference": row["context_ref"], "path": ["details"], "offset": 11999})
        self.assertEqual(page["data"][0], {"gap": -0.008, "name": "11999"})
        self.assertIsNone(page["next_offset"])

    def test_compaction_keeps_author_capabilities_and_recovery_instruction(self):
        context=self.large()
        context.update(authoring_capabilities={"packages":{"scipy":"developer_fixture"},"max_files":512},
                       task_recovery={"instruction":"keep current source and required interfaces"})
        view=self.view(context)
        self.assertEqual(view.context["authoring_capabilities"],context["authoring_capabilities"])
        self.assertEqual(view.context["task_recovery"],context["task_recovery"])

    def test_all_pending_item_ids_and_exact_source_survive(self):
        c = self.large()
        c.update(current_model_py="# exact Unicode source 中文\n" * 5000,
                 deferred_checks=[{"id": str(i), "stage": "scene", "detail": "must verify"} for i in range(200)])
        v = self.view(c)
        self.assertEqual(v.context["current_model_py"], c["current_model_py"])
        self.assertEqual(v.context["deferred_checks"], c["deferred_checks"])

    def test_oversized_mandatory_source_is_not_silently_cut(self):
        with self.assertRaises(ContextLimitError):
            self.view({"current_model_py": "x" * 1000000})
        receipt = json.loads(next(self.root.rglob("receipt.json")).read_text("utf-8"))
        self.assertEqual(receipt["status"], "blocked_mandatory_context_or_images")

    def test_every_distinct_image_retained_duplicate_pixels_have_aliases(self):
        images = [{"path": "a.png", "data_url": "data:A", "sha256": "A"},
                  {"path": "b.png", "data_url": "data:A", "sha256": "A"},
                  {"path": "c.png", "data_url": "data:C", "sha256": "C"}]
        v = self.view({}, images)
        self.assertEqual([im["data_url"] for im in v.images], ["data:A", "data:C"])
        self.assertEqual(v.context["image_aliases"]["a.png"], ["a.png", "b.png"])
        paged = self.view({}, [{"path": str(i), "data_url": str(i)} for i in range(54)])
        self.assertEqual(len(paged.all_images),54)
        self.assertEqual(len(paged.image_pages),3)
        self.assertTrue(all(len(p)<=self.config.max_images for p in paged.image_pages))

    def test_references_are_scoped_and_hash_checked(self):
        v = self.view(self.large())
        other = RequestContext(self.root, {"implementation": "changed"}, "fixture",
                               {"tiny": True}, [], "author", self.config)
        with self.assertRaises(ValidationError):
            other.read({"reference": v.source_ref})
        with self.assertRaises(ValidationError):
            v.read({"reference": "../state"})
        with self.assertRaises(ValidationError):
            v.read({"reference": v.source_ref, "path": ["feedback", "checks", -1]})
        (v.root / (v.source_ref + ".json")).write_text("{}", encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            v.read({"reference": v.source_ref})

    def test_current_evidence_replaces_full_transcript_without_erasing_failures(self):
        c = self.large()
        c["supplemental_evidence"] = [
            {"inventory": {"large": "x" * 100000}, "results": [
                {"request": {"kind": "support", "instance": "tube"}, "status": "incomplete",
                 "complete": False, "error": "measurement_error", "details": "y" * 10000}]}] * 5
        v = self.view(c)
        self.assertNotIn("x" * 1000, encoded(v.context).decode())
        self.assertEqual(v.context["_context"]["check_index"][0]["status"], "fail")
        saved = json.loads((v.root / (v.source_ref + ".json")).read_text("utf-8"))
        self.assertEqual(saved, c)

    def model(self, transport, config=None, calls=20):
        store = Store(self.root / "model", self.identity, Config(max_model_calls=calls, max_reserved_tokens=10000000))
        return GenerationModel(Config(), config or self.config, {"HARNESS_MODEL_API_KEY": "synthetic"},
                               store, transport)

    def test_detail_reads_share_ledger_replace_pages_and_resume_from_cache(self):
        outer = self
        class Transport:
            seen = []
            def post(self, url, payload, key, timeout):
                context = json.loads(payload["messages"][1]["content"][0]["text"])
                self.seen.append(context)
                if len(self.seen) == 1:
                    answer = {"context_read": {"reference": context["_context"]["source_ref"],
                                              "path": ["feedback", "checks", 0, "details"],
                                              "offset": 11999, "limit": 1}}
                else:
                    outer.assertEqual(context["context_page"]["data"][0]["gap"], -0.008)
                    answer = {"action": "repair", "synthetic": True}
                return {"model": "fixture", "choices": [{"message": {"content": json.dumps(answer)}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        t = Transport()
        m = self.model(t)
        result, call = m.ask("fixture", self.large(), [], "scene_repair")
        self.assertEqual(result["action"], "repair")
        self.assertEqual(len(m.store.state["calls"]), 2)
        self.assertTrue(all(c["requested_model"] == m.config.model for c in m.store.state["calls"]))
        self.assertEqual(m.ask("fixture", self.large(), [], "scene_repair")[0], result)
        self.assertEqual(len(t.seen), 2)
        self.assertEqual(len(m.store.state["calls"]), 2)

    def test_repeated_pages_only_defer_after_stalled_window_and_decision_prompt(self):
        class Transport:
            def post(self, url, payload, key, timeout):
                c = json.loads(payload["messages"][1]["content"][0]["text"])
                answer = {"context_read": {"reference": c["_context"]["source_ref"], "limit": 1}}
                return {"choices": [{"message": {"content": json.dumps(answer)}}]}
        m = self.model(Transport())
        with self.assertRaisesRegex(ContextLimitError, "keep adding nothing new"):
            m.ask("fixture", self.large(), [], "author")
        self.assertEqual(len(m.store.state["calls"]), 10)
        receipt=json.loads(next((m.store.root/"request_context").rglob("receipt.json")).read_text("utf-8"))
        self.assertEqual(receipt["status"],"detail_task_deferred")
        self.assertEqual(sum(r["reused"] for r in receipt["reads"]),8)

    def test_unknown_paid_request_cannot_be_bypassed_by_compaction(self):
        t = SimpleNamespace(post=lambda *a: self.fail("must not send"))
        m = self.model(t)
        m.store.state["calls"].append({"status": "pending", "phase": "author", "reserved_tokens": 1000})
        with self.assertRaises(UncertainRequest):
            m.ask("fixture", self.large(), [], "scene_repair")
        self.assertEqual(m.store.state["calls"][0]["reserved_tokens"], 1000)

    def test_detail_read_cannot_bypass_total_call_budget(self):
        class Transport:
            def post(self, url, payload, key, timeout):
                c = json.loads(payload["messages"][1]["content"][0]["text"])
                return {"choices": [{"message": {"content": json.dumps({
                    "context_read": {"reference": c["_context"]["source_ref"], "limit": 1}})}}]}
        m = self.model(Transport(), calls=1)
        with self.assertRaises(BudgetExceeded):
            m.ask("fixture", self.large(), [], "scene_repair")
        self.assertEqual(len(m.store.state["calls"]), 1)

    def test_text_and_image_metrics_do_not_claim_exact_tokens(self):
        v = self.view({"text": "中文"})
        self.assertFalse(v.after["exact_token_count"])
        self.assertEqual(v.after["required"], v.after["input_upper_estimate"] + 16000 + 32768)

    def test_dependency_contract_interfaces_and_coordinates_stay_inline(self):
        c = self.large()
        dep = {"manifest": {"asset_ids": ["A01"], "interfaces": [{"id": "hole", "size": [0.01, 0.02]}],
                            "assumptions": ["Unverified material"], "connections": [{"id": "hinge"}]},
               "geometry": {"unit": "m", "interfaces": [{"name": "hole", "position": [0.1, 0.2, 0.3]}],
                            "bodies": [{"name": str(i), "bounds": [0, 1]} for i in range(3000)]},
               "instances": [{"id": "rack", "position": [1, 2, 3], "yaw": 0.5}],
               "version": "v1", "source_requirements": "Do not change hole size", "status": "unassessed"}
        c["dependencies"] = {"rack_job": dep}
        v = self.view(c)
        kept = v.context["dependencies"]["rack_job"]
        self.assertEqual(kept["manifest"], dep["manifest"])
        self.assertEqual(kept["instances"], dep["instances"])
        self.assertEqual(kept["geometry"]["interfaces"], dep["geometry"]["interfaces"])
        self.assertEqual(kept["source_requirements"], dep["source_requirements"])
        page = v.read({"reference": kept["geometry"]["bodies"]["context_ref"], "offset": 2999})
        self.assertEqual(page["data"], [{"name": "2999", "bounds": [0, 1]}])

    def test_all_generation_phases_use_same_request_manager(self):
        outer = self
        class Transport:
            def post(self, url, payload, key, timeout):
                c = json.loads(payload["messages"][1]["content"][0]["text"])
                outer.assertIn("_context", c)
                outer.assertEqual(c["_context"]["check_index"][0]["status"], "fail")
                return {"choices": [{"message": {"content": "{}"}}]}
        m = self.model(Transport())
        for phase in ("scene_plan", "author", "asset:fixture", "visual_review",
                      "asset_review:rack", "scene_review", "scene_review:evidence_plan", "scene_repair"):
            m.ask("fixture", self.large(), [], phase)
        self.assertEqual(len(m.store.state["calls"]), 8)

    def test_detail_round_limit_does_not_reset_or_clear_reservations(self):
        class Transport:
            count = 0
            def post(self, url, payload, key, timeout):
                c = json.loads(payload["messages"][1]["content"][0]["text"])
                self.count += 1
                answer = {"context_read": {"reference": c["_context"]["source_ref"],
                          "path": ["feedback", "checks", 0, "details"], "offset": self.count, "limit": 1}}
                return {"choices": [{"message": {"content": json.dumps(answer)}}]}
        m = self.model(Transport(), config=replace(self.config, max_context_reads=1), calls=4)
        with self.assertRaises(BudgetExceeded):
            m.ask("fixture", self.large(), [], "scene_review")
        self.assertEqual(len(m.store.state["calls"]), 4)
        self.assertTrue(all(c["reserved_tokens"] > 0 for c in m.store.state["calls"]))

    def test_cancel_after_detail_response_sends_no_followup(self):
        outer = self
        class Transport:
            def post(self, url, payload, key, timeout):
                c = json.loads(payload["messages"][1]["content"][0]["text"])
                return {"choices": [{"message": {"content": json.dumps({
                    "context_read": {"reference": c["_context"]["source_ref"], "limit": 1}})}}]}
        m = self.model(Transport())
        def guard():
            if m.store.state["calls"] and m.store.state["calls"][-1]["status"] == "done":
                raise KeyboardInterrupt()
        m.execution_guard = guard
        with self.assertRaises(KeyboardInterrupt):
            m.ask("fixture", self.large(), [], "scene_review")
        self.assertEqual(len(m.store.state["calls"]), 1)
        self.assertEqual(m.store.state["calls"][0]["status"], "done")

    def test_lossless_interface_table_codec_handles_nested_and_reserved_keys(self):
        from asset_generation.request_context import pack_tables, unpack_tables
        original = [{"name": str(i), "local_position": [i * 0.1, 0, -0.003],
                     "unit": "m", "note": "中文接口", "nullable": None,
                     "extra": {"_context_encoding": "rows_v1", "columns": ["untrusted"]}}
                    for i in range(30)]
        packed = pack_tables(original)
        self.assertEqual(unpack_tables(packed), original)
        self.assertLess(len(encoded(packed)), len(encoded(original)))

    def test_small_optional_text_is_repacked_when_combined_budget_is_tight(self):
        v = self.view({"optional_details": "x" * 30000}, max_context_tokens=70000)
        self.assertTrue(v.projected)
        self.assertLessEqual(v.after["required"], 70000)

    def test_detail_page_adapts_to_remaining_bytes_without_skipping_records(self):
        v = self.view(self.large())
        request = {"reference": v.source_ref, "path": ["feedback", "checks", 0, "details"],
                   "offset": 11990, "limit": 16}
        page = v.read(request, budget=1024)
        self.assertLessEqual(len(encoded(page)), 1024)
        self.assertEqual(page["data"][0]["name"], "11990")
        self.assertEqual(len(page["data"]), min(10, 16))
        text = "中文🙂" * 10000
        ref = v.reference(text)
        page = v.read({"reference": ref, "limit": 8192}, budget=2048)
        self.assertLessEqual(len(encoded(page)), 2048)
        self.assertEqual(page["data"], text[:page["next_offset"]])

    def test_working_notes_carry_prior_page_facts_without_replaying_transcript(self):
        outer = self
        class Transport:
            seen = []
            def post(self, url, payload, key, timeout):
                c = json.loads(payload["messages"][1]["content"][0]["text"])
                self.seen.append(c)
                reference = c["_context"]["source_ref"]
                if len(self.seen) == 1:
                    answer = {"context_read": {"reference": reference,
                              "path": ["feedback", "checks", 0, "details"], "offset": 0, "limit": 1}}
                elif len(self.seen) == 2:
                    answer = {"context_read": {"reference": reference,
                              "path": ["feedback", "checks", 0, "details"], "offset": 1, "limit": 1},
                              "context_notes": [{"reference": reference, "text": "First component gap is -0.008 m"}]}
                else:
                    outer.assertEqual(c["context_page"]["data"][0]["name"], "1")
                    outer.assertEqual(c["context_working_notes"]["items"][0]["text"], "First component gap is -0.008 m")
                    outer.assertEqual(c["context_working_notes"]["provenance"], "model_working_notes_not_verified_evidence")
                    outer.assertEqual(len(c["context_page"]["data"]), 1)
                    answer = {"result": "developer fixture only"}
                return {"choices": [{"message": {"content": json.dumps(answer)}}]}
        t = Transport()
        m = self.model(t)
        first = m.ask("fixture", self.large(), [], "scene_repair")[0]
        self.assertEqual(m.ask("fixture", self.large(), [], "scene_repair")[0], first)
        self.assertEqual(len(t.seen), 3)

    def test_working_notes_are_bounded_and_cannot_reference_other_inputs(self):
        v = self.view(self.large())
        with self.assertRaises(ValidationError):
            v.notes([{"reference": "a" * 64, "text": "untrusted"}])
        with self.assertRaises(ValidationError):
            v.notes([{"reference": v.source_ref, "text": "x" * 2000}] * 5)
        with self.assertRaises(ValidationError):
            v.notes([{"reference": v.source_ref, "text": "changed", "status": "pass"}])
        original = self.large()
        self.assertEqual(v.original, original)


if __name__ == "__main__":
    unittest.main()

