"""模型联网与静态产品页的合成测试；所有内容由开发者编写，不调用真实服务。"""

import copy
import json
from dataclasses import replace
from unittest.mock import patch

from asset_prep.config import Config
from asset_prep.contracts import validate_selection
from asset_prep.errors import BudgetExceeded, ProviderError, StaleEvidence, UncertainRequest, ValidationError
from asset_prep.network import Transport
from asset_prep.pages import PageFetcher, parse_page
from asset_prep.pipeline import execute
from asset_prep.providers import combine_references
from asset_prep.search import ModelWebSearch, Search, normalize_model_search
from asset_prep.storage import Store
from fixtures import PROVENANCE, FakeTransport, documents, inventory, references, selection
from test_boundaries import TempCase


def native_response():
    # 模拟实际探测的最小 trace 结构；无 action/sources 字段，不能要求供应商必填。
    return {"model": "fixture-search-model", "status": "completed", "usage": {"total_tokens": 12000},
            "output": [{"id": "ws_fixture", "type": "web_search_call", "status": "completed"},
                       {"type": "message", "content": [{"type": "output_text",
                         "text": "Unverified model summary says capacity 999 mL. https://invented.example/not-cited",
                         "annotations": [{"type": "url_citation", "url": "https://manufacturer.example/tube",
                                          "title": "Tube X"}]}]}]}


class NativeTransport(FakeTransport):
    def __init__(self, replies=None, search_replies=None, html=None):
        super().__init__(replies)
        self.search_replies = list(search_replies) if search_replies is not None else [native_response()]
        self.page_requests, self.native_auth = [], []
        self.html = html if html is not None else '<html><head><meta property="og:image" content="/tube.png"></head><body>Tube X capacity 0.5 mL</body></html>'

    def post(self, url, payload, key, timeout):
        if url.endswith("/responses"):
            self.search_calls += 1
            self.requests.append((url, copy.deepcopy(payload)))
            self.native_auth.append(key)
            if not self.search_replies:
                raise AssertionError("Unexpected search request")
            result = self.search_replies.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return super().post(url, payload, key, timeout)

    def request(self, url, **kwargs):
        if kwargs.get("accept", "").startswith("text/html"):
            self.page_requests.append((url, kwargs))
            if isinstance(self.html, BaseException):
                raise self.html
            return self.html.encode("utf-8"), "text/html; charset=utf-8"
        return super().request(url, **kwargs)


class SearchTests(TempCase):
    def adapter(self, config=None, transport=None):
        config = config or Config()
        store = Store(self.root, {}, config)
        transport = transport or NativeTransport()
        return Search(config, {"HARNESS_MODEL_API_KEY": "fixture-key"}, store, transport), store, transport

    def test_default_uses_native_endpoint_and_only_model_key(self):
        adapter, store, transport = self.adapter()
        self.assertIsInstance(adapter, ModelWebSearch)
        result = adapter.query("tube")
        self.assertTrue(transport.requests[0][0].endswith("/responses"))
        request = transport.requests[0][1]
        self.assertEqual(request["tools"], [{"type": "web_search"}])
        self.assertEqual(request["max_tool_calls"], 1)
        self.assertEqual(request["model"], "GPT-6-astra")
        self.assertEqual(transport.native_auth, ["fixture-key"])
        self.assertEqual(result["sources"][0]["content"], "")
        self.assertEqual(store.totals()["model_calls"], 1)
        self.assertEqual(store.totals()["search_calls"], 1)
        self.assertEqual(store.totals()["reported_total_tokens"], 12000)

    def test_summary_is_not_source_text_and_body_links_ignored(self):
        result = normalize_model_search(native_response(), "tube", 5)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["content"], "")
        self.assertFalse(result["search_records"][0]["summary_is_source_text"])
        bad = selection()
        bad["image_ids"] = []
        bad["specifications"][0]["quote"] = "capacity 999 mL"
        with self.assertRaises(ValidationError):
            validate_selection(bad, inventory()["assets"][0], result)

    def test_requires_completed_trace_not_claim_or_citation_alone(self):
        for status in (None, "failed", "in_progress"):
            response = native_response()
            if status is None:
                response["output"].pop(0)
            else:
                response["output"][0]["status"] = status
            with self.subTest(status=status), self.assertRaises(ProviderError):
                normalize_model_search(response, "tube", 5)

    def test_incomplete_response_rejected(self):
        response = native_response()
        response["status"] = "incomplete"
        with self.assertRaises(ProviderError):
            normalize_model_search(response, "tube", 5)

    def test_no_annotations_does_not_scrape_answer_urls(self):
        response = native_response()
        response["output"][1]["content"][0]["annotations"] = []
        with self.assertRaises(ProviderError):
            normalize_model_search(response, "tube", 5)

    def test_action_sources_and_duplicate_citations_merge(self):
        response = native_response()
        response["output"][0]["action"] = {"sources": [{"url": "https://manufacturer.example/tube#spec"}]}
        result = normalize_model_search(response, "tube", 5)
        self.assertEqual(len(result["sources"]), 1)

    def test_invalid_citation_url_ignored(self):
        response = native_response()
        response["output"][1]["content"][0]["annotations"].insert(0, {"type": "url_citation", "url": "https://[broken"})
        result = normalize_model_search(response, "tube", 5)
        self.assertEqual(len(result["sources"]), 1)

    def test_ignored_tool_limit_stops_instead_of_more_queries(self):
        response = native_response()
        response["output"].insert(0, {"id": "extra", "type": "web_search_call", "status": "completed"})
        adapter, store, _ = self.adapter(transport=NativeTransport(search_replies=[response]))
        with self.assertRaises(ValidationError):
            adapter.query("tube")
        self.assertEqual(store.state["calls"][0]["status"], "done")

    def test_search_success_resume_without_duplicate_charge(self):
        adapter, store, transport = self.adapter()
        adapter.query("tube")
        adapter.query("tube")
        self.assertEqual(transport.search_calls, 1)
        self.assertEqual(store.totals()["model_calls"], 1)

    def test_search_reserves_both_limits_atomically(self):
        adapter, store, transport = self.adapter(config=replace(Config(), max_search_calls=1))
        store.reserve("other", "search")
        with self.assertRaises(BudgetExceeded):
            adapter.query("tube")
        self.assertEqual(transport.search_calls, 0)
        self.assertEqual(store.totals()["model_calls"], 0)

    def test_search_timeout_requires_explicit_retry(self):
        adapter, store, transport = self.adapter(transport=NativeTransport(search_replies=[ProviderError("timeout", uncertain=True)]))
        with self.assertRaises(UncertainRequest):
            adapter.query("tube")
        with self.assertRaises(UncertainRequest):
            adapter.query("tube")
        self.assertEqual(transport.search_calls, 1)
        self.assertEqual(store.state["calls"][0]["status"], "pending")

    def test_more_actual_tokens_stops_and_response_can_resume(self):
        config = replace(Config(), max_reserved_tokens=30000)
        response = native_response()
        response["usage"]["total_tokens"] = 50000
        adapter, store, transport = self.adapter(config, NativeTransport(search_replies=[response]))
        with self.assertRaises(BudgetExceeded):
            adapter.query("tube")
        self.assertEqual(store.state["calls"][0]["status"], "done")
        store.config = replace(config, max_reserved_tokens=60000)
        adapter.query("tube")
        self.assertEqual(transport.search_calls, 1)
        self.assertEqual(store.totals()["accounted_tokens"], 50000)


class PageTests(TempCase):
    def fetcher(self, transport=None, config=None):
        config = config or Config()
        store = Store(self.root, {}, config)
        transport = transport or NativeTransport()
        return PageFetcher(config, {"HARNESS_MODEL_API_KEY": "fixture-key"}, store, transport), store, transport

    def test_static_text_metadata_relative_and_lazy_images(self):
        html = '<meta property="og:image" content="/product.png"><p>capacity 0.5 mL</p><img data-src="../side.jpg" alt="side"><img srcset="/small.webp 1x, /large.webp 2x">'
        result = parse_page(html, "https://example.org/products/tube")
        self.assertIn("capacity 0.5 mL", result["content"])
        self.assertEqual([i["url"] for i in result["images"]], ["https://example.org/product.png", "https://example.org/side.jpg", "https://example.org/small.webp", "https://example.org/large.webp"])

    def test_product_json_ld_without_running_scripts(self):
        html = '<script>alert("injected")</script><style>fake spec</style><script type="application/ld+json">{"@graph":[{"@type":"Product","name":"Tube X","image":[{"contentUrl":"/tube.png"}]}]}</script><p>real spec</p>'
        result = parse_page(html, "https://example.org/product")
        self.assertEqual(result["content"], "real spec")
        self.assertEqual(result["images"][0]["discovery"], "page_product_data")

    def test_invalid_json_ld_keeps_visible_text(self):
        result = parse_page('<script type="application/ld+json">{bad}</script>real', "https://example.org/")
        self.assertEqual(result["content"], "real")
        self.assertTrue(result["warnings"])

    def test_product_image_survives_many_decorations(self):
        html = ''.join(f'<img src="/decoration{i}.png">' for i in range(120))
        html += '<script type="application/ld+json">{"@type":"Product","image":"/tube.png"}</script>'
        result = parse_page(html, "https://example.org/")
        self.assertEqual(result["images"][0]["url"], "https://example.org/tube.png")
        self.assertLessEqual(len(result["images"]), 30)
        self.assertTrue(result["warnings"])

    def test_duplicate_image_upgrades_to_product_evidence(self):
        html = '<img src="/tube.png"><script type="application/ld+json">{"@type":"Product","name":"Tube X","image":"/tube.png"}</script>'
        result = parse_page(html, "https://example.org/")
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["discovery"], "page_product_data")
        self.assertEqual(result["images"][0]["description"], "Tube X")

    def test_unsafe_image_schemes_and_credentials_rejected(self):
        html = '<img src="file:///secret"><img src="data:image/png;base64,AA"><img src="javascript:bad"><img src="https://user:secret@example.org/pic">'
        self.assertEqual(parse_page(html, "https://example.org/")["images"], [])

    def test_no_key_sent_to_website_and_snapshot_cached(self):
        fetcher, store, transport = self.fetcher()
        page = fetcher.fetch("https://manufacturer.example/tube")
        self.assertEqual(page["content"], "Tube X capacity 0.5 mL")
        self.assertNotIn("key", transport.page_requests[0][1])
        fetcher.fetch("https://manufacturer.example/tube")
        self.assertEqual(len(transport.page_requests), 1)
        self.assertEqual(store.totals()["page_fetches"], 1)

    def test_page_tampering_invalidates_evidence(self):
        fetcher, store, _ = self.fetcher()
        fetcher.fetch("https://manufacturer.example/tube")
        (self.root / store.state["calls"][0]["file"]).write_text("{}", encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            fetcher.fetch("https://manufacturer.example/tube")

    def test_failed_page_keeps_citation_not_summary_as_spec(self):
        fetcher, _, _ = self.fetcher(NativeTransport(html=ProviderError("HTTP 403")))
        refs = normalize_model_search(native_response(), "tube", 5)
        fetcher.enrich(refs)
        self.assertEqual(refs["sources"][0]["content"], "")
        self.assertEqual(refs["sources"][0]["page_status"], "unavailable")
        self.assertTrue(refs["warnings"])
        self.assertFalse(refs["images"])

    def test_page_limit_is_visible_and_not_silent_truncation(self):
        fetcher, _, transport = self.fetcher(config=replace(Config(), pages_per_asset=1))
        refs = normalize_model_search(native_response(), "tube", 5)
        refs["sources"].append(dict(refs["sources"][0], id="r1", url="https://manufacturer.example/other"))
        fetcher.enrich(refs)
        self.assertEqual(len(transport.page_requests), 1)
        self.assertEqual(refs["sources"][1]["page_status"], "not_fetched")
        self.assertTrue(refs["warnings"])

    def test_private_page_blocked_without_connection(self):
        fetcher, store, _ = self.fetcher(Transport())
        with patch("asset_prep.network.public_target", side_effect=ValidationError("拒绝内网")), patch("asset_prep.network.PinnedHTTPS") as connection:
            with self.assertRaises(ValidationError):
                fetcher.fetch("https://127.0.0.1/secret")
            connection.assert_not_called()
        self.assertEqual(store.state["calls"][0]["status"], "failed")

    def test_shared_image_keeps_all_source_associations(self):
        batch1, batch2 = references(), references()
        batch2["sources"][0]["url"] = "https://manufacturer.example/other"
        combined = combine_references([batch1, batch2])
        self.assertEqual(len(combined["images"]), 1)
        self.assertEqual(combined["images"][0]["source_ids"], ["r0", "r1"])
        chosen = selection()
        chosen["source_ids"] = ["r1"]
        for spec in chosen["specifications"]:
            spec["source_id"] = "r1"
        validate_selection(chosen, inventory()["assets"][0], combined)


class NativePipelineTests(TempCase):
    def execute(self, transport, config=None, **kwargs):
        return execute(documents(), config or Config(), {"HARNESS_MODEL_API_KEY": "fixture-key"},
                       self.root, transport, provenance=PROVENANCE, **kwargs)

    def test_native_pipeline_one_key_and_reusable_evidence(self):
        transport = NativeTransport()
        package = self.execute(transport)
        self.assertEqual(package["search_provider"], "model_web")
        self.assertTrue(package["search_url"].endswith("/responses"))
        self.assertEqual(package["budget"]["model_calls"], 3)
        self.assertEqual(package["budget"]["search_calls"], 1)
        self.assertEqual(package["budget"]["page_fetches"], 1)
        self.assertIn("fixture-search-model", package["returned_models"])
        item = package["assets"][0]
        self.assertEqual(item["references"]["sources"][0]["evidence_kind"], "fetched_page")
        self.assertTrue(item["downloaded_images"])
        self.assertFalse(item["downloaded_images"][0]["visual_identity_verified"])
        replay = NativeTransport(replies=[], search_replies=[])
        self.execute(replay, resume=True)
        self.assertEqual((replay.model_calls, replay.search_calls, replay.image_calls, len(replay.page_requests)), (0, 0, 0, 0))

    def test_budget_plan_stops_before_search_then_resume(self):
        with self.assertRaises(BudgetExceeded):
            self.execute(NativeTransport(), replace(Config(), max_model_calls=2))
        replay = NativeTransport(replies=[selection()])
        package = self.execute(replay, resume=True)
        self.assertEqual(replay.search_calls, 1)
        self.assertEqual(package["budget"]["model_calls"], 3)

    def test_page_failure_still_outputs_generation_requirements(self):
        no_spec = selection()
        no_spec.update(mode="reference", specifications=[], adaptations=[], image_ids=[])
        transport = NativeTransport(replies=[inventory(), no_spec], html=ProviderError("HTTP 403"))
        package = self.execute(transport)
        self.assertEqual(package["status"], "completed_with_gaps")
        self.assertEqual(package["assets"][0]["target"]["quantity"]["value"], 2)
        self.assertTrue(package["assets"][0]["warnings"])

    def test_page_budget_resumes_mid_asset_without_repeating_evidence(self):
        response = native_response()
        response["output"][1]["content"][0]["annotations"].append(
            {"type": "url_citation", "url": "https://manufacturer.example/other", "title": "Other"})
        first = NativeTransport(search_replies=[response])
        with self.assertRaises(BudgetExceeded):
            self.execute(first, replace(Config(), max_page_fetches=1))
        partial = json.loads((self.root / "asset_package.json").read_text(encoding="utf-8"))
        self.assertEqual(partial["status"], "paused")
        self.assertEqual(partial["assets"][0]["references"]["sources"][0]["page_status"], "fetched")
        replay = NativeTransport(replies=[selection()], search_replies=[])
        package = self.execute(replay, resume=True)
        self.assertEqual(replay.search_calls, 0)
        self.assertEqual([url for url, _ in replay.page_requests], ["https://manufacturer.example/other"])
        self.assertEqual(package["budget"]["page_fetches"], 2)
        self.assertEqual(package["assets"][0]["references"]["images"][0]["source_ids"], ["r0", "r1"])

    def test_tool_limit_violation_halts_remaining_queries_and_selection(self):
        response = native_response()
        response["output"].insert(0, {"type": "web_search_call", "status": "completed"})
        inv = inventory()
        inv["assets"][0]["queries"].append("second query must not execute")
        transport = NativeTransport(replies=[inv], search_replies=[response])
        with self.assertRaises(ValidationError):
            self.execute(transport)
        self.assertEqual((transport.model_calls, transport.search_calls), (1, 1))
        self.assertEqual(transport.page_requests, [])
        partial = json.loads((self.root / "asset_package.json").read_text(encoding="utf-8"))
        self.assertEqual(partial["status"], "failed")
        self.assertTrue(partial["assets"][0]["target"])
