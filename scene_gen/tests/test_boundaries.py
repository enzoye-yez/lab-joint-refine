"""工程边界测试；样例均为开发者编写，不调用远端服务。"""

import json
import hashlib
import os
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from asset_prep.config import Config, credentials
from asset_prep.documents import chunks, load_documents
from asset_prep.errors import BudgetExceeded, ProviderError, StaleEvidence, UncertainRequest, ValidationError
from asset_prep.images import image_extension
from asset_prep.network import Transport, public_target
from asset_prep.providers import Model, combine_references, normalize_search, redact
from asset_prep.storage import Store, atomic_json, run_directory, run_lock, safe_path
from fixtures import PDF_TEXT, write_pdf


class TempCase(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache/tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)


class DocumentTests(TempCase):
    def prepare(self, xml="<protocol>Place three tubes.</protocol>", text=PDF_TEXT):
        pdf, markup = self.root / "input.pdf", self.root / "input.xml"
        write_pdf(pdf, text)
        markup.write_text(xml, encoding="utf-8")
        return pdf, markup

    def test_pdf_and_xml_keep_separate_evidence(self):
        value = load_documents(*self.prepare(), 10000, xml_reason='核对数量', xml_queries=['three'])
        self.assertEqual(value["sources"][0]["kind"], "pdf")
        self.assertIn("two", value["sources"][0]["text"])
        self.assertIn("three", value["sources"][1]["text"])

    def test_no_silent_xml_fallback_on_scanned_pdf(self):
        with self.assertRaises(ValidationError):
            load_documents(*self.prepare(text=""), 10000)

    def test_xml_external_entity_rejected(self):
        xml = '<!DOCTYPE x [<!ENTITY leak SYSTEM "file:///C:/secret">]><x>&leak;</x>'
        with self.assertRaises(ValidationError):
            load_documents(*self.prepare(xml), 10000, xml_reason='补证测试', xml_queries=['x'])

    def test_document_limit_not_silent_truncation(self):
        with self.assertRaises(ValidationError):
            load_documents(*self.prepare(), 10)

    def test_chunks_keep_all_text(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        result = chunks([{"id": "pdf:p1", "kind": "pdf", "text": text}], 7)
        self.assertEqual("".join(p["text"] for c in result for p in c), text)

    def test_hash_matches_parsed_snapshot_even_if_input_changes(self):
        from pypdf import PdfReader
        pdf, xml = self.prepare()
        expected = hashlib.sha256(pdf.read_bytes()).hexdigest()
        def mutate_after_snapshot(stream):
            pdf.write_bytes(b"changed while parsing")
            return PdfReader(stream)
        with patch("asset_prep.documents.PdfReader", side_effect=mutate_after_snapshot):
            result = load_documents(pdf, xml, 10000)
        self.assertEqual(result["files"]["pdf"]["sha256"], expected)
        self.assertIn("two", result["sources"][0]["text"])


class StorageTests(TempCase):
    def test_budget_charged_before_call_and_on_failure(self):
        store = Store(self.root, {"input": "a"}, replace(Config(), max_model_calls=1))
        call = store.reserve("a", "model", 10)
        store.fail(call, "network")
        with self.assertRaises(BudgetExceeded):
            store.reserve("b", "model", 10)
        self.assertEqual(store.totals()["reserved_tokens"], 10)

    def test_unknown_request_requires_explicit_retry(self):
        store = Store(self.root, {}, Config())
        store.reserve("a", "model", 10)
        with self.assertRaises(UncertainRequest):
            Store(self.root, {}, Config(), resume=True).cached("a")
        resumed = Store(self.root, {}, Config(), resume=True, retry_uncertain=True)
        self.assertIsNone(resumed.cached("a"))
        resumed.reserve("a", "model", 10)
        self.assertEqual(resumed.totals()["reserved_tokens"], 20)

    def test_changed_input_invalidates_evidence(self):
        Store(self.root, {"input": "a"}, Config())
        with self.assertRaises(StaleEvidence):
            Store(self.root, {"input": "b"}, Config(), resume=True)

    def test_tampered_cache_rejected(self):
        store = Store(self.root, {}, Config())
        call = store.reserve("a", "model")
        store.finish(call, {"hello": 1})
        (self.root / call["file"]).write_text("{}", encoding="utf-8")
        with self.assertRaises(StaleEvidence):
            store.cached("a")

    def test_token_budget_survives_restart(self):
        store = Store(self.root, {}, replace(Config(), max_reserved_tokens=10))
        store.reserve("a", "model", 10)
        resumed = Store(self.root, {}, replace(Config(), max_reserved_tokens=10), resume=True)
        with self.assertRaises(BudgetExceeded):
            resumed.reserve("b", "model", 1)

    def test_new_run_refuses_to_overwrite_existing_state(self):
        Store(self.root, {}, Config())
        with self.assertRaises(ValidationError):
            Store(self.root, {}, Config())

    def test_output_path_traversal_and_windows_drive_rejected(self):
        for bad in ("../outside", "D:/outside", "C:alternate", "../../.env.local"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                safe_path(self.root, bad)
        with self.assertRaises(ValidationError):
            run_directory(self.root, "../bad")
        with self.assertRaises(ValidationError):
            run_directory(self.root, "CON")

    def test_concurrent_run_lock(self):
        with run_lock(self.root):
            with self.assertRaises(ValidationError):
                with run_lock(self.root):
                    pass


class NetworkTests(TempCase):
    def resolver(self, ip):
        return lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    def test_internal_network_and_credentials_in_url_rejected(self):
        for ip in ("127.0.0.1", "10.0.0.2", "169.254.169.254", "::1"):
            with self.subTest(ip=ip), self.assertRaises(ValidationError):
                public_target("https://example.org/image", self.resolver(ip))
        with self.assertRaises(ValidationError):
            public_target("https://secret@example.org/image", self.resolver("8.8.8.8"))

    def test_public_target_returns_pinned_address(self):
        parsed, address = public_target("https://example.org/image", self.resolver("8.8.8.8"))
        self.assertEqual(address, "8.8.8.8")
        self.assertEqual(parsed.hostname, "example.org")

    def test_redirect_not_followed_and_error_body_not_logged(self):
        class Connection:
            sock = None
            def __init__(self, *args): pass
            def request(self, *args): pass
            def getresponse(self):
                class Response:
                    status = 302
                return Response()
            def close(self): pass
        with patch("asset_prep.network.public_target", return_value=(__import__("urllib.parse", fromlist=["urlsplit"]).urlsplit("https://example.org/"), "8.8.8.8")), patch("asset_prep.network.PinnedHTTPS", Connection):
            with self.assertRaisesRegex(ProviderError, "302"):
                Transport().request("https://example.org/", payload={}, key="never-print-this")

    def test_credentials_read_only_project_env_and_process(self):
        (self.root / ".env.local").write_text("HARNESS_MODEL_API_KEY=local-value\nUNRELATED=ignored", encoding="utf-8")
        with patch.dict(os.environ, {"HARNESS_MODEL_API_KEY": "process-value"}):
            result = credentials(self.root)
            self.assertEqual(result["HARNESS_MODEL_API_KEY"], "process-value")
            self.assertNotIn("UNRELATED", result)

    def test_response_redacts_secrets_recursively(self):
        self.assertEqual(redact({"x": ["secret123"]}, ["secret123"]), {"x": ["[REDACTED]"]})

    def test_search_images_preserve_source_linkage(self):
        result = normalize_search({"results": [{"url": "https://example.org/product", "content": "spec",
             "images": [{"url": "https://example.org/a.png"}]}], "images": ["https://example.org/b.png"]}, "query")
        merged = combine_references([result, result])
        self.assertEqual(len(merged["sources"]), 1)
        self.assertEqual(len(merged["images"]), 2)
        self.assertEqual(merged["images"][0]["source_id"], "r0")
        self.assertIsNone(merged["images"][1]["source_id"])

    def test_bad_config_limits(self):
        for kwargs in ({"max_model_calls": 0}, {"max_model_calls": True}, {"timeout_seconds": -1}, {"download_images": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                Config(**kwargs)

    def test_script_or_html_cannot_be_saved_as_image(self):
        for data in (b"<svg><script>bad</script></svg>", b"<html>not image</html>", b"print('bad')"):
            with self.assertRaises(ProviderError):
                image_extension(data)
