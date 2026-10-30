"""开发者合成服务错误；禁止真实网络和真实密钥。"""

import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from asset_generation.provider_diagnostics import DiagnosticTransport, safe_details
from asset_prep.errors import ProviderError
from asset_prep.network import Transport


class ProviderDiagnosticsTests(unittest.TestCase):
    def error(self, raw, status=400, read_error=None, transport=None):
        response = Mock(status=status)
        response.getheader.side_effect = lambda key: {"x-request-id": "request-123", "retry-after": "60"}.get(key)
        reader = io.BytesIO(raw)
        response.read1.side_effect = read_error or reader.read1
        connection = Mock()
        connection.getresponse.return_value = response
        with patch("asset_prep.network.public_target", return_value=(urlsplit("https://example.org/api"), "8.8.8.8")), \
             patch("asset_prep.network.PinnedHTTPS", return_value=connection):
            with self.assertRaises(ProviderError) as caught:
                (transport or DiagnosticTransport()).post("https://example.org/api", {}, "synthetic-secret", 10)
        connection.close.assert_called_once()
        return caught.exception, response

    def test_json_error_is_redacted_and_keeps_status(self):
        body = json.dumps({"error": {"code": "invalid_request", "type": "bad_parameter",
            "message": "bad synthetic-secret and Bearer another-token", "param": "model",
            "request": {"secret": "do not store"}}, "extra": "do not store"}).encode()
        error, _ = self.error(body)
        details = error.diagnostics["provider_error"]
        self.assertEqual(error.diagnostics["http_status"], 400)
        self.assertEqual(details["code"], "invalid_request")
        self.assertEqual(details["request_id"], "request-123")
        self.assertNotIn("synthetic-secret", str(error) + str(details))
        self.assertNotIn("another-token", str(error) + str(details))
        self.assertNotIn("do not store", str(details))
        self.assertFalse(error.uncertain)

    def test_error_body_timeout_does_not_mask_http_status(self):
        error, _ = self.error(b"", 503, TimeoutError("synthetic-secret"))
        self.assertEqual(error.diagnostics["http_status"], 503)
        self.assertEqual(error.diagnostics["provider_error"]["error_body_status"], "read_failed")
        self.assertTrue(error.uncertain)

    def test_non_json_and_oversized_responses_are_not_saved(self):
        for data, expected in ((b"<html>synthetic-secret</html>", "not_json"),
                               (b"x" * 50000, "too_large")):
            with self.subTest(expected=expected):
                error, response = self.error(data)
                details = error.diagnostics["provider_error"]
                self.assertEqual(details["error_body_status"], expected)
                self.assertLessEqual(int(details["error_body_bytes"]), 32769)
                self.assertNotIn("message", details)

    def test_default_asset_prep_transport_remains_body_free(self):
        error, response = self.error(b"secret body", transport=Transport())
        response.read1.assert_not_called()
        self.assertNotIn("provider_error", error.diagnostics)

    def test_redaction_precedes_truncation(self):
        secret = "secret" * 200
        result = safe_details({"message": "prefix " + secret, "unknown": "ignored"}, [secret])
        self.assertEqual(result, {"message": "prefix [REDACTED]"})


if __name__ == "__main__":
    unittest.main()
