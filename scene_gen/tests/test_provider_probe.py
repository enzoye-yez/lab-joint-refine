"""服务诊断脚本的离线测试，不访问网络或真实凭据。"""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from asset_generation.contracts import GenerationConfig
from asset_prep.config import Config
from asset_prep.errors import ProviderError

spec = importlib.util.spec_from_file_location("provider_probe_script",
    Path(__file__).resolve().parents[1] / "scripts/probe_generation_provider.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def run_probe(self, execute, response=None, error=None):
        transport = Mock()
        transport.post.return_value = response
        transport.post.side_effect = error
        saved = []
        with patch.object(probe.Config, "load", return_value=Config()), \
             patch.object(probe.GenerationConfig, "load", return_value=GenerationConfig()), \
             patch.object(probe, "credentials", return_value={"HARNESS_MODEL_API_KEY":"synthetic-secret"}) as keys, \
             patch.object(probe, "atomic_json", side_effect=lambda path, value: saved.append(dict(value))), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            code = probe.main(["--execute"] if execute else [], transport)
        self.assertNotIn("synthetic-secret", output.getvalue())
        return code, transport, saved, keys

    def test_default_does_not_read_credentials_or_call(self):
        code, transport, saved, keys = self.run_probe(False)
        self.assertEqual(code, 0)
        transport.post.assert_not_called()
        keys.assert_not_called()
        self.assertEqual(saved, [])

    def test_failure_is_reported_once_without_retry(self):
        error = ProviderError("rejected", diagnostics={"http_status":400,
            "provider_error":{"code":"bad_request", "message":"bad synthetic-secret"}})
        code, transport, saved, _ = self.run_probe(True, error=error)
        self.assertEqual(code, 2)
        transport.post.assert_called_once()
        self.assertEqual(saved[-1]["http_status"], 400)
        self.assertEqual(saved[-1]["provider_error"]["message"], "bad [REDACTED]")

    def test_success_only_claims_short_text_connectivity(self):
        code, transport, saved, _ = self.run_probe(True, response={"choices":[{"message":{"content":"{}"}}]})
        self.assertEqual(code, 0)
        self.assertEqual(saved[-1]["scope"], "short_text_request_only")
        transport.post.assert_called_once()
