"""Developer fixtures: credential selection never sends a real request."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
from asset_prep.config import Config, credentials
from asset_prep.errors import ProviderError, ValidationError
from asset_prep.providers import redact

# 开发者编写的 HTTP 兼容服务配置，仅用于本地替身传输，不发真实请求。
HTTP = dict(provider="compatible", base_url="https://api.example.test/v1", model="test-http-model",
            api_key_env="HARNESS_MODEL_API_KEY")

class CredentialSelectionTests(unittest.TestCase):
    def test_separate_slots_and_missing_new_does_not_fallback(self):
        old=Config(**HTTP); new=Config(**{**HTTP, "api_key_env": "HARNESS_MODEL_API_KEY2"})
        keys={"HARNESS_MODEL_API_KEY":"fixture-old", "HARNESS_MODEL_API_KEY2":"fixture-new"}
        self.assertEqual(old.model_key(keys),"fixture-old")
        self.assertEqual(new.model_key(keys),"fixture-new")
        with self.assertRaises(ProviderError):new.model_key({"HARNESS_MODEL_API_KEY":"fixture-old"})
        with self.assertRaises(ProviderError):new.model_key({"HARNESS_MODEL_API_KEY2":"  "})
        self.assertEqual(redact("fixture-old fixture-new",keys.values()),"[REDACTED] [REDACTED]")

    def test_local_keys_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{"HARNESS_MODEL_API_KEY2":"fixture-process"},clear=True):
            p=Path(d);(p/".env.local").write_text("HARNESS_MODEL_API_KEY=fixture-old\nHARNESS_MODEL_API_KEY2=fixture-local\nUNRELATED=ignored\n",encoding="utf-8")
            self.assertEqual(credentials(p),{"HARNESS_MODEL_API_KEY":"fixture-old","HARNESS_MODEL_API_KEY2":"fixture-process"})

    def test_legacy_identity_preserved_and_new_slot_bound(self):
        c=Config(**HTTP); expected=asdict(c)
        for k in ("api_key_env","codex_agent_author","max_model_calls","max_search_calls","max_image_downloads","max_reserved_tokens","max_page_fetches","timeout_seconds"):expected.pop(k)
        self.assertEqual(c.identity(),expected)
        self.assertEqual(Config(**{**HTTP, "api_key_env": "HARNESS_MODEL_API_KEY2"}).identity()["api_key_env"],"HARNESS_MODEL_API_KEY2")
        with self.assertRaises(ValidationError):Config(**{**HTTP, "api_key_env": "UNRELATED_SECRET"})

    def test_worker_does_not_inherit_second_key(self):
        from asset_generation.local import LocalBackend
        from asset_generation.contracts import GenerationConfig
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{"HARNESS_MODEL_API_KEY2":"fixture-new"}):
            env=LocalBackend(GenerationConfig(),Path(__file__).resolve().parents[1]).worker_environment(Path(d))
            self.assertNotIn("HARNESS_MODEL_API_KEY2",env)

    def test_official_profile_selects_only_official_key(self):
        c=Config.load(Path(__file__).resolve().parents[1]/"config/openai_astra.json")
        self.assertEqual((c.provider,c.base_url,c.model,c.api_key_env),
            ("openai","https://api.openai.com/v1","gpt-6-astra","OPENAI_API_KEY"))
        keys={"HARNESS_MODEL_API_KEY":"fixture-old","HARNESS_MODEL_API_KEY2":"fixture-tatu"}
        with self.assertRaises(ProviderError):c.model_key(keys)
        keys["OPENAI_API_KEY"]="fixture-official"
        self.assertEqual(c.model_key(keys),"fixture-official")
        self.assertEqual(c.identity()["api_key_env"],"OPENAI_API_KEY")
        self.assertEqual(redact("fixture-official fixture-tatu",keys.values()),"[REDACTED] [REDACTED]")

    def test_official_local_credential_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{},clear=True):
            p=Path(d);(p/".env.local").write_text("OPENAI_API_KEY=fixture-local\n",encoding="utf-8")
            self.assertEqual(credentials(p)["OPENAI_API_KEY"],"fixture-local")
            os.environ["OPENAI_API_KEY"]="fixture-process"
            self.assertEqual(credentials(p)["OPENAI_API_KEY"],"fixture-process")

    def test_official_key_not_in_build_worker_environment(self):
        from asset_generation.local import LocalBackend
        from asset_generation.contracts import GenerationConfig
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{"OPENAI_API_KEY":"fixture-official"}):
            env=LocalBackend(GenerationConfig(),Path(__file__).resolve().parents[1]).worker_environment(Path(d))
            self.assertNotIn("OPENAI_API_KEY",env)
