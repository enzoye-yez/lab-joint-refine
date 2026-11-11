"""Codex 传输：开发者编写的假 codex 可执行文件，不发起真实请求。"""

import base64
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from asset_prep.config import Config
from asset_prep.errors import ProviderError, ValidationError
from asset_generation.codex_transport import CodexTransport, strip_fence
from asset_generation.transport import GenerationTransport, transport_for

PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()

FAKE = r'''#!{python}
import json, sys
from pathlib import Path
args = sys.argv[1:]
prompt = sys.stdin.read()
images = [args[i + 1] for i, a in enumerate(args) if a == "--image"]
Path({log!r}).write_text(json.dumps({{"args": args, "prompt": prompt,
    "images": [Path(p).read_bytes().hex() for p in images]}}))
mode = {mode!r}
if mode == "agent":
    cwd = Path(args[args.index("--cd") + 1])
    assert args[args.index("--sandbox") + 1] == "workspace-write"
    assert (cwd / "AGENT_TASK.md").is_file() and (cwd / "check").is_file()
    job = json.loads((cwd / "selfcheck.json").read_text())
    (cwd / "model.py").write_text("def build(output_dir):\n    return " + repr(job["asset_ids"]) + "\n")
    print(json.dumps({{"type": "item.completed", "item": {{"type": "command_execution", "command": "./check"}}}}))
    print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "{{\"done\": true}}"}}}}))
    print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 900, "output_tokens": 70}}}}))
    sys.exit(0)
if mode == "fail":
    print(json.dumps({{"type": "turn.failed", "error": {{"message": "rate limited"}}}}))
    sys.exit(1)
text = "```json\n{{\"ok\": true}}\n```" if mode == "fenced" else "{{\"ok\": true}}"
print(json.dumps({{"type": "thread.started"}}))
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": text}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 120, "output_tokens": 7}}}}))
'''


class CodexTransportTests(unittest.TestCase):
    def fake(self, folder, mode="ok"):
        path = Path(folder) / "codex"
        path.write_text(FAKE.format(python=sys.executable, log=str(Path(folder) / "call.json"), mode=mode))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return str(path)

    def payload(self):
        return {"model": "gpt-6.1-sol", "stream": False, "max_completion_tokens": 100, "messages": [
            {"role": "system", "content": "Return {\"ok\": true}."},
            {"role": "user", "content": [{"type": "text", "text": "check"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}]}]}

    def test_response_matches_chat_completions_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            response = CodexTransport(self.fake(folder, "fenced")).post("unused", self.payload(), "codex-login", 30)
            call = json.loads((Path(folder) / "call.json").read_text())
        self.assertEqual(json.loads(response["choices"][0]["message"]["content"]), {"ok": True})
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["prompt_tokens"], 120)
        self.assertEqual(response["model"], "gpt-6.1-sol")
        self.assertIn("--ephemeral", call["args"])
        self.assertEqual(call["args"][call["args"].index("--sandbox") + 1], "read-only")
        self.assertEqual(call["args"][call["args"].index("--model") + 1], "gpt-6.1-sol")
        self.assertEqual(call["images"], [base64.b64decode(PNG).hex()])
        self.assertIn("[attached image #1]", call["prompt"])
        self.assertIn("Return {\"ok\": true}.", call["prompt"])

    def author_payload(self, previous):
        context = {"job": {"asset_ids": ["A15"]}, "profile": {"validation_mode": "static"},
                   "current_model_py": previous, "feedback": None}
        payload = self.payload()
        payload["messages"][1]["content"][0]["text"] = json.dumps(context)
        return payload

    def test_agent_author_returns_script_under_author_contract(self):
        from asset_generation.contracts import author_result
        with tempfile.TemporaryDirectory() as folder:
            transport = CodexTransport(self.fake(folder, "agent"), agent=True, workspace_root=folder)
            transport.phase = "asset:asset15"
            first = transport.post("unused", self.author_payload(None), "codex-login", 30)
            source = author_result(json.loads(first["choices"][0]["message"]["content"]), None, 500000)
            self.assertIn("['A15']", source)
            self.assertEqual(first["agent_session"]["checks_run"], 1)
            self.assertTrue(Path(first["agent_session"]["workspace"]).is_relative_to(folder))
            repair = transport.post("unused", self.author_payload("def build(output_dir):\n    pass\n"), "codex-login", 30)
            value = json.loads(repair["choices"][0]["message"]["content"])
            self.assertEqual(author_result(value, "def build(output_dir):\n    pass\n", 500000), source)
            # Reviews and other phases keep the plain read-only request (the fake rejects it).
            transport.phase = "visual_review"
            with self.assertRaises(ProviderError):
                transport.post("unused", self.author_payload(None), "codex-login", 30)

    def test_agent_mode_moves_long_text_into_a_file(self):
        payload = self.payload()
        payload["messages"][1]["content"][0]["text"] = "x" * 70000
        with tempfile.TemporaryDirectory() as folder:
            transport = CodexTransport(self.fake(folder), agent=True)
            self.assertTrue(transport.whole_context("asset_review:tool"))
            transport.post("unused", payload, "codex-login", 30)
            call = json.loads((Path(folder) / "call.json").read_text())
        self.assertIn("input_1.txt", call["prompt"])
        self.assertNotIn("x" * 70000, call["prompt"])
        self.assertEqual(call["args"][call["args"].index("--sandbox") + 1], "read-only")

    def test_failure_is_retryable_provider_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ProviderError) as caught:
                CodexTransport(self.fake(folder, "fail")).post("unused", self.payload(), "codex-login", 30)
        self.assertFalse(caught.exception.uncertain)
        self.assertEqual(caught.exception.diagnostics["failure"], "connection_error")
        self.assertIn("rate limited", str(caught.exception))

    def test_config_selects_transport_without_api_key(self):
        config = Config()
        self.assertEqual((config.provider, config.model), ("codex", "gpt-6.1-sol"))
        self.assertEqual(config.model_key({}), "codex-login")
        self.assertIsInstance(transport_for(config), CodexTransport)
        with self.assertRaises(ValidationError):
            Config(api_key_env="OPENAI_API_KEY")
        openai = Config(provider="openai", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
        self.assertIsInstance(transport_for(openai), GenerationTransport)

    def test_strip_fence(self):
        self.assertEqual(strip_fence("```json\n{\"a\": 1}\n```"), "{\"a\": 1}")
        self.assertEqual(strip_fence(" {\"a\": 1} "), "{\"a\": 1}")


if __name__ == "__main__":
    unittest.main()
