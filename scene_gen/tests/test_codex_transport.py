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
