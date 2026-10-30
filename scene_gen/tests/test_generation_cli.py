import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_generation.cli import main
from asset_generation.sandbox import SandboxUnavailable
from generation_fixtures import package


class GenerationCliTests(unittest.TestCase):
    def test_preflight_does_not_read_credentials_or_call_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = package(root)
            output = io.StringIO()
            with patch("asset_generation.cli.run_directory", return_value=root/"preflight"), \
                 patch("asset_generation.cli.credentials", side_effect=AssertionError("No credential reads")), \
                 patch("asset_generation.cli.Transport", side_effect=AssertionError("No model transport")), \
                 patch("asset_generation.cli.DockerSandbox.preflight", side_effect=SandboxUnavailable("fixture missing docker")), \
                 contextlib.redirect_stdout(output):
                code = main(["--package", str(source), "--asset", "A01", "--run", "fixture"])
            self.assertEqual(code, 0)
            report = json.loads((root/"preflight/preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model_calls"], 0)
            self.assertFalse(report["sandbox"]["available"])

    def test_execute_refuses_before_key_read_when_sandbox_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = package(root)
            with patch("asset_generation.cli.run_directory", return_value=root/"run"), \
                 patch("asset_generation.cli.credentials", side_effect=AssertionError("No credential reads")), \
                 patch("asset_generation.cli.DockerSandbox.preflight", side_effect=SandboxUnavailable("fixture missing docker")), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--package", str(source), "--asset", "A01", "--run", "fixture", "--execute"]), 3)


if __name__ == "__main__":
    unittest.main()
