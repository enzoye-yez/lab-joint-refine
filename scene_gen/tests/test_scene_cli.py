import contextlib
import io
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from asset_generation.sandbox import SandboxUnavailable
from scene_generation.cli import main


class SceneCliTests(unittest.TestCase):
    def test_explicit_local_preflight_never_touches_docker_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"package.md"
            source.write_text("### A01｜合成\n固定工程样例。",encoding="utf-8")
            with patch("scene_generation.cli.run_directory",return_value=root/"run"), \
                 patch("scene_generation.cli.DockerSandbox",side_effect=AssertionError("No Docker")), \
                 patch("scene_generation.cli.credentials",side_effect=AssertionError("No keys")), \
                 patch("scene_generation.cli.transport_for",side_effect=AssertionError("No model")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--package",str(source),"--run","fixture","--backend","local"]),0)
            report = json.loads((root/"run/preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(report["sandbox"]["backend"],"local")
            self.assertEqual(report["model_calls"],0)

    def test_preflight_never_reads_key_or_calls_extraction_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"package.md"
            source.write_text("### A01｜合成\n固定工程样例。",encoding="utf-8")
            with patch("scene_generation.cli.run_directory",return_value=root/"run"), \
                 patch("scene_generation.cli.credentials",side_effect=AssertionError("No keys")), \
                 patch("scene_generation.cli.transport_for",side_effect=AssertionError("No model")), \
                 patch("asset_prep.extraction.extract_inventory",side_effect=AssertionError("No extraction")), \
                 patch("scene_generation.cli.DockerSandbox.preflight",side_effect=SandboxUnavailable("fixture")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--package",str(source),"--run","fixture"]),0)
            report = json.loads((root/"run/preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model_calls"],0)
            self.assertEqual(report["extraction_calls"],0)
            self.assertEqual(report["search_calls"],0)

    def test_execute_requires_sandbox_before_credentials(self):
        from asset_prep.implementation import source_manifest, implementation_digest
        identity = implementation_digest(source_manifest(Path(__file__).resolve().parents[1]))
        active = getattr(sys, '_harness_runtime', None) or {
            'base':str(Path(__file__).resolve().parents[1]/'src'), 'implementation':identity}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"package.md"
            source.write_text("### A01｜合成\n固定工程样例。",encoding="utf-8")
            with patch("scene_generation.cli.run_directory",return_value=root/"run"), \
                 patch.object(sys,'_harness_runtime',active,create=True), \
                 patch("scene_generation.cli.credentials",side_effect=AssertionError("No keys")), \
                 patch("scene_generation.cli.DockerSandbox.preflight",side_effect=SandboxUnavailable("fixture")), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--package",str(source),"--run","fixture","--execute",
                                       '--expected-implementation',identity]),3)

    def test_execute_without_approved_identity_never_reads_credentials(self):
        with patch('scene_generation.cli.credentials',side_effect=AssertionError('No keys')), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--package','synthetic.md','--run','synthetic','--execute']),2)
