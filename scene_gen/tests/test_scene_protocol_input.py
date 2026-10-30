import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from asset_generation.contracts import GenerationConfig
from asset_prep.config import Config
from asset_prep.errors import ValidationError, StaleEvidence
from fixtures import write_pdf
from generation_fixtures import FakeTransport, PROVENANCE
from scene_fixtures import plan, environment, SceneSandbox
from scene_generation.inputs import catalog
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.pipeline import SceneRun


class ProtocolInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root/"asset_package.md"
        self.package.write_text("### A01｜合成仪器\n工程输入。",encoding="utf-8")
        self.pdf = self.root/"source.pdf"
        write_pdf(self.root/"first.pdf","FIRST_PAGE_MECHANICAL_CONTEXT")
        write_pdf(self.root/"second.pdf","SECOND_PAGE_MECHANICAL_CONTEXT")
        writer = PdfWriter()
        writer.append(self.root/"first.pdf")
        writer.append(self.root/"second.pdf")
        writer.write(self.pdf)

    def test_all_pdf_pages_reach_model_and_are_archived_without_extraction(self):
        generation = GenerationConfig()
        project = Path(__file__).resolve().parents[1]
        with patch("asset_prep.extraction.extract_inventory",side_effect=AssertionError("No extraction")), \
             patch("asset_prep.search.Search",side_effect=AssertionError("No search")):
            inventory = catalog(self.package,generation,self.pdf)
            review = {"verdict":"accept","findings":[],"unverified":[]}
            transport = FakeTransport([plan(),{"model_py":"def build(output_dir):\n    pass\n"},
                {"model_py":"def build(output_dir):\n    # TOOL_FIXTURE\n    pass\n"},review,review,review])
            env = environment(self.root/"robot")
            args = [inventory,env,numerical_contract({}),Config(),generation,
                    SceneConfig(validation_mode="static"),{"HARNESS_MODEL_API_KEY":"fixture"},self.root/"run",
                    transport,SceneSandbox(),self.root/"robot",project]
            result = SceneRun(*args,provenance=PROVENANCE).run()
        self.assertEqual(result["status"],"static_candidate_ready")
        for request in transport.requests:
            context = json.loads(request["messages"][1]["content"][0]["text"])
            protocol = context.get("protocol") or context["input"]["protocol"]
            self.assertEqual([s["id"] for s in protocol["sources"]],["pdf:p1","pdf:p2"])
            self.assertIn("SECOND_PAGE_MECHANICAL_CONTEXT",protocol["sources"][1]["text"])
        archived = json.loads((self.root/"run/stages/00001_plan/input.json").read_text(encoding="utf-8"))
        self.assertEqual(archived["protocol"],inventory["protocol"])
        write_pdf(self.pdf,"REVISED_MECHANICAL_CONTEXT")
        args[0] = catalog(self.package,generation,self.pdf)
        with self.assertRaises(StaleEvidence):
            SceneRun(*args,resume=True,provenance=PROVENANCE)

    def test_unreadable_pdf_rejected_instead_of_silent_package_only_run(self):
        write_pdf(self.pdf,"")
        with self.assertRaises(ValidationError):
            catalog(self.package,GenerationConfig(),self.pdf)

    def test_pdf_text_limit_never_truncates(self):
        with self.assertRaises(ValidationError):
            catalog(self.package,GenerationConfig(),self.pdf,max_protocol_chars=10)

