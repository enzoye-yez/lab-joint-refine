import json
import tempfile
import unittest
from pathlib import Path

from asset_prep.errors import ValidationError
from asset_generation.contracts import GenerationConfig, author_result, check_profile
from asset_generation.inputs import load_package
from generation_fixtures import SOURCE, package


class GenerationInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_curated_input_sends_pixels_and_scopes_asset_text(self):
        bundle = load_package(package(self.root), ["A01"], GenerationConfig())
        self.assertEqual(bundle["asset_ids"], ["A01"])
        self.assertNotIn("另一个资产", bundle["source"]["selected_sections"])
        self.assertIn("A01", bundle["source"]["selected_sections"])
        self.assertIn("global_requirements", bundle["source"])
        self.assertEqual(len(bundle["images"]), 1)
        self.assertTrue(bundle["images"][0]["data_url"].startswith("data:image/png;base64,"))

    def test_unknown_asset_and_missing_images_fail(self):
        path = package(self.root)
        with self.assertRaises(ValidationError):
            load_package(path, ["A99"], GenerationConfig())
        (self.root/"reference.png").unlink()
        with self.assertRaises(ValidationError):
            load_package(path, ["A01"], GenerationConfig())

    def test_image_escape_rejected(self):
        path = package(self.root)
        path.write_text(path.read_text(encoding="utf-8").replace("reference.png", "../private.png"), encoding="utf-8")
        with self.assertRaises(ValidationError):
            load_package(path, ["A01"], GenerationConfig())

    def test_asset_prep_adapter_keeps_gaps_and_relations(self):
        from asset_prep.storage import file_hash
        package(self.root)
        value = {"status": "completed_with_gaps", "inventory": {"relations": [{"a": "A01", "b": "A02"}]},
                 "assets": [{"asset_id": "A01", "target": {"name": "合成按钮"}, "warnings": ["未知质量"],
                             "downloaded_images": [{"local_path": "reference.png", "sha256": file_hash(self.root/"reference.png")}]}]}
        path = self.root/"asset_package.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        result = load_package(path, ["A01"], GenerationConfig())
        self.assertIn("未知质量", result["warnings"])
        self.assertEqual(result["source"]["inventory"], value["inventory"])
        value["status"] = "paused"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(ValidationError):
            load_package(path, ["A01"], GenerationConfig())

    def test_unique_patch_only_and_no_other_files(self):
        source = author_result({"model_py": SOURCE}, None, 10000)
        patched = author_result({"edits": [{"old": "    pass", "new": "    return None"}]}, source, 10000)
        self.assertIn("return None", patched)
        with self.assertRaises(ValidationError):
            author_result({"edits": [{"old": "", "new": "evil"}]}, source, 10000)
        with self.assertRaises(ValidationError):
            author_result({"model_py": SOURCE, "checks.json": {}}, None, 10000)

    def test_profile_rejects_nonfinite_and_unknown_rules(self):
        with self.assertRaises(ValidationError):
            check_profile({"penetration_tolerance_m": float("nan")})
        with self.assertRaises(ValidationError):
            check_profile({"ignore_all_collisions": True})


if __name__ == "__main__":
    unittest.main()
