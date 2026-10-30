"""Preview regression fixtures, not formal experiment results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from asset_prep.storage import atomic_json, digest
from asset_prep.errors import ValidationError
from asset_generation.pipeline import snapshot
from scene_generation.preview import resolve_artifact, write_preview_index


class PreviewEntryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.run=self.root/"runs/example";self.artifact=self.run/"artifact";self.artifact.mkdir(parents=True)
        (self.artifact/"model.xml").write_text("synthetic artifact hash fixture")
        self.result={"status":"incomplete_scene", "scene_artifact":"artifact",
            "scene_sha256":digest(snapshot(self.artifact)),"validation_mode":"static",
            "assembly":{"available":True,"complete":False,"missing_jobs":["tool"],"quality_accepted":False,
                        "issues":[{"detail":"x"*2100000}]}, "unverified":["missing tool"]}
        atomic_json(self.run/"result.json",self.result)

    def test_legacy_large_report_resolves_partial_scene_without_mutation(self):
        before=snapshot(self.run)
        _,artifact,result=resolve_artifact(self.root,"example")
        self.assertEqual(artifact,self.artifact)
        self.assertFalse(result["assembly"]["complete"])
        self.assertFalse(result["assembly"]["quality_accepted"])
        self.assertEqual(result["assembly"]["missing_jobs"],["tool"])
        self.assertNotIn("issues",result["assembly"])
        self.assertEqual(snapshot(self.run),before)

    def test_index_does_not_parse_full_report_and_remains_small(self):
        write_preview_index(self.run,self.result)
        self.assertLess((self.run/"preview.json").stat().st_size,2000)
        from scene_generation import preview
        read=preview.read_json
        def bounded_read(path,*args):
            self.assertEqual(Path(path).name,"preview.json")
            return read(path,*args)
        with patch.object(preview,"read_json",side_effect=bounded_read):
            self.assertFalse(resolve_artifact(self.root,"example")[2]["assembly"]["quality_accepted"])

    def test_stale_index_and_changed_artifact_cannot_reuse_quality(self):
        write_preview_index(self.run,self.result)
        self.result["status"]="changed";atomic_json(self.run/"result.json",self.result)
        with self.assertRaisesRegex(ValidationError,"版本不一致"):resolve_artifact(self.root,"example")
        write_preview_index(self.run,self.result)
        (self.artifact/"model.xml").write_text("changed")
        with self.assertRaisesRegex(ValidationError,"场景文件已变化"):resolve_artifact(self.root,"example")

    def test_missing_invalid_oversized_and_no_scene_have_distinct_errors(self):
        p=self.run/"result.json"
        p.unlink()
        with self.assertRaisesRegex(ValidationError,"缺少结果文件"):resolve_artifact(self.root,"example")
        p.write_text("broken")
        with self.assertRaisesRegex(ValidationError,"无效的结构化文件"):resolve_artifact(self.root,"example")
        p.write_text("[]")
        with self.assertRaisesRegex(ValidationError,"必须为对象"):resolve_artifact(self.root,"example")
        with p.open("wb") as f:f.truncate(64000001)
        with self.assertRaisesRegex(ValidationError,"64 MB"):resolve_artifact(self.root,"example")
        atomic_json(p,{"scene_artifact":None})
        with self.assertRaisesRegex(ValidationError,"尚无已发布场景"):resolve_artifact(self.root,"example")

    def test_paths_remain_confined_and_missing_model_is_not_report_size_error(self):
        for name in ("../outside","C:/outside"):
            with self.assertRaises(ValidationError):resolve_artifact(self.root,name)
        self.result["scene_artifact"]="../outside";atomic_json(self.run/"result.json",self.result)
        with self.assertRaises(ValidationError):resolve_artifact(self.root,"example")
        self.result["scene_artifact"]="artifact";atomic_json(self.run/"result.json",self.result)
        (self.artifact/"model.xml").unlink()
        with self.assertRaisesRegex(ValidationError,"model.xml 文件缺失"):resolve_artifact(self.root,"example")

    def test_failed_index_write_keeps_old_index_and_detects_publication_gap(self):
        write_preview_index(self.run,self.result)
        original=(self.run/"preview.json").read_bytes()
        self.result["status"]="new_result";atomic_json(self.run/"result.json",self.result)
        with patch("scene_generation.preview.atomic_json",side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):write_preview_index(self.run,self.result)
        self.assertEqual((self.run/"preview.json").read_bytes(),original)
        with self.assertRaisesRegex(ValidationError,"版本不一致"):resolve_artifact(self.root,"example")
        write_preview_index(self.run,self.result)
        self.assertEqual(resolve_artifact(self.root,"example")[2]["status"],"new_result")
