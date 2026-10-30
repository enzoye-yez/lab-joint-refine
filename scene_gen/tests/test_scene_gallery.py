"""部分产物展示的离线测试，不调用模型、不修改资产。"""
import json
from pathlib import Path
import tempfile
import unittest

from asset_prep.storage import atomic_json, file_hash
from scene_generation.gallery import export


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / "runs/synthetic"
        stage = self.root / "stages/00001_asset"
        stage.mkdir(parents=True)
        self.png = stage / "view 1.png"
        self.png.write_bytes(b"synthetic image for integrity test")
        self.state = {"scene":{"status":"paused", "plan":{"jobs":[{"id":"asset_A01", "role":"asset", "asset_ids":["A01"]}]},
            "stages":[{"directory":"stages/00001_asset", "files":{"view 1.png":file_hash(self.png)}}],
            "jobs":{"asset_A01":{"status":"assembly_ready", "stage":"stages/00001_asset",
                    "preview_paths":["stages/00001_asset/view 1.png"]}}}}
        atomic_json(self.root / "state.json", self.state)

    def test_paused_run_gallery_is_refreshable_and_preserves_evidence(self):
        before = file_hash(self.root / "state.json"), file_hash(self.png)
        page = export(self.project, "synthetic", refresh=True)
        export(self.project, "synthetic", refresh=True)
        self.assertEqual(before, (file_hash(self.root / "state.json"), file_hash(self.png)))
        self.assertIn("data:image/png;base64,", page.read_text(encoding="utf-8"))
        record = json.loads((page.parent / "sources.json").read_text(encoding="utf-8"))
        self.assertEqual(len(record["assets"]), 1)
        self.assertEqual(record["model_calls"], 0)

    def test_empty_run_has_explicit_placeholder(self):
        atomic_json(self.root / "state.json", {"scene":{"jobs":{}, "stages":[]}})
        page = export(self.project, "synthetic", refresh=True, allow_empty=True)
        self.assertIn("尚无已归档资产图片", page.read_text(encoding="utf-8"))

    def test_windows_job_reference_matches_normalized_archive_directory(self):
        self.state['scene']['jobs']['asset_A01']['stage']='stages\\00001_asset'
        atomic_json(self.root/'state.json',self.state)
        before=file_hash(self.root/'state.json')
        page=export(self.project,'synthetic',refresh=True)
        self.assertIn('data:image/png;base64,',page.read_text(encoding='utf-8'))
        self.assertEqual(before,file_hash(self.root/'state.json'))

    def test_tampered_png_does_not_replace_previous_gallery(self):
        page = export(self.project, "synthetic")
        before = file_hash(page)
        self.png.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "摘要不一致"):
            export(self.project, "synthetic", refresh=True)
        self.assertEqual(file_hash(page), before)
