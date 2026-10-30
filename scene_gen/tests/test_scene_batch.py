"""合成队列测试；不发送模型请求，不执行建模或渲染。"""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scene_batch_script", PROJECT / "scripts/run_scene_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / "config").mkdir()
        for name in ("generation.json", "default.json", "scene.json", "scene_environment.json"):
            (self.project / "config" / name).write_bytes((PROJECT / "config" / name).read_bytes())
        self.inputs = self.project / "inputs"
        for i in range(1, 6):
            source = self.inputs / f"NP-{i:03d}"
            (source / "images").mkdir(parents=True)
            (source / "source.pdf").write_bytes(b"synthetic input, never parsed")
            (source / "asset_package.md").write_text("synthetic package", encoding="utf-8")
            (source / "images/ref.png").write_bytes(b"synthetic image, never opened")
        self.args = argparse.Namespace(batch="synthetic", protocols=[f"NP-{i:03d}" for i in range(1, 6)],
                                       input_root=self.inputs, asset_iterations=30, scene_iterations=40,
                                       max_model_calls=400, max_reserved_tokens=120000000)
        self.root = self.project / "runs/batches/synthetic"
        self.manifest = batch.prepare(self.project, self.root, self.args)

    def result(self, run_root, status="static_preview_ready"):
        batch.atomic_json(run_root / "result.json", dict(status=status, scene_artifact="synthetic",
                          budget=dict(model_calls=7, reported_total_tokens=80, accounted_tokens=150)))

    def render(self, command):
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        (destination / "index.html").write_text("synthetic preview", encoding="utf-8")

    def test_batch_model_override_is_frozen_without_changing_shared_default(self):
        old=batch.read(self.project/"config/default.json")
        self.args.model="GPT-6.1-sol"
        manifest=batch.prepare(self.project,self.root,self.args)
        self.assertEqual(batch.read(self.root/"config/provider.json")["model"],"GPT-6.1-sol")
        self.assertEqual(batch.read(self.project/"config/default.json"),old)
        batch.verify(manifest,self.project)
        provider=batch.read(self.root/"config/provider.json");provider["model"]="different"
        batch.atomic_json(self.root/"config/provider.json",provider)
        with self.assertRaises(ValueError):batch.verify(manifest,self.project)

    def test_batch_digest_matches_runtime_and_detects_operation_changes(self):
        from asset_prep.implementation import source_manifest, implementation_digest
        frozen = batch.hashes(batch.code_files(PROJECT))
        manifest = {Path(p).relative_to(PROJECT / "src").as_posix(): sha for p, sha in frozen.items()}
        self.assertEqual(implementation_digest(manifest), implementation_digest(source_manifest(PROJECT)))
        self.assertTrue(any(p.startswith("robot_operation/") for p in manifest))
        source = self.project / "src/robot_operation/new_module.py"
        source.parent.mkdir(parents=True)
        source.write_text("# developer fixture", encoding="utf-8")
        with self.assertRaises(ValueError): batch.verify(self.manifest, self.project)
        manifest = batch.prepare(self.project, self.root, self.args)
        source.write_text("# changed developer fixture", encoding="utf-8")
        with self.assertRaises(ValueError): batch.verify(manifest, self.project)

    def test_saved_budgets_and_no_shared_config_mutation(self):
        self.assertEqual(batch.read(self.root / "config/generation.json")["max_iterations"], 30)
        scene = batch.read(self.root / "config/scene.json")
        self.assertEqual((scene["max_iterations"], scene["max_model_calls"], scene["max_reserved_tokens"]),
                         (40, 400, 120000000))
        self.assertNotEqual(batch.read(self.project / "config/generation.json")["max_iterations"], 30)
        self.assertEqual(len({e["run"] for e in self.manifest["entries"]}), 5)

    def test_explicit_provider_is_frozen_without_changing_default(self):
        original=batch.read(self.project/'config/default.json')
        provider={**original,'provider':'openai','base_url':'https://api.openai.com/v1',
                  'model':'gpt-6-astra','api_key_env':'OPENAI_API_KEY',
                  'output_token_parameter':'max_completion_tokens'}
        batch.atomic_json(self.project/'config/official.json',provider)
        self.args.provider_config=Path('config/official.json')
        manifest=batch.prepare(self.project,self.root,self.args)
        self.assertEqual(batch.read(self.root/'config/provider.json'),provider)
        self.assertEqual(batch.read(self.project/'config/default.json'),original)
        batch.verify(manifest,self.project)

    def test_robot_override_is_frozen_and_shared_default_is_unchanged(self):
        original=batch.read(self.project/"config/scene_environment.json")
        optional=batch.read(PROJECT/"config/scene_environment_tiago.json")
        batch.atomic_json(self.project/"config/optional.json",optional)
        self.args.environment=Path("config/optional.json")
        manifest=batch.prepare(self.project,self.root,self.args)
        self.assertEqual(batch.read(self.root/"config/environment.json"),optional)
        self.assertEqual(batch.read(self.project/"config/scene_environment.json"),original)
        batch.verify(manifest,self.project)
        changed=batch.read(self.root/"config/environment.json")
        changed["robot"]["name"]="different"
        batch.atomic_json(self.root/"config/environment.json",changed)
        with self.assertRaises(ValueError):batch.verify(manifest,self.project)

    def test_explicit_profiles_are_copied_and_execution_uses_frozen_copies(self):
        self.args.scene_config = Path("config/custom_scene.json")
        self.args.generation_config = Path("config/custom_generation.json")
        for name, source in ((self.args.scene_config, "scene.json"),
                             (self.args.generation_config, "generation.json")):
            profile = batch.read(self.project / "config" / source)
            profile["build_timeout_seconds"] = 600
            batch.atomic_json(self.project / name, profile)
        manifest = batch.prepare(self.project, self.root, self.args)
        manifest["entries"] = manifest["entries"][:1]
        # 原配置后续变化不影响已复制的配置；篡改冻结副本则必须拒绝。
        batch.atomic_json(self.project / self.args.scene_config, {"changed": True})
        seen = []
        def runner(command, log, project, run_root):
            if command[2] == "scene_generation.preview":
                self.render(command)
            else:
                for flag in ("--config", "--generation-config"):
                    config_path = Path(command[command.index(flag) + 1])
                    self.assertEqual(config_path.parent, self.root / "config")
                    self.assertEqual(batch.read(config_path)["build_timeout_seconds"], 600)
                seen.append(command)
                self.result(run_root)
            return 0
        self.assertEqual(batch.execute(manifest, self.root, self.project, runner), 0)
        self.assertEqual(len(seen), 1)
        batch.atomic_json(self.root / "config/scene.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "已变化"):
            batch.verify(manifest, self.project)

    def test_missing_or_invalid_explicit_profile_never_falls_back(self):
        self.args.scene_config = Path("config/missing.json")
        with self.assertRaises(FileNotFoundError):
            batch.prepare(self.project, self.root, self.args)
        batch.atomic_json(self.project / self.args.scene_config, {"unexpected": True})
        with self.assertRaises(TypeError):
            batch.prepare(self.project, self.root, self.args)

    def test_unaccepted_scene_gets_preview_without_becoming_success(self):
        def runner(command,log,project,run_root):
            if command[2]=="scene_generation.preview":
                self.render(command);return 0
            self.result(run_root,status="needs_check_support");return 3
        self.assertEqual(batch.execute(self.manifest,self.root,self.project,runner),3)
        self.assertTrue(all(e["status"]=="diagnostic_available" and e.get("preview") for e in self.manifest["entries"]))
        self.assertEqual(self.manifest["status"],"completed_with_failures")

    def test_failure_does_not_stop_later_jobs_or_create_false_success(self):
        calls = []

        def runner(command, log, project, run_root):
            calls.append((command[2], run_root.name))
            if command[2] == "scene_generation.preview":
                self.render(command)
                return 0
            self.assertIn("--protocol-pdf", command)
            self.assertIn("--static-only", command)
            self.assertNotIn("--retry-uncertain", command)
            if run_root.name.endswith("np002"):
                return 3
            self.result(run_root)
            return 0

        self.assertEqual(batch.execute(self.manifest, self.root, self.project, runner), 3)
        self.assertEqual([s for _, s in calls if _ == "scene_generation"],
                         [f"synthetic_np00{i}" for i in range(1, 6)])
        self.assertEqual([e["status"] for e in self.manifest["entries"]],
                         ["completed", "failed", "completed", "completed", "completed"])
        self.assertEqual(self.manifest["status"], "completed_with_failures")
        self.assertTrue((self.root / "index.html").is_file())
        # 接续时不对已经结束的成功项、失败项重复付费调用。
        batch.execute(self.manifest, self.root, self.project,
                      lambda *args: self.fail("terminal experiment was retried"))

    def test_interrupted_scene_resumes_without_unsafe_retry_flag(self):
        self.manifest["entries"] = self.manifest["entries"][:1]
        entry = self.manifest["entries"][0]
        entry["status"] = "running"
        run_root = self.project / "runs" / entry["run"]
        batch.atomic_json(run_root / "state.json", {"calls": [{"status": "pending"}]})

        def runner(command, *unused):
            self.assertIn("--resume", command)
            self.assertNotIn("--retry-uncertain", command)
            return 3  # 模拟原有入口拒绝重发未知结果请求。

        batch.execute(self.manifest, self.root, self.project, runner)
        self.assertEqual(entry["status"], "failed")

    def test_request_failure_pauses_queue_and_can_resume_same_entry(self):
        calls = []
        def rejected(command, log, project, run_root):
            calls.append(command)
            batch.atomic_json(run_root / "state.json", {"calls": []})
            batch.atomic_json(run_root / "result.json", {
                "status":"paused", "budget":{"model_calls":1, "reported_total_tokens":0},
                "request_failure":{"kind":"provider_request", "diagnostics":{"http_status":400},
                                   "provider_error":{"message":"synthetic rejection"}}})
            return 2
        self.assertEqual(batch.execute(self.manifest, self.root, self.project, rejected), 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.manifest["status"], "paused_provider")
        self.assertEqual([e["status"] for e in self.manifest["entries"]], ["paused_provider"] + ["queued"] * 4)
        self.assertIn("synthetic rejection", (self.root / "index.html").read_text(encoding="utf-8"))

        def recovered(command, log, project, run_root):
            if command[2] == "scene_generation.preview":
                self.render(command)
            else:
                if run_root.name.endswith("np001"):
                    self.assertIn("--resume", command)
                self.result(run_root)
            return 0
        self.assertEqual(batch.execute(self.manifest, self.root, self.project, recovered), 0)
        self.assertNotIn("request_failure", self.manifest["entries"][0])

    def test_render_resume_never_regenerates_and_preserves_partial_images(self):
        self.manifest["entries"] = self.manifest["entries"][:1]
        entry = self.manifest["entries"][0]
        entry["status"] = "rendering"
        run_root = self.project / "runs" / entry["run"]
        self.result(run_root)
        (run_root / "preview").mkdir()
        old = run_root / "preview/partial.png"
        old.write_bytes(b"preserve")

        def runner(command, *unused):
            self.assertEqual(command[2], "scene_generation.preview")
            self.render(command)
            return 0

        self.assertEqual(batch.execute(self.manifest, self.root, self.project, runner), 0)
        self.assertEqual(old.read_bytes(), b"preserve")
        self.assertIn("preview_batch_1", entry["preview"])

    def test_preview_failure_recorded_separately(self):
        self.manifest["entries"] = self.manifest["entries"][:1]

        def runner(command, log, project, run_root):
            if command[2] == "scene_generation.preview":
                return 2
            self.result(run_root)
            return 0

        batch.execute(self.manifest, self.root, self.project, runner)
        self.assertEqual(self.manifest["entries"][0]["status"], "preview_failed")
        self.assertEqual(self.manifest["entries"][0]["result_status"], "static_preview_ready")

    def test_zero_exit_without_ready_artifact_is_failure(self):
        self.manifest["entries"] = self.manifest["entries"][:1]
        batch.execute(self.manifest, self.root, self.project, lambda *args: 0)
        self.assertEqual(self.manifest["entries"][0]["status"], "failed")

    def test_changed_input_stops_before_calls(self):
        (self.inputs / "NP-003/images/ref.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "已变化"):
            batch.execute(self.manifest, self.root, self.project,
                          lambda *args: self.fail("called model with changed input"))

    def test_missing_input_blocks_entire_batch_before_calls(self):
        (self.inputs / "NP-005/source.pdf").unlink()
        with self.assertRaisesRegex(ValueError, "NP-005"):
            batch.prepare(self.project, self.root, self.args)

    def test_control_c_stops_before_next_experiment(self):
        calls = []

        def runner(command, *unused):
            calls.append(command)
            return 130

        with self.assertRaises(KeyboardInterrupt):
            batch.execute(self.manifest, self.root, self.project, runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.manifest["entries"][1]["status"], "queued")

    def test_real_child_nonzero_exit_is_logged_without_ending_parent(self):
        log = self.root / "synthetic_child.log"
        code = batch.run_child([sys.executable, "-c", "print('synthetic child'); raise SystemExit(3)"],
                               log, self.project, self.project / "runs/synthetic")
        self.assertEqual(code, 3)
        self.assertIn("synthetic child", log.read_text(encoding="utf-8"))

    def test_scene_child_uses_source_bootstrap_and_keeps_expected_version(self):
        command=[sys.executable,'-m','scene_generation','--execute','--expected-implementation','approved']
        with patch.object(batch.subprocess,'Popen',return_value=SimpleNamespace(wait=lambda timeout:0)) as spawn, patch('asset_generation.process_tree.ProcessTree'):
            batch.run_child(command,self.root/'bootstrap.log',self.project,self.project/'runs/synthetic')
        actual=spawn.call_args.args[0]
        self.assertEqual(actual[1:3],['-B','-P'])
        self.assertIn(str(self.project/'src/runtime_bootstrap.py'),actual)
        self.assertIn('--expected-implementation',actual)
        self.assertEqual(actual[-1],'approved')


if __name__ == "__main__":
    unittest.main()
