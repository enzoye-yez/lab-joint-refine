import io
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from asset_prep.errors import ValidationError
from asset_generation.contracts import GenerationConfig
from asset_generation.sandbox import DockerSandbox, SandboxUnavailable, extract_artifacts


def archive(name, content=b"{}", kind=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tf:
        item = tarfile.TarInfo(name)
        item.size = len(content)
        if kind:
            item.type = kind
            item.linkname = "../../private"
        tf.addfile(item, io.BytesIO(content))
    buffer.seek(0)
    return buffer


class GenerationSandboxTests(unittest.TestCase):
    def test_archive_rejects_escape_links_and_unbounded_data(self):
        with tempfile.TemporaryDirectory() as root:
            config = GenerationConfig()
            for name, kind in [("../model.xml", None), ("/model.xml", None), ("x.json", tarfile.SYMTYPE),
                               ("x.json", tarfile.LNKTYPE), ("C:\\private.json", None), ("code.py", None)]:
                with self.subTest(name=name, kind=kind), self.assertRaises(ValidationError):
                    extract_artifacts(archive(name, kind=kind), root, config)
            with self.assertRaises(ValidationError):
                extract_artifacts(archive("model.xml", b"oversized"), root, replace(config, max_artifact_bytes=2))

    def test_only_artifact_files_are_extracted(self):
        with tempfile.TemporaryDirectory() as root:
            files = extract_artifacts(archive("./interfaces.json"), root, GenerationConfig())
            self.assertEqual(files, ["interfaces.json"])
            self.assertEqual((Path(root)/files[0]).read_text(), "{}")

    def test_container_command_is_readonly_without_host_credentials(self):
        sandbox = DockerSandbox(GenerationConfig(), Path(__file__).resolve().parents[1])
        sandbox.executable = "docker"
        sandbox.image_id = "sha256:fixture"
        command = sandbox.command("fixture", [(Path("example/model.py"), "/input/model.py")], "build")
        for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--log-driver=none"):
            self.assertIn(flag, command)
        mounts = [command[i+1] for i, x in enumerate(command) if x == "--mount"]
        self.assertTrue(all(m.endswith(",readonly") for m in mounts))
        self.assertNotIn(".env", " ".join(command))
        self.assertNotIn("--privileged", command)
        # 渲染器和BLAS线程必须在容器进程上限内，否则真实检查会卡住。
        self.assertIn("--pids-limit=64",command)
        self.assertIn("--env=LP_NUM_THREADS=2",command)
        self.assertIn("--env=OPENBLAS_NUM_THREADS=1",command)

    def test_missing_docker_has_no_local_execution_fallback(self):
        with patch("asset_generation.sandbox.shutil.which", return_value=None):
            sandbox = DockerSandbox(GenerationConfig(), Path.cwd())
        with self.assertRaises(SandboxUnavailable):
            sandbox.preflight()

    def test_artifacts_stream_from_running_tmpfs_and_container_is_removed(self):
        # 本地替身核对容器生命周期，不宣称实际Docker服务已经验证。
        sandbox = DockerSandbox(GenerationConfig(), Path(__file__).resolve().parents[1])
        sandbox.executable = "docker"
        sandbox.image_id = "sha256:fixture"
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="0\n")
        process = SimpleNamespace(stdout=archive("model.xml", b"<mujoco/>"),
                                  wait=lambda timeout: 0, poll=lambda: 0, kill=lambda: None)
        with tempfile.TemporaryDirectory() as folder, \
             patch("asset_generation.sandbox.subprocess.run", side_effect=run), \
             patch("asset_generation.sandbox.subprocess.Popen", return_value=process) as popen:
            root = Path(folder)/"output"
            root.mkdir()
            files = sandbox._run("build", [], root)
            self.assertEqual(files, ["model.xml"])
            stream_command = popen.call_args.args[0]
            self.assertEqual(stream_command[1], "exec")
            self.assertIn("tar", stream_command)
            self.assertNotIn("cp", stream_command)
            self.assertEqual(calls[-1][1:3], ["rm", "--force"])


if __name__ == "__main__":
    unittest.main()
