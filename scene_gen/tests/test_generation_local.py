import os
import time
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from asset_generation.contracts import GenerationConfig, check_profile
from asset_generation.local import LocalBackend, artifact_files
from asset_prep.errors import ValidationError


class LocalBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = LocalBackend(GenerationConfig(), Path(__file__).resolve().parents[1])

    def test_environment_does_not_inherit_credentials_and_reports_no_isolation(self):
        with patch.dict(os.environ, {"HARNESS_MODEL_API_KEY": "fixture", "OPENAI_API_KEY": "fixture"}):
            env = self.backend.worker_environment(self.root)
        self.assertNotIn("HARNESS_MODEL_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertFalse(self.backend.preflight()["os_isolation"])

    def test_real_worker_exception_is_reported_and_logged(self):
        source = self.root/"model.py"
        source.write_text("def build(output_dir):\n    raise RuntimeError('fixture build failure')\n", encoding="utf-8")
        report, renders = self.backend.build_and_check(source, self.root, ["A01"], check_profile({}))
        self.assertEqual(report["status"], "fail")
        self.assertIsNone(renders)
        self.assertIn("fixture build failure", report["checks"][0]["detail"])
        self.assertTrue((self.root/"local-build.log").is_file())
        import json
        receipt=json.loads((self.root/"local-build-exit.json").read_text())
        self.assertEqual(receipt["status"],"process_failed")
        self.assertNotEqual(receipt["exit_code"],0)
        self.assertFalse(list(self.root.glob("local-worker-*")))

    def test_real_worker_timeout_does_not_become_a_candidate(self):
        self.backend.config = replace(self.backend.config, build_timeout_seconds=1)
        source = self.root/"model.py"
        source.write_text("import time\ndef build(output_dir):\n    time.sleep(60)\n", encoding="utf-8")
        report, _ = self.backend.build_and_check(source, self.root, ["A01"], check_profile({}))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report['error_kind'],'backend_execution')
        self.assertIn("timed out", ' '.join(report['coverage']))
        import json
        self.assertEqual(json.loads((self.root/"local-build-exit.json").read_text())["status"],"timeout")
        self.assertFalse(list(self.root.glob("local-worker-*")))

    def test_pause_interrupts_worker_without_waiting_for_work_budget(self):
        from asset_prep.errors import PrepError
        source=self.root/'model.py'
        source.write_text('import time\ndef build(output_dir):\n    time.sleep(60)\n',encoding='utf-8')
        output=self.root/'output';output.mkdir()
        started=time.monotonic()
        def cancel():
            if time.monotonic()-started > 1:
                raise PrepError('pause requested')
        self.backend.cancel_guard=cancel
        with self.assertRaisesRegex(PrepError,'pause requested'):
            self.backend._run('build',[(source,'/input/model.py')],output)
        self.assertLess(time.monotonic()-started,10)
        self.assertFalse(list(self.root.glob('local-worker-*')))

    def test_excessive_worker_logging_is_bounded(self):
        source = self.root/"model.py"
        source.write_text("def build(output_dir):\n    print('x' * 2000000)\n    print('final-log-marker')\n    (output_dir/'model.xml').write_text('<mujoco/>')\n", encoding="utf-8")
        output = self.root/"output"
        output.mkdir()
        self.backend._run("build", [(source, "/input/model.py")], output)
        self.assertLessEqual((self.root/"local-build.log").stat().st_size, 1024 * 1024)
        self.assertTrue((output/"model.xml").is_file())
        self.assertIn("final-log-marker",(self.root/"local-build.log").read_text())

    def test_descendant_is_killed_even_after_worker_exits(self):
        source=self.root/"model.py"
        source.write_text("import subprocess,sys\ndef build(output_dir):\n"
            "    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
            "    (output_dir/'child.txt').write_text(str(child.pid))\n",encoding="utf-8")
        output=self.root/"output"
        output.mkdir()
        started=time.monotonic()
        self.backend._run("build",[(source,"/input/model.py")],output)
        self.assertLess(time.monotonic()-started,10)
        pid=int((output/"child.txt").read_text())
        if os.name == "nt":
            import ctypes as c
            from ctypes import wintypes as w
            api=c.WinDLL("kernel32",use_last_error=True)
            api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
            api.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];api.WaitForSingleObject.restype=w.DWORD
            api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL
            handle=api.OpenProcess(0x100000,False,pid)
            if handle:
                try:
                    self.assertEqual(api.WaitForSingleObject(handle,2000),0)
                finally:
                    api.CloseHandle(handle)
        self.assertFalse(list(self.root.glob("local-worker-*")))

    def test_output_limits_and_hardlinks_are_rejected_before_copy(self):
        (self.root/"model.xml").write_text("oversize", encoding="utf-8")
        with self.assertRaises(ValidationError):
            artifact_files(self.root, replace(self.backend.config, max_artifact_bytes=2))
        os.link(self.root/"model.xml", self.root/"link.xml")
        with self.assertRaises(ValidationError):
            artifact_files(self.root, self.backend.config)


    def test_abrupt_worker_exit_keeps_failure_receipt(self):
        import json
        source=self.root/"model.py"
        source.write_text("import os\ndef build(output_dir):\n    os._exit(23)\n",encoding="utf-8")
        report,_=self.backend.build_and_check(source,self.root,["A01"],check_profile({}))
        receipt=json.loads((self.root/"local-build-exit.json").read_text())
        self.assertEqual(report["status"],"error")
        self.assertEqual(report["error_kind"],"backend_execution")
        self.assertEqual(receipt["status"],"process_failed")
        self.assertEqual(receipt["exit_code"],23)
