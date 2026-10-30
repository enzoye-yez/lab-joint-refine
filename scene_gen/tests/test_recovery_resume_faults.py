"""Developer fixtures: program faults must never trigger geometric authoring."""
import json
import unittest
from asset_generation.pipeline import snapshot
from generation_fixtures import FixtureSandbox,SOURCE
from scene_fixtures import SceneSandbox
import test_scene_pipeline as scene_tests
import test_generation_pipeline as generation_tests

class SceneFaultResumeTests(unittest.TestCase):
    setUp=scene_tests.ScenePipelineTests.setUp
    run_loop=scene_tests.ScenePipelineTests.run_loop
    def test_program_failure_resume_is_read_only(self):
        class Broken(SceneSandbox):
            def build_and_check(self,*args):
                r,p=super().build_and_check(*args)
                r.update(status="error",execution_failure={"kind":"program","components":["synthetic"],"retryable":False})
                return r,p
            def recheck_asset(self,*args):
                return {"status":"error","checks":[],"renders":[],"error_kind":"check_execution",
                    "execution_failure":{"kind":"program","components":["synthetic"],"retryable":False}},None
        first,_=self.run_loop(self.base_replies[:2],sandbox=Broken())
        self.assertEqual(first["status"],"needs_environment")
        before=json.loads((self.root/"run/state.json").read_text(encoding="utf-8"))
        artifact=self.root/"run"/before["scene"]["jobs"]["instrument"]["artifact"]
        files=snapshot(artifact);backend=Broken()
        result,transport=self.run_loop([],resume=True,sandbox=backend)
        self.assertEqual(result["status"],"needs_environment")
        self.assertEqual(backend.build_calls,0);self.assertEqual(transport.requests,[])
        self.assertEqual(files,snapshot(artifact))
        self.assertEqual(before["calls"],json.loads((self.root/"run/state.json").read_text(encoding="utf-8"))["calls"])

class StandaloneFaultResumeTests(unittest.TestCase):
    setUp=generation_tests.GenerationPipelineTests.setUp
    run_loop=generation_tests.GenerationPipelineTests.run_loop
    def test_program_failure_resume_is_read_only(self):
        class Broken(FixtureSandbox):
            def build_and_check(self,*args):
                r,p=super().build_and_check(*args)
                r.update(status="error",execution_failure={"kind":"program","components":["synthetic"],"retryable":False})
                return r,p
        result,_=self.run_loop([{"model_py":SOURCE}],sandbox=Broken())
        self.assertEqual(result["status"],"needs_environment")
        files=snapshot(self.run/"revisions/0001");backend=Broken()
        result,transport=self.run_loop([],resume=True,sandbox=backend)
        self.assertEqual(result["status"],"needs_environment")
        self.assertEqual(backend.calls,0);self.assertEqual(transport.requests,[])
        self.assertEqual(files,snapshot(self.run/"revisions/0001"))
