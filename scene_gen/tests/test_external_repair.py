"""Robot-feedback repair: planner rows reach a static scene and dirty only the named job."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asset_generation.contracts import GenerationConfig
from asset_prep.errors import ValidationError
from scene_generation.contracts import SceneConfig
from scene_generation.external_repair import inject, layout_changes
from scene_generation.feedback import PLANNER_POLICY, planning_feedback
from scene_generation.pipeline import SceneRun
from test_mobile_layout import mobile_fixture

ROW = {"kind": "mechanism", "instances": ["station"],
       "detail": "Lid hinge has no frictionloss; requirement: stay open at any angle >= 45 deg",
       "evidence": "pushed open to 60 deg, released, fell back to 0 deg"}


class ExternalRepairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env, self.plan = mobile_fixture(self.root)
        # holder depends on the instrument; rack is independent; the tool serves every asset.
        self.plan["jobs"][1:1] = [
            {"id": "holder", "asset_ids": ["A02"], "role": "asset", "depends_on": ["instrument"], "requirements": "fixture"},
            {"id": "rack", "asset_ids": ["A03"], "role": "asset", "depends_on": [], "requirements": "fixture"}]
        self.plan["instances"] += [
            {"id": "holder_1", "job": "holder", "placement": "table", "table_id": "bench", "position": [-.5, 0, .8], "yaw": 0},
            {"id": "rack_1", "job": "rack", "placement": "table", "table_id": "bench", "position": [0, .5, .8], "yaw": 0}]
        tool = self.plan["jobs"][-1]
        tool["depends_on"], tool["asset_ids"] = ["instrument", "holder", "rack"], ["A01", "A02", "A03"]
        (self.root/"source.py").write_text("# developer fixture", encoding="utf-8")

    def run_fixture(self, feedback):
        requests = []
        commits = []

        def ask(instruction, context, images, phase):
            requests.append((instruction, copy.deepcopy(context)))
            return {"jobs": ["instrument"], "placements": [], "reason": "add hinge friction"}, {}
        job = lambda stage: {"source": "source.py", "stage": stage, "status": "candidate_ready", "dirty": False,
                             "assembly_readiness": {"loadable": True}}
        review = {"verdict": "repair", "affected_instances": ["rack_1"], "affected_layout": [], "findings": ["rack tilted"],
                  "unverified": []}
        scene = {"plan": self.plan, "order": ["instrument", "holder", "rack", "tool"], "rounds": ["scene"], "stages": [],
                 "status": "repair_tasks_deferred", "feedback": {"status": "fail", "checks": []},
                 "latest": {"directory": "scene", "report": {"status": "fail", "checks": []}, "review": review},
                 "jobs": {"instrument": job("a1"), "holder": job("a2"), "rack": job("a3"), "tool": job("a4")}}
        inject(scene, feedback, "sha")
        run = SimpleNamespace(root=self.root, env=self.env, contract={}, generation=GenerationConfig(),
            before_repair=lambda action: None, robot_joints={}, catalogue={"asset_ids": ["A01", "A02", "A03"]},
            limits=SceneConfig(validation_mode="static"), check_pause=lambda: None, repair_input_key=lambda: "fixture",
            progress=lambda phase, event: None, implementation_guard=SimpleNamespace(expected="fixture"),
            validate_prepared_plan=lambda plan: None,
            store=SimpleNamespace(state={"calls": []}, save=lambda: None), model=SimpleNamespace(ask=ask),
            begin=lambda kind: self.root, commit=lambda kind, record: commits.append(record), state=scene)
        self.assertTrue(SceneRun.route_repair(run, use_review=False))
        return run, commits, requests

    def test_one_feedback_row_dirties_only_its_job_and_revalidates_dependents(self):
        feedback = planning_feedback({"scene_sha256": "abc", "failures": [ROW]}, "abc", self.plan)
        run, commits, requests = self.run_fixture(feedback)
        jobs = run.state["jobs"]
        self.assertEqual(commits[-1]["dirty_jobs"], ["instrument"])
        self.assertTrue(jobs["instrument"]["dirty"] and jobs["instrument"]["regenerate"])
        for key in ("holder", "tool"):
            self.assertEqual(jobs[key]["status"], "review_pending")
            self.assertTrue(jobs[key]["dirty"])
            self.assertNotIn("regenerate", jobs[key])
        self.assertEqual(jobs["rack"], {"source": "source.py", "stage": "a3", "status": "candidate_ready", "dirty": False,
                                        "assembly_readiness": {"loadable": True}})
        self.assertEqual(sorted(commits[-1]["layout_revalidation_jobs"]), ["holder", "tool"])
        # The visual verdict naming rack_1 is not in scope; the planner rows and their policy are.
        instruction, context = requests[-1]
        self.assertTrue(instruction.endswith(PLANNER_POLICY))
        self.assertEqual(context["allowed_jobs"], ["instrument"])
        self.assertIsNone(context["visual"])
        task = context["repair_tasks"][0]
        self.assertEqual(task["check_id"], "planner:mechanism:0")
        self.assertEqual(task["external_feedback"]["detail"], ROW["detail"])
        self.assertEqual(run.state["repair_history"][-1]["target_checks"], ["planner:mechanism:0"])
        self.assertEqual(layout_changes(self.plan, run.state["plan"])["moved_instances"], [])

    def test_stale_or_foreign_feedback_is_refused(self):
        with self.assertRaises(ValidationError):
            planning_feedback({"scene_sha256": "old", "failures": [ROW]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            planning_feedback({"scene_sha256": "abc", "failures": [{**ROW, "instances": ["unknown"]}]}, "abc", self.plan)

    def test_static_run_still_refuses_planner_feedback(self):
        run = SimpleNamespace(limits=SceneConfig(validation_mode="static"))
        with self.assertRaises(ValidationError):
            SceneRun.run(run, {"scene_sha256": "abc", "failures": [ROW]})


if __name__ == "__main__":
    unittest.main()
