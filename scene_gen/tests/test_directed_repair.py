"""Directed repair: verified edits reach a static scene deterministically, without a model call."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asset_generation.contracts import GenerationConfig
from asset_prep.errors import ValidationError
from scene_generation.arrangement import arrange
from scene_generation.contracts import SceneConfig, validate_plan
from scene_generation.directed_repair import check_edits, plan_edits, record_repair
from scene_generation.layout import local_xy
from test_arrangement import box
from test_mobile_layout import mobile_fixture

ASSETS = ["A01", "A02", "A03"]
SHAPES = {"station": box(.4, .3), "holder_1": box(.3, .3), "rack_1": box(.2, .2)}


def edit(kind, instance, params, **extra):
    return {"kind": kind, "instance": instance, "params": params, "requirement": "verified in simulation",
            "evidence": {"trials": 5, "success": 5}, **extra}


class DirectedRepairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env, plan = mobile_fixture(self.root)
        # holder depends on the instrument; rack is independent; the tool serves every asset.
        plan["jobs"][1:1] = [
            {"id": "holder", "asset_ids": ["A02"], "role": "asset", "depends_on": ["instrument"], "requirements": "fixture"},
            {"id": "rack", "asset_ids": ["A03"], "role": "asset", "depends_on": [], "requirements": "fixture"}]
        plan["instances"] += [
            {"id": "holder_1", "job": "holder", "placement": "table", "table_id": "bench", "position": [-.5, 0, .8], "yaw": 0},
            {"id": "rack_1", "job": "rack", "placement": "table", "table_id": "bench", "position": [0, .5, .8], "yaw": 0}]
        tool = plan["jobs"][-1]
        tool["depends_on"], tool["asset_ids"] = ["instrument", "holder", "rack"], list(ASSETS)
        self.limits = SceneConfig(validation_mode="static")
        # The published scene is an arranged one.
        self.plan = arrange(plan, SHAPES, self.env)
        validate_plan(self.plan, ASSETS, self.limits, self.env)

    def apply(self, *edits):
        checked = check_edits({"scene_sha256": "abc", "edits": list(edits)}, "abc", self.plan)
        return plan_edits(self.plan, checked, SHAPES, self.env, self.limits, ASSETS), checked

    def test_move_changes_only_the_targeted_instance(self):
        (updated, value, _), _ = self.apply(edit("move_instance", "holder_1", {"dx": .05, "dy": .02, "dyaw": .1}))
        self.assertEqual(value["jobs"], [])
        before = {i["id"]: i for i in self.plan["instances"]}
        after = {i["id"]: i for i in updated["instances"]}
        for key in ("station", "rack_1", "gripper"):
            self.assertEqual(after[key], before[key])
        self.assertAlmostEqual(after["holder_1"]["position"][0]-before["holder_1"]["position"][0], .05, places=6)
        self.assertAlmostEqual(after["holder_1"]["position"][1]-before["holder_1"]["position"][1], .02, places=6)
        self.assertAlmostEqual(after["holder_1"]["yaw"]-before["holder_1"]["yaw"], .1, places=6)
        self.assertEqual(updated["layout"]["tables"], self.plan["layout"]["tables"])
        docks = lambda p: {d["instance"]: d for d in p["layout"]["docks"]}
        self.assertEqual({k: v for k, v in docks(updated).items() if k != "holder_1"},
                         {k: v for k, v in docks(self.plan).items() if k != "holder_1"})
        # The override is part of the plan: a later re-arrangement keeps the verified pose.
        self.assertEqual(arrange(updated, SHAPES, self.env)["instances"], updated["instances"])
        validate_plan(updated, ASSETS, self.limits, self.env)

    def test_clearance_widens_only_the_named_gap(self):
        (updated, _, _), _ = self.apply(edit("clearance", "rack_1", {"gap_m": .3}))
        table = next(t for t in updated["layout"]["tables"] if t["id"] == "bench")
        old_table = next(t for t in self.plan["layout"]["tables"] if t["id"] == "bench")
        x = lambda plan, t, key: local_xy(next(i for i in plan["instances"] if i["id"] == key)["position"], t)[0]
        order = sorted(("station", "holder_1", "rack_1"), key=lambda k: x(self.plan, old_table, k))
        self.assertEqual(order, sorted(order, key=lambda k: x(updated, table, k)))
        widths = {"station": .4, "holder_1": .3, "rack_1": .2}
        gap = lambda plan, t, a, b: x(plan, t, b)-x(plan, t, a)-(widths[a]+widths[b])/2
        for a, b in zip(order, order[1:]):
            expected = .3 if "rack_1" in (a, b) else .1
            self.assertAlmostEqual(gap(updated, table, a, b), expected, places=6)
            self.assertAlmostEqual(gap(self.plan, old_table, a, b), .1, places=6)

    def test_joint_edit_marks_only_its_job_for_regeneration(self):
        (updated, value, order), checked = self.apply(
            edit("joint_param", "rack_1", {"joint": "lid_hinge", "frictionloss": .4}, requirement="Lid stays open at >= 45 deg"))
        self.assertEqual(value["jobs"], ["rack"])
        self.assertEqual(updated["instances"], self.plan["instances"])
        rack = next(j for j in updated["jobs"] if j["id"] == "rack")
        self.assertIn("Lid stays open at >= 45 deg", rack["requirements"])
        self.assertIn('"frictionloss": 0.4', rack["requirements"])
        for job in updated["jobs"]:
            if job["id"] != "rack":
                self.assertEqual(job, next(j for j in self.plan["jobs"] if j["id"] == job["id"]))
        commits = []
        job = lambda stage: {"source": "source.py", "stage": stage, "status": "candidate_ready", "dirty": False,
                             "assembly_readiness": {"loadable": True}}
        scene = {"plan": self.plan, "order": order, "rounds": ["scene"], "stages": [], "status": "repairing_scene",
                 "latest": {"directory": "scene", "assessed": True},
                 "jobs": {"instrument": job("a1"), "holder": job("a2"), "rack": job("a3"), "tool": job("a4")}}
        untouched = copy.deepcopy(scene["jobs"])
        run = SimpleNamespace(root=self.root, env=self.env, generation=GenerationConfig(), robot_joints={},
            before_repair=lambda action: None, catalogue={"asset_ids": ASSETS}, limits=self.limits,
            repair_input_key=lambda: "fixture", validate_prepared_plan=lambda plan: None,
            store=SimpleNamespace(state={"calls": []}, save=lambda: None),
            begin=lambda kind: self.root, commit=lambda kind, record: commits.append(record), state=scene)
        record_repair(run, updated, value, checked)
        jobs = scene["jobs"]
        self.assertEqual(commits[-1]["dirty_jobs"], ["rack"])
        self.assertEqual([k for k, v in jobs.items() if v.get("regenerate")], ["rack"])
        # The tool depends on the rack: it is revalidated, never re-authored.
        self.assertEqual(commits[-1]["layout_revalidation_jobs"], ["tool"])
        self.assertEqual(jobs["tool"]["status"], "review_pending")
        for key in ("instrument", "holder"):
            self.assertEqual(jobs[key], untouched[key])
        entry = scene["repair_history"][-1]
        self.assertEqual(entry["target_checks"], [checked["edits"][0]["id"]])
        self.assertEqual(entry["baseline"]["jobs"]["rack"], "a3")
        self.assertEqual(scene["repair_tasks"]["rack"][0]["parameters"], {"joint": "lid_hinge", "frictionloss": .4})
        self.assertEqual(scene["plan"], updated)

    def test_bad_scene_digest_unknown_instance_and_out_of_range_are_refused(self):
        move = edit("move_instance", "holder_1", {"dx": .05})
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "old", "edits": [move]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "abc", "edits": [{**move, "instance": "unknown"}]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "abc", "edits": [edit("move_instance", "holder_1", {"dx": 5})]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "abc", "edits": [edit("clearance", "rack_1", {"gap_m": -1})]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "abc", "edits": [edit("move_instance", "gripper", {"dx": .1})]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "abc", "edits": [edit("teleport", "holder_1", {"dx": .1})]}, "abc", self.plan)
        with self.assertRaises(ValidationError):
            check_edits({"scene_sha256": "abc", "edits": []}, "abc", self.plan)

    def test_invalid_arrangement_override_is_refused_by_the_plan_contract(self):
        broken = copy.deepcopy(self.plan)
        broken["instances"][0]["arrangement"] = {"gap_m": 5}
        with self.assertRaises(ValidationError):
            validate_plan(broken, ASSETS, self.limits, self.env)


if __name__ == "__main__":
    unittest.main()
