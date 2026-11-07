import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_refine import diagnose, feedback, judge  # noqa: E402

INSTANCES = [{"id": "tips_small", "job": "J05", "configuration": "box"}, {"id": "balance", "job": "J02", "configuration": "x"}]
FINDING = {"id": "asset:M4:tips_small", "attribution": "asset", "kind": "mechanism", "instance": "tips_small",
           "blocked_tasks": ["tips_small__lid_open"], "measured": {"frictionloss": 0.0, "gravity_torque_nm": 0.031},
           "requirement": {"hinge_holds_open_from_deg": 45}}


def failure(**over):
    row = {"kind": "mechanism", "instances": ["tips_small"],
           "detail": "Lid hinge frictionloss 0.0 must exceed the 0.031 N m gravity torque at 45 degrees.",
           "evidence": "Blocks 1 task."}
    row.update(over)
    return row


class FeedbackValidation(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(feedback.validate({"failures": [failure()]}, [FINDING], INSTANCES), [])

    def test_invented_number(self):
        errors = feedback.validate({"failures": [failure(detail="Raise frictionloss to 0.07 N m.")]}, [FINDING], INSTANCES)
        self.assertTrue(any("0.07" in e for e in errors))

    def test_unknown_instance_and_kind(self):
        errors = feedback.validate({"failures": [failure(instances=["ghost"], kind="paint")]}, [FINDING], INSTANCES)
        self.assertTrue(any("ghost" in e for e in errors) and any("paint" in e for e in errors))

    def test_instance_digits_are_not_numbers(self):
        self.assertEqual(feedback.numbers("pipette1 and A04 are 3 mm apart"), {3.0})


def write_round(root, results, findings):
    (root / "trajectory").mkdir(parents=True)
    (root / "diagnosis").mkdir(parents=True)
    (root / "trajectory/results.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    (root / "diagnosis/findings.json").write_text(json.dumps({"findings": findings}))


class Judge(unittest.TestCase):
    def test_outcomes(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "r0", Path(t) / "r1"
            r = lambda task, ok: {"task": task, "success": ok, "reason": None if ok else "x"}
            write_round(a, [r("t1", False), r("t2", False), r("t3", True)], [
                {"id": "asset:M4:i1", "attribution": "asset", "kind": "mechanism", "instance": "i1", "blocked_tasks": ["t1"]},
                {"id": "layout:R1:i2", "attribution": "layout", "kind": "unreachable", "instance": "i2", "blocked_tasks": ["t2"]}])
            write_round(b, [r("t1", True), r("t2", False), r("t3", False)], [
                {"id": "layout:R1:i2", "attribution": "layout", "kind": "unreachable", "instance": "i2", "blocked_tasks": ["t2"]},
                {"id": "asset:C3:i3", "attribution": "asset", "kind": "collision", "instance": "i3", "blocked_tasks": ["t3"]}])
            s = judge.judge(a, b)
            self.assertEqual(s["transitions"], {"fixed": 1, "still_fail": 1, "regressed": 1})
            self.assertEqual(s["findings"], {"resolved": 1, "persisting": 1})
            self.assertEqual(s["new_findings"], ["asset:C3:i3"])


class Attribution(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(diagnose.attribution("M4_falls_back"), ("asset", "mechanism"))
        self.assertEqual(diagnose.attribution("L1_layout_beyond_reach"), ("layout", "unreachable"))
        self.assertEqual(diagnose.attribution("E1_arm_path_or_tracking")[0], "method")
        self.assertEqual(diagnose.attribution("M4_rotation_beyond_wrist")[0], "method")


if __name__ == "__main__":
    unittest.main()
