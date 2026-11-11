"""Regression tests for the NP-001/NP-002 convergence failures; developer fixtures only."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco

from scene_generation.evidence import (evidence_progress, question_alias, question_identity,
                                       semantic_text, working_state)


def round_with(unresolved, requests=1, failed=0):
    results = [{"request": {"kind": "view"}, "error": "x"} for _ in range(failed)]
    results += [{"request": {"kind": "view"}} for _ in range(requests - failed)]
    return {"unresolved": unresolved, "requests": [{"kind": "view"}] * requests,
            "evidence": {"results": results}}


class EvidenceProgressTests(unittest.TestCase):
    def test_successful_render_without_answered_question_is_not_progress(self):
        rounds = [round_with({"q1": {"e": 1}}), round_with({"q1": {"e": 1}, "q2": {"e": 2}})]
        progress = evidence_progress({}, rounds)
        self.assertFalse(progress["progress"])
        self.assertEqual(progress["added"], 1)

    def test_answered_question_counts_as_progress(self):
        rounds = [round_with({"q1": {"e": 1}, "q2": {"e": 2}}), round_with({"q2": {"e": 2}})]
        self.assertTrue(evidence_progress({}, rounds)["progress"])

    def test_answer_offset_by_more_new_gaps_is_not_progress(self):
        rounds = [round_with({"q1": {}}), round_with({"q2": {}, "q3": {}})]
        self.assertFalse(evidence_progress({}, rounds)["progress"])

    def test_all_failed_window_is_saturated(self):
        rounds = [round_with({"q1": {}}, 2, 2), round_with({}, 2, 2)]
        self.assertTrue(evidence_progress({}, rounds)["saturated"])


class QuestionIdentityTests(unittest.TestCase):
    def test_canonical_identity_is_unchanged_for_saved_sessions(self):
        from asset_prep.storage import digest
        reason = "缺少内部视图"
        self.assertEqual(question_identity(reason), "issue_" + digest(reason)[:16])

    def test_punctuation_and_spacing_rewrites_share_an_alias(self):
        a = "盖锁 卡扣，未观察到；需要剖视。"
        b = "盖锁卡扣,未观察到;需要剖视"
        self.assertEqual(question_alias(a), question_alias(b))
        self.assertNotEqual(question_alias(a), question_alias("底脚连接未观察到"))

    def test_reworded_blocker_signature_matches(self):
        self.assertEqual(semantic_text("材料 量测，失败。"), semantic_text("材料量测,失败"))


class WorkingStateTests(unittest.TestCase):
    def test_working_state_is_bounded_and_keeps_outcomes(self):
        actions = {f"a{i}": {"status": "completed", "effective": i % 2 == 0, "failure": "x" * 900}
                   for i in range(60)}
        state = {"actions": actions, "capability_gaps": {"q": {"kind": "observation_incomplete"}}}
        unresolved = {f"u{i}": {"request": {"kind": "view"}, "error": "e"} for i in range(60)}
        value = working_state(state, unresolved, {f"issue_{i}": "r" for i in range(60)})
        self.assertEqual(len(value["attempted_actions"]), 24)
        self.assertEqual(len(value["open_questions"]), 24)
        self.assertEqual(len(value["questions"]), 24)
        self.assertTrue(all(len(r.get("failure", "")) <= 300 for r in value["attempted_actions"]))
        self.assertIn("q", value["capability_gaps"])



class SegmentationFallbackTests(unittest.TestCase):
    """MuJoCo segmentation maps scene-local slots to global ids; hidden parts can overflow it."""

    XML = '''<mujoco><compiler angle="radian"/><worldbody>
    <body name="base"><geom name="floor_g" type="box" size=".2 .2 .01"/>
    <body name="cover" pos="0 0 .1"><geom name="cover_g" type="box" size=".05 .05 .005"/></body>
    <body name="part" pos="0 0 .05"><geom name="part_g" type="box" size=".02 .02 .02"/></body>
    </body></worldbody></mujoco>'''

    def test_index_error_falls_back_to_depth_visibility(self):
        from asset_generation import observation
        from asset_generation.evidence import descendants
        model = mujoco.MjModel.from_xml_string(self.XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        request = {"kind": "view", "joints": {}, "hide_bodies": ["cover"], "focus_body": "part",
                   "azimuth": 45, "elevation": -45}
        original = mujoco.Renderer.render

        def overflowing(self, *args, **kwargs):
            if self._segmentation_rendering:
                raise IndexError("index 269 is out of bounds for axis 0 with size 261")
            return original(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(mujoco.Renderer, "render", overflowing):
            result = observation.render_observation(model, data, request, temporary, 0, descendants)
        self.assertGreater(result["quality"]["target_only_pixels"], 0)
        self.assertGreater(result["quality"]["target_pixels"], 0)
        self.assertTrue(result["quality"]["observable"], result["quality"])
        self.assertTrue((model.geom_group == mujoco.MjModel.from_xml_string(self.XML).geom_group).all(),
                        "visibility probing must restore the model's geom groups")


def collect_run(jobs, attempts=None):
    return SimpleNamespace(
        state={"plan": {"jobs": jobs}, "order": [j["id"] for j in jobs], "jobs": {},
               "candidate_attempts": attempts or {}},
        generation=SimpleNamespace(max_iterations=60), env={"robot": {}},
        implementation_guard=SimpleNamespace(expected="developer-test"),
        store=SimpleNamespace(save=lambda: None), check_pause=lambda: None, sandbox=SimpleNamespace())


class CollectProgressTests(unittest.TestCase):
    def test_failing_job_publishes_between_bounded_repair_attempts(self):
        from scene_generation import scene_first
        calls = []
        run = collect_run([{"id": "lid", "depends_on": [], "role": "asset", "asset_ids": ["A1"]}])

        def generate_job(job, feedback, draft):
            calls.append(job["id"])
            run.state["jobs"][job["id"]] = {"stage": f"s{len(calls)}", "feedback": {}}
            return False
        run.generate_job = generate_job
        run.generation.max_iterations=3
        publications=[]
        with patch.object(scene_first, "usable", lambda run, record: False), \
             patch.object(scene_first,"publish",side_effect=lambda *args:publications.append(len(calls))):
            missing = scene_first.collect(run)
        self.assertEqual(missing, ["lid"])
        self.assertEqual(calls, ["lid"]*3)
        self.assertEqual(publications,[1,2,3])
        self.assertEqual(run.state['blocked_jobs']['lid']['kind'],'allocation_exhausted')

    def test_dependency_that_stopped_does_not_block_a_fixture(self):
        from scene_generation import scene_first
        calls = []
        exhausted = [{"cycle": "c", "status": "completed"}] * 60
        run = collect_run([{"id": "stuck", "depends_on": [], "role": "asset", "asset_ids": ["A1"]},
                           {"id": "fixture", "depends_on": ["stuck"], "role": "fixture", "asset_ids": ["A1"]}],
                          {"stuck": exhausted})

        def generate_job(job, feedback, draft):
            calls.append(job["id"])
            run.state["jobs"][job["id"]] = {"stage": "s", "feedback": {}}
            return False
        run.generate_job = generate_job
        with patch.object(scene_first, "usable", lambda run, record: False), \
             patch.object(scene_first,"publish",return_value=None), \
             patch("scene_generation.layout_planning.refine", lambda run: None):
            scene_first.collect(run)
        self.assertEqual(run.state["blocked_jobs"]["stuck"]["kind"], "allocation_exhausted")
        self.assertIn("fixture", calls)


if __name__ == "__main__":
    unittest.main()
