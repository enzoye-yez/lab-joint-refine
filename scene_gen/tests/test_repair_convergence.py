"""Repair convergence: mechanism-level rejection, partial progress, and history injection."""
import unittest

from scene_generation.capabilities import repair_mechanism, repair_signature, repeated_repair
from scene_generation.progress import failed_approaches, repair_comparison

KEY = "problem-1"
INPUT = "input-1"


def evaluated(action, outcome, mechanism=None, problem=KEY, input_key=INPUT):
    return {"status": "evaluated", "outcome": outcome, "problem_key": problem,
            "input_key": input_key, "output_key": input_key + "-out",
            "action_key": repair_signature(action),
            "mechanism": mechanism if mechanism is not None else repair_mechanism(action),
            "action": action,
            "comparison": {"outcome": outcome, "new_defects": [], "resolved_checks": [],
                           "remaining_checks": [], "coverage_lost": []}}


class MechanismTests(unittest.TestCase):
    def test_same_placement_with_new_numbers_is_one_mechanism(self):
        a = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}], "reason": "move left"}
        b = {"placements": [{"id": "station", "position": [0.2, 0, 0.8], "yaw": 0.3}], "reason": "move right"}
        self.assertEqual(repair_mechanism(a), repair_mechanism(b))
        self.assertNotEqual(repair_signature(a), repair_signature(b))

    def test_different_lever_is_a_different_mechanism(self):
        move = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}]}
        geometry = {"jobs": ["asset01"], "geometry_changes": [{"job": "asset01", "operation": "shorten lug",
                                                             "parameters": {"depth_m": 0.002}}]}
        self.assertNotEqual(repair_mechanism(move), repair_mechanism(geometry))

    def test_exact_action_repeat_is_rejected_immediately(self):
        action = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}]}
        history = [evaluated(action, "regressed")]
        verdict = repeated_repair(history, INPUT, KEY, repair_signature(action),
                                  mechanism=repair_mechanism(action))
        self.assertEqual(verdict["reason"], "exact_action_repeated")

    def test_repeated_mechanism_does_not_forbid_new_parameters(self):
        first = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}]}
        second = {"placements": [{"id": "station", "position": [0.2, 0, 0.8], "yaw": 0}]}
        third = {"placements": [{"id": "station", "position": [0.3, 0, 0.8], "yaw": 0}]}
        # One prior failure leaves the direction open: a single bad attempt can be noise.
        one = [evaluated(first, "regressed")]
        self.assertIsNone(repeated_repair(one, INPUT, KEY, repair_signature(second),
                                          mechanism=repair_mechanism(second)))
        # Failures of two parameter choices do not disprove a third choice.
        two = [evaluated(first, "regressed"), evaluated(second, "ineffective")]
        verdict = repeated_repair(two, INPUT, KEY, repair_signature(third),
                                  mechanism=repair_mechanism(third))
        self.assertIsNone(verdict)

    def test_a_different_mechanism_stays_available_after_placements_are_exhausted(self):
        moves = [evaluated({"placements": [{"id": "station", "position": [i / 10, 0, 0.8], "yaw": 0}]},
                           "regressed") for i in (1, 2, 3)]
        geometry = {"jobs": ["asset01"],
                    "geometry_changes": [{"job": "asset01", "operation": "reduce lug", "parameters": {"m": 1}}]}
        verdict = repeated_repair(moves, INPUT, KEY, repair_signature(geometry),
                                  mechanism=repair_mechanism(geometry))
        self.assertIsNone(verdict, "the author must be able to change approach, not just retry")

    def test_failures_on_another_problem_do_not_block_this_one(self):
        action = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}]}
        history = [evaluated(action, "regressed", problem="other-problem")]
        self.assertIsNone(repeated_repair(history, INPUT, KEY, repair_signature(action),
                                          mechanism=repair_mechanism(action)))



def row(identifier, status, **extra):
    return {"id": identifier, "status": status, **extra}


class PartialProgressTests(unittest.TestCase):
    """Removing some named defects is progress even when the check still fails."""

    def test_fewer_named_collisions_than_before_is_adopted(self):
        previous = {"checks": [row("scene_collisions", "fail", collision_count=3, collisions=[
            {"geom1": "a", "geom2": "b"}, {"geom1": "c", "geom2": "d"}, {"geom1": "e", "geom2": "f"}])]}
        updated = {"checks": [row("scene_collisions", "fail", collision_count=2, collisions=[
            {"geom1": "a", "geom2": "b"}, {"geom1": "c", "geom2": "d"}])]}
        comparison = repair_comparison(previous, updated)
        self.assertEqual(comparison["outcome"], "improved_partial")
        self.assertTrue(comparison["adopt"])
        self.assertEqual(comparison["partially_improved_checks"][0]["check"], "scene_collisions")

    def test_same_count_with_different_pairs_is_not_progress(self):
        previous = {"checks": [row("scene_collisions", "fail", collision_count=1, collisions=[
            {"geom1": "a", "geom2": "b"}])]}
        updated = {"checks": [row("scene_collisions", "fail", collision_count=1, collisions=[
            {"geom1": "c", "geom2": "d"}])]}
        comparison = repair_comparison(previous, updated)
        self.assertNotEqual(comparison["outcome"], "improved_partial")
        self.assertIn(comparison["outcome"], {"regressed", "mixed", "ineffective"})

    def test_swapping_one_defect_for_another_is_not_partial_progress(self):
        previous = {"checks": [row("scene_collisions", "fail", collision_count=1, collisions=[
            {"geom1": "a", "geom2": "b"}])]}
        updated = {"checks": [row("scene_collisions", "fail", collision_count=2, collisions=[
            {"geom1": "c", "geom2": "d"}, {"geom1": "e", "geom2": "f"}])]}
        comparison = repair_comparison(previous, updated)
        self.assertNotEqual(comparison["outcome"], "improved_partial")
        self.assertFalse(comparison["adopt"])

    def test_full_resolution_still_counts_as_improved(self):
        previous = {"checks": [row("support:x", "fail", unsupported_components=[{"body": "leg"}])]}
        updated = {"checks": [row("support:x", "pass", complete=True)]}
        comparison = repair_comparison(previous, updated)
        self.assertEqual(comparison["outcome"], "improved")
        self.assertTrue(comparison["adopt"])


class FailedApproachTests(unittest.TestCase):
    def test_only_this_problem_and_only_rejected_attempts_are_returned(self):
        move = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}], "reason": "left"}
        history = [evaluated(move, "regressed"),
                   evaluated(move, "improved", problem="other-problem")]
        rows = failed_approaches(history, KEY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "regressed")
        self.assertIn("placements", rows[0]["action"])

    def test_applied_but_unevaluated_attempts_are_not_reported_as_failed(self):
        move = {"placements": [{"id": "station", "position": [0.1, 0, 0.8], "yaw": 0}]}
        history = [{"status": "applied", "problem_key": KEY, "input_key": INPUT,
                    "action": move, "action_key": "k", "mechanism": "m"}]
        self.assertEqual(failed_approaches(history, KEY), [])


if __name__ == "__main__":
    unittest.main()
