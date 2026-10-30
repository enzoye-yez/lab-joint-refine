import unittest

from asset_prep.errors import ValidationError
from scene_generation.review import scoped_review


class StageReviewTests(unittest.TestCase):
    def test_other_pose_verification_is_explicitly_pending_only_for_preview(self):
        deferred = [{"id":"one","stage":"scene"}]
        review = {"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[],"deferred_results":[
            {"id":"one","status":"out_of_scope","target_stage":"static_fit","evidence":"需要装管后的另一姿态"}]}
        self.assertEqual(scoped_review(review,"scene",deferred,preview=True),review)
        with self.assertRaises(ValidationError):
            scoped_review(review,"scene",deferred)
        review["deferred_results"][0]["target_stage"] = "scene"
        with self.assertRaises(ValidationError):
            scoped_review(review,"scene",deferred,preview=True)

    def test_repeated_instances_cannot_accept_multiple_units_in_one_prototype(self):
        for verdict in ("accept","defer"):
            review = {"verdict":verdict,"findings":[],"unverified":[],"prototype_units":2}
            if verdict == "defer":
                review["deferred"] = [{"stage":"scene","detail":"稍后检查"}]
            with self.subTest(verdict=verdict),self.assertRaises(ValidationError):
                scoped_review(review,"asset",expected_units=1)
        review = {"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[],"prototype_units":1}
        self.assertEqual(scoped_review(review,"asset",expected_units=1),review)

    def test_defer_verdict_and_invalid_targets_are_rejected(self):
        # Cross-instance items may accompany accept; a defer verdict never does.
        self.assertEqual(scoped_review({"verdict":"accept","motion_coverage":[],"findings":[],"unverified":[],
                                        "deferred":[{"stage":"scene","detail":"装配"}]},"asset")["verdict"],"accept")
        for review in (
            {"verdict":"defer","findings":["本体穿模"],"unverified":[],"deferred":[{"stage":"scene","detail":"装配"}]},
            {"verdict":"defer","findings":[],"unverified":[],"deferred":[]},
            {"verdict":"defer","findings":[],"unverified":[],"deferred":[{"stage":"never","detail":"跳过"}]},
            {"verdict":"uncertain","findings":[],"unverified":["装配还没有开始"]},
        ):
            with self.subTest(review=review), self.assertRaises(ValidationError):
                scoped_review(review,"asset")

    def test_scene_cannot_resolve_unknown_operation_or_missing_items(self):
        deferred = [{"id":"one","stage":"scene"},{"id":"two","stage":"operation"}]
        for results in ([],[{"id":"two","status":"resolved","evidence":"图片"}],
                        [{"id":"one","status":"unresolved","evidence":"缺少组合图"}],
                        [{"id":"one","status":"resolved","evidence":""}]):
            with self.subTest(results=results), self.assertRaises(ValidationError):
                scoped_review({"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[],"deferred_results":results},"scene",deferred)

    def test_known_unverified_measurements_do_not_become_false_success_claims(self):
        review = {"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":["精确孔径未量测"]}
        self.assertEqual(scoped_review(review,"asset"),review)
