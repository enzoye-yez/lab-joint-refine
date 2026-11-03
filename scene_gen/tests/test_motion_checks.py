"""Asset motion, exact-surface and review-contract checks; developer-authored fixtures only."""
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from asset_prep.errors import ValidationError
from asset_generation import motion
from asset_generation.material import integrity
from asset_generation.static_state import geometry_state


LOCKED_LID = '''<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
<body name="base"><geom name="base_g" type="box" size=".1 .1 .05" pos="0 0 .05"/>
  <body name="lid" pos="0 -.1 .1"><joint name="lid_hinge" type="hinge" axis="1 0 0" range="0 1.5"/>
    <geom name="lid_g" type="box" size=".1 .1 .005" pos="0 .1 .005" mass=".2"/></body></body>
</worldbody><equality><joint name="lid_lock" joint1="lid_hinge"/></equality></mujoco>'''

BLOCKED_SLIDE = '''<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
<body name="base"><geom name="base_g" type="box" size=".1 .1 .02" pos="0 0 .02"/>
  <geom name="stop" type="box" size=".005 .05 .02" pos=".03 0 .06"/>
  <body name="drawer" pos="0 0 .06"><joint name="drawer_slide" type="slide" axis="1 0 0" range="0 .08"/>
    <geom name="drawer_g" type="box" size=".02 .02 .02" mass=".1"/></body></body>
</worldbody></mujoco>'''


def compiled(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def mech(id_, joints, **kw):
    return {"id": id_, "requirement": "开发者合成要求："+id_, "joints": joints, "lock": None,
            "actuation": "manual", **kw}


class MechanismContractTests(unittest.TestCase):
    def test_every_movable_joint_needs_a_requirement_bound_mechanism(self):
        m, _ = compiled(LOCKED_LID)
        self.assertEqual(motion.mechanism_contract(m, {"mechanisms": []})["status"], "fail")
        row = motion.mechanism_contract(m, {"mechanisms": [mech("lid", ["lid_hinge"], lock="lid_lock")]})
        self.assertEqual(row["status"], "pass", row)
        bad = motion.mechanism_contract(m, {"mechanisms": [mech("lid", ["lid_hinge"], lock="no_such_lock")]})
        self.assertEqual(bad["status"], "fail")

    def test_unlimited_rotation_must_be_declared_continuous(self):
        m, _ = compiled(LOCKED_LID.replace(' range="0 1.5"', ' limited="false"'))
        self.assertEqual(motion.mechanism_contract(m, {"mechanisms": [mech("lid", ["lid_hinge"])]})["status"], "fail")
        self.assertEqual(motion.mechanism_contract(m, {"mechanisms": [mech("lid", ["lid_hinge"], continuous=True)]})["status"], "pass")


class MechanismDriveTests(unittest.TestCase):
    def latched(self, stop=""):
        xml = LOCKED_LID.replace('<body name="lid"', '<body name="latch" pos=".12 0 .05"><joint name="latch_slide" type="slide" '
                                 'axis="0 0 1" range="0 .01"/><geom name="latch_g" type="box" size=".005 .005 .005" mass=".01"/></body>'
                                 + stop + '<body name="lid"')
        manifest = {"mechanisms": [mech("lid", ["lid_hinge"], lock="lid_lock", release_event="unlatch"), mech("latch", ["latch_slide"])],
                    "interfaces": [{"id": "tab", "body": "latch", "kind": "press"}, {"id": "rim", "body": "lid", "kind": "grasp"}],
                    "mechanical_events": [{"id": "unlatch", "interface": "tab", "action": "release", "joint": "latch_slide", "at": .008}]}
        m, d = compiled(xml)
        return m, d, manifest

    def test_lock_holds_and_is_released_by_driving_its_latch(self):
        m, d, manifest = self.latched()
        row = motion.mechanism_drive(m, manifest, d.qpos.copy())[0]
        self.assertEqual(row["status"], "pass", row)
        self.assertTrue(row["lock_holds"])
        self.assertTrue(row["unlock"]["crossed"])
        self.assertTrue(row["reached"]["high"])

    def test_jammed_latch_keeps_the_lock_and_fails(self):
        m, d, manifest = self.latched('<geom name="jam" type="box" size=".005 .005 .005" pos=".12 0 .064"/>')
        row = motion.mechanism_drive(m, manifest, d.qpos.copy())[0]
        self.assertEqual(row["status"], "fail", row)
        self.assertFalse(row["unlock"]["crossed"])

    def test_lock_needs_a_release_event_and_a_manual_one_must_act_on_the_locked_part(self):
        m, d = compiled(LOCKED_LID)
        row = motion.mechanism_drive(m, {"mechanisms": [mech("lid", ["lid_hinge"], lock="lid_lock")]}, d.qpos.copy())[0]
        self.assertEqual(row["status"], "fail")
        self.assertIn("release event", row["unlock"])
        for body, status in (("lid", "pass"), ("base", "fail")):
            manifest = {"mechanisms": [mech("lid", ["lid_hinge"], lock="lid_lock", release_event="pop")],
                        "interfaces": [{"id": "rim", "body": body, "kind": "grasp"}],
                        "mechanical_events": [{"id": "pop", "interface": "rim", "action": "release"}]}
            self.assertEqual(motion.mechanism_drive(m, manifest, d.qpos.copy())[0]["status"], status, body)

    def test_sliding_door_must_stay_on_its_track_unless_it_extends(self):
        xml = ('<mujoco><option timestep="0.002"/><worldbody><body name="frame"><geom name="track" type="box" size=".1 .01 .005" pos="0 0 .005"/>'
               '<body name="door" pos="0 0 .02"><joint name="door_slide" type="slide" axis="1 0 0" range="0 {}"/>'
               '<geom name="door_g" type="box" size=".05 .005 .01" mass=".1"/></body></body></worldbody></mujoco>')
        for travel, extends, status in ((.08, False, "pass"), (.16, False, "fail"), (.12, True, "pass"), (.16, True, "fail")):
            m, d = compiled(xml.format(travel))
            row = motion.mechanism_retention(m, {"mechanisms": [mech("door", ["door_slide"], extends=extends)]}, d.qpos.copy())[0]
            self.assertEqual(row["status"], status, (travel, extends, row))

    def test_blocked_mechanism_fails_with_the_blocking_contact(self):
        m, d = compiled(BLOCKED_SLIDE)
        rows = motion.mechanism_drive(m, {"mechanisms": [mech("drawer", ["drawer_slide"])]}, d.qpos.copy())
        self.assertEqual(rows[0]["status"], "fail")
        names = {(c["geom1"], c["geom2"]) for c in rows[0]["high_blocking_contacts"]["contacts"]}
        self.assertTrue(any("stop" in pair for pair in names), rows)

    def test_sweep_reports_new_penetration_along_the_range(self):
        m, d = compiled(BLOCKED_SLIDE)
        rows, poses = motion.mechanism_sweep(m, {"mechanisms": [mech("drawer", ["drawer_slide"])]}, d.qpos.copy(), .0005)
        self.assertEqual(rows[0]["status"], "fail")
        self.assertEqual([name for name, _ in poses], ["mech_drawer_low", "mech_drawer_high"])


class SettleAndReleaseTests(unittest.TestCase):
    def write(self, root, xml):
        root.mkdir(parents=True, exist_ok=True)
        (root/"model.xml").write_text(xml, encoding="utf-8")
        return root/"model.xml"

    def test_resting_free_part_is_stable_and_hovering_one_drifts(self):
        with tempfile.TemporaryDirectory() as t:
            resting = self.write(Path(t)/"a", '<mujoco><worldbody><body name="cup" pos="0 0 .02"><freejoint/>'
                                 '<geom name="cup_g" type="box" size=".02 .02 .02" mass=".05"/></body></worldbody></mujoco>')
            hovering = self.write(Path(t)/"b", '<mujoco><worldbody><body name="cup" pos="0 0 .2"><freejoint/>'
                                  '<geom name="cup_g" type="box" size=".02 .02 .02" mass=".05"/></body></worldbody></mujoco>')
            self.assertEqual(motion.settle(resting)["status"], "pass")
            row = motion.settle(hovering)
            self.assertEqual(row["status"], "fail")
            self.assertEqual(row["drifting_bodies"][0]["body"], "cup")

    def test_released_part_is_removable_and_held_while_connected(self):
        xml = ('<mujoco><worldbody><body name="base"><geom name="base_g" type="box" size=".05 .05 .01" pos="0 0 .01"/></body>'
               '<body name="cap" pos="0 0 .03"><freejoint/><geom name="cap_g" type="box" size=".02 .02 .01" mass=".02"/></body>'
               '</worldbody><equality><weld name="cap_lock" body1="base" body2="cap"/></equality></mujoco>')
        m, d = compiled(xml)
        manifest = {"mechanical_events": [{"id": "lift", "interface": "grip", "action": "release"}],
                    "connections": [{"id": "cap", "equality": "cap_lock", "release_event": "lift"}]}
        rows = motion.release_probes(m, manifest, d.qpos.copy())
        self.assertEqual(rows[0]["status"], "pass", rows)

    def test_pull_is_measured_in_the_anchor_frame_and_sized_on_the_carried_mass(self):
        xml = ('<mujoco><worldbody><body name="sheet"><freejoint/><geom name="s" type="box" size=".05 .05 .002" mass="1"/></body>'
               '<body name="quarter" pos=".1 0 0"><freejoint/><geom name="q" type="box" size=".02 .02 .002" mass=".02"/></body>'
               '<body name="tip" pos=".1 0 .02"><freejoint/><geom name="t" type="box" size=".005 .005 .01" mass=".05"/></body>'
               '</worldbody><equality><weld name="seam" body1="sheet" body2="quarter"/><weld name="seat" body1="quarter" body2="tip"/></equality></mujoco>')
        m, d = compiled(xml)
        quarter, sheet = m.body("quarter").id, m.body("sheet").id
        # The welded sheet turns about 160 degrees while it is pulled at one corner (about 200 mm
        # in world coordinates); seen from the sheet the quarter stays within the 2 mm held limit.
        moved, _ = motion._pull(m, d.qpos.copy(), np.ones(m.neq), {quarter: np.array([0, 0, 1.])}, {quarter: sheet})
        self.assertLess(moved[quarter], .002, moved)
        self.assertAlmostEqual(motion._carried_mass(m, quarter, np.ones(m.neq), exclude={sheet}), .07)

    def test_diverged_pull_fails_instead_of_reading_the_reset_state(self):
        xml = ('<mujoco><option timestep="0.01"/><worldbody><body name="base"><geom name="b" type="box" size=".05 .05 .01"/></body>'
               '<body name="cap" pos="0 0 .05"><joint name="s" type="slide" axis="0 0 1" stiffness="1e9"/>'
               '<geom name="c" type="box" size=".01 .01 .01" mass=".001"/></body>'
               '</worldbody><equality><weld name="lock" body1="base" body2="cap"/></equality></mujoco>')
        m, d = compiled(xml)
        manifest = {"mechanical_events": [{"id": "lift", "interface": "grip", "action": "release"}],
                    "connections": [{"id": "cap", "equality": "lock", "release_event": "lift"}]}
        row = motion.release_probes(m, manifest, d.qpos.copy())[0]
        self.assertEqual(row["status"], "fail", row)
        self.assertTrue(row["diverged"])
        self.assertIsNone(row["moved_when_active_mm"])

    def test_shared_release_run_still_finds_the_trapped_part(self):
        walls = "".join(f'<geom name="w{i}" type="box" size="{sx} {sy} {sz}" pos="{x} {y} {z}"/>' for i, (sx, sy, sz, x, y, z) in
                        enumerate([(.014, .014, .003, .1, 0, .0235), (.003, .014, .01, .1135, 0, .01), (.003, .014, .01, .0865, 0, .01),
                                   (.014, .003, .01, .1, .0135, .01), (.014, .003, .01, .1, -.0135, .01)]))
        xml = ('<mujoco><worldbody><body name="base"><geom name="base_g" type="box" size=".15 .05 .005" pos="0 0 -.005"/>'
               + walls + '</body>'
               '<body name="free_cap" pos="-.1 0 .01"><freejoint/><geom name="free_g" type="box" size=".01 .01 .01" mass=".02"/></body>'
               '<body name="boxed_cap" pos=".1 0 .01"><freejoint/><geom name="boxed_g" type="box" size=".01 .01 .01" mass=".02"/></body>'
               '</worldbody><equality><weld name="free_lock" body1="base" body2="free_cap"/>'
               '<weld name="boxed_lock" body1="base" body2="boxed_cap"/></equality></mujoco>')
        m, d = compiled(xml)
        manifest = {"mechanical_events": [{"id": "lift", "interface": "grip", "action": "release"}],
                    "connections": [{"id": "free", "equality": "free_lock", "release_event": "lift"},
                                    {"id": "boxed", "equality": "boxed_lock", "release_event": "lift"}]}
        rows = {r["id"]: r for r in motion.release_probes(m, manifest, d.qpos.copy())}
        self.assertEqual(rows["release_probe:free"]["status"], "pass", rows)
        self.assertEqual(rows["release_probe:boxed"]["status"], "fail", rows)
        self.assertEqual(rows["release_probe:boxed"]["trapped_by"], ["base"])


class ExactSurfaceTests(unittest.TestCase):
    def test_floating_hinged_part_is_a_decided_defect_and_seated_part_is_not(self):
        base = '<mujoco><worldbody><body name="base"><geom name="base_g" type="box" size=".05 .05 .05" pos="0 0 .05"/>{}</body></worldbody></mujoco>'
        floating = base.format('<body name="lid" pos="0 0 .13"><joint name="h" axis="0 1 0" range="0 1"/>'
                               '<geom name="lid_g" type="box" size=".05 .05 .01"/></body>')
        seated = base.format('<body name="lid" pos="0 0 .11"><joint name="h" axis="0 1 0" range="0 1"/>'
                             '<geom name="lid_g" type="box" size=".05 .05 .01"/></body>')
        for xml, status in ((floating, "fail"), (seated, "pass")):
            m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); geometry_state(m, d)
            row = integrity(m, d)
            self.assertEqual(row["status"], status, row)

    def test_open_shell_crossing_is_proven_on_original_triangles(self):
        import trimesh
        from asset_generation.distance import DistanceQuery
        from asset_generation.mesh_exact import exact
        tube = trimesh.creation.annulus(r_min=.019, r_max=.02, height=.05, sections=24)
        open_faces = tube.faces[np.abs(tube.face_normals[:, 2]) < .5]  # drop the end caps: an open shell
        vertices = " ".join(map(str, tube.vertices.reshape(-1))); faces = " ".join(map(str, open_faces.reshape(-1)))
        xml = (f'<mujoco><asset><mesh name="tube" vertex="{vertices}" face="{faces}"/></asset><worldbody>'
               '<body name="rack"><geom name="pad" type="box" size=".003 .003 .003" pos=".02 0 0"/></body>'
               '<body name="t"><freejoint/><geom name="shell" type="mesh" mesh="tube" contype="0" conaffinity="0"/></body>'
               '</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); geometry_state(m, d)
        evidence = exact(DistanceQuery(m, d), 0, 1)
        self.assertTrue(evidence["crossing"], evidence)


class ReviewContractTests(unittest.TestCase):
    def test_asset_review_cannot_defer_and_accept_needs_motion_coverage(self):
        from scene_generation.review import scoped_review
        with self.assertRaises(ValidationError):
            scoped_review({"verdict": "defer", "findings": [], "unverified": [],
                           "deferred": [{"stage": "scene", "detail": "x"}]}, "asset")
        with self.assertRaises(ValidationError):
            scoped_review({"verdict": "accept", "findings": [], "unverified": []}, "asset")
        ok = {"verdict": "accept", "findings": [], "unverified": [], "motion_coverage": [],
              "deferred": [{"stage": "scene", "detail": "与其他仪器的插接"}]}
        self.assertEqual(scoped_review(ok, "asset")["verdict"], "accept")
        missing = {"verdict": "accept", "findings": [], "unverified": [], "motion_coverage": [
            {"requirement": "盖可打开", "mechanism": None, "verdict": "missing", "evidence": "无铰链"}]}
        with self.assertRaises(ValidationError):
            scoped_review(missing, "asset")

    def test_deferred_items_accept_a_quoted_requirement_and_reject_unknown_keys(self):
        from scene_generation.review import scoped_review
        value = {"verdict": "accept", "findings": [], "unverified": [], "motion_coverage": [],
                 "deferred": [{"stage": "scene", "requirement": "插入管架"},
                              {"stage": "operation", "requirement": "取管", "detail": "验证抓取轨迹"}]}
        self.assertEqual(scoped_review(value, "asset")["deferred"],
                         [{"stage": "scene", "detail": "插入管架"}, {"stage": "operation", "detail": "取管: 验证抓取轨迹"}])
        with self.assertRaisesRegex(ValidationError, "stage"):
            scoped_review({**value, "deferred": [{"stage": "scene", "note": "x"}]}, "asset")

    def test_asset_findings_must_name_part_evidence_and_change(self):
        from scene_generation.review import scoped_review
        value = {"verdict": "repair", "unverified": [], "motion_coverage": []}
        with self.assertRaisesRegex(ValidationError, "finding"):
            scoped_review({**value, "findings": ["the door looks wrong"]}, "asset")
        good = ("left_door_g: slides out of its rail at the upper limit; evidence: mech_door_high_45.png; "
                "suggested fix: lower the slide limit from 0.14 to 0.08")
        self.assertEqual(scoped_review({**value, "findings": [good]}, "asset")["findings"], [good])

    def test_failed_asset_gets_its_own_repairs_and_stops_without_improvement(self):
        from types import SimpleNamespace
        from unittest import mock
        from scene_generation import scene_first
        record = lambda n: {"artifact": f"v{n}", "status": "failed",
                            "feedback": {"numerical": {"checks": [{"id": "x", "status": "fail"}]}}}
        run = SimpleNamespace(state={"plan": {"jobs": [{"id": "a"}]}, "jobs": {"a": record(0)}},
                              store=SimpleNamespace(save=lambda: None))
        with mock.patch.object(scene_first, "usable", lambda run, r: bool(r)):
            for n in range(3):
                self.assertEqual(scene_first.schedule_asset_repairs(run), ["a"] if n < 2 else [])
                self.assertEqual(run.state["jobs"]["a"].get("regenerate"), "quality" if n < 2 else None)
                run.state["jobs"]["a"] = record(n+1)
        self.assertIn("did not reduce", run.state["asset_repairs_unresolved"]["a"]["reason"])

    def test_review_key_changes_only_with_what_the_review_judged(self):
        from scene_generation.scene_first import review_key
        job = {"id": "a", "role": "asset", "depends_on": []}
        report = {"checks": [{"id": "settle", "status": "pass"}], "execution_timing": {"seconds": 1}}
        base = review_key(job, {"artifact_sha256": "x"}, None, {}, report, ["i2", "i1"])
        self.assertEqual(base, review_key(job, {"artifact_sha256": "x"}, None, {}, {**report, "execution_timing": {}}, ["i1", "i2"]))
        self.assertNotEqual(base, review_key(job, {"artifact_sha256": "x"}, None, {}, {"checks": [{"id": "settle", "status": "fail"}]}, ["i1", "i2"]))
        self.assertNotEqual(base, review_key(job, {"artifact_sha256": "x"}, None, {}, report, ["i1", "i3"]))
        self.assertNotEqual(base, review_key(job, {"artifact_sha256": "y"}, None, {}, report, ["i1", "i2"]))

    def test_plan_rejects_fixture_jobs(self):
        from scene_generation.contracts import validate_plan, SceneConfig
        plan = {"jobs": [{"id": "a", "asset_ids": ["A01"], "role": "asset", "depends_on": [], "requirements": "x"},
                         {"id": "f", "asset_ids": ["A01"], "role": "fixture", "depends_on": ["a"], "requirements": "x"}],
                "instances": [{"id": "a", "job": "a", "placement": "table", "position": [0, 0, .8], "yaw": 0}],
                "assumptions": []}
        with self.assertRaisesRegex(ValidationError, "fixture"):
            validate_plan(plan, ["A01"], SceneConfig())


if __name__ == "__main__":
    unittest.main()
