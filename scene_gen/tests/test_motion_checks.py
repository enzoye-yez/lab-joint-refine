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
    def test_lock_holds_under_operator_effort_and_releases(self):
        m, d = compiled(LOCKED_LID)
        rows = motion.mechanism_drive(m, {"mechanisms": [mech("lid", ["lid_hinge"], lock="lid_lock")]}, d.qpos.copy())
        self.assertEqual(rows[0]["status"], "pass", rows)
        self.assertTrue(rows[0]["lock_holds"])
        self.assertTrue(rows[0]["reached"]["high"])

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
