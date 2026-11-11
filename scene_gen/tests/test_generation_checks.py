import tempfile
import copy
import unittest
from pathlib import Path

import mujoco

from asset_generation.checks import load_model, validate_artifact, validate_manifest
from asset_generation.contracts import check_profile
from asset_generation.runtime import InteractionRuntime
from asset_prep.errors import ValidationError
from generation_fixtures import MANIFEST, XML, write_artifact


HINGED_LID = '''<mujoco><compiler angle="radian"/><worldbody>
<body name="body"><geom name="body_g" type="box" size=".1 .1 .05" pos="0 0 .05"/>
<site name="base_catch" pos=".1 0 .1"/>
<body name="lid" pos="-.1 0 .1"><joint name="hinge" type="hinge" axis="0 1 0" range="0 1.6" limited="true"/>
<geom name="lid_g" type="box" size=".1 .1 .005" pos=".1 0 .005"/><site name="lid_catch" pos=".2 0 0"/></body>
<body name="button" pos="0 -.1 .05"><joint name="press" type="slide" axis="0 -1 0" range="0 .004" limited="true"/>
<geom name="button_g" type="box" size=".01 .005 .01"/></body>
</body></worldbody>
<equality><joint name="lid_lock" joint1="hinge" polycoef="0 0 0 0 0"/></equality></mujoco>'''


def lid_manifest(**connection):
    value = {"schema_version": "1.0", "asset_ids": ["A01"], "parts": [{"id": "body", "body": "body"}],
             "interfaces": [], "assumptions": [], "approximations": [],
             "buttons": [{"id": "release", "joint": "press", "geom": "button_g",
                          "pressed_at": .003, "released_at": .0005, "event": "unlock"}],
             "connections": [{"id": "lid_lock", "kind": "latch", "equality": "lid_lock",
                              "release_event": "unlock", "irreversible": False}]}
    value["connections"][0].update(connection)
    return value


class LatchContractTests(unittest.TestCase):
    """A hinged lid lock is not a detachable part and must not be forced into that contract."""

    def model(self):
        return mujoco.MjModel.from_xml_string(HINGED_LID)

    def test_hinged_lid_lock_is_a_valid_latch(self):
        validate_manifest(lid_manifest(), self.model(), ["A01"])

    def test_latch_releases_and_the_lid_stays_hinged(self):
        xml = HINGED_LID.replace('</body></worldbody>',
                                 '</body><body name="finger" pos="0 -.112 .05"><geom name="finger_g" type="box" size=".004 .004 .004"/></body></worldbody>')
        model = mujoco.MjModel.from_xml_string(xml)
        manifest = lid_manifest()
        validate_manifest(manifest, model, ["A01"])
        data = mujoco.MjData(model)
        lock = model.equality("lid_lock").id
        mujoco.mj_forward(model, data)
        self.assertTrue(data.eq_active[lock])
        runtime = InteractionRuntime(model, manifest, actor_geoms=["finger_g"])
        self.assertEqual(runtime.update(data), [])
        data.qpos[model.joint("press").qposadr[0]] = .0035
        mujoco.mj_forward(model, data)
        records = runtime.update(data)
        self.assertIn("connection_released", [r["type"] for r in records], records)
        self.assertFalse(data.eq_active[lock])
        # A latch only frees the hinge: the lid is still parented to the instrument body.
        self.assertEqual(model.body_parentid[model.body("lid").id], model.body("body").id)

    def test_weld_declared_as_latch_names_the_correct_constraint(self):
        xml = HINGED_LID.replace('<joint name="lid_lock" joint1="hinge" polycoef="0 0 0 0 0"/>',
                                 '<weld name="lid_lock" body1="lid" body2="body"/>')
        with self.assertRaisesRegex(ValidationError, "joint-type constraint"):
            validate_manifest(lid_manifest(), mujoco.MjModel.from_xml_string(xml), ["A01"])

    def test_detachable_in_same_tree_points_author_to_latch(self):
        xml = HINGED_LID.replace('<joint name="lid_lock" joint1="hinge" polycoef="0 0 0 0 0"/>',
                                 '<weld name="lid_lock" body1="lid" body2="body"/>')
        with self.assertRaisesRegex(ValidationError, "kind=latch"):
            validate_manifest(lid_manifest(kind="detachable"), mujoco.MjModel.from_xml_string(xml), ["A01"])

    def test_latch_cannot_be_irreversible(self):
        with self.assertRaisesRegex(ValidationError, "irreversible"):
            validate_manifest(lid_manifest(irreversible=True), self.model(), ["A01"])

    def test_unknown_connection_kind_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "detachable"):
            validate_manifest(lid_manifest(kind="hinge"), self.model(), ["A01"])

    def test_latch_reattach_sites_span_lid_and_support(self):
        manifest = lid_manifest(attach_event="lock", capture_sites=["lid_catch", "base_catch"],
                                position_tolerance_m=.002, angle_tolerance_rad=.05)
        manifest["buttons"].append({"id": "relock", "joint": "press", "geom": "button_g",
                                    "pressed_at": .004, "released_at": .001, "event": "lock"})
        model = self.model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        gap = data.site_xpos[model.site("lid_catch").id] - data.site_xpos[model.site("base_catch").id]
        self.assertLess(float(abs(gap).max()), 1e-9)
        validate_manifest(manifest, model, ["A01"])
        manifest["connections"][0]["capture_sites"] = ["lid_catch", "lid_catch"]
        with self.assertRaises(ValidationError):
            validate_manifest(manifest, model, ["A01"])

    def test_omitted_kind_keeps_detachable_contract(self):
        model = load_model_from_fixture()
        manifest = copy.deepcopy(MANIFEST)
        for connection in manifest["connections"]:
            connection.pop("kind", None)
        validate_manifest(manifest, model, ["A01"])


def load_model_from_fixture():
    return mujoco.MjModel.from_xml_string(XML)


def dimension_profile(width=0.1):
    return check_profile({"dimensions": [{"id": "width", "geoms": ["base_geom"], "axis": 0,
                                          "expected_m": width, "tolerance_m": 0.000001}],
                          "required_interfaces": ["press"],
                          "joints": [{"name": "button_slide", "type": "slide", "range": [0, 0.004], "tolerance": 0.000001}]})


class GenerationCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_artifact(self.root)

    def test_actual_geometry_passes_and_wrong_width_fails(self):
        result = validate_artifact(self.root, ["A01"], dimension_profile())
        self.assertEqual(result["status"], "pass", result)
        wrong = validate_artifact(self.root, ["A01"], dimension_profile(0.12))
        check = next(c for c in wrong["checks"] if c["id"] == "width")
        self.assertEqual(check["status"], "fail")
        self.assertAlmostEqual(check["actual_m"], 0.1)

    def test_missing_interface_is_not_silently_accepted(self):
        profile = check_profile({"required_interfaces": ["missing"]})
        result = validate_artifact(self.root, ["A01"], profile)
        self.assertEqual(result["status"], "fail")

    def test_invalid_scope_and_asset_external_include_fail(self):
        result = validate_artifact(self.root, ["A02"], check_profile({}))
        self.assertEqual(result["status"], "fail")
        (self.root/"model.xml").write_text('<mujoco><include file="../secret.xml"/></mujoco>')
        with self.assertRaises(ValidationError):
            load_model(self.root)
        (self.root/"model.xml").write_text('<mujoco><compiler angle="radian"/><asset><texture name="cube" type="cube" fileleft="../private.png"/></asset></mujoco>')
        with self.assertRaises(ValidationError):
            load_model(self.root)

    def test_button_needs_actor_contact_not_position_alone(self):
        # 用真实MuJoCo接触数据测试事件检测；直接置位仅是工程探测，不是机器人执行。
        xml = XML.replace('<light pos="0 -1 2"/>', '<geom name="probe" type="box" pos="0 0 0.061" size="0.009 0.009 0.003"/>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        runtime = InteractionRuntime(model, MANIFEST, actor_geoms=["probe"])
        j = model.joint("button_slide").qposadr[0]
        data.qpos[j] = 0.003
        mujoco.mj_forward(model, data)
        no_actor = InteractionRuntime(model, MANIFEST)
        self.assertEqual(no_actor.update(data), [])
        events = runtime.update(data)
        self.assertIn("release", [x.get("name") for x in events])
        self.assertFalse(data.eq_active[model.equality("cap_lock").id])
        self.assertEqual(runtime.update(data), [])
        data.qpos[j] = 0
        mujoco.mj_forward(model, data)
        runtime.update(data)
        data.qpos[j] = 0.003
        mujoco.mj_forward(model, data)
        self.assertEqual(sum(e["type"] == "event" for e in runtime.update(data)), 1)

    def test_undeclared_software_event_rejected(self):
        model = load_model(self.root)
        runtime = InteractionRuntime(model, MANIFEST)
        with self.assertRaises(ValueError):
            runtime.update(mujoco.MjData(model), ["release"])

    def test_disabling_contacts_is_not_a_repair(self):
        xml = XML.replace('<option timestep="0.002"/>', '<option timestep="0.002"><flag contact="disable"/></option>')
        write_artifact(self.root, xml)
        report = validate_artifact(self.root, ["A01"], check_profile({}))
        self.assertEqual(report["status"], "fail")
        self.assertIn("must not be disabled", report["checks"][0]["detail"])

    def test_real_multiview_render(self):
        result = validate_artifact(self.root, ["A01"], dimension_profile(), self.root/"renders")
        self.assertEqual(result["status"], "pass", result)
        self.assertGreaterEqual(len(result["renders"]), 3)
        self.assertTrue(all((self.root/"renders"/name).stat().st_size > 1000 for name in result["renders"]))

    def test_physical_probe_triggers_button_and_releases_connection(self):
        from asset_generation.probes import interaction_probes
        rows = interaction_probes(self.root,MANIFEST)
        self.assertEqual([r["status"] for r in rows],["pass","pass"])
        button = rows[0]
        self.assertTrue(button["triggered"])
        self.assertGreaterEqual(button["peak_position"],MANIFEST["buttons"][0]["pressed_at"])

    def test_physical_probe_cannot_pass_when_button_contact_disabled(self):
        from asset_generation.probes import interaction_probes
        xml = XML.replace('name="button_geom"', 'name="button_geom" contype="0" conaffinity="0"')
        write_artifact(self.root,xml)
        rows = interaction_probes(self.root,MANIFEST)
        self.assertEqual([r["status"] for r in rows],["fail","fail"])

    def test_button_on_a_free_instrument_is_pressed_not_dropped_and_not_overdriven(self):
        from asset_generation.probes import interaction_probes, PRESS_OVERTRAVEL
        write_artifact(self.root,XML.replace('<body name="base">','<body name="base"><freejoint/>'))
        button = interaction_probes(self.root,MANIFEST)[0]
        self.assertEqual(button["status"],"pass",button)
        self.assertTrue(button["triggered"])
        self.assertLessEqual(button["peak_position"],.004+PRESS_OVERTRAVEL+1e-4)

    def test_button_sagging_past_released_at_under_its_own_weight_fails(self):
        from asset_generation.probes import interaction_probes
        write_artifact(self.root,XML.replace('stiffness="400"','stiffness="100"'))
        button = interaction_probes(self.root,MANIFEST)[0]
        self.assertEqual(button["status"],"fail")
        self.assertIn("rest position",button["detail"])

    def test_button_that_stays_pressed_after_retreat_fails(self):
        from asset_generation.probes import interaction_probes
        write_artifact(self.root,XML.replace('stiffness="400"','stiffness="0"'))
        button = interaction_probes(self.root,MANIFEST)[0]
        self.assertTrue(button["triggered"])
        self.assertFalse(button["released_after_retreat"])
        self.assertEqual(button["status"],"fail")
        self.assertIn("not reset",button["detail"])

    def test_probe_reads_the_configuration_model_file(self):
        from asset_generation.probes import interaction_probes
        (self.root/"model.xml").rename(self.root/"pump.xml")
        rows = interaction_probes(self.root,MANIFEST,"pump.xml")
        self.assertEqual([r["status"] for r in rows],["pass","pass"])

    def test_reconnection_requires_real_alignment_and_allows_later_release(self):
        xml = XML.replace('<body name="base">','<body name="base"><site name="socket" pos=".1 0 .07"/>')
        manifest = copy.deepcopy(MANIFEST)
        manifest["interfaces"].extend([{"id":"unlock","asset_id":"A01","kind":"software","body":"base"},
                                       {"id":"lock","asset_id":"A01","kind":"software","body":"base"}])
        manifest["connections"][0].update(release_event="unlock",attach_event="lock",capture_sites=["socket","cap_grasp"],
            position_tolerance_m=.001,angle_tolerance_rad=.02)
        model = mujoco.MjModel.from_xml_string(xml)
        validate_manifest(manifest,model,["A01"])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        runtime = InteractionRuntime(model,manifest)
        self.assertTrue(any(r["type"] == "connection_released" for r in runtime.update(data,["unlock"])))
        address = model.jnt_qposadr[model.body("cap").jntadr[0]]
        data.qpos[address] += .02
        mujoco.mj_forward(model,data)
        self.assertFalse(any(r["type"] == "connection_attached" for r in runtime.update(data,["lock"])))
        data.qpos[address] -= .02
        mujoco.mj_forward(model,data)
        self.assertTrue(any(r["type"] == "connection_attached" for r in runtime.update(data,["lock"])))
        self.assertTrue(any(r["type"] == "connection_released" for r in runtime.update(data,["unlock"])))
        manifest["connections"][0]["irreversible"] = True
        with self.assertRaises(ValidationError):
            validate_manifest(manifest,model,["A01"])


if __name__ == "__main__":
    unittest.main()
