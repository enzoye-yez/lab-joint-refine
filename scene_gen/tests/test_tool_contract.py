"""Developer-authored active detachable tool; never formal model output."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from asset_generation.checks import load_model, validate_manifest, validate_artifact
from asset_generation.contracts import read_json, check_profile
from asset_generation.pipeline import snapshot
from asset_generation.static_state import geometry_state
from asset_generation.runtime import InteractionRuntime
from asset_prep.errors import ValidationError
from asset_prep.storage import atomic_json
from scene_generation.assembly import assemble
from scene_generation.candidate import inspect_candidate
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.checks import scene_checks, reach_target
from scene_generation.inspection import mount
from scene_generation.tool_contract import tool_roots
from scene_fixtures import environment, plan, write_component

XML = '''<mujoco><compiler angle="radian"/><worldbody>
<body name="base"><geom name="base_geom" type="box" size=".02 .02 .02" mass=".2"/>
 <site name="socket" pos="0 0 .04"/><site name="tcp" pos="0 0 .06"/>
 <body name="finger" pos=".03 0 0"><joint name="drive" type="slide" axis="1 0 0" range="0 .02"/>
 <geom name="finger_geom" type="box" size=".01 .01 .02" mass=".05"/></body></body>
<body name="tip" pos="0 0 .04"><freejoint name="tip_free"/>
 <geom name="tip_geom" type="box" size=".02 .02 .02" mass=".1"/><site name="capture"/></body>
</worldbody><equality><weld name="lock" body1="base" body2="tip"/></equality>
<actuator><position name="motor" joint="drive" kp="10"/></actuator></mujoco>'''
MANIFEST = {"schema_version":"1.0", "asset_ids":["A01"],
 "parts":[{"id":n,"body":n} for n in ("base","finger","tip")],
 "interfaces":[{"id":n,"asset_id":"A01","body":"base","kind":"software"} for n in ("unlock","lock")],
 "assumptions":["Developer fixture explicitly permits software release and attach"],"approximations":[],"buttons":[],
 "connections":[{"id":"tip_lock","equality":"lock","release_event":"unlock","irreversible":False,
 "attach_event":"lock","capture_sites":["socket","capture"],"position_tolerance_m":.001,"angle_tolerance_rad":.02}]}


class ToolContractTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root=Path(temp.name)
        self.tool=self.root/"tool"; self.tool.mkdir()
        self.write()
        self.env=environment(self.root/"robot")
        # A rotated flange and nonzero robot home expose accidental local/world mixing.
        p=self.root/"robot/model.xml"
        p.write_text(p.read_text().replace('name="attachment_site"', 'name="attachment_site" quat=".7071067811865476 0 0 .7071067811865476"'))
        self.env["robot"]["home"]=[.2,-.15]

    def write(self, xml=XML, manifest=None):
        (self.tool/"model.xml").write_text(xml,encoding="utf-8")
        atomic_json(self.tool/"interfaces.json",manifest or MANIFEST)

    def build(self):
        return assemble(plan(),self.env,{"instrument":write_component(self.root/"asset"),"tool":self.tool},
                        self.root/"robot",self.root/"scene")

    def test_active_detachable_tool_passes_admission_and_real_assembly_contract(self):
        before=snapshot(self.tool)
        report=inspect_candidate(self.tool,["A01"],{"assembly_role":"tool"})
        self.assertTrue(report["assembly_readiness"]["loadable"])
        self.assertEqual(report["status"],"partial")
        with patch("mujoco.mj_step",side_effect=AssertionError("No simulation")):
            output=self.build()
            model=load_model(output,scope="scene"); data=mujoco.MjData(model)
            meta=read_json(output/"scene.json"); data.qpos[:]=meta["home_qpos"]; geometry_state(model,data)
            manifest=validate_manifest(read_json(output/"interfaces.json"),model,["A01"])
            measured=mount(model,data,meta,{"instance":"gripper","distance_m":.005})
            self.assertTrue(measured["attachment_transform_matches"],measured)
            self.assertEqual(model.body_parentid[model.body("gripper__tip").id],0)
            self.assertNotEqual(model.body_parentid[model.body("gripper__base").id],0)
            self.assertEqual(model.actuator("gripper__motor").trnid[0],model.joint("gripper__drive").id)
            np.testing.assert_allclose(data.site("gripper__socket").xpos,data.site("gripper__capture").xpos,atol=1e-9)
            self.assertEqual(snapshot(self.tool),before)
            mujoco.mj_forward(model,data)
            np.testing.assert_allclose(data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY],0,atol=1e-8)
            motor=model.actuator("gripper__motor").id
            data.ctrl[motor]=.01; mujoco.mj_forward(model,data)
            self.assertGreater(abs(data.actuator_force[motor]),.01)
            runtime=InteractionRuntime(model,manifest)
            runtime.update(data,["gripper__unlock"])
            self.assertFalse(data.eq_active[model.equality("gripper__lock").id])
            address=model.joint("gripper__tip_free").qposadr[0]
            data.qpos[address]+=.02; geometry_state(model,data)
            self.assertFalse(runtime.update(data,["gripper__lock"])[-1]["type"] == "connection_attached")
            data.qpos[address]-=.02; geometry_state(model,data)
            self.assertTrue(any(r["type"]=="connection_attached" for r in runtime.update(data,["gripper__lock"])))
        # Quality checks remain real and account for detached parts in the payload.
        report=scene_checks(output,SceneConfig(validation_mode="static"),numerical_contract({}))
        payload=next(r for r in report["checks"] if r["id"]=="payload")
        self.assertAlmostEqual(payload["tool_mass_kg"],.35)

    def test_release_does_not_leave_a_permanent_parent_link(self):
        output=self.build(); model=load_model(output,scope="scene"); data=mujoco.MjData(model)
        data.qpos[:]=read_json(output/"scene.json")["home_qpos"]; geometry_state(model,data)
        runtime=InteractionRuntime(model,read_json(output/"interfaces.json"))
        runtime.update(data,["gripper__unlock"])
        tip=data.xpos[model.body("gripper__tip").id].copy()
        data.qpos[model.joint("robot__x").qposadr[0]]+=.1; geometry_state(model,data)
        np.testing.assert_allclose(data.xpos[model.body("gripper__tip").id],tip)
        self.assertGreater(np.linalg.norm(data.site("gripper__socket").xpos-data.site("gripper__capture").xpos),.09)

    def test_orphan_free_root_rejected_by_both_entry_points(self):
        m=copy.deepcopy(MANIFEST);m["connections"]=[];self.write(manifest=m)
        for action in (lambda:inspect_candidate(self.tool,["A01"],{"assembly_role":"tool"}),self.build):
            with self.assertRaisesRegex(ValidationError,"未连接") : action()

    def test_all_free_tool_and_same_tree_release_are_rejected(self):
        self.write(XML.replace('<body name="base">','<body name="base"><freejoint name="base_free"/>'))
        with self.assertRaisesRegex(ValidationError,"固定安装主体"):
            inspect_candidate(self.tool,["A01"],{"assembly_role":"tool"})
        self.write(XML.replace('<body name="tip"', '<body name="tip"').replace('<freejoint name="tip_free"/>',''))
        with self.assertRaisesRegex(ValidationError,"neither end can move freely"):
            inspect_candidate(self.tool,["A01"],{"assembly_role":"tool"})

    def test_general_connection_constraints_are_not_relaxed(self):
        for change,match in ((lambda m:m["connections"][0].update(position_tolerance_m=.02),"tolerance"),
                             (lambda m:m["connections"][0].update(release_event="invented"),"release_event")):
            m=copy.deepcopy(MANIFEST);change(m);self.write(manifest=m)
            with self.assertRaisesRegex(ValidationError,match):inspect_candidate(self.tool,["A01"],{"assembly_role":"tool"})

    def test_delivered_scope_feedback_names_expected_and_actual_ids(self):
        m=copy.deepcopy(MANIFEST);m["asset_ids"]=["tool"]
        with self.assertRaisesRegex(ValidationError,"A01.*tool.*service scope"):
            validate_manifest(m,load_model(self.tool),["A01"])

    def test_coupled_tool_cannot_receive_false_kinematic_acceptance(self):
        output=self.build();model=load_model(output,scope="scene");home=read_json(output/"scene.json")["home_qpos"]
        result=reach_target(model,home,self.env,{"tool_site":"gripper__tcp"},SceneConfig())
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["coverage_gaps"][0]["error_kind"],"unsupported_check")
        report=scene_checks(output,SceneConfig(settle_steps=1),numerical_contract({}))
        row=next(r for r in report["checks"] if r["id"]=="scene_joint_sweeps")
        self.assertEqual(row["status"],"partial")
        self.assertTrue(row["coverage_gaps"])

    def test_child_frames_and_external_mesh_survive_reparenting(self):
        (self.tool/"part.obj").write_text("v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
        xml=XML.replace('<worldbody>','<asset><mesh name="part" file="part.obj"/></asset><worldbody><frame pos=".1 .2 .3" quat=".7071067811865476 0 .7071067811865476 0">')
        xml=xml.replace('</worldbody>','</frame></worldbody>').replace('<site name="capture"/>','<site name="capture"/><geom name="mesh_geom" type="mesh" mesh="part" contype="0" conaffinity="0" mass="0"/>')
        self.write(xml)
        output=self.build();model=load_model(output,scope="scene");data=mujoco.MjData(model);meta=read_json(output/"scene.json")
        data.qpos[:]=meta["home_qpos"];geometry_state(model,data)
        self.assertEqual(model.nmesh,1)
        self.assertTrue(mount(model,data,meta,{"instance":"gripper","distance_m":.005})["attachment_transform_matches"])
        np.testing.assert_allclose(data.site("gripper__socket").xpos,data.site("gripper__capture").xpos,atol=1e-9)

    def test_permanent_tree_connection_is_rejected_before_assembly(self):
        xml=XML.replace('<body name="tip" pos="0 0 .04"><freejoint name="tip_free"/>', '<body name="tip" pos="0 0 .04"><joint name="tip_hinge" range="0 .1"/>')
        xml=xml.replace('</body></body>\n<body name="tip"','</body>\n<body name="tip"').replace('</worldbody>','</body></worldbody>')
        self.write(xml)
        with self.assertRaisesRegex(ValidationError,"same permanent kinematic tree"):
            inspect_candidate(self.tool,["A01"],{"assembly_role":"tool"})
