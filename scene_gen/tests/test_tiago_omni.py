"""Official model integration checks; fixtures are engineering data, not experiments."""
import copy
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from asset_generation.checks import load_model
from asset_generation.contracts import read_json
from asset_prep.errors import ValidationError
from scene_generation.assembly import assemble
from scene_generation.contracts import environment
from scene_generation.layout import validate_mobile
from scene_generation.robot import validate_integrated, prepare_environment
from scene_fixtures import write_component
from test_mobile_layout import mobile_fixture


class IntegratedAdapterTests(unittest.TestCase):
    def test_flag_must_be_boolean_and_old_mobile_contract_remains_valid(self):
        value={"base_size":[.7,.5,.3],"mount_height":.9,"mass_kg":70,
               "source":"synthetic engineering fixture","approximations":[]}
        validate_mobile(value)
        validate_mobile({**value,"integrated":True})
        with self.assertRaises(ValidationError):
            validate_mobile({**value,"integrated":"true"})

    def test_missing_coordinates_do_not_silently_add_another_base(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=".1"/></body></worldbody></mujoco>')
        with self.assertRaises(ValidationError):
            validate_integrated(m)

    def test_wrong_coordinate_axis_and_order_are_rejected(self):
        xml='<mujoco><worldbody><body><geom size=".1"/>{}</body></worldbody></mujoco>'
        joints=['<joint name="base_x" type="slide" axis="1 0 0"/>',
                '<joint name="base_y" type="slide" axis="0 1 0"/>',
                '<joint name="base_yaw" axis="0 0 1"/>']
        validate_integrated(mujoco.MjModel.from_xml_string(xml.format("".join(joints))))
        for invalid in ([joints[1],joints[0],joints[2]],
                        [joints[0].replace('1 0 0','0 0 1'),*joints[1:]]):
            with self.assertRaises(ValidationError):
                validate_integrated(mujoco.MjModel.from_xml_string(xml.format("".join(invalid))))


class TiagoOfficialResourceTests(unittest.TestCase):
    def setUp(self):
        self.project=Path(__file__).resolve().parents[1]
        self.root=self.project/"data/robots/tiago_omni"
        if not self.root.is_dir():
            self.skipTest("TIAGo official resources not installed; no network during tests")
        self.env=environment(read_json(self.project/"config/scene_environment_tiago.json"))
        self.assertEqual(self.env["robot"]["directory"],"data/robots/tiago_omni")

    def test_conversion_preserves_manufacturer_kinematics_mass_and_limits(self):
        reference=mujoco.MjModel.from_xml_path(str(self.root/"import_stock_gripper.urdf"))
        model=load_model(self.root,model_file="stock_gripper.xml")
        self.assertAlmostEqual(reference.body_mass.sum(),model.body_mass.sum(),places=8)
        for lift in (0,.2,.35):
            a,b=mujoco.MjData(reference),mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model,b,0)
            b.qpos[model.joint("torso_lift_joint").qposadr[0]]=lift
            for j in range(reference.njnt):
                target=model.joint(reference.joint(j).name)
                np.testing.assert_allclose(reference.jnt_range[j],target.range)
                a.qpos[reference.jnt_qposadr[j]]=b.qpos[target.qposadr[0]]
            mujoco.mj_forward(reference,a);mujoco.mj_forward(model,b)
            for body in range(1,reference.nbody):
                target=model.body(reference.body(body).name).id
                np.testing.assert_allclose(a.xpos[body],b.xpos[target],atol=2e-6)
                np.testing.assert_allclose(a.xmat[body],b.xmat[target],atol=2e-6)

    def test_provenance_stock_gripper_and_independent_joint_motion(self):
        self.assertEqual(prepare_environment(self.project,self.env),self.root)
        model=load_model(self.root,model_file="stock_gripper.xml")
        self.assertEqual(model.njnt,19)  # 3 planar + 4 wheels + lift + 7 arm + 2 head + 2 fingers.
        data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
        mujoco.mj_forward(model,data)
        before=data.site_xpos[model.site("attachment_site").id].copy()
        data.qpos[model.joint("torso_lift_joint").qposadr[0]]+=.1
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.site_xpos[model.site("attachment_site").id]-before,[0,0,.1],atol=1e-8)
        joint=model.joint("torso_lift_joint")
        np.testing.assert_allclose(joint.range,[0,.35])
        before=data.xpos[model.body("gripper_left_finger_link").id].copy()
        data.qpos[model.joint("gripper_left_finger_joint").qposadr[0]]=.01
        mujoco.mj_forward(model,data)
        self.assertGreater(np.linalg.norm(data.xpos[model.body("gripper_left_finger_link").id]-before),.02)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_whole_robot_assembles_without_added_hull_and_tool_follows_lift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            _,plan=mobile_fixture(root)
            plan["layout"]["robot_home"]=[1,-2,.4]
            output=assemble(plan,self.env,{"instrument":write_component(root/"instrument"),
                "tool":write_component(root/"tool",True)},self.root,root/"scene")
            model=load_model(output)
            names={model.geom(g).name for g in range(model.ngeom)}
            self.assertNotIn("robot__base_hull",names)
            self.assertNotIn("robot__mast",names)
            self.assertIn("robot__base_link__visual_0",names)
            self.assertNotIn("robot__gripper_link__visual_0",names)
            meta=read_json(output/"scene.json")
            data=mujoco.MjData(model);data.qpos[:]=meta["home_qpos"];mujoco.mj_forward(model,data)
            before=data.xpos.copy()
            tool_bodies=[b for b in range(1,model.nbody) if (model.body(b).name or "").startswith("gripper__")]
            self.assertTrue(tool_bodies)
            data.qpos[model.joint("robot__torso_lift_joint").qposadr[0]]+=.1
            mujoco.mj_forward(model,data)
            for b in tool_bodies:
                np.testing.assert_allclose(data.xpos[b]-before[b],[0,0,.1],atol=1e-7)
            self.assertTrue((output/"robot_provenance.json").is_file())
            self.assertTrue((output/"robot_LICENSE.txt").is_file())
