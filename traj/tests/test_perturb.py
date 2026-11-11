"""Counterfactual scene edits: only the edited bodies change, carried parts follow, joints and sizes update."""
import unittest

import mujoco
import numpy as np

from np002_traj.scene import perturb
from support import world


def _rack(spec):
    rack = spec.worldbody.add_body(name="rack__base", pos=[1.5, 0.0, 0.40])
    rack.add_geom(name="rack__top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.10, 0.10, 0.02])
    lid = rack.add_body(name="rack__lid", pos=[0.0, 0.0, 0.05])
    lid.add_joint(name="rack__hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.5, frictionloss=0.2)
    lid.add_geom(name="rack__lid", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.005], mass=0.05)
    tube = spec.worldbody.add_body(name="tube__body", pos=[1.5, 0.0, 0.445])
    tube.add_freejoint(name="tube__free")
    tube.add_geom(name="tube__body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.01, 0.02], mass=0.01)
    far = spec.worldbody.add_body(name="far__base", pos=[-2.0, 0.0, 0.3])
    far.add_geom(name="far__box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
    for key in spec.keys:
        key.qpos = list(key.qpos) + [0.0] + [1.5, 0.0, 0.445, 1, 0, 0, 0]


class PerturbTest(unittest.TestCase):
    def setUp(self):
        self.model, self.data, self.robot = world(extra=_rack)

    def edited(self, edits):
        model, report = perturb.apply(self.model, edits, self.robot, key="home", settle_s=0.2)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        return model, data, report

    def test_move_instance_carries_resting_parts_only(self):
        model, data, report = self.edited([{"kind": "move_instance", "instance": "rack", "dxyz": [0.1, 0.0, 0.0]}])
        rack, tube, far = (model.body(n).id for n in ("rack__base", "tube__body", "far__base"))
        np.testing.assert_allclose(data.xpos[rack], self.data.xpos[rack] + [0.1, 0, 0], atol=1e-9)
        self.assertAlmostEqual(data.xpos[tube][0] - self.data.xpos[tube][0], 0.1, delta=0.005)
        np.testing.assert_allclose(data.xpos[far], self.data.xpos[far], atol=1e-9)
        self.assertEqual(report["edits"][0]["carried"], 1)
        np.testing.assert_allclose(self.model.body_pos[rack], [1.5, 0, 0.4])      # the source is untouched

    def test_joint_and_geometry_edits(self):
        model, _, report = self.edited([
            {"kind": "joint", "joint": "rack__hinge", "frictionloss": {"scale": 0.1}, "damping": {"value": 0.0}},
            {"kind": "scale_geom", "geom": "rack__lid", "factor": [1.0, 1.0, 2.0]},
            {"kind": "disable_collision", "geoms": ["far__box"]}])
        dof = model.jnt_dofadr[model.joint("rack__hinge").id]
        self.assertAlmostEqual(model.dof_frictionloss[dof], 0.02)
        self.assertEqual(model.dof_damping[dof], 0.0)
        lid = model.geom("rack__lid").id
        self.assertAlmostEqual(model.geom_size[lid][2], 0.01)
        self.assertAlmostEqual(model.geom_aabb[lid][5], 0.01)
        self.assertEqual(model.geom_contype[model.geom("far__box").id], 0)
        self.assertEqual(self.model.dof_frictionloss[dof], 0.2)

    def test_unknown_edit_is_rejected(self):
        with self.assertRaises(perturb.EditError):
            perturb.apply(self.model, [{"kind": "melt"}], self.robot, key="home")
        with self.assertRaises(perturb.EditError):
            perturb.apply(self.model, [{"kind": "move_instance", "instance": "nothing"}], self.robot, key="home")


if __name__ == "__main__":
    unittest.main()
