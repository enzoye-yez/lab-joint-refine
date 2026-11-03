"""Presentation animation plan: mechanisms move in an order without penetration; developer-authored fixtures only."""
import unittest

import mujoco

from asset_generation import studio
from asset_generation.motion import _ranges

# A drawer behind a hinged door; the door has no collision geometry (drawn only), which the
# plan must still respect.
CABINET = '''<mujoco><compiler angle="radian"/><worldbody>
<body name="cabinet"><geom name="floor_panel" type="box" size=".2 .2 .01" pos="0 0 0"/>
  <body name="door" pos=".21 -.2 .1"><joint name="door_hinge" type="hinge" axis="0 0 -1" range="0 1.6"/>
    <geom name="door_panel" type="box" size=".005 .2 .09" pos="0 .2 0" contype="0" conaffinity="0" group="1"/></body>
  <body name="drawer" pos="0 0 .1"><joint name="drawer_slide" type="slide" axis="1 0 0" range="0 .25"/>
    <geom name="drawer_box" type="box" size=".18 .15 .05"/></body></body>
</worldbody></mujoco>'''

MECHANISMS = [{"id": "drawer", "joints": ["drawer_slide"]}, {"id": "door", "joints": ["door_hinge"]}]


def plan(door_open):
    model = mujoco.MjModel.from_xml_string(CABINET)
    data = mujoco.MjData(model)
    if door_open:
        data.qpos[model.jnt_qposadr[model.joint("door_hinge").id]] = 1.6
    mujoco.mj_forward(model, data)
    items = [(m["id"], _ranges(model, m)) for m in MECHANISMS]
    return studio.mechanism_plan(model, data, items), model, data


class MechanismPlanTest(unittest.TestCase):
    def test_drawer_waits_for_its_door_and_comes_back_first(self):
        (moves, held), _, _ = plan(door_open=False)
        self.assertEqual(moves, [("door", 1), ("drawer", 1), ("drawer", -1), ("door", -1)])
        self.assertEqual(held["drawer"], ["door"])

    def test_door_open_at_home_closes_only_after_the_drawer_is_back(self):
        (moves, held), _, _ = plan(door_open=True)
        self.assertEqual(moves, [("drawer", 1), ("drawer", -1), ("door", 1), ("door", -1)])
        self.assertEqual(held["door"], [])

    def test_plan_restores_the_model_and_pose(self):
        _, model, data = plan(door_open=False)
        self.assertEqual(int(model.geom_contype[model.geom("door_panel").id]), 0)
        self.assertEqual(float(data.qpos[model.jnt_qposadr[model.joint("drawer_slide").id]]), 0.)

    def test_schedule_holds_prerequisites_open_for_the_stills(self):
        model = mujoco.MjModel.from_xml_string(CABINET)
        slots = studio.mechanism_schedule(model, MECHANISMS)
        outward = [(ids, held) for ids, _, direction, held in slots if direction > 0]
        self.assertEqual([ids for ids, _ in outward], [["door"], ["drawer"]])
        self.assertEqual(len(outward[1][1]), 1)


if __name__ == "__main__":
    unittest.main()
