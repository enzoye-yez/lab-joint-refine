"""Detail renders look at an instance's operating face and fade what stands between it and the camera."""
import unittest

import mujoco
import numpy as np

from scene_generation.checks import fade_occluders, operating_azimuth, segment_hits_box

XML = """<mujoco><worldbody>
<body name="station__base" pos="0 0 .5" euler="0 0 90">
  <geom name="station__base_g" type="box" size=".1 .1 .1"/>
</body>
<body name="cabinet__wall" pos=".6 0 .5"><geom name="cabinet__wall_g" type="box" size=".02 .5 .5"/></body>
<body name="shelf__back" pos="-.6 0 .5"><geom name="shelf__back_g" type="box" size=".02 .5 .5"/></body>
</worldbody></mujoco>"""


class DetailViewTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_camera_faces_the_operating_face(self):
        # A quarter-turn yaw points local -y at world +x; the camera looks back along -x, 25 degrees off square.
        self.assertAlmostEqual(operating_azimuth(self.model, self.data, "station") % 360, 205, places=3)

    def test_segment_box_intersection(self):
        lo, hi = np.array([-1., -1, -1]), np.array([1., 1, 1])
        self.assertTrue(segment_hits_box(np.array([-3., 0, 0]), np.array([3., 0, 0]), lo, hi))
        self.assertFalse(segment_hits_box(np.array([-3., 2, 0]), np.array([3., 2, 0]), lo, hi))
        self.assertFalse(segment_hits_box(np.array([-3., 0, 0]), np.array([-2., 0, 0]), lo, hi))

    def test_only_geometry_between_camera_and_target_is_faded(self):
        scene = mujoco.MjvScene(self.model, maxgeom=100)
        camera = mujoco.MjvCamera()
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        lo, hi = np.array([-.1, -.1, .4]), np.array([.1, .1, .6])
        fade_occluders(self.model, self.data, scene, "station", np.array([1.5, 0, .5]), lo, hi)
        alpha = {self.model.geom(scene.geoms[i].objid).name: scene.geoms[i].rgba[3] for i in range(scene.ngeom)
                 if scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM}
        self.assertLess(alpha["cabinet__wall_g"], .2)
        self.assertEqual(alpha["shelf__back_g"], 1)
        self.assertEqual(alpha["station__base_g"], 1)


if __name__ == "__main__":
    unittest.main()
