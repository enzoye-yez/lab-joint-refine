import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asset_generation.checks import load_model
from asset_prep.errors import ValidationError


class SceneCapacityTests(unittest.TestCase):
    def test_real_composite_model_exceeds_asset_limit_but_loads_as_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geoms = ''.join(f'<geom name="g{i}" pos="{i*.003} 0 0" size=".001"/>' for i in range(10001))
            (root/"model.xml").write_text('<mujoco><compiler angle="radian"/><worldbody>'+geoms+'</worldbody></mujoco>',encoding="utf-8")
            with self.assertRaisesRegex(ValidationError,"ngeom=10001/10000"):
                load_model(root)
            self.assertEqual(load_model(root,scope="scene").ngeom,10001)

    def test_scene_limit_remains_bounded_and_reports_actual_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"model.xml").write_text('<mujoco><compiler angle="radian"/></mujoco>',encoding="utf-8")
            model = SimpleNamespace(ngeom=50001,nbody=9,nv=7)
            with patch('mujoco.MjModel.from_xml_path',return_value=model), self.assertRaisesRegex(ValidationError,"ngeom=50001/50000, nbody=9/4096, nv=7/2048"):
                load_model(root,scope="scene")
