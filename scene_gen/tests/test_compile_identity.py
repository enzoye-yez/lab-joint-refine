import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_prep import implementation
from asset_generation.checks import measure_model
from scene_generation.checkpoints import Checkpoints
from scene_generation.contracts import SceneConfig


class CompileIdentityTests(unittest.TestCase):
    def test_planner_changes_invalidate_findings_but_not_identical_xml_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifact=root/'artifact';artifact.mkdir()
            xml=artifact/'model.xml'
            xml.write_text('<mujoco><compiler angle="radian"/><worldbody><body><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
            sources=implementation.package_manifest(Path(__file__).resolve().parents[1]/'src')
            def cache():return Checkpoints(root/'cache',artifact,SceneConfig(validation_mode='static'),{},False)
            first=cache();first.load_model(lambda:measure_model(artifact,static=True))
            changed={**sources,'scene_generation/progress.py':'changed-planner'}
            with patch('asset_prep.implementation.package_manifest',return_value=changed):
                second=cache()
                self.assertNotEqual(first.key,second.key)
                self.assertEqual(first.model_key,second.model_key)
                second.load_model(lambda:self.fail('Unchanged compiler inputs must reuse validated model'))
            changed['asset_generation/checks.py']='changed-loader-policy'
            with patch('asset_prep.implementation.package_manifest',return_value=changed):
                third=cache();self.assertNotEqual(first.model_key,third.model_key)
                with patch('asset_generation.checks.measure_model',wraps=measure_model) as compute:
                    third.load_model(lambda:compute(artifact,static=True))
                    self.assertEqual(compute.call_count,1)
            xml.write_text(xml.read_text().replace('.1 .1 .1','.2 .1 .1'))
            fourth=cache();self.assertNotEqual(first.model_key,fourth.model_key)
            with patch('asset_generation.checks.measure_model',wraps=measure_model) as compute:
                fourth.load_model(lambda:compute(artifact,static=True))
                self.assertEqual(compute.call_count,1)
