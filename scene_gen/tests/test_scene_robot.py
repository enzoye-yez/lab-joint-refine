import tempfile
import unittest
from pathlib import Path

import mujoco
from asset_generation.checks import load_model,render_views
from asset_generation.contracts import read_json
from scene_generation.assembly import assemble
from scene_fixtures import plan,write_component


class PublicRobotEngineeringTests(unittest.TestCase):
    def test_pinned_fr3_assembles_and_reloads_resources_when_available(self):
        project = Path(__file__).resolve().parents[1]
        robot = project/"data/robots/fr3"
        if not robot.is_dir():
            self.skipTest("公开FR3资源尚未准备；不联网下载测试依赖")
        env = read_json(project/"config/scene_environment_fixed.json")
        provenance = read_json(robot/"provenance.json")
        self.assertEqual(provenance["revision"],"0059d4335f8156206f63a35662313385f7ad6d74")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = plan()
            if env["robot"].get("mobile"):
                from test_mobile_layout import mobile_fixture
                _, candidate = mobile_fixture(root)
            output = assemble(candidate,env,{"instrument":write_component(root/"instrument"),
                "tool":write_component(root/"tool",True)},robot,root/"scene")
            model = load_model(output)
            self.assertGreater(model.nmesh,20)
            self.assertGreaterEqual(model.site("robot__attachment_site").id,0)
            meta = read_json(output/"scene.json")
            frames = render_views(model,[("fr3_fixture",meta["home_qpos"])],root/"renders")
            self.assertEqual(len(frames),3)
            for frame in frames:
                self.assertGreater((root/"renders"/frame).stat().st_size,1000)
