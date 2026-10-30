"""开发者合成几何的只读补证据测试；不调用正式模型。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asset_generation.evidence import inspect_artifact, requests_valid
from asset_generation.local import LocalBackend
from asset_generation.contracts import GenerationConfig
from asset_generation.pipeline import snapshot
from asset_prep.errors import ValidationError
from asset_prep.storage import atomic_json
from generation_fixtures import write_artifact


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.artifact=self.root/"artifact"
        write_artifact(self.artifact)

    def test_static_pose_cutaway_does_not_step_or_mutate_asset(self):
        before=snapshot(self.artifact)
        request={"kind":"view","joints":{"button_slide":.003},"hide_bodies":["cap"],
                 "focus_body":"base","azimuth":45,"elevation":-90}
        with patch("mujoco.mj_step",side_effect=AssertionError("must not simulate")):
            report=inspect_artifact(self.artifact,"asset",[request],self.root/"evidence")
        self.assertEqual(report["status"],"pass")
        self.assertEqual(report["simulation_steps"],0)
        self.assertTrue((self.root/"evidence/renders/inspection_0.png").is_file())
        self.assertEqual(before,snapshot(self.artifact))
        self.assertEqual(report["results"][0]["request"]["hide_bodies"],["cap"])

    def test_invalid_pose_stays_incomplete(self):
        request={"kind":"view","joints":{"button_slide":20},"hide_bodies":[],
                 "focus_body":None,"azimuth":45,"elevation":-35}
        report=inspect_artifact(self.artifact,"asset",[request],self.root/"bad")
        self.assertEqual(report["status"],"incomplete")
        self.assertFalse(report["renders"])

    def scene(self, positions):
        xml='<mujoco><compiler angle="radian"/><worldbody>'
        for name,x,flags in positions:
            xml+=f'<body name="{name}__body" pos="{x} 0 0"><geom name="{name}__geom" type="box" size=".1 .1 .1" {flags}/></body>'
        xml+='</worldbody></mujoco>'
        write_artifact(self.artifact,xml=xml)
        atomic_json(self.artifact/"scene.json",{"home_qpos":[],"home_ctrl":[]})

    def test_clearance_includes_filtered_visual_geoms_and_robot(self):
        self.scene([("device",0,''),("robot",.25,'contype="0" conaffinity="0"'),("table",0,'')])
        report=inspect_artifact(self.artifact,"scene",[{"kind":"clearance","instance":"device","distance_m":.1}],self.root/"distance")
        result=report["results"][0]
        self.assertAlmostEqual(result["minimum_distance_m"],.05,places=6)
        self.assertEqual(result["other_instances"],["robot"])

    def test_disconnected_station_envelope_is_not_used_as_solid(self):
        xml='''<mujoco><compiler angle="radian"/><worldbody><geom name="environment__floor" type="plane" size="5 5 .1"/>
        <body name="device__body"><geom name="device__g" type="box" size=".1 .1 .1"/></body>
        <body name="station__left" pos="-.5 0 0"><geom name="station__g1" type="box" size=".1 .1 .1"/></body>
        <body name="station__right" pos=".5 0 0"><geom name="station__g2" type="box" size=".1 .1 .1"/></body></worldbody></mujoco>'''
        write_artifact(self.artifact,xml=xml)
        atomic_json(self.artifact/"scene.json",{"home_qpos":[],"home_ctrl":[]})
        report=inspect_artifact(self.artifact,"scene",[{"kind":"clearance","instance":"device","distance_m":.05}],self.root/"distance")
        self.assertEqual(report["results"][0]["lower_bound_m"],.05)
        self.assertIsNone(report["results"][0]["minimum_distance_m"])

    def test_backend_inventory_uses_worker_not_model_script(self):
        # 若误执行建模代码则失败；检查后端只读取已编译产物。
        (self.artifact/"model.py").write_text("raise AssertionError('do not execute')",encoding="utf-8")
        backend=LocalBackend(GenerationConfig(),Path(__file__).resolve().parents[1])
        report,_=backend.inspect_artifact(self.artifact,self.root/"inspect","asset",[{"kind":"inventory"}])
        self.assertEqual(report["status"],"pass")
        self.assertIn("button",report["inventory"]["bodies"])

    def test_no_scripts_or_unbounded_requests(self):
        for requests in ([{"kind":"python","code":"x"}], [{"kind":"inventory"}]*5,
                         [{"kind":"clearance","instance":"x","distance_m":float("nan")}]):
            with self.assertRaises(ValidationError):
                requests_valid(requests,"scene")


if __name__ == "__main__":
    unittest.main()
