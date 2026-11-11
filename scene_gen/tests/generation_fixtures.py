"""开发者编写的合成资产，不是模型生成结果或正式实验。"""

import copy
import json
from pathlib import Path

from PIL import Image

PROVENANCE = "developer_authored_test_fixture"

XML = '''<mujoco model="synthetic_button">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -1 2"/>
    <body name="base"><geom name="base_geom" type="box" pos="0 0 0.025" size="0.05 0.05 0.025" rgba="0.6 0.7 0.8 1"/>
      <geom name="cap_bracket" type="box" pos="0.065 0 0.0575" size="0.025 0.012 0.0175"/>
      <geom name="button_guide" type="box" pos="0.015 0 0.055" size="0.005 0.01 0.01"/>
      <body name="button" pos="0 0 0.06">
        <joint name="button_slide" type="slide" axis="0 0 -1" range="0 0.004" damping="1" stiffness="400"/>
        <geom name="button_geom" type="box" size="0.01 0.01 0.002" mass="0.01"/>
        <site name="press_site" pos="0 0 0.002" size="0.002"/>
      </body>
    </body>
    <body name="cap" pos="0.1 0 0.07"><freejoint/>
      <geom name="cap_geom" type="box" size="0.01 0.01 0.005" mass="0.02"/>
      <site name="cap_grasp" size="0.002"/>
    </body>
  </worldbody>
  <equality><weld name="cap_lock" body1="base" body2="cap"/></equality>
</mujoco>'''

MANIFEST = {"schema_version": "1.0", "asset_ids": ["A01"],
    "parts": [{"id": "base", "body": "base"}, {"id": "button", "body": "button"}, {"id": "cap", "body": "cap"}],
    "interfaces": [{"id": "press", "asset_id": "A01", "kind": "press", "body": "button", "site": "press_site"}],
    "assumptions": ["开发者合成工程样例"], "approximations": [],
    "buttons": [{"id": "button", "joint": "button_slide", "geom": "button_geom", "pressed_at": 0.002,
                 "released_at": 0.0005, "event": "release"}],
    "connections": [{"id": "cap_connection", "equality": "cap_lock", "release_event": "release", "irreversible": False}],
    "mechanisms": [{"id": "button", "requirement": "开发者合成：按钮按下并回弹", "joints": ["button_slide"],
                    "lock": None, "actuation": "button"}]}

SOURCE = 'def build(output_dir):\n    # Developer-authored fixture; not executed by host tests.\n    pass\n'


def write_artifact(root, xml=XML, manifest=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root/"model.xml").write_text(xml, encoding="utf-8")
    (root/"interfaces.json").write_text(json.dumps(manifest or MANIFEST), encoding="utf-8")


def package(root):
    root = Path(root)
    Image.new("RGB", (24, 24), "white").save(root/"reference.png")
    path = root/"asset_package.md"
    path.write_text("# 合成资料\n资产尺寸由合成测试指定。\n### A01｜按钮测试件\n![参考](reference.png)\n宽100毫米。\n"
                    "### A02｜另一个资产\n本次不生成。\n", encoding="utf-8")
    return path


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.urls = []

    def post(self, url, payload, key, timeout):
        self.requests.append(copy.deepcopy(payload))
        self.urls.append(url)
        if not self.replies:
            raise AssertionError("Unexpected request; tests must not contact a live service")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return {"model": "developer-fixture-model", "choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(reply)}}], "usage": {"total_tokens": 100, "completion_tokens": 80}}


class FixtureSandbox:
    """只写入已知合成MJCF，不执行任何传入的模型代码。"""
    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def preflight(self):
        return {"backend": "developer_fixture_no_code_execution"}

    def build_and_check(self, script, destination, asset_ids, profile):
        from asset_generation.checks import validate_artifact
        self.calls += 1
        artifact = Path(destination)/"artifact"
        xml = XML.replace('size="0.05 0.05 0.025"', 'size="0.06 0.05 0.025"') if self.fail_first and self.calls == 1 else XML
        write_artifact(artifact, xml)
        report = validate_artifact(artifact, asset_ids, profile)
        renders = Path(destination)/"renders"
        renders.mkdir()
        Image.new("RGB", (24, 24), "blue").save(renders/"fixture.png")
        report["renders"] = ["fixture.png"]
        return report, renders
