"""所有几何和响应都是开发者合成样例，绝不是模型实验。"""

from pathlib import Path
from PIL import Image
from asset_prep.storage import atomic_json

ROBOT = '''<mujoco><compiler angle="radian"/><worldbody><body name="base">
<geom type="box" size=".05 .05 .03" pos="0 0 .03"/>
<body name="slide_x" pos="0 0 .3"><joint name="x" type="slide" axis="1 0 0" range="-.9 .9"/>
<geom type="sphere" size=".025" mass="1"/><body name="slide_y">
<joint name="y" type="slide" axis="0 1 0" range="-.9 .9"/>
<geom type="sphere" size=".02" mass="1"/><site name="attachment_site" pos="0 0 .1"/>
</body></body></body></worldbody>
<actuator><position joint="x" kp="1000"/><position joint="y" kp="1000"/></actuator></mujoco>'''
ASSET = '''<mujoco><compiler angle="radian"/><worldbody><body name="base">
<geom name="base_geom" type="box" size=".05 .05 .05" pos="0 0 .05"/>
<site name="operation" pos="0 0 .43"/>
</body></worldbody></mujoco>'''
TOOL = '''<mujoco><compiler angle="radian"/><worldbody><body name="base">
<geom name="tool_geom" type="box" size=".015 .015 .015" pos="0 0 .015" mass=".1"/>
<site name="tcp" pos="0 0 .03"/>
</body></worldbody></mujoco>'''


def manifest(tool=False):
    return {"schema_version":"1.0","asset_ids":["A01"],"parts":[{"id":"base","body":"base"}],
            "interfaces":[] if tool else [{"id":"operation","asset_id":"A01","kind":"press","body":"base","site":"operation"}],
            "buttons":[],"connections":[],"assumptions":["开发者合成场景"],"approximations":[]}


def write_component(root, tool=False):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    (root/"model.xml").write_text(TOOL if tool else ASSET,encoding="utf-8")
    atomic_json(root/"interfaces.json",manifest(tool))
    return root


def environment(root):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    (root/"model.xml").write_text(ROBOT,encoding="utf-8")
    return {"table":{"size":[2,1.6,.06],"height":.8},"robot":{"name":"developer_fixture","directory":str(root),
            "attachment_site":"attachment_site","joints":["x","y"],"home":[0,0],"payload_kg":3,"reach_m":1}}


def plan():
    return {"jobs":[{"id":"instrument","asset_ids":["A01"],"role":"asset","depends_on":[],"requirements":"合成"},
                    {"id":"tool","asset_ids":["A01"],"role":"tool","depends_on":["instrument"],"requirements":"合成装置"}],
            "instances":[{"id":"station","job":"instrument","placement":"table","position":[.5,0,.8],"yaw":0},
                         {"id":"gripper","job":"tool","placement":"robot","position":[0,0,0],"yaw":0}],"assumptions":[]}


class SceneSandbox:
    """固定合成产物替身，绝不执行传入脚本；隔离后端由独立测试检查。"""
    def __init__(self, fail_scene_once=False):
        self.build_calls = 0
        self.scene_calls = 0
        self.fail_scene_once = fail_scene_once
        self.candidate_reports = {}

    def preflight(self):
        return {"backend":"developer_fixture_no_generated_code_execution"}

    def build_and_check(self, script, destination, asset_ids, profile):
        self.build_calls += 1
        tool = "TOOL_FIXTURE" in script.read_text(encoding="utf-8")
        write_component(Path(destination)/"artifact",tool)
        renders = Path(destination)/"renders"
        renders.mkdir()
        Image.new("RGB",(24,24),"white").save(renders/"fixture.png")
        return {"status":"pass","checks":[{"id":"developer_fixture","status":"pass"}],"renders":["fixture.png"],"manifest":manifest(tool),"coverage":["替身仅供调度测试"]},renders

    def assemble_and_check(self, plan, env, artifacts, robot, destination, limits, contract):
        from scene_generation.assembly import assemble
        self.scene_calls += 1
        output = Path(destination)/"scene"
        assemble(plan,env,artifacts,robot,output/"artifact")
        renders = output/"renders"
        renders.mkdir()
        Image.new("RGB",(24,24),"blue").save(renders/"fixture.png")
        failed = self.fail_scene_once and self.scene_calls == 1
        return {"status":"fail" if failed else "pass", "checks":[{"id":"fixture_collision","status":"fail" if failed else "pass",
                    "instances":["station"],"detail":"开发者合成失败"}],"renders":["fixture.png"],"coverage":["替身不认证场景质量"]},renders

    def build_candidate(self, script, destination, asset_ids, profile):
        from scene_generation.candidate import inspect_candidate
        report,renders=self.build_and_check(script,destination,asset_ids,profile)
        artifact=Path(destination)/'artifact'
        self.candidate_reports[str(artifact)]=(report,renders)
        if not artifact.is_dir():return report,renders
        admission=inspect_candidate(artifact,asset_ids,profile)
        admission['renders']=report.get('renders',[])
        return admission,renders

    def assemble_candidate(self,*args):
        report,renders=self.assemble_and_check(*args)
        artifact=Path(args[4])/'scene'/'artifact'
        self.candidate_reports[str(artifact)]=(report,renders)
        return {'status':'partial','checks':[{'id':'quality_pending','status':'partial'}],
                'renders':[],'assembly_readiness':{'loadable':True}},None

    def check_scene(self,artifact,destination,limits,contract):
        if str(artifact) in self.candidate_reports:return self.candidate_reports[str(artifact)]
        # Resume a persisted fixture without reconstructing model outputs.
        return {'status':'pass','checks':[{'id':'fixture_collision','status':'pass'}],
                'renders':['fixture.png']},Path(artifact).parent/'renders'

    def recheck_asset(self,artifact,destination,asset_ids,profile):
        if str(artifact) in self.candidate_reports:return self.candidate_reports[str(artifact)]
        from asset_generation.contracts import read_json
        return {'status':'pass','checks':[{'id':'developer_fixture','status':'pass'}],
            'renders':['fixture.png'],'manifest':read_json(Path(artifact)/'interfaces.json')},Path(artifact).parent/'renders'

    def admit_asset(self,artifact,destination,asset_ids,profile,**kwargs):
        from scene_generation.candidate import inspect_candidate
        return inspect_candidate(artifact,asset_ids,profile),None
