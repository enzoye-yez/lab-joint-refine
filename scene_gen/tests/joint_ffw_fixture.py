"""Original pinned FFW-SG2 and developer fixture; no model-generated assets or liquid simulation."""
from pathlib import Path
import json,mujoco,numpy as np
from asset_prep.storage import atomic_json,digest
from scene_generation.assembly import assemble
from test_mobile_layout import mobile_fixture
from robot_operation.common import read,identity,Settings
from robot_operation.scene import prepare,load
from robot_operation.runtime import Session
from operation_fixtures import task_graph

PROJECT=Path(__file__).resolve().parents[1]

def build(root):
    root=Path(root);root.mkdir(parents=True)
    env=read(PROJECT/'config/scene_environment_ffw.json');_,plan=mobile_fixture(root)
    plan['jobs']=plan['jobs'][:1];plan['instances']=plan['instances'][:1]
    plan['end_effectors']={a:{'mode':'stock','instance':None,'reason':'developer fixture'} for a in ('left','right')}
    plan['layout']['tables']=[{'id':'bench','size':[.36,.9,.04],'height':.52,'position':[.62,-2.0],'yaw':0,'reason':'developer physical contact fixture'}]
    plan['instances'][0]['position']=[0,0,0]
    asset=root/'asset';asset.mkdir(exist_ok=True)
    (asset/'model.xml').write_text('<mujoco><compiler angle="radian"/><worldbody>\n<body name="frame"><geom name="rail" type="box" pos=".55 -2.227 .525" size=".12 .1 .005"/><site name="frame_site"/>\n<body name="lid" pos=".53 -2.227 .545"><joint name="slide" type="slide" axis="1 0 0" range="0 .08" damping=".2" frictionloss=".02"/>\n<geom name="lid_geom" type="box" size=".06 .09 .005" mass=".03"/>\n<geom name="handle" type="box" pos="0 0 .04" size=".005 .09 .035" mass=".01"/><site name="handle_site" pos="0 0 .04"/></body></body>\n<body name="object" pos=".50 -1.7725 .56"><freejoint name="object_free"/><geom name="cube" type="box" size=".018 .02 .04" mass=".02" friction="1 .01 .001" condim="6"/><site name="grasp_site"/></body>\n</worldbody></mujoco>',encoding='utf-8')
    manifest={'schema_version':'1.0','asset_ids':['A01'],'parts':[{'id':'frame','body':'frame'},{'id':'lid','body':'lid'},{'id':'object','body':'object'}],
    'interfaces':[{'id':'grasp','asset_id':'A01','kind':'grasp','body':'object','site':'grasp_site'},
    {'id':'open','asset_id':'A01','kind':'press','body':'lid','site':'handle_site'}],
    'assumptions':['Developer fixture, not model-generated; sample moved 30 mm toward bench front after measured wrist-bench contact, dimensions and required travel unchanged; object six-dimensional frictional contact, coefficients 1 / 0.01 / 0.001 are explicit modeling assumptions'], 'approximations':[], 'buttons':[], 'connections':[]}
    atomic_json(asset/'interfaces.json',manifest)
    source=assemble(plan,env,{'instrument':asset},PROJECT/'data/robots/ffw_sg2',root/'scene')
    # Task landmarks are fixed developer references, not robot geometry changes.
    model=mujoco.MjModel.from_xml_path(str(source/'model.xml'));data=mujoco.MjData(model)
    meta=read(source/'scene.json');data.qpos[:]=meta['home_qpos'];mujoco.mj_forward(model,data)
    from xml.etree.ElementTree import parse,SubElement
    xml=parse(source/'model.xml');frame=next(b for b in xml.iter('body') if b.get('name')=='station__frame')
    base=next(b for b in xml.iter('body') if b.get('name')=='robot__layout_root')
    SubElement(base,'site',name='robot__dock_reference',pos='0 0 0',size='.003')
    dock=data.xpos[model.body('robot__layout_root').id]+[.02,0,0]
    for name,pos in [('dock_target',dock),('place_target',[.50,-1.7375,.56])]:
     SubElement(frame,'site',name='station__'+name,pos=' '.join(str(float(x)) for x in pos),size='.003')
    xml.write(source/'model.xml',encoding='unicode')
    r=prepare(source,root/'dynamic',identity(source),Settings());return source,r
