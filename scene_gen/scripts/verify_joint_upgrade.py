"""Offline original-robot developer acceptance. No API client or credentials; no fabricated passes."""
import argparse,json,sys,os
from pathlib import Path
from copy import deepcopy
PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'tests'),str(PROJECT/'scripts')]
from test_offline import deny_external
sys.addaudithook(deny_external)
for key in list(os.environ):
    if key.endswith('API_KEY'):os.environ.pop(key)
import mujoco,numpy as np
from asset_prep.storage import atomic_json,digest,file_hash
from asset_prep import implementation
from robot_operation.common import read,identity,Settings
from robot_operation.runtime import Session
from robot_operation.scene import load
from robot_operation.improve import open_run
from robot_operation.requests import import_exchange
from robot_operation.recording import replay
from joint_ffw_fixture import build

def teach(root,dynamic,sha,task,settings):
    """Developer precomputed strategy. Recording is later rerun from the full initial state."""
    m,d,meta,manifest=load(dynamic,sha);s=Session(m,d,meta,manifest,task,settings);nodes=[]
    def act(op,p,stage,timeout=45):
        node={'id':'n'+str(len(nodes)),'operation':op,'stage':stage,'parameters':deepcopy(p),
              'precondition':{'kind':'base_stopped'},'completion':{'kind':'base_stopped'},
              'timeout':timeout,'on_success':'success','on_failure':'failed'}
        if nodes:nodes[-1]['on_success']=node['id']
        nodes.append(node);s.begin(node)
        atomic_json(root/'developer_actions.json',nodes)
        try:s.execute(node)
        except Exception:
            atomic_json(root/'teaching_failure_state.json',s.snapshot());print("failure evidence",getattr(s,"last_collision",None),getattr(s,"last_tracking",None),flush=True);raise
        print('developer strategy',op,float(d.time),flush=True)
    act('wait',{'seconds':1.},'dock')
    targets=[]
    for arm in ('left','right'):
        sid=m.site(s.tcps[arm]).id;q=np.zeros(4);mujoco.mju_mat2Quat(q,d.site_xmat[sid])
        targets.append({'arm':arm,'position':(d.site_xpos[sid]+[0,0,.1]).tolist(),'quaternion':q.tolist()})
    act('cartesian',{'targets':targets},'dock')
    transport={'arms':['left','right'],'positions':{j:float(np.clip(d.qpos[m.joint(j).qposadr[0]],*m.joint(j).range)) for j in s.joints_for(['left','right'])}}
    act('navigate',{'goal':[.0205,-2,0],'bounds':[[-.3,-2.5],[.25,-1.5]],'transport':transport},'dock')
    allowed_right=[[m.body(m.geom_bodyid[g]).name,'station__lid'] for g in s.actors['right']]
    sid=m.site(s.tcps['right']).id;q=np.zeros(4);mujoco.mju_mat2Quat(q,d.site_xmat[sid])
    act('cartesian',{'targets':[{'arm':'right','position':[.54,-2.227,.58],'quaternion':q.tolist()}],'allowed_contacts':allowed_right},'open')
    act('press',{'arm':'right','axis':[1,0,0],'distance':.04,'hold_arms':['left'],'allowed_contacts':allowed_right},'open')
    act('press',{'arm':'right','axis':[0,0,1],'distance':.07,'hold_arms':['left'],'allowed_contacts':allowed_right},'grasp')
    allowed_left=[[m.body(m.geom_bodyid[g]).name,'station__object'] for g in s.actors['left']]+[['station__object','environment__table_bench']]
    sid=m.site(s.tcps['left']).id;q=np.zeros(4);mujoco.mju_mat2Quat(q,d.site_xmat[sid])
    right=m.site(s.tcps['right']).id;rq=np.zeros(4);mujoco.mju_mat2Quat(rq,d.site_xmat[right])
    hold={'arm':'right','position':d.site_xpos[right].tolist(),'quaternion':rq.tolist()}
    left={'arm':'left','position':[.50,-1.7725,.66],'quaternion':[float(np.sqrt(.5)),0,float(np.sqrt(.5)),0]}
    act('cartesian',{'targets':[left,hold],'allowed_contacts':allowed_left,'include_lift':True},'grasp')
    left={**left,'position':[.50,-1.7725,.56]}
    act('cartesian',{'targets':[left,hold],'allowed_contacts':allowed_left,'include_lift':True,'contact':True},'grasp')
    act('grasp',{'arm':'left','body':'station__object','controls':{'robot__gripper_l_joint1':.85},'closure_speed':.05,'contact_force_n':1.,'allowed_contacts':allowed_left},'grasp')
    for axis,distance in (([0,0,1],.04),([0,1,0],.035)):
        act('press',{'arm':'left','axis':axis,'distance':distance,'hold_arms':['right'],'allowed_contacts':allowed_left},'transfer')
    act('press',{'arm':'left','axis':[0,0,-1],'distance':.04,'hold_arms':['right'],'allowed_contacts':allowed_left,'stop_when':{'kind':'contact','body_a':'station__object','body_b':'environment__table_bench'}},'release')
    act('place',{'arm':'left','body':'station__object','controls':{'robot__gripper_l_joint1':0},'support':'environment__table_bench','allowed_contacts':allowed_left},'release')
    return {'version':1,'scene_sha256':sha,'task_sha256':digest(task),'provenance':'developer_fixture','entry':'n0','nodes':nodes}

def preview(run,out):
    from PIL import Image,ImageDraw
    frames=[];images=sorted((run/'images').glob('*_overview.png'))
    stride=max(1,len(images)//140)
    for path in images[::stride]:
        prefix=path.name.split('_')[0];canvas=Image.new('RGB',(960,264),'white')
        for i,view in enumerate(('overview','left','right')):
            with Image.open(path.with_name(prefix+'_'+view+'.png')) as im:canvas.paste(im,(i*320,24))
        ImageDraw.Draw(canvas).text((8,5),'Developer fixture | original FFW-SG2 | overview / left / right',fill='black')
        frames.append(canvas)
    if frames:frames[0].save(out,save_all=True,append_images=frames[1:],duration=int(stride*read(run/"recording.json")["settings"]["image_period"]*1000),loop=0)
    return len(frames)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--developer-strategy',type=Path);args=parser.parse_args()
    root=args.output.resolve()
    if root.exists():raise ValueError('Use a new output directory; prior attempts are immutable')
    root.mkdir(parents=True)
    code=implementation.implementation_digest(implementation.source_manifest(PROJECT))
    settings=Settings(image_period=.5)
    source,prepared=build(root/'fixture');dynamic=root/'fixture/dynamic';sha=prepared['scene_sha256']
    stopped={'kind':'base_stopped'}
    dock={'kind':'relative_pose','site_a':'robot__dock_reference','site_b':'station__dock_target','position_tolerance':.003}
    opened={'kind':'joint_range','joint':'station__slide','range':[.025,.08]}
    held={'kind':'held','arm':'left','body':'station__object','tcp':'robot__left_tcp','duration':.2}
    transfer={'kind':'relative_pose','site_a':'station__grasp_site','site_b':'station__place_target','offset':[0,0,.04],'position_tolerance':.01}
    released={'kind':'all','items':[{'kind':'supported','body':'station__object','support':'environment__table_bench','duration':.2},
        {'kind':'relative_pose','site_a':'station__grasp_site','site_b':'station__place_target','position_tolerance':.01}]}
    task={'version':1,'provenance':'developer_fixture','input_sha256':'developer_fixture','public_profile_sha256':'developer_fixture',
          'bindings':{'sample':'station__object','lid':'station__lid'},'initial':stopped,'stages':[],'final':{'kind':'all','items':[opened,released]}}
    for name,goal in [('dock',dock),('open',opened),('grasp',held),('transfer',transfer),('release',released)]:
        task['stages'].append({'id':name,'source':'developer short-chain specification, not a real protocol','depends_on':[] if not task['stages'] else [task['stages'][-1]['id']],'goal':goal})
    # Freeze requirements BEFORE any candidate operation graph or teaching execution exists.
    atomic_json(root/'independent_task.json',task)
    try:
        if args.developer_strategy:
            nodes=read(args.developer_strategy);atomic_json(root/"strategy_source.json",{"path":str(args.developer_strategy.resolve()),"sha256":file_hash(args.developer_strategy),"provenance":"developer_fixture"})
            graph={"version":1,"scene_sha256":sha,"task_sha256":digest(task),"provenance":"developer_fixture","entry":nodes[0]["id"],"nodes":nodes}
        else:graph=teach(root,dynamic,sha,task,settings)
    except Exception as exc:
        atomic_json(root/'verification.json',{'status':'failed','phase':'developer_strategy','detail':str(exc),'implementation':code,'network_calls':0});raise
    taskroot=root/'task_inputs';taskroot.mkdir();(taskroot/'protocol.txt').write_text('Developer task: wheel dock 20 mm forward; physically open the sliding lid at least 25 mm; left gripper pick; move 40 mm up and 35 mm lateral while right arm holds; return height and release on the bench. Object 36 x 40 x 80 mm. No liquid simulation. Static pose tolerances are engineering fixture bounds.',encoding='utf-8')
    atomic_json(taskroot/'input_manifest.json',{'task_id':'ffw_developer_short_chain','files':[{'path':'protocol.txt','sha256':file_hash(taskroot/'protocol.txt')}]})
    profile=root/'public_profile.json';atomic_json(profile,{'version':1,'robot':'original pinned FFW-SG2','workspace':'developer bench and fixture'})
    joint=open_run(root/'joint',source,identity(source),taskroot,profile,PROJECT,developer=True,settings=settings)
    joint.advance();task.update(input_sha256=joint.bundle['input_sha256'],public_profile_sha256=joint.bundle['public_profile_sha256'])
    def reply(value):
        pending=read(Path(joint.store.state['pending_exchange']['path']))
        value={'request_id':pending['request_id'],'delivered_batches':list(range(len(pending['batches']))),'provenance':'developer_fixture','value':value}
        atomic_json(root/'reply.json',value);import_exchange(joint.root,root/'reply.json');return joint.advance()
    reply({'task':task,'unverified':[]})
    graph.update(scene_sha256=joint.state['current']['scene_sha256'],task_sha256=digest(task))
    bad=deepcopy(graph)
    next(n for n in bad['nodes'] if n['operation']=='grasp')['parameters']['controls']={'robot__gripper_l_joint1':0}
    next(n for n in bad['nodes'] if n['operation']=='grasp')['timeout']=2
    first=reply({'graph':bad});atomic_json(root/'injected_empty_grasp.json',first)
    failed=read(joint.root/joint.state['last_run']/'failure_state.json')
    injected_node=next(n['id'] for n in bad['nodes'] if n['operation']=='grasp')
    injection_observed=failed['node']==injected_node and first['operation_result'].get('code')=='timeout'
    if injection_observed and joint.state['phase']=='route':reply({'operation':{'reason':'Developer correction: restore actual closure at 0.05 rad/s, same frozen task and scene'}})
    result=reply({'graph':graph}) if injection_observed and joint.state['phase']=='graph' else first
    final=joint.root/'final/1';replayed=None;frames=0
    preview_run=final if final.exists() else joint.root/joint.state.get('last_run','executions/0000')
    if (preview_run/'recording.json').exists():
        replayed=replay(preview_run);frames=preview(preview_run,root/'short_chain.gif')
    receipt={'status':'pass' if result['status']=='verified_task' and replayed and replayed['status']=='pass' else 'failed',
             'implementation':code,'network_calls':0,'provenance':'developer_fixture','model_autonomous_success':False,
             'original_robot_resource':str(PROJECT/'data/robots/ffw_sg2'), 'result':result,'control_replay':replayed,'preview_frames':frames,'preview_run':str(preview_run),'preview_is_final':final.exists(),
             'injection':{'kind':'empty grasp then explicit canned correction','observed':injection_observed,'failed_node':failed['node']},'frozen_task':str(root/'joint/task.json')}
    atomic_json(root/'verification.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2));return 0 if receipt['status']=='pass' else 2

if __name__=='__main__':raise SystemExit(main())
