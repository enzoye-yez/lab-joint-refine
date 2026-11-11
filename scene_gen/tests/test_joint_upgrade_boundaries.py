"""Failure-oriented developer fixtures for the joint workflow, never API experiments."""
import tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import mujoco,numpy as np
from asset_prep.storage import Store,atomic_json,digest
from asset_prep.config import Config
from asset_generation.contracts import GenerationConfig
from asset_generation.pipeline import snapshot
from robot_operation.common import read,identity,Settings,OperationError
from robot_operation.planning import rrt_connect
from robot_operation.bindings import resolve_task
from robot_operation.joint_state import Budget,compare,duplicate,mechanism
from robot_operation.requests import Exchange,AwaitingResponse,import_exchange
from robot_operation.scene_bridge import seed,SceneBridge
from robot_operation.scene import prepare,load
from robot_operation.workflow import run as execute
from robot_operation.diagnosis import inspect_scene,diagnose_run
from operation_fixtures import fixture,task_graph

class BindingTests(unittest.TestCase):
    def test_explicit_semantic_rebinding_preserves_conditions(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="new"><joint name="hinge" range="0 90"/><geom size=".1"/><site name="new_site"/></body></worldbody></mujoco>')
        old={"interfaces":[{"id":"handle","asset_id":"A","kind":"grasp","body":"old","site":"old_site"}]}
        new={"interfaces":[{"id":"handle_v2","replaces":["handle"],"asset_id":"A","kind":"grasp","body":"new","site":"new_site"}]}
        task={"bindings":{"sample":"old"},"final":{"kind":"relative_pose","site_a":"old_site","site_b":"old_site","position":[1,2,3]}}
        original=deepcopy(task);bound,receipt=resolve_task(task,old,new,model)
        self.assertEqual(task,original);self.assertEqual(bound['final']['position'],[1,2,3]);self.assertEqual(bound['bindings']['sample'],'new')
        self.assertFalse(receipt['requirements_changed'])
        new['interfaces'][0]['kind']='press'
        with self.assertRaises(OperationError):resolve_task(task,old,new,model)
    def test_shrinking_range_and_ambiguous_alias_are_rejected(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="part"><joint name="j" range="0 90"/><geom size=".1"/><site name="s"/></body></worldbody></mujoco>')
        item={"id":"open","asset_id":"A","kind":"rotate","body":"part","site":"s","operation":{"joint":"j","range":[0,1.]}}
        old={'interfaces':[item]};new=deepcopy(old);new['interfaces'][0]['operation']['range']=[0,.5]
        with self.assertRaises(OperationError):resolve_task({},old,new,model)
        new={'interfaces':[{**item,'id':'a','replaces':['open']},{**item,'id':'b','replaces':['open']}]}
        with self.assertRaises(OperationError):resolve_task({},old,new,model)

class NumericalBoundaryTests(unittest.TestCase):
    def test_soft_limit_observation_is_not_an_out_of_range_command(self):
        initial=np.array([-1e-5]);limits=np.array([[0.,1.]])
        path=rrt_connect(initial,[.2],limits,lambda q:True,np.random.default_rng(0),start_tolerance=.025)
        np.testing.assert_array_equal(initial,[-1e-5]);self.assertTrue(all(q[0]>=0 for q in path[1:]))
        with self.assertRaises(OperationError):rrt_connect([-.1],[.2],limits,lambda q:True,np.random.default_rng(0),start_tolerance=.025)
        with self.assertRaises(OperationError):rrt_connect([-.001],[.2],limits,lambda q:q[0]>=0,np.random.default_rng(0),start_tolerance=.025)
    def test_goals_never_gain_soft_limit_tolerance(self):
        with self.assertRaises(OperationError):rrt_connect([.1],[-.001],np.array([[0.,1.]]),lambda q:True,np.random.default_rng(0),start_tolerance=.025)
    def test_no_old_two_revision_cap_and_same_input_mechanism_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp),{'fixture':True},Config());budget=Budget(store)
            for i in range(5):budget.round(str(i),'operation')
            self.assertEqual(budget.remaining()['rounds'],75)
        graph={'operation':'press','distance':.1};h={'input_sha256':'a','evidence_sha256':'b','mechanism':digest(mechanism(graph)),'outcome':'ineffective','action_sha256':digest(graph)}
        self.assertTrue(duplicate([h,h],{**graph,'distance':.2},'a','b'))
        self.assertFalse(duplicate([h,h],{**graph,'distance':.2},'new','b'))
    def test_new_static_defect_does_not_overwrite_operation_progress(self):
        base={'result':{'status':'failed','completed_stages':[]},'diagnosis':{'rows':[],'coverage_gaps':[]},'static_report':{'checks':[{'id':'support:a','status':'pass'}]}}
        candidate=deepcopy(base);candidate['result']['completed_stages']=['grasp'];candidate['static_report']['checks'][0]['status']='fail'
        self.assertEqual(compare(base,candidate),'exploration')

class EvidenceBoundaryTests(unittest.TestCase):
    def test_failure_snapshot_keeps_pre_stop_controls_and_is_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source');dynamic=root/'dynamic';r=prepare(source,dynamic,identity(source),Settings(settle_seconds=.1))
            task,graph=task_graph(r['scene_sha256']);graph['nodes'][0]['operation']='check';graph['nodes'][0]['completion']={'kind':'joint_range','joint':'robot__left_x','range':[.3,.4]};graph['nodes'][0]['timeout']=.03
            result=execute(dynamic,r['scene_sha256'],task,graph,root/'execution',Settings(settle_seconds=.1),images=False)
            self.assertEqual(result['status'],'failed');report=diagnose_run(root/'execution');self.assertEqual(report['failure']['scope'],'operation')
            state=read(root/'execution/failure_state.json');state['holding']={'left':'fiction'};atomic_json(root/'execution/failure_state.json',state)
            with self.assertRaises(OperationError):diagnose_run(root/'execution')
    def test_pause_propagates_with_complete_interrupted_receipt(self):
        from robot_operation.improve import Paused
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source');r=prepare(source,root/'dynamic',identity(source),Settings(settle_seconds=.1));task,graph=task_graph(r['scene_sha256'])
            def cancel():raise Paused('developer pause')
            with self.assertRaises(Paused):execute(root/'dynamic',r['scene_sha256'],task,graph,root/'execution',images=False,cancel_guard=cancel)
            self.assertEqual(read(root/'execution/result.json')['status'],'interrupted');self.assertTrue((root/'execution/files.json').exists())
    def test_mechanism_blocking_is_measured_not_inferred_from_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source');xml=(source/'model.xml').read_text()
            xml=xml.replace('</worldbody>','<body name="device__fixed"><geom name="device__block" pos=".5 1 .1" size=".05 .05 .05"/><body name="device__door" pos=".2 1 .1"><joint name="device__slide" type="slide" axis="1 0 0" range="0 .3"/><geom name="device__door_geom" size=".04 .04 .04" type="box"/><site name="device__handle"/></body></body></worldbody>')
            (source/'model.xml').write_text(xml);meta=read(source/'scene.json');meta['plan']['instances'].append({'id':'device','job':'device','placement':'floor'})
            model=mujoco.MjModel.from_xml_path(str(source/'model.xml'));meta['home_qpos']=mujoco.MjData(model).qpos.tolist();atomic_json(source/'scene.json',meta)
            manifest=read(source/'interfaces.json');manifest['interfaces'].append({'id':'device__open','asset_id':'A2','kind':'press','body':'device__door','site':'device__handle','operation':{'joint':'device__slide','range':[0,.3]}});atomic_json(source/'interfaces.json',manifest)
            r=prepare(source,root/'dynamic',identity(source),Settings(settle_seconds=.1));before=identity(root/'dynamic')
            report=inspect_scene(root/'dynamic',before,queries=[{'kind':'mechanism','interface':'device__open'}])
            row=report['rows'][-1];self.assertTrue(row['repair_authorized']);self.assertEqual(row['scope'],'asset');self.assertEqual(identity(root/'dynamic'),before)

class BridgeTests(unittest.TestCase):
    def test_existing_scene_repair_exports_imports_reassembles_on_shared_store(self):
        from test_scene_pipeline import ScenePipelineTests,REVIEW
        from scene_generation.pipeline import SceneRun
        from scene_generation.contracts import SceneConfig,numerical_contract
        from scene_fixtures import SceneSandbox
        from generation_fixtures import PROVENANCE
        helper=ScenePipelineTests();helper.setUp();self.addCleanup(helper.doCleanups)
        helper.run_loop(helper.base_replies+[REVIEW]);source=helper.root/'run';root=helper.root/'joint'
        store=Store(root,{'implementation':read(source/'state.json')['identity']['implementation'],'provenance':'developer_fixture'},Config(max_model_calls=800,max_reserved_tokens=240000000))
        inherited=seed(source,root);store.state.update(inherited);store.state['joint']={};store.state['joint_backend']=inherited['source_identity']['backend'];store.save()
        budget=Budget(store);exchange=Exchange(store,helper.generation);backend=SceneSandbox()
        bridge=SceneBridge(store,exchange,budget,helper.project,helper.source,'local',helper.generation,Config(),lambda:None)
        def runner():
            return SceneRun(helper.catalog,helper.env,numerical_contract({}),Config.load(helper.project/'config/default.json'),helper.generation,SceneConfig(),{},root,None,backend,helper.root/'robot',helper.project,provenance=PROVENANCE,shared_store=store,shared_model=exchange,before_candidate=budget.asset,before_repair=lambda a:budget.round(digest(a),'scene'))
        bridge.runner=runner
        diagnosis={'rows':[{'kind':'collision','instances':['station'],'repair_authorized':True,'geoms':['a','b']}],'coverage_gaps':[]}
        old_calls=len(store.state['calls']);original=snapshot(source)
        for _ in range(8):
            try:result=bridge.repair(diagnosis);break
            except AwaitingResponse:
                req=read(Path(store.state['pending_exchange']['path']));value=({'jobs':[],'placements':[{'id':'station','position':[.6,0,.8],'yaw':0}],'reason':'developer fixture local repair'} if req['phase']=='scene_repair' else {'edits':[{'old':'    pass','new':'    return None'}]} if req['phase'].startswith('asset:') else REVIEW)
                response={'request_id':req['request_id'],'delivered_batches':list(range(len(req['batches']))),'provenance':'developer_fixture','value':value}
                atomic_json(root/'reply.json',response);import_exchange(root,root/'reply.json')
        else:self.fail('offline scene repair did not finish')
        self.assertEqual(result['status'],'candidate',{'result':result,'jobs':store.state['scene']['jobs']});self.assertEqual(store.state['scene']['plan']['instances'][0]['position'],[.6,0,.8])
        self.assertEqual(len(store.state['calls']),old_calls);self.assertEqual(snapshot(source),original)
        self.assertEqual(len(budget.state['rounds']),1);self.assertTrue((Path(result['scene'])/'model.xml').is_file())

class AdditionalBoundaryTests(unittest.TestCase):
    def test_child_exchange_rejects_fixture_in_formal_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root,{'provenance':'model_generated'},Config());exchange=Exchange(store,GenerationConfig())
            with self.assertRaises(AwaitingResponse):exchange.ask('asset repair',{},[],'asset')
            request=read(store.state['pending_exchange']['path'])
            atomic_json(root/'reply.json',{'request_id':request['request_id'],'delivered_batches':list(range(len(request['batches']))),'provenance':'developer_fixture','value':{}})
            import_exchange(root,root/'reply.json')
            with self.assertRaises(OperationError):exchange.ask('asset repair',{},[],'asset')
            self.assertEqual(store.totals()['model_calls'],0)
    def test_run_cannot_pollute_source_before_validation(self):
        from robot_operation.improve import open_run
        from asset_prep.storage import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source');task=root/'task';task.mkdir();profile=root/'profile.json'
            atomic_json(profile,{});before=identity(source)
            with self.assertRaises(OperationError):open_run(source/'bad',source,before,task,profile,Path(__file__).resolve().parents[1])
            self.assertEqual(identity(source),before);self.assertFalse((source/'bad').exists())
    def test_frozen_task_mutation_rejected(self):
        from test_joint_upgrade import JointTests
        helper=JointTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,taskroot,profile=helper.inputs(root);run=helper.create(root,source,taskroot,profile);run.advance()
            task,_=task_graph(run.state['current']['scene_sha256']);task.update(input_sha256=run.bundle['input_sha256'],public_profile_sha256=run.bundle['public_profile_sha256'])
            helper.reply(run,{'task':task,'unverified':[]});run.advance()
            task['final']={'kind':'base_stopped'};atomic_json(run.root/'task.json',task)
            with self.assertRaises(OperationError):run.current_task()
    def test_preview_rejects_wrong_version_and_path_escape(self):
        from robot_operation.views import render_current
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);view=root/'views'/'a';atomic_json(view/'receipt.json',{'scene_sha256':'wrong','images':[],'error':None})
            with self.assertRaises(OperationError):render_current(root,'a',view)
    def test_contact_stop_is_bounded_and_does_not_authorize_other_operations(self):
        from robot_operation.runtime import Session
        from robot_operation.contracts import validate_graph
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source');r=prepare(source,root/'dynamic',identity(source),Settings(settle_seconds=.1));m,d,meta,manifest=load(root/'dynamic',r['scene_sha256']);task,graph=task_graph(r['scene_sha256']);s=Session(m,d,meta,manifest,task,Settings())
            s.deadline=2;before=d.qpos.copy();sid=m.site(s.tcps['left']).id
            condition={'kind':'base_stopped'}
            s.cartesian({'targets':[{'arm':'left','position':(d.site_xpos[sid]+[.02,0,0]).tolist(),'quaternion':[1,0,0,0]}],'stop_when':condition},True)
            np.testing.assert_array_equal(d.qpos,before)
            graph['nodes'][0]['parameters']['stop_when']=condition
            with self.assertRaises(OperationError):validate_graph(graph,task,r['scene_sha256'],Settings())
    def test_held_drift_is_failed_on_same_sample_not_erased_by_window_reset(self):
        from robot_operation.evidence import Evidence
        xml='<mujoco><worldbody><body name="object"><freejoint/><geom size=".01"/></body><body name="hand"><geom size=".01"/><site name="tcp"/></body></worldbody></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        task={'initial':{'kind':'relative_pose','site_a':'tcp','site_b':'tcp'},'final':{'kind':'relative_pose','site_a':'tcp','site_b':'tcp'},'stages':[]}
        e=Evidence(m,d,{'left':set()},task,Settings());rule={'kind':'held','arm':'left','body':'object','tcp':'tcp','duration':.2}
        with patch.object(e,'grip_contacts',return_value=(True,set())):
            self.assertFalse(e.evaluate(rule));d.time=.3;self.assertTrue(e.evaluate(rule));e.lock_hold(rule)
            d.qpos[0]+=.02;mujoco.mj_forward(m,d)
            immediate={**rule,'duration':0}
            self.assertFalse(e.evaluate(immediate));self.assertFalse(e.evaluate(immediate))
            e.update(0);self.assertFalse(e.evaluate(immediate))

class SelectionAuditTests(unittest.TestCase):
    def test_execution_collision_is_regression_even_without_asset_repair_permission(self):
        old={'result':{'status':'failed','completed_stages':[]},'diagnosis':{'rows':[],'coverage_gaps':[]}}
        new=deepcopy(old);new['result']['completed_stages']=['grasp'];new['diagnosis']['rows']=[{'kind':'collision','status':'confirmed_observation','repair_authorized':False,'geoms':['object','robot']}]
        self.assertEqual(compare(old,new),'exploration')
    def test_per_step_cancellation_does_not_rehash_source(self):
        from robot_operation.improve import Coordinator
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            obj=object.__new__(Coordinator);obj.root=Path(tmp);obj.store=SimpleNamespace(state={'calls':[]});obj.guard_impl=Mock()
            for _ in range(100):obj.cancel()
            obj.guard_impl.check.assert_not_called();obj.guard();obj.guard_impl.check.assert_called_once()
    def test_invalid_task_binding_is_recoverable_before_freeze(self):
        from test_joint_upgrade import JointTests
        helper=JointTests()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,taskroot,profile=helper.inputs(root);run=helper.create(root,source,taskroot,profile);run.advance()
            task,_=task_graph(run.state['current']['scene_sha256']);task.update(input_sha256=run.bundle['input_sha256'],public_profile_sha256=run.bundle['public_profile_sha256']);task['bindings']['object']='missing'
            helper.reply(run,{'task':task,'unverified':[]});result=run.advance()
            self.assertEqual(result['status'],'awaiting_response');self.assertEqual(run.state['phase'],'task');self.assertFalse((run.root/'task.json').exists())


class ServoLoadTests(unittest.TestCase):
    def test_joint_path_tracks_gravity_load_through_controls(self):
        import mujoco, numpy as np
        from robot_operation.runtime import Session
        from robot_operation.common import Settings
        model=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/><worldbody><body pos="0 0 .5"><joint name="lift" type="slide" axis="0 0 1" range="-1 1" damping="2"/><geom type="box" size=".02 .02 .02" mass="1"/></body></worldbody><actuator><position joint="lift" kp="100" kv="20" ctrlrange="-1 1"/></actuator></mujoco>')
        for compensated in (False,True):
            with self.subTest(compensated=compensated):
                session=object.__new__(Session);session.model=model;session.data=mujoco.MjData(model)
                session.settings=Settings();session.position_actuators={0:0}
                if compensated:session.data.ctrl[0]=.0981
                controls=[]
                def advance(seconds):
                    for _ in range(round(seconds/model.opt.timestep)):
                        controls.append(float(session.data.ctrl[0]));mujoco.mj_step(model,session.data)
                    mujoco.mj_forward(model,session.data)
                session.advance=advance
                session.follow(['lift'],[np.array([0.]),np.array([.1])])
                self.assertLessEqual(abs(session.data.qpos[0]-.1),session.settings.joint_tolerance)
                self.assertGreater(max(controls),.15)
                self.assertTrue(all(-1<=x<=1 for x in controls))
                self.assertGreater(session.data.time,0)


class ContactManifoldTests(unittest.TestCase):
    def test_derived_contact_manifold_is_recorded_and_resolves_surface_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source');before=identity(source)
            prepared=prepare(source,root/'dynamic',before,Settings(settle_seconds=.1))
            model,_,meta,_=load(root/'dynamic',prepared['scene_sha256'])
            flags=int(model.opt.enableflags)
            self.assertTrue(flags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD))
            self.assertEqual(meta['operation_conversion']['solver']['derived_enableflags'],flags)
            self.assertEqual(identity(source),before)
        verts=' '.join(str(v) for x in (-.02,.02) for y in (-.04,.04) for z in (-.04,.04) for v in (x,y,z))
        xml=f'<mujoco><asset><mesh name="pad" vertex="{verts}"/></asset><worldbody><geom type="mesh" mesh="pad" pos="-.039 0 0"/><body><freejoint/><geom type="box" size=".02 .03 .03"/></body></worldbody></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        one=data.ncon;pose=data.qpos.copy();model.opt.enableflags=flags;mujoco.mj_forward(model,data)
        self.assertGreater(data.ncon,one);self.assertGreaterEqual(data.ncon,4)
        self.assertTrue(np.array_equal(pose,data.qpos))
