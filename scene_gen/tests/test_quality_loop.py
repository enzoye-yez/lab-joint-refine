"""Engineering boundary cases; real archived candidates are replayed separately."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import mujoco
from asset_generation.material import integrity
from scene_generation.progress import repair_comparison, connectivity_delta, repair_tasks
from scene_generation.checks import representative
from scene_generation.repair_schedule import schedule, action_tasks
from asset_prep.phase_budget import enter, PhaseBudgetExceeded
from asset_prep.storage import Store
from asset_prep.config import Config
from robot_operation.quality import check_node
from robot_operation.common import OperationError
from asset_generation.pipeline import snapshot
from asset_prep.storage import digest


def material(bridge=False):
    xml='<mujoco><worldbody><body name="base"><geom name="a" type="sphere" size=".02"/>'
    xml+='<geom name="b" type="sphere" size=".02" pos=".1 0 0"/>'
    if bridge:xml+='<geom name="bridge" type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>'
    xml+='<body name="moving" pos="0 0 .3"><joint name="hinge"/><geom name="c" type="sphere" size=".02"/></body></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    return integrity(m,d)


def precheck_run(root,recheck):
    """A run whose last applied repair changed job a from stage old to new; recheck(artifact) gives each report."""
    records={}
    for key in ('old','new'):
        artifact=root/key;artifact.mkdir();(artifact/'geometry').write_text(key)
        records[key]={'directory':key,'artifact':key,
            'assembly_readiness':{'loadable':True,'artifact_sha256':digest(snapshot(artifact))}}
    entry={'status':'applied','baseline':{'scene':'scene','jobs':{'a':'old'}},
        'action':{'jobs':['a'],'placements':[]},'target_checks':['material_integrity:x'],'call_start':0}
    run=SimpleNamespace(root=root,contract={'jobs':{}},implementation_guard=SimpleNamespace(expected='code'),
        state={'repair_history':[entry],'stages':[records['old']], 'jobs':{'a':records['new']},
            'plan':{'jobs':[{'id':'a','asset_ids':['A01']}],'instances':[{'id':'x','job':'a'}]}},
        store=SimpleNamespace(state={'calls':[]},save=lambda:None),repair_input_key=lambda:'candidate',
        sandbox=SimpleNamespace(recheck_asset=lambda artifact,*args:recheck(artifact)),selected=[])
    def begin(kind):
        run.destination=root/('check'+str(len(run.state['stages'])))
        run.destination.mkdir();return run.destination
    def commit(kind,record):
        stage={**record,'directory':run.destination.name,'files':snapshot(run.destination)}
        run.state['stages'].append(stage);return stage
    run.begin=begin
    run.commit=commit
    run.select_repair_candidate=lambda:run.selected.append(True)
    return run,entry


class QualityLoopTests(unittest.TestCase):
    def test_local_regression_stops_before_assembly_and_reuses_checks(self):
        from scene_generation.local_repair import precheck
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            def recheck(artifact):
                calls.append(artifact.name)
                row=material(artifact.name=='old')
                return {'checks':[row],'status':row['status']},None
            run,entry=precheck_run(Path(directory),recheck)
            self.assertTrue(precheck(run));self.assertEqual(calls,['old','new']);self.assertTrue(run.selected)
            entry['status']='applied'
            self.assertTrue(precheck(run));self.assertEqual(calls,['old','new'])

    def test_repaired_asset_whose_recheck_times_out_is_rejected_without_pausing(self):
        from scene_generation.local_repair import precheck
        from asset_generation.sandbox import WorkerTimeout
        with tempfile.TemporaryDirectory() as directory:
            def recheck(artifact):
                if artifact.name=='new':raise WorkerTimeout('本地工作进程 validate 超时')
                row=material(True)
                return {'checks':[row],'status':row['status']},None
            run,entry=precheck_run(Path(directory),recheck)
            self.assertTrue(precheck(run));self.assertTrue(run.selected)
            self.assertEqual(entry['outcome'],'regressed')
            self.assertEqual(entry['comparison']['new_defects'][0]['added'],['worker_timeout'])

    def test_recorded_candidates_retain_unknowns_while_improving(self):
        import json
        for path in (Path(__file__).parent/'fixtures/quality_loop').glob('*.json'):
            record=json.loads(path.read_text('utf-8'))
            self.assertTrue(record['sources'])
            result=repair_comparison(record['before'],record['after'])
            self.assertEqual(result['outcome'],'improved_partial',path.name)
            self.assertFalse(record['after']['checks'][0]['complete'])
    def test_rigid_repair_is_adopted_while_floating_articulation_remains(self):
        # The hinged sphere touches nothing: a decided floating articulation, not an unknown.
        old,new=material(),material(True)
        self.assertEqual(old['obligations']['articulated_connection'],'fail')
        self.assertEqual(new['obligations']['articulated_connection'],'fail')
        self.assertEqual(old['disconnected_count'],2)
        self.assertEqual(new['disconnected_count'],0)
        self.assertEqual(new['floating_count'],1)
        result=repair_comparison({'checks':[old]},{'checks':[new]})
        self.assertTrue(result['adopt'],result)
        self.assertEqual(result['resolved_checks'],[])

    def test_delete_material_or_new_collision_rejected(self):
        old,new=material(),material(True)
        new['connectivity']['geoms'].remove('a')
        self.assertFalse(repair_comparison({'checks':[old]},{'checks':[new]})['adopt'])
        new=material(True);new.update(status='fail',collisions=[{'geom1':'a','geom2':'c','distance_m':-.1}],collision_count=1)
        self.assertFalse(repair_comparison({'checks':[old]},{'checks':[new]})['adopt'])

    def test_rebuilt_connector_is_not_a_split(self):
        old,new=material(),material(True)
        for record in (new['connectivity'],):
            record['geoms']=[g if g!='b' else 'b_tip' for g in record['geoms']]
            for group in record['rigid_groups'].values():
                for key in ('confirmed','possible'):
                    group[key]=[[('b_tip' if n=='b' else n) for n in island] for island in group[key]]
        self.assertIsNone(connectivity_delta(old,new))

    def test_collisions_split_into_one_task_per_instance_pair(self):
        hits=[{'geom1':'p__x','geom2':'q__y','distance_m':-.01,'instances':['p','q']},
              {'geom1':'r__x','geom2':'s__y','distance_m':-.003,'instances':['r','s']}]
        row={'id':'scene_collisions','status':'fail','instances':['p','q','r','s'],'collisions':hits,'collision_count':2}
        tasks=repair_tasks({'checks':[row]},None,{'instances':[{'id':i,'job':i} for i in 'pqrs']},'key')
        self.assertEqual(sorted(t['check_id'] for t in tasks),['scene_collisions:p|q','scene_collisions:r|s'])
        self.assertEqual(representative(list(reversed(hits)),limit=2)[0]['instances'],['p','q'])

    def test_round_on_another_job_does_not_count_against_a_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a.py').write_text('a');(root/'b.py').write_text('b')
            run=SimpleNamespace(root=root,implementation_guard=SimpleNamespace(expected='code'),contract={},env={},
                state={'plan':{'jobs':[{'id':k,'depends_on':[]} for k in ('a','b')],
                    'instances':[{'id':k,'job':k} for k in ('a','b')]},
                    'jobs':{k:{'source':k+'.py','assembly_readiness':{'artifact_sha256':k}} for k in ('a','b')}})
            tasks=[{'id':'a','check_id':'a','instances':['a'],'repair_level':'asset_structure'}]
            binding=schedule(run,deepcopy(tasks))[0]['binding']
            run.state['repair_history']=[{'status':'evaluated','task_bindings':{'a':binding},'selection':'baseline_preserved',
                                          'action':{'jobs':['b']}}]*3
            self.assertEqual([t['id'] for t in schedule(run,deepcopy(tasks))],['a'])

    def test_checker_version_mismatch_requires_remeasurement(self):
        old,new=material(),material(True);old.pop('connectivity')
        self.assertEqual(repair_comparison({'checks':[old]},{'checks':[new]})['outcome'],'coverage_lost')

    def test_schedule_fairness_deferral_and_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a.py').write_text('a');(root/'b.py').write_text('b')
            run=SimpleNamespace(root=root,implementation_guard=SimpleNamespace(expected='code'),contract={},env={},
                state={'plan':{'jobs':[{'id':k,'depends_on':[]} for k in ('a','b')],
                    'instances':[{'id':k,'job':k} for k in ('a','b')]},
                    'jobs':{k:{'source':k+'.py','assembly_readiness':{'artifact_sha256':k}} for k in ('a','b')}})
            tasks=[{'id':k,'check_id':k,'instances':[k],'repair_level':'asset_structure'} for k in ('a','b')]
            first=schedule(run,deepcopy(tasks));binding=first[0]['binding']
            run.state['repair_history']=[{'status':'evaluated','task_bindings':{'a':binding},'selection':'baseline_preserved','mechanism':'m',
                                          'action':{'jobs':['a']}}]
            self.assertEqual(schedule(run,deepcopy(tasks))[0]['id'],'b')
            run.state['repair_history']*=3
            self.assertEqual([t['id'] for t in schedule(run,deepcopy(tasks))],['b'])
            (root/'b.py').write_text('unrelated change')
            self.assertEqual([t['id'] for t in schedule(run,deepcopy(tasks))],['b'])
            (root/'a.py').write_text('relevant change')
            self.assertIn('a',[t['id'] for t in schedule(run,deepcopy(tasks))])

    def test_only_edited_targets_are_charged(self):
        plan={'instances':[{'id':'x','job':'a'},{'id':'y','job':'b'}]}
        tasks=[{'id':i,'instances':[i]} for i in ('x','y')]
        self.assertEqual([t['id'] for t in action_tasks(tasks,{'jobs':['a']},plan,plan)],['x'])

    def test_phase_budget_is_persisted_and_carries_forward(self):
        with tempfile.TemporaryDirectory() as root:
            config=Config(max_model_calls=10,max_reserved_tokens=1000)
            s=Store(root,{},config);enter(s,'scene')
            for i in range(7):s.reserve(str(i),'model',100)
            s=Store(root,{},config,resume=True);enter(s,'scene')
            s.reserve('7','model',100)
            enter(s,'operation')
            s.reserve('op','model',100)
            with self.assertRaises(PhaseBudgetExceeded):s.reserve('final reserved','model',1)
            enter(s,'final');s.reserve('final','model',100)
            self.assertEqual(s.totals()['model_calls'],10)

    def test_unused_mechanical_event_does_not_block_independent_node(self):
        session=SimpleNamespace(holding={},manifest={'interfaces':[{'id':'lid','body':'a__lid'}],
            'mechanical_events':[{'id':'opened','interface':'lid'}]},unbound_events=['opened'],
            meta={'plan':{'instances':[{'id':'a'},{'id':'b'}]}},quality_report={})
        check_node(session,{'parameters':{'body':'b__object'}})
        with self.assertRaises(OperationError):check_node(session,{'parameters':{'interface':'lid'}})
        session.quality_report={'checks':[{'id':'support:a','instances':['a'],'status':'partial'}]}
        check_node(session,{'parameters':{'body':'b__object'}})
        with self.assertRaises(OperationError):check_node(session,{'parameters':{'body':'a__object'}})
        session.unbound_events=[]
        with self.assertRaises(OperationError):check_node(session,{'parameters':{'interface':'lid'}})
        session.quality_report={'status':'error','checks':[]}
        with self.assertRaises(OperationError):check_node(session,{'parameters':{'body':'b__object'}})


if __name__=='__main__':unittest.main()
