"""开发者合成模型、响应和故障注入；无正式模型请求、无实验操作。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asset_generation.distance import DistanceQuery, MeasurementError
from asset_generation.pipeline import snapshot
from asset_generation.local import LocalBackend
from asset_generation.contracts import GenerationConfig
from asset_prep.errors import ValidationError, StaleEvidence
from asset_prep.storage import atomic_json
from scene_generation.capabilities import (REGISTRY, register, merge, proposal, repair_signature,
    repeated_repair, problem_identity)
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.dispatch import recover
from scene_generation.recovery import decide
from scene_generation.recovery_geometry import CertificateQuery
from test_distance_evidence import compiled
import test_scene_pipeline as pipeline_tests
from test_scene_pipeline import REVIEW
from test_main_check_recovery import RecoveringBackend


def unknown(check='scene_collisions'):
    return {'status':'partial','checks':[{'id':check,'status':'partial','instances':[],
            'coverage_gaps':[{'geom1':'a__g','geom2':'b__g','error_kind':'measurement_error'}]}],
            'renders':['original.png'],'manifest':{'keep':True},'coverage':['original qualification']}


class Backend:
    def __init__(self):
        self.calls = []

    def recheck_scene(self,*args):
        self.calls.append('full_recheck_v2')
        return unknown()

    def recover_scene(self, artifact, destination, limits, contract, capability, request, input_key):
        self.calls.append(capability)
        return {**REGISTRY[capability].inspect(artifact,limits,contract,request),'input_key':input_key}


class DispatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.artifact = self.root/'scene'
        self.artifact.mkdir()
        self.plan = {'instances':[{'id':'a','job':'first','placement':'table'},
                                  {'id':'b','job':'second','placement':'table','support_instance':'a'}]}
        self.write_scene('.15 0 0')
        self.backend = Backend()
        self.run = SimpleNamespace(root=self.root, state={'plan':self.plan},sandbox=self.backend,
            limits=SceneConfig(validation_mode='static'),contract=numerical_contract({}),check_pause=lambda:None)
        self.saved = None
        def save():
            self.saved = copy.deepcopy(self.run.state)
        self.run.store = SimpleNamespace(save=save)

    def write_scene(self, pos):
        (self.artifact/'model.xml').write_text('<mujoco><compiler angle="radian"/><worldbody>'
            '<body name="a__body"><geom name="a__g" type="sphere" size=".1"/></body>'
            f'<body name="b__body"><geom name="b__g" type="sphere" size=".1" pos="{pos}"/></body></worldbody></mujoco>',encoding='utf-8')
        atomic_json(self.artifact/'scene.json',{'plan':self.plan,'home_qpos':[],'home_ctrl':[]})

    def recover(self, report=None):
        return recover(self.run,report or unknown(),self.artifact,self.root/'stage')

    def session(self):
        return list(self.run.state['check_recovery'].values())[-1]

    def test_second_real_capability_executes_after_ineffective_first_and_changes_decision(self):
        before = snapshot(self.artifact)
        with patch.object(DistanceQuery,'measure',side_effect=MeasurementError('injected native failure')),patch('asset_generation.mesh_exact.exact',return_value=None):
            result = self.recover()
        self.assertEqual(self.backend.calls,['full_recheck_v2','collision_certificates_v1'])
        self.assertEqual(decide(result),'repair')
        self.assertAlmostEqual(result['checks'][0]['collisions'][0]['distance_m'],-.05)
        self.assertFalse(self.session()['actions'][0]['effective'])
        self.assertTrue(self.session()['actions'][1]['effective'])
        self.assertEqual(snapshot(self.artifact),before)
        self.assertEqual(result['manifest'],{'keep':True})

    def test_support_uses_contact_evidence_and_not_clearance(self):
        self.write_scene('0 0 .2')
        self.backend.recheck_scene = lambda *args: unknown('support:b')
        with patch.object(DistanceQuery,'measure',side_effect=MeasurementError('injected')),patch('asset_generation.mesh_exact.exact',return_value=None):
            result = self.recover(unknown('support:b'))
        self.assertEqual(result['checks'][0]['status'],'pass')
        self.assertEqual(result['checks'][0]['contact_count'],1)
        self.assertEqual(self.backend.calls,['support_certificates_v1'])
        self.assertNotEqual(decide(result),'candidate_ready')  # 还没有视觉审查。

    def test_locates_only_confirmed_geometries_before_repair(self):
        report = {'status':'fail','checks':[{'id':'scene_collisions','status':'fail','instances':[],
            'collisions':[{'geom1':'a__g','geom2':'environment__table','distance_m':-.01}],
            'coverage_gaps':[{'geom1':'b__g','geom2':'unknown'}]}]}
        self.backend.recheck_scene = lambda *args: report
        # 本测试隔离定位消费；不让独立碰撞扫描引入新缺陷。
        with patch.dict(REGISTRY,{'locate_defects_v1':REGISTRY['locate_defects_v1']},clear=True):
            result = self.recover(report)
        self.assertEqual(result['checks'][0]['instances'],['a'])
        self.assertEqual(result['checks'][0]['coverage_gaps'],report['checks'][0]['coverage_gaps'])
        self.assertEqual(self.backend.calls,['locate_defects_v1'])

    def test_new_capability_registration_needs_no_pipeline_branch(self):
        class New:
            id = 'developer_registered_measurement'
            def propose(self, ctx, report):
                if report['checks'][0]['status']=='partial':
                    yield proposal(['scene_collisions'],pairs=[['a__g','b__g']])
            def execute(self, ctx, request, destination):
                # 真实解析球体证据来自相同的编译模型，而非直接声明通过。
                return {**REGISTRY['collision_certificates_v1'].inspect(ctx.artifact,ctx.limits,ctx.contract,
                        {**request,'max_pairs':100}), 'input_key':ctx.input_key}
        original = REGISTRY.copy()
        self.addCleanup(lambda:(REGISTRY.clear(),REGISTRY.update(original)))
        REGISTRY.clear()
        register(New())
        # 执行函数使用原可信几何实现；选择仅通过新增注册项完成。
        impl = original['collision_certificates_v1']
        with patch.object(New,'execute',lambda cap,ctx,req,dest:{**impl.inspect(ctx.artifact,ctx.limits,ctx.contract,
                {**req,'max_pairs':100}),'input_key':ctx.input_key}):
            result = self.recover()
        self.assertEqual(decide(result),'repair')
        self.assertEqual(self.session()['actions'][0]['capability'],New.id)

    def test_interruption_after_first_action_keeps_scene_and_does_not_repeat_first(self):
        real = self.backend.recover_scene
        with patch.object(self.backend,'recover_scene',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            self.recover()
        self.run.state = copy.deepcopy(self.saved)
        with patch.object(self.backend,'recover_scene',side_effect=real):
            result = self.recover()
        self.assertEqual(self.backend.calls.count('full_recheck_v2'),1)
        self.assertEqual(decide(result),'repair')
        self.assertEqual(self.session()['actions'][1]['attempts'],2)

    def test_receipt_survives_interruption_before_state_commit(self):
        save = self.run.store.save
        def interrupt():
            if any(a.get('status')=='completed' for h in self.run.state.get('check_recovery',{}).values() for a in h['actions']):
                raise KeyboardInterrupt()
            save()
        with patch.object(self.run.store,'save',side_effect=interrupt),self.assertRaises(KeyboardInterrupt):
            self.recover()
        self.run.state = copy.deepcopy(self.saved)
        result = self.recover()
        self.assertEqual(self.backend.calls.count('full_recheck_v2'),1)
        self.assertEqual(decide(result),'repair')

    def test_budget_stop_then_resume_runs_remaining_action(self):
        self.run.limits = SceneConfig(validation_mode='static',max_check_actions=1)
        result = self.recover()
        self.assertEqual(self.session()['status'],'budget_exhausted')
        self.assertEqual(result['status'],'partial')
        self.run.limits = SceneConfig(validation_mode='static',max_check_actions=2)
        self.assertEqual(decide(self.recover()),'repair')
        self.assertEqual(self.backend.calls.count('full_recheck_v2'),1)

    def test_persistent_unknown_has_exhaustion_evidence_and_cannot_pass(self):
        with patch.object(REGISTRY['collision_certificates_v1'],'inspect',return_value={'scope':'checks','checks':unknown()['checks']}):
            result = self.recover()
            self.recover()
        self.assertEqual(result['status'],'partial')
        self.assertEqual(self.session()['exhaustion']['reason'],'no_executable_action')
        self.assertEqual(len(self.backend.calls),2)

    def test_checker_bug_blocks_geometry_repair_and_retains_defect(self):
        with patch.object(REGISTRY['collision_certificates_v1'],'inspect',side_effect=TypeError('checker bug')):
            result = self.recover()
        self.assertEqual(decide(result),'needs_environment')
        self.assertEqual(self.session()['actions'][-1]['failure']['error_kind'],'check_execution')

    def test_new_scene_version_invalidates_previous_pass_and_executes_again(self):
        self.recover()
        self.write_scene('.14 0 0')
        self.recover()
        self.assertEqual(self.backend.calls.count('full_recheck_v2'),2)
        self.assertEqual(len(self.run.state['check_recovery']),2)

    def test_readonly_violation_is_rejected(self):
        def corrupt(*args):
            (self.artifact/'model.xml').write_text('changed',encoding='utf-8')
            return unknown()
        with patch.object(self.backend,'recheck_scene',side_effect=corrupt),self.assertRaises(StaleEvidence):
            self.recover()

    def test_registered_capability_runs_through_real_local_worker(self):
        backend = LocalBackend(GenerationConfig(),Path(__file__).resolve().parents[1])
        before = snapshot(self.artifact)
        result = backend.recover_scene(self.artifact,self.root/'local',self.run.limits,self.run.contract,
            'collision_certificates_v1',{'checks':['scene_collisions'],'max_pairs':100},'version')
        self.assertEqual(result['input_key'],'version')
        self.assertEqual(result['checks'][0]['status'],'fail')
        self.assertEqual(snapshot(self.artifact),before)


class MergeAndIdentityTests(unittest.TestCase):
    def test_partial_pairs_preserve_other_gaps_defects_checks_and_metadata(self):
        report = unknown()
        report['checks'][0]['coverage_gaps'] += [{'geom1':'c','geom2':'d'}, {'unknown_pair_count':906}]
        report['checks'] += [{'id':'support:b','status':'partial','coverage_gaps':[{'error_kind':'measurement_error'}]}]
        result = {'input_key':'v','scope':'pairs','covered_pairs':[['a__g','b__g']],
                  'checks':[{'id':'scene_collisions','status':'pass','coverage_gaps':[],'collisions':[]}]}
        merged = merge(report,result,proposal(['scene_collisions'],pairs=[['a__g','b__g']]),'v')
        self.assertEqual(len(merged['checks'][0]['coverage_gaps']),2)
        self.assertEqual(merged['checks'][1],report['checks'][1])
        self.assertEqual(merged['status'],'partial')
        self.assertEqual(merged['renders'],report['renders'])
        result['checks'][0]['status']='error'
        failed = merge(report,result,proposal(['scene_collisions'],pairs=[['a__g','b__g']]),'v')
        self.assertEqual(failed['checks'][0]['coverage_gaps'],report['checks'][0]['coverage_gaps'])

    def test_stale_out_of_scope_and_clearance_as_support_rejected(self):
        for result,request in (
            ({'input_key':'old','scope':'checks','checks':[]},proposal(['scene_collisions'])),
            ({'input_key':'v','scope':'checks','checks':[{'id':'support:b','status':'pass'}]},proposal(['scene_collisions'])),
            ({'input_key':'v','scope':'pairs','checks':[{'id':'support:b','status':'pass'}]},proposal(['support:b']))):
            report = unknown()
            report['checks'] += [{'id':'support:b','status':'partial'}]
            with self.subTest(result=result),self.assertRaises(ValidationError):
                merge(report,result,request,'v')

    def test_same_version_pass_cannot_erase_confirmed_defect(self):
        report = {'status':'fail','checks':[{'id':'x','status':'fail','instances':['a']}]}
        result = merge(report,{'input_key':'v','scope':'checks','checks':[{'id':'x','status':'pass'}]},proposal(['x']),'v')
        self.assertEqual(result['status'],'fail')
        report['checks'][0]['collisions']=[{'geom1':'a','geom2':'b','method':'first','distance_m':-.01}]
        result = merge(report,{'input_key':'v','scope':'checks','checks':[{'id':'x','status':'fail',
            'collisions':[{'geom1':'b','geom2':'a','method':'second','distance_m':-.01}]}]},proposal(['x']),'v')
        self.assertEqual(len(result['checks'][0]['collisions']),1)

    def test_reason_rewrite_deduplicated_but_distinct_parameters_and_proposals_not_blocked(self):
        a = {'jobs':[],'placements':[{'id':'a','position':[1,0,0],'yaw':0}],'reason':'first'}
        b = {**a,'reason':'another explanation'}
        self.assertEqual(repair_signature(a),repair_signature(b))
        h = {'input_key':'v','output_key':'w','problem_key':'p','action_key':repair_signature(a),'status':'applied'}
        self.assertFalse(repeated_repair([h],'v','p',repair_signature(b)))
        h.update(status='evaluated',outcome='ineffective')
        self.assertTrue(repeated_repair([h],'w','p',repair_signature(b)))
        b['placements'] = [{'id':'a','position':[2,0,0],'yaw':0}]
        self.assertFalse(repeated_repair([h],'w','p',repair_signature(b)))
        self.assertFalse(repeated_repair([h],'new_version','p',repair_signature(a)))
        a = {'jobs':['j'],'placements':[],'reason':'design',
             'geometry_changes':[{'job':'j','operation':'resize','parameters':{'part':'base','width':.1}}]}
        b = copy.deepcopy(a);b['geometry_changes'][0]['parameters']['width'] = .2
        self.assertNotEqual(repair_signature(a),repair_signature(b))

    def test_projection_does_not_invent_contact_normal(self):
        model,data = compiled('<mujoco><worldbody><geom type="box" size=".1 .1 .1"/>'
                              '<geom type="cylinder" size=".1 .1" pos="0 0 .201"/></worldbody></mujoco>')
        result = CertificateQuery(model,data).measure(0,1)
        self.assertGreater(result['lower_bound_m'],0)
        self.assertIsNone(result['normal'])
        with self.assertRaises(MeasurementError):
            CertificateQuery(model,data,support=True).measure(0,1,.004)

    def test_penetration_lower_bound_is_not_complete_support_evidence(self):
        from scene_generation.checks import support_evidence
        from scene_generation.geometry import bounds
        model,data = compiled('<mujoco><worldbody><body name="a__body"><geom name="a__g" type="sphere" size=".1"/></body>'
                              '<body name="b__body"><geom name="b__g" type="sphere" size=".1" pos="0 0 .19"/></body></worldbody></mujoco>')
        query = SimpleNamespace(measure=lambda *args:{'distance_m':None,'lower_bound_m':None,
                            'normal':None,'penetration_lower_bound_m':.001})
        result = support_evidence(model,data,{'id':'b','support_instance':'a'},.003,.0005,
                                  {g:bounds(model,data,g) for g in range(model.ngeom)},query)
        self.assertFalse(result['complete'])
        self.assertFalse(result['supported'])
        self.assertEqual(result['measurement_error_count'],1)


class PipelineDispatchTests(unittest.TestCase):
    setUp = pipeline_tests.ScenePipelineTests.setUp
    run_loop = pipeline_tests.ScenePipelineTests.run_loop

    def backend(self):
        local = LocalBackend(self.generation,self.project)
        class RealSupplement(RecoveringBackend):
            def recover_scene(self,*args):
                return local.recover_scene(*args)
        return RealSupplement('partial')

    def test_pipeline_consumes_second_capability_before_visual_review(self):
        backend = self.backend()
        result,transport = self.run_loop(self.base_replies+[REVIEW],sandbox=backend)
        self.assertEqual(result['status'],'candidate_ready')
        self.assertEqual(backend.rechecks,1)
        self.assertEqual(backend.build_calls,2)
        actions = result['check_recovery'][0]['actions']
        self.assertEqual([a['capability'] for a in actions],['full_recheck_v2','collision_certificates_v1'])
        self.assertTrue(actions[-1]['effective'])
        self.assertEqual(transport.requests[-1]['messages'][0]['role'],'system')

    def test_pipeline_localizes_without_expanding_editable_dependencies(self):
        backend = self.backend()
        original = backend.assemble_and_check
        def failure(*args):
            report,renders = original(*args)
            if backend.scene_calls==1:
                report.update(status='fail',checks=[{'id':'fixture_collision','status':'fail','instances':[],
                    'geom1':'station__base_geom','geom2':'environment__table_top','distance_m':-.001}])
            else:
                report.update(status='pass',checks=[{'id':'fixture_collision','status':'pass'}])
            return report,renders
        backend.assemble_and_check = failure
        repair = {'jobs':[],'placements':[{'id':'station','position':[.6,0,.8],'yaw':0}],'reason':'合成布局修复'}
        result,transport = self.run_loop(self.base_replies+[repair,REVIEW,REVIEW],sandbox=backend)
        self.assertEqual(result['status'],'candidate_ready')
        context = json.loads(transport.requests[5]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['allowed_jobs'],['instrument'])
        self.assertEqual(context['allowed_instances'],['station'])
        self.assertEqual(result['check_recovery'][0]['actions'][0]['capability'],'locate_defects_v1')

    def test_needs_review_does_not_become_geometry_repair(self):
        from scene_generation.pipeline import SceneRun
        with patch('scene_generation.pipeline.decide',return_value='needs_review'), \
             patch.object(SceneRun,'route_repair',side_effect=AssertionError('must not repair')):
            result,transport = self.run_loop(self.base_replies)
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(len(transport.requests),5)

    def test_second_capability_interrupt_keeps_scene_and_model_ledger(self):
        backend = self.backend()
        with patch.object(backend,'recover_scene',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.base_replies,sandbox=backend)
        state = json.loads((self.root/'run/state.json').read_text(encoding='utf-8'))
        old_calls = copy.deepcopy(state['calls'])
        with patch.object(backend,'assemble_and_check',side_effect=AssertionError('must reuse scene')):
            result,transport = self.run_loop([REVIEW],resume=True,sandbox=backend)
        state = json.loads((self.root/'run/state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['calls'][:len(old_calls)],old_calls)
        self.assertEqual(len(transport.requests),1)
        self.assertEqual(backend.rechecks,1)
        self.assertEqual(backend.scene_calls,1)
        self.assertEqual(result['status'],'candidate_ready')

    def test_actual_ineffective_repair_cannot_repeat_by_changing_reason(self):
        from scene_fixtures import SceneSandbox
        backend = SceneSandbox(fail_scene_once=True)
        real = backend.assemble_and_check
        def two_failures(*args):
            report,renders = real(*args)
            if backend.scene_calls<=2:
                report['status']='fail'; report['checks'][0]['status']='fail'
            return report,renders
        backend.assemble_and_check = two_failures
        repair = {'jobs':['instrument'],'placements':[],'reason':'first'}
        repeated = {**repair,'reason':'same proposal with different prose'}
        different = {**repair,'geometry_changes':[{'job':'instrument','operation':'resize',
                                                  'parameters':{'part':'base','width':.2}}]}
        p1 = {'edits':[{'old':'    pass','new':'    return None'}]}
        p2 = {'edits':[{'old':'    return None','new':'    return 0'}]}
        replies = self.base_replies+[repair,p1,REVIEW,REVIEW,repeated,different,p2,REVIEW,REVIEW,REVIEW]
        result,transport = self.run_loop(replies,sandbox=backend)
        self.assertEqual(result['status'],'candidate_ready')
        state = json.loads((self.root/'run/state.json').read_text(encoding='utf-8'))
        history = state['scene']['repair_history']
        self.assertEqual(len(history),2)
        self.assertEqual(history[0]['status'],'evaluated')
        self.assertEqual(history[0]['outcome'],'ineffective')
        self.assertEqual(history[1]['outcome'],'changed')
        attempts=state['scene']['repair_planning']['attempts']
        self.assertEqual(len(attempts),2)
        self.assertIn('This repair was already executed',attempts[0]['error'])
        self.assertEqual(attempts[1]['status'],'accepted')
        self.assertEqual(attempts[1]['response'],different)
