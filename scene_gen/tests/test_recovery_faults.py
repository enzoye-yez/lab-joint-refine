"""开发者合成故障注入，验证故障义务、覆盖证据和执行资格；不调用正式模型。"""

import copy
import errno
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_recovery_dispatch as fixtures
from scene_generation.capabilities import REGISTRY, proposal
from scene_generation.dispatch import consume
from scene_generation.faults import add_failure, exception_failure
from scene_generation.contracts import SceneConfig
from scene_generation.recovery import decide


def passing():
    return {'status':'pass','checks':[{'id':'scene_collisions','status':'pass','coverage_gaps':[]}]}


class FaultDispatchTests(unittest.TestCase):
    setUp = fixtures.DispatchTests.setUp
    write_scene = fixtures.DispatchTests.write_scene
    recover = fixtures.DispatchTests.recover
    session = fixtures.DispatchTests.session

    def only_full(self):
        return patch.dict(REGISTRY,{'full_recheck_v2':REGISTRY['full_recheck_v2']},clear=True)

    def test_program_failure_then_independent_real_capability_can_pass(self):
        self.write_scene('.3 0 0')
        with patch.object(self.backend,'recheck_scene',side_effect=TypeError('injected isolated implementation bug')) as first:
            result = self.recover()
        self.assertEqual(first.call_count,1)
        self.assertEqual(decide(result,{'verdict':'accept'}),'candidate_ready')
        fault = result['recovery_failures'][0]
        self.assertEqual(fault['status'],'resolved')
        self.assertEqual(fault['affected_checks'],['scene_collisions'])
        self.assertEqual(fault['resolutions'][0]['capability'],'collision_certificates_v1')
        self.assertEqual(fault['diagnostic']['error_type'],'TypeError')
        self.assertNotIn('error_kind',result)

    def test_independently_confirmed_defect_can_enter_repair_after_tool_error(self):
        with patch.object(self.backend,'recheck_scene',side_effect=TypeError('isolated')):
            result = self.recover()
        self.assertEqual(decide(result),'repair')
        self.assertEqual(result['recovery_failures'][0]['status'],'resolved')
        self.assertTrue(result['checks'][0]['collisions'])

    def test_partial_coverage_only_releases_matching_obligation(self):
        self.write_scene('.3 0 0')
        report = fixtures.unknown()
        report['checks'].append(fixtures.unknown('support:b')['checks'][0])
        selected = {k:REGISTRY[k] for k in ('full_recheck_v2','collision_certificates_v1')}
        with patch.dict(REGISTRY,selected,clear=True),patch.object(self.backend,'recheck_scene',side_effect=TypeError('isolated')):
            result = self.recover(report)
        self.assertEqual(result['recovery_failures'][0]['remaining_checks'],['support:b'])
        self.assertEqual(decide(result),'needs_environment')
        self.assertEqual(result['checks'][0]['status'],'pass')

    def test_shared_checker_bug_cannot_be_cleared_by_dependent_capability(self):
        self.write_scene('.3 0 0')
        failure = {'status':'error','execution_failure':{'kind':'program','retryable':False,
                   'components':['scene_generation.geometry']}}
        with patch.object(self.backend,'recheck_scene',return_value=failure):
            result = self.recover()
        self.assertEqual(result['checks'][0]['status'],'pass')
        self.assertEqual(result['recovery_failures'][0]['status'],'open')
        self.assertEqual(decide(result),'needs_environment')
        self.assertEqual(self.session()['exhaustion']['reason'],'no_executable_action')

    def test_unknown_failure_source_cannot_be_cleared(self):
        self.write_scene('.3 0 0')
        with patch.object(self.backend,'recheck_scene',return_value={'status':'error','detail':'no scope provenance'}):
            result = self.recover()
        self.assertEqual(result['recovery_failures'][0]['components'],['*'])
        self.assertEqual(decide(result),'needs_environment')

    def test_transient_failure_retries_and_does_not_replay_failed_receipt_as_success(self):
        with self.only_full(),patch.object(self.backend,'recheck_scene',side_effect=[TimeoutError('temporary'),passing()]) as call:
            result = self.recover()
        self.assertEqual(call.call_count,2)
        self.assertEqual(decide(result,{'verdict':'accept'}),'candidate_ready')
        action = self.session()['actions'][0]
        self.assertEqual(action['attempts'],2)
        self.assertEqual([e['status'] for e in action['executions']],['failed','completed'])
        self.assertEqual(action['executions'][1]['eligibility'],'temporary_retry')
        self.assertNotEqual(action['executions'][0]['receipt'],action['executions'][1]['receipt'])

    def test_unused_alternative_precedes_transient_retry(self):
        self.write_scene('.3 0 0')
        with patch.object(self.backend,'recheck_scene',side_effect=TimeoutError('temporary')) as call:
            result = self.recover()
        self.assertEqual(call.call_count,1)
        self.assertEqual(decide(result,{'verdict':'accept'}),'candidate_ready')
        self.assertEqual(len(self.session()['actions']),2)

    def test_retry_budget_survives_resume_and_explicit_budget_increase(self):
        with self.only_full(),patch.object(self.backend,'recheck_scene',side_effect=TimeoutError('temporary')) as call:
            first = self.recover()
            self.run.state = copy.deepcopy(self.saved)
            second = self.recover()
        self.assertEqual(call.call_count,3)
        self.assertEqual(decide(first),'needs_environment')
        self.assertEqual(decide(second),'needs_environment')
        self.assertEqual(self.session()['status'],'budget_exhausted')
        self.assertEqual(self.session()['exhaustion']['reason'],'recovery_retry_budget')
        self.run.limits = SceneConfig(validation_mode='static',max_check_retries=3)
        with self.only_full(),patch.object(self.backend,'recheck_scene',return_value=passing()) as call:
            result = self.recover()
        self.assertEqual(call.call_count,1)
        self.assertEqual(len(self.run.state['check_recovery']),1)
        self.assertEqual(self.session()['actions'][0]['attempts'],4)
        self.assertTrue(all(f['status']=='resolved' for f in result['recovery_failures']))

    def test_changed_observed_environment_allows_retry_without_resetting_total_budget(self):
        version = {'executable_available':False}
        self.backend.recovery_environment = lambda:copy.deepcopy(version)
        self.run.limits = SceneConfig(validation_mode='static',max_check_retries=0,max_check_actions=2)
        with self.only_full(),patch.object(self.backend,'recheck_scene',side_effect=FileNotFoundError('missing')) as call:
            self.recover(); self.recover()
        self.assertEqual(call.call_count,1)
        self.assertEqual(self.session()['exhaustion']['capabilities'][0]['excluded'][0]['reason'],'environment_change_required')
        version['executable_available']=True
        with self.only_full(),patch.object(self.backend,'recheck_scene',return_value=passing()):
            result = self.recover()
        action = self.session()['actions'][0]
        self.assertEqual(action['attempts'],2)
        self.assertEqual(action['executions'][1]['eligibility'],'observed_environment_change')
        self.assertEqual(result['status'],'pass')

    def test_program_bug_is_not_retried_on_environment_change_or_reason_text(self):
        version = {'executable_available':False}
        self.backend.recovery_environment = lambda:copy.deepcopy(version)
        with self.only_full(),patch.object(self.backend,'recheck_scene',side_effect=TypeError('temporary timeout please retry')) as call:
            self.recover()
            version['executable_available']=True
            self.recover()
        self.assertEqual(call.call_count,1)
        self.assertEqual(self.session()['exhaustion']['capabilities'][0]['excluded'][0]['reason'],'implementation_fix_required')

    def test_failure_receipt_replayed_after_state_commit_interruption_without_reexecution(self):
        save = self.run.store.save
        def interrupt():
            if any(a['status']=='failed' for h in self.run.state.get('check_recovery',{}).values() for a in h['actions']):
                raise KeyboardInterrupt()
            save()
        with self.only_full(),patch.object(self.backend,'recheck_scene',side_effect=[TimeoutError('temporary'),passing()]) as call:
            with patch.object(self.run.store,'save',side_effect=interrupt),self.assertRaises(KeyboardInterrupt):
                self.recover()
            self.run.state=copy.deepcopy(self.saved)
            result=self.recover()
        self.assertEqual(call.call_count,2)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(self.session()['actions'][0]['attempts'],2)

    def test_interrupted_retry_has_its_own_bound_not_total_attempt_number(self):
        with self.only_full(),patch.object(self.backend,'recheck_scene',side_effect=[TimeoutError('temporary'),KeyboardInterrupt(),passing()]) as call:
            with self.assertRaises(KeyboardInterrupt):
                self.recover()
            self.run.state=copy.deepcopy(self.saved)
            result=self.recover()
        self.assertEqual(call.call_count,3)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(self.session()['actions'][0]['attempts'],3)

    def test_unscoped_main_failure_needs_complete_scene_evidence(self):
        initial={'status':'error','error_kind':'backend_execution','checks':[],
                 'execution_failure':{'kind':'temporary','retryable':True,'components':['worker']}}
        with patch.object(self.backend,'recheck_scene',return_value=passing()):
            result=self.recover(initial)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['recovery_failures'][0]['resolutions'][0]['scope'],'scene')
        self.assertTrue(self.session()['actions'][0]['parameters']['full_scene'])

    def test_empty_full_scene_result_cannot_discharge_failure(self):
        initial={'status':'error','checks':[],'execution_failure':{'kind':'temporary','retryable':True}}
        with patch.object(self.backend,'recheck_scene',return_value={'status':'pass','checks':[]}):
            result=self.recover(initial)
        self.assertEqual(decide(result),'needs_environment')
        self.assertEqual(result['recovery_failures'][0]['status'],'open')


class FaultContractTests(unittest.TestCase):
    def test_classification_is_typed_and_walks_exception_cause(self):
        from asset_generation.sandbox import BackendExecutionError
        wrapped=BackendExecutionError('execution failed')
        wrapped.__cause__=OSError(errno.EBUSY,'busy')
        self.assertEqual(exception_failure(wrapped)['kind'],'temporary')
        wrapped.__cause__=TypeError('underlying checker bug')
        self.assertEqual(exception_failure(wrapped)['kind'],'program')
        self.assertEqual(exception_failure(ValueError('timeout'))['kind'],'program')
        self.assertEqual(exception_failure(OSError(errno.ENOSPC,'no space'))['kind'],'environment')

    def test_local_pairs_cannot_discharge_whole_check_or_scene_fault(self):
        cap=SimpleNamespace(id='independent',dependencies=set())
        for scope in ('checks','scene'):
            report=add_failure(fixtures.unknown(),{'kind':'temporary','retryable':True},input_key='v',
                               action_key='failed',attempt=1,capability='old',request=proposal(['scene_collisions']),scope=scope)
            result={'input_key':'v','scope':'pairs','covered_pairs':[['a__g','b__g']],
                    'checks':[{'id':'scene_collisions','status':'pass','coverage_gaps':[],'collisions':[]}]}
            merged=consume(report,result,proposal(['scene_collisions'],pairs=[['a__g','b__g']]),cap,'new','v')
            self.assertEqual(merged['recovery_failures'][0]['status'],'open')
            self.assertEqual(decide(merged),'needs_environment')

    def test_whole_scene_claim_cannot_omit_known_checks(self):
        report=fixtures.unknown()
        report['checks'].append({'id':'support:b','status':'partial'})
        request=proposal(['scene_collisions','support:b'],full_scene=True)
        report=add_failure(report,{'kind':'temporary','retryable':True},input_key='v',
                           action_key='failed',attempt=1,capability='old',request=request,scope='scene')
        result={'input_key':'v','scope':'scene','complete_scene':True,'checks':passing()['checks']}
        merged=consume(report,result,request,SimpleNamespace(id='new'), 'new','v')
        self.assertEqual(merged['recovery_failures'][0]['status'],'open')
        self.assertEqual(merged['checks'][1]['status'],'partial')

    def test_undeclared_dependencies_cannot_claim_program_failure_independence(self):
        report=add_failure(fixtures.unknown(),{'kind':'program','components':['original_checker']},input_key='v',
                           action_key='failed',attempt=1,capability='old',request=proposal(['scene_collisions']))
        result={'input_key':'v','scope':'checks','checks':passing()['checks']}
        updated=consume(report,result,proposal(['scene_collisions']),SimpleNamespace(id='undeclared'),'new','v')
        self.assertEqual(updated['recovery_failures'][0]['status'],'open')

    def test_pair_scoped_failure_requires_all_affected_pairs(self):
        cap=SimpleNamespace(id='independent',dependencies=set())
        report=fixtures.unknown()
        report['checks'][0]['coverage_gaps'].append({'geom1':'c','geom2':'d'})
        request=proposal(['scene_collisions'],pairs=[['a__g','b__g'],['c','d']])
        report=add_failure(report,{'kind':'temporary','retryable':True},input_key='v',action_key='failed',attempt=1,
                           capability='old',request=request,scope='pairs')
        for pair,status in ((['a__g','b__g'],'open'),(['c','d'],'resolved')):
            result={'input_key':'v','scope':'pairs','covered_pairs':[pair],
                    'checks':[{'id':'scene_collisions','status':'pass','coverage_gaps':[],'collisions':[]}]}
            report=consume(report,result,proposal(['scene_collisions'],pairs=[pair]),cap,'new','v')
            self.assertEqual(report['recovery_failures'][0]['status'],status)

    def test_unrelated_or_stale_evidence_never_discharges_fault(self):
        from asset_prep.errors import ValidationError
        cap=SimpleNamespace(id='independent',dependencies=set())
        report=fixtures.unknown()
        report['checks'].append({'id':'support:b','status':'partial'})
        report=add_failure(report,{'kind':'temporary','retryable':True},input_key='v',action_key='failed',attempt=1,
                           capability='old',request=proposal(['scene_collisions']))
        result={'input_key':'v','scope':'checks','checks':[{'id':'support:b','status':'pass'}]}
        merged=consume(report,result,proposal(['support:b']),cap,'new','v')
        self.assertEqual(merged['recovery_failures'][0]['status'],'open')
        result['input_key']='old'
        with self.assertRaises(ValidationError):
            consume(report,result,proposal(['support:b']),cap,'new','v')


class FaultPipelineTests(unittest.TestCase):
    setUp = fixtures.PipelineDispatchTests.setUp
    run_loop = fixtures.PipelineDispatchTests.run_loop
    backend = fixtures.PipelineDispatchTests.backend

    def test_failed_first_tool_then_real_worker_allows_visual_review(self):
        backend=self.backend()
        with patch.object(backend,'recheck_scene',side_effect=TypeError('isolated full-check implementation')):
            result,transport=self.run_loop(self.base_replies+[fixtures.REVIEW],sandbox=backend)
        self.assertEqual(result['status'],'candidate_ready')
        self.assertEqual(len(transport.requests),6)
        self.assertEqual(backend.scene_calls,1)
        self.assertEqual(backend.build_calls,2)
        actions=result['check_recovery'][0]['actions']
        self.assertEqual([a['status'] for a in actions],['failed','completed'])
        report=json.loads((self.root/'run'/result['scene_artifact']).parent.parent.joinpath('checks.json').read_text(encoding='utf-8'))
        self.assertEqual(report['recovery_failures'][0]['status'],'resolved')

    def test_retry_budget_pause_resume_preserves_artifact_and_model_ledger(self):
        backend=self.backend()
        limited=SceneConfig(max_check_retries=0)
        selected={'full_recheck_v2':REGISTRY['full_recheck_v2']}
        with patch.dict(REGISTRY,selected,clear=True),patch.object(backend,'recheck_scene',side_effect=TimeoutError('temporary')) as call:
            result,transport=self.run_loop(self.base_replies,limits=limited,sandbox=backend)
            again,transport=self.run_loop([],resume=True,limits=limited,sandbox=backend)
        self.assertEqual(call.call_count,1)
        self.assertEqual(result['status'],'check_recovery_budget_exhausted')
        self.assertEqual(again['status'],'check_recovery_budget_exhausted')
        self.assertEqual(transport.requests,[])
        old=json.loads((self.root/'run/state.json').read_text(encoding='utf-8'))['calls']
        artifact=self.root/'run'/result['scene_artifact']
        before=fixtures.snapshot(artifact)
        with patch.dict(REGISTRY,selected,clear=True),patch.object(backend,'recheck_scene',return_value=passing()), \
             patch.object(backend,'assemble_and_check',side_effect=AssertionError('must reuse scene')):
            result,transport=self.run_loop([fixtures.REVIEW],resume=True,limits=SceneConfig(max_check_retries=1),sandbox=backend)
        self.assertEqual(result['status'],'candidate_ready')
        self.assertEqual(result['check_recovery'][0]['actions'][0]['attempts'],2)
        self.assertEqual(fixtures.snapshot(artifact),before)
        new=json.loads((self.root/'run/state.json').read_text(encoding='utf-8'))['calls']
        self.assertEqual(new[:len(old)],old)
        self.assertEqual(len(transport.requests),1)

    def test_program_failure_stays_tool_failure_in_pipeline_report(self):
        backend=self.backend()
        selected={'full_recheck_v2':REGISTRY['full_recheck_v2']}
        with patch.dict(REGISTRY,selected,clear=True),patch.object(backend,'recheck_scene',side_effect=TypeError('bug')):
            result,transport=self.run_loop(self.base_replies,sandbox=backend)
        self.assertEqual(result['status'],'needs_environment')
        self.assertEqual(result['check_failure']['kind'],'check_execution')
        self.assertEqual(result['check_failure']['failures'][0]['status'],'open')
        self.assertEqual(len(transport.requests),5)

    def test_local_backend_preserves_typed_process_failure(self):
        from asset_generation.local import LocalBackend
        from asset_generation.sandbox import BackendExecutionError
        from scene_generation.contracts import numerical_contract
        backend=LocalBackend(self.generation,self.project)
        wrapped=BackendExecutionError('worker failed')
        wrapped.__cause__=subprocess.TimeoutExpired('synthetic-worker',1)
        with patch.object(LocalBackend,'_run',side_effect=wrapped):
            report=backend.recheck_scene(self.root/'unused',self.root/'supplement',SceneConfig(),numerical_contract({}))
        self.assertEqual(report['execution_failure']['kind'],'temporary')
        self.assertTrue(report['execution_failure']['retryable'])
        observed=backend.recovery_environment()
        self.assertTrue(observed['executable_available'])
        self.assertTrue(observed['worker_available'])
