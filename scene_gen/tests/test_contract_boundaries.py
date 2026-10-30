"""契约边界回归：合成数据及带来源摘要的真实收据，不调用模型或执行生成脚本。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asset_prep.errors import ValidationError
from asset_generation.check_report import (CheckContractError, validate_report, validate_rows,
    checked_report, report_status, incomplete, inspection_results)
from asset_generation.contracts import check_profile, visual_result, author_result
from asset_generation.evidence import requests_valid
from scene_generation.capabilities import merge, proposal, REGISTRY
from scene_generation.contracts import SceneConfig, validate_plan, repair_action, numerical_contract, environment
from scene_generation.feedback import planning_feedback
from scene_generation.review import scoped_review
from scene_generation.dispatch import consume
from scene_generation.faults import exception_failure, add_failure
from scene_generation.recovery import decide
import scene_fixtures
import test_recovery_dispatch as dispatch_tests


class ReportBoundaryTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).parent/'data/np001_static_recheck_contract.json').read_text(encoding='utf-8'))

    def test_real_np001_static_receipt_roundtrip_preserves_defects_and_gaps(self):
        data=self.fixture()
        validate_report(data['report'],domain='scene',mode='static')
        merged=merge(data['report'],data['result'],data['request'],data['input_key'])
        self.assertEqual(merged['status'],'fail')
        self.assertEqual(decide(merged),'repair')
        collision=next(r for r in merged['checks'] if r['id']=='scene_collisions')
        self.assertEqual(collision['collision_count'],5)
        self.assertEqual(len(collision['coverage_gaps']),3)
        self.assertEqual([r['id'] for r in merged['checks'] if r['status']=='skipped'],
                         ['scene_joint_sweeps','settling','robot_reachability'])

    def test_static_exclusions_are_mode_and_check_specific(self):
        for mode,name in [('full','settling'),('static','scene_collisions'),(None,'settling')]:
            with self.subTest(mode=mode,name=name),self.assertRaises(CheckContractError):
                validate_rows([{'id':name,'status':'skipped'}],mode=mode)

    def test_skip_cannot_erase_existing_defect(self):
        old={'validation_mode':'static','status':'fail','checks':[{'id':'settling','status':'fail'}]}
        with self.assertRaises(CheckContractError):
            merge(old,{'input_key':'v','scope':'checks','checks':[{'id':'settling','status':'skipped'}]},proposal(['settling']),'v')

    def test_incomplete_summaries_cannot_pass_without_expanded_diagnostics(self):
        for field in ({'complete':False},{'measurement_error_count':2},{'unexamined_pairs':1},
                      {'measurement_errors':[{}]},{'coverage_gaps':[{}]}):
            row={'id':'scene_collisions','status':'pass',**field}
            with self.subTest(field=field):
                self.assertEqual(report_status([row]),'partial')
                with self.assertRaises(CheckContractError):
                    validate_rows([row])

    def test_empty_and_contradictory_reports_are_tool_errors(self):
        for value in ({'status':'pass','checks':[]},{'status':'pass','checks':[{'id':'x','status':'fail'}]},
                      {'status':'pass'},{'status':'pass','checks':[{'id':'x','status':'unknown'}]},
                      {'status':'pass','checks':[{'id':'settling','status':'skipped'}]}):
            self.assertEqual(checked_report(value,domain='scene',mode='static')['error_kind'],'check_execution')

    def test_mixed_defect_and_program_error_prioritizes_tool_error(self):
        rows=[{'id':'a','status':'fail'},{'id':'b','status':'error'}]
        self.assertEqual(report_status(rows),'error')
        self.assertEqual(decide({'status':'error','checks':rows}),'needs_environment')

    def test_missing_previews_are_tool_errors_and_keep_numerical_findings(self):
        from asset_generation.check_report import checked_previews
        row={'id':'scene_collisions','status':'fail','collision_count':1}
        report={'status':'fail','checks':[row],'renders':['missing.png']}
        def missing(*args):
            raise FileNotFoundError('synthetic missing render')
        result,images=checked_previews(report,Path('.'),missing)
        self.assertEqual(result['checks'],[row])
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['error_kind'],'backend_execution')
        self.assertEqual(images,[])
        self.assertEqual(report['status'],'fail')
        with self.assertRaises(CheckContractError):
            validate_report({'status':'pass','checks':[{'id':'x','status':'pass'}],'renders':[]},
                            domain='scene',mode='static',require_renders=True)

    def test_manifest_references_require_names_not_numeric_indices(self):
        from generation_fixtures import XML, MANIFEST
        from test_distance_evidence import compiled
        from asset_generation.checks import validate_manifest
        model,_=compiled(XML)
        for ref in (1,True):
            value=copy.deepcopy(MANIFEST)
            value['parts'][0]['body']=ref
            with self.subTest(ref=ref),self.assertRaises(ValidationError):
                validate_manifest(value,model,['A01'])

    def test_duplicate_checks_rejected_at_profile_boundary(self):
        item={'id':'width','geoms':['body'],'axis':0,'expected_m':.1,'tolerance_m':.001}
        with self.assertRaises(ValidationError):
            check_profile({'dimensions':[item,item]})

    def test_missing_scope_rejected_even_for_empty_result(self):
        with self.assertRaises(CheckContractError):
            merge({'checks':[{'id':'a','status':'partial'}]}, {'input_key':'v','checks':[]},proposal(['a']),'v')

    def test_shared_contract_fault_cannot_be_cleared_by_other_user_of_contract(self):
        try:
            validate_rows([{'id':'x','status':'invalid'}])
        except CheckContractError as exc:
            failure=exception_failure(exc,'isolated_capability')
        self.assertEqual(failure['components'],['asset_generation.check_report'])
        report=add_failure({'checks':[{'id':'x','status':'partial'}]},failure,input_key='v',
                           action_key='old',attempt=1,capability='old',request=proposal(['x']))
        new=consume(report,{'input_key':'v','scope':'checks','checks':[{'id':'x','status':'pass'}]},
                    proposal(['x']),SimpleNamespace(id='different',dependencies=set()),'new','v')
        self.assertEqual(new['recovery_failures'][0]['status'],'open')

    def test_partial_pair_result_cannot_clear_unexamined_candidates(self):
        old={'checks':[{'id':'scene_collisions','status':'partial','complete':False,'unexamined_pairs':4,
                        'coverage_gaps':[{'geom1':'a','geom2':'b'}]}]}
        new=merge(old,{'input_key':'v','scope':'pairs','covered_pairs':[['a','b']],
                       'checks':[{'id':'scene_collisions','status':'pass'}]},proposal(['scene_collisions'],pairs=[['a','b']]),'v')
        self.assertEqual(new['status'],'partial')
        self.assertEqual(new['checks'][0]['unexamined_pairs'],4)

    def test_pair_merge_preserves_unrelated_errors_without_summary_count(self):
        old={'checks':[{'id':'scene_collisions','status':'partial','coverage_gaps':[{'geom1':'a','geom2':'b'}],
                       'measurement_errors':[{'geom1':'c','geom2':'d','error':'synthetic'}]}]}
        result={'input_key':'v','scope':'pairs','covered_pairs':[['a','b']],
                'checks':[{'id':'scene_collisions','status':'pass'}]}
        merged=merge(old,result,proposal(['scene_collisions'],pairs=[['a','b']]),'v')
        validate_rows(merged['checks'])
        self.assertEqual(merged['status'],'partial')
        self.assertEqual(len(merged['checks'][0]['measurement_errors']),1)

    def test_positive_collision_evidence_cannot_be_reported_as_pass(self):
        for status in ('pass','partial','skipped'):
            with self.subTest(status=status),self.assertRaises(CheckContractError):
                validate_rows([{'id':'settling','status':status,'collision_count':1}],mode='static')

    def test_inspection_requires_actual_matching_results(self):
        request={'kind':'support','instance':'rack'}
        for result in ({'status':'pass','results':[]},
                       {'status':'pass','results':[{'request':request,'complete':False}]},
                       {'status':'pass','results':[{'request':request},{'request':request}]},
                       {'status':'pass','results':[{'request':{'kind':'support','instance':'elsewhere'}}]}):
            with self.subTest(result=result),self.assertRaises(CheckContractError):
                inspection_results(result,[request])

    def test_planner_duplicate_failure_kinds_get_distinct_check_ids(self):
        item={'kind':'collision','instances':['station'],'detail':'synthetic','evidence':'synthetic'}
        report=planning_feedback({'scene_sha256':'v','failures':[item,item]},'v',scene_fixtures.plan())
        validate_report(report,domain='scene',mode='full')


def mutations(value, path=()):
    yield path
    if isinstance(value,dict):
        for key,item in value.items():
            yield from mutations(item,path+(key,))
    elif isinstance(value,list):
        for index,item in enumerate(value):
            yield from mutations(item,path+(index,))


class ModelContractShapeTests(unittest.TestCase):
    def test_model_and_input_validators_classify_malformed_shapes(self):
        plan=scene_fixtures.plan(); limits=SceneConfig()
        profile={'dimensions':[{'id':'width','geoms':['g'],'axis':0,'expected_m':.1,'tolerance_m':.001}],
                 'joints':[{'name':'hinge','type':'hinge','range':[0,1],'tolerance':.01}]}
        fixed={'fits':[{'id':'fit','outer':['a','b'],'inner':['c','d'],'axis':0,'clearance_m':[0,.1],'evidence':'source'}],
               'targets':[{'id':'reach','site':'a','tool_site':'b','offset':[0,0,0],'quaternion':[1,0,0,0],
                           'position_tolerance_m':.01,'angle_tolerance_rad':.01,'allowed_contacts':[['a','b']]}]}
        review={'verdict':'repair','findings':['defect'],'unverified':[],'affected_instances':['station']}
        action={'jobs':['instrument'],'placements':[{'id':'station','position':[.4,0,.8],'yaw':0}],'reason':'synthetic'}
        with tempfile.TemporaryDirectory() as root:
            env=scene_fixtures.environment(Path(root)/'robot')
        cases=[('plan',plan,lambda v:validate_plan(v,['A01'],limits)),
               ('profile',profile,check_profile),('fixed',fixed,numerical_contract),('environment',env,environment),
               ('review',review,lambda v:scoped_review(v,'scene',instance_ids={'station'})),
               ('visual',{k:review[k] for k in ('verdict','findings','unverified')},visual_result),
               ('repair',action,lambda v:repair_action(v,plan,['instrument'],['station'],limits)),
               ('inspection',[{'kind':'pairs','pairs':[['a','b']],'distance_m':.01}],lambda v:requests_valid(v,'scene'))]
        from scene_generation.evidence import evidence_plan
        from scene_generation.capabilities import validate_result, validate_request
        row={'id':'scene_collisions','status':'fail','collision_count':1,'complete':False,
             'collisions':[{'geom1':'a','geom2':'b','instances':['station']}],
             'coverage_gaps':[{'geom1':'c','geom2':'d'}]}
        cases.extend([
            ('report',{'status':'fail','checks':[row],'renders':[]},lambda v:validate_report(v,domain='scene',mode='static')),
            ('recovery_request',proposal(['scene_collisions'],pairs=[['c','d']]),validate_request),
            ('recovery_result',{'input_key':'v','scope':'checks','checks':[row]},
                 lambda v:validate_result(v,proposal(['scene_collisions']),'v','static')),
            ('inspection_result',{'status':'pass','results':[{'request':{'kind':'support','instance':'station'}}],'renders':[]},
                 lambda v:inspection_results(v,[{'kind':'support','instance':'station'}])),
            ('evidence_plan',{'requests':[{'kind':'support','instance':'station'}],
                              'blocked':{'kind':'measurement_error','detail':'synthetic'}},lambda v:evidence_plan(v,'scene')),
        ])
        from asset_generation.checks import validate_manifest
        from generation_fixtures import MANIFEST, XML
        from test_distance_evidence import compiled
        model,_=compiled(XML)
        cases.append(('manifest',MANIFEST,lambda v:validate_manifest(v,model,['A01'])))
        from scene_generation.table import validate_cuts
        from scene_generation.edge_mounts import validate_mounts
        fixture_plan=copy.deepcopy(plan); fixture_plan['jobs'][0]['role']='fixture'
        cases.extend([
            ('author',{'model_py':'def build(output_dir):\n    pass\n'},lambda v:author_result(v,None,10000)),
            ('patch',{'edits':[{'old':'pass','new':'return None'}]},lambda v:author_result(v,'def build(output_dir):\n    pass\n',10000)),
            ('table',[{'id':'cut','instance':'station','center':[.5,0],'size':[.02,.02],'depth':.01,'reason':'synthetic'}],lambda v:validate_cuts(v,plan)),
            ('edge',[{'instance':'station','geoms':['foot'],'edge':'x+','reason':'synthetic'}],lambda v:validate_mounts(v,fixture_plan)),
            ('feedback',{'scene_sha256':'v','failures':[{'kind':'collision','instances':['station'],'detail':'x','evidence':'x'}]},
                        lambda v:planning_feedback(v,'v',plan)),
        ])
        errors=[]; count=0
        for name,original,validator in cases:
            validator(copy.deepcopy(original))
            for path in mutations(original):
                for replacement in (None,[],{},True,0,'invalid'):
                    value=copy.deepcopy(original)
                    if not path:
                        value=replacement
                    else:
                        target=value
                        for key in path[:-1]: target=target[key]
                        target[path[-1]]=copy.deepcopy(replacement)
                    count+=1
                    try:
                        validator(value)
                    except ValidationError:
                        pass
                    except Exception as exc:
                        errors.append((name,path,type(replacement).__name__,type(exc).__name__))
        self.assertGreater(count,1000)
        self.assertEqual(errors,[],str(errors[:30]))


class StaticRecoveryProducerTests(unittest.TestCase):
    setUp=dispatch_tests.PipelineDispatchTests.setUp
    run_loop=dispatch_tests.PipelineDispatchTests.run_loop

    def test_actual_local_static_report_flows_through_dispatch_and_review(self):
        from asset_generation.local import LocalBackend
        from test_main_check_recovery import RecoveringBackend
        local=LocalBackend(self.generation,self.project)
        class Supplement(RecoveringBackend):
            def recheck_scene(self,*args):
                return local.recheck_scene(*args)
        result,_=self.run_loop(self.base_replies+[dispatch_tests.REVIEW],sandbox=Supplement('partial'),
                               limits=SceneConfig(validation_mode='static'))
        self.assertEqual(result['status'],'static_candidate_ready')
        actions=result['check_recovery'][0]['actions']
        self.assertEqual(len(actions),1)
        self.assertEqual(actions[0]['status'],'completed')

    def test_invalid_asset_report_does_not_trigger_model_geometry_repair(self):
        backend=scene_fixtures.SceneSandbox()
        real=backend.build_and_check
        def invalid(*args):
            report,renders=real(*args)
            report['checks']=[]
            return report,renders
        with patch.object(backend,'build_and_check',side_effect=invalid):
            result,transport=self.run_loop(self.base_replies[:2],sandbox=backend)
        self.assertEqual(result['status'],'needs_environment')
        self.assertEqual(len(transport.requests),2)


class WorkerContractTests(unittest.TestCase):
    def current_identity(self):
        from asset_prep import implementation
        return implementation.implementation_digest(implementation.source_manifest(
            Path(__file__).resolve().parents[1]))

    def test_worker_refuses_to_start_without_implementation_binding(self):
        from asset_generation.worker import main
        with tempfile.TemporaryDirectory() as root, \
             patch('sys.argv',['worker','validate','--input',str(root),'--output',str(root)]), \
             self.assertRaises(ValidationError):
            main()

    def test_worker_refuses_unbound_source_drift(self):
        from asset_generation.worker import main
        from asset_prep.implementation import ImplementationDrift
        with tempfile.TemporaryDirectory() as root, \
             patch('sys.argv',['worker','validate','--input',str(root),'--output',str(root),
                               '--implementation','0'*64]), self.assertRaises(ImplementationDrift):
            main()

    def test_asset_checker_bug_survives_worker_as_tool_error(self):
        from asset_generation.worker import main
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); source=root/'input'; output=root/'output'
            source.mkdir(); output.mkdir()
            (source/'job.json').write_text(json.dumps({'asset_ids':['A01'],'profile':{},
                                                      'implementation':self.current_identity()}),encoding='utf-8')
            with patch('sys.argv',['worker','validate','--input',str(source),'--output',str(output),
                                   '--implementation',self.current_identity()]), \
                 patch('asset_generation.checks.validate_artifact',side_effect=TypeError('checker implementation bug')):
                main()
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'error')
            self.assertEqual(report['execution_failure']['kind'],'program')
            self.assertEqual(report['checks'],[])

    def test_malformed_generated_xml_remains_asset_feedback(self):
        from asset_generation.checks import validate_artifact
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'model.xml').write_text('<mujoco><broken>',encoding='utf-8')
            report=validate_artifact(root,['A01'],check_profile({}))
        self.assertEqual(report['status'],'fail')
        self.assertEqual(report['checks'][0]['id'],'compile_and_interfaces')


class DispatchContractTests(unittest.TestCase):
    setUp=dispatch_tests.DispatchTests.setUp
    write_scene=dispatch_tests.DispatchTests.write_scene
    recover=dispatch_tests.DispatchTests.recover
    session=dispatch_tests.DispatchTests.session

    def test_invalid_proposal_is_persisted_without_repeated_crash(self):
        class Invalid:
            id='invalid_proposer'
            def propose(self,ctx,report):
                yield {'checks':[[]]}
        with patch.dict(REGISTRY,{Invalid.id:Invalid()},clear=True):
            first=self.recover()
            self.run.state=copy.deepcopy(self.saved)
            second=self.recover()
        self.assertEqual(decide(first),'needs_environment')
        self.assertEqual(decide(second),'needs_environment')
        self.assertEqual(len(second['recovery_failures']),1)
        self.assertIn(Invalid.id,self.session()['query_failures'])
