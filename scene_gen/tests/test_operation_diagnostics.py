"""开发者合成反例；验证证据机制，不代表模型自主诊断能力。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from asset_prep.storage import atomic_json, digest
from robot_operation.common import Settings, OperationError, identity, read
from robot_operation.scene import prepare
from robot_operation.workflow import run
from robot_operation.diagnostic_tools import DiagnosticContext, normalize
from robot_operation.diagnostic_ledger import EvidenceLedger
from robot_operation.diagnostic_service import collect, merge, repair_ticket, check_obligations
from operation_fixtures import fixture, task_graph


class LedgerTests(unittest.TestCase):
    def test_equivalent_queries_reuse_across_resume_and_wording(self):
        state={}; ledger=EvidenceLedger(state,Settings())
        measure=Mock(return_value={"status":"valid","observations":{"x":1},"coverage":{"region":"a"}})
        a=normalize({"kind":"fit","interface":"i","body":"b","distance":1,"question":"为什么卡住"})
        b=normalize({"kind":"fit","interface":"i","body":"b","distance":1.0,"samples":9,"hypotheses":["另一种措辞"]})
        first=ledger.acquire(a,{"scene_sha256":"v1"},measure)
        second=EvidenceLedger(state,Settings()).acquire(b,{"scene_sha256":"v1"},measure)
        self.assertEqual(first['id'],second['id']);self.assertEqual(second['delivery'],'reused');measure.assert_called_once()

    def test_provable_sample_coverage_reuses_but_not_different_path(self):
        ledger=EvidenceLedger({},Settings());measure=Mock(return_value={"status":"valid","observations":{},"coverage":{"samples":17}})
        a=normalize({"kind":"fit","interface":"i","body":"b","distance":.1,"samples":17})
        ledger.acquire(a,{},measure)
        self.assertEqual(ledger.acquire({**a,"samples":9},{},measure)['delivery'],'covered_by_existing')
        self.assertEqual(ledger.acquire({**a,"distance":.2},{},measure)['delivery'],'measured')
        self.assertEqual(measure.call_count,2)

    def test_scene_state_and_tool_version_invalidate_reuse(self):
        ledger=EvidenceLedger({},Settings());measure=Mock(return_value={"status":"valid","observations":{},"coverage":{}})
        for binding in ({"scene_sha256":"a","state_sha256":"x","tool_version":1},
                        {"scene_sha256":"b","state_sha256":"x","tool_version":1},
                        {"scene_sha256":"b","state_sha256":"y","tool_version":1},
                        {"scene_sha256":"b","state_sha256":"y","tool_version":2}):
            ledger.acquire({"kind":"contacts","bodies":[]},binding,measure)
        self.assertEqual(measure.call_count,4)

    def test_invalid_measurement_cached_not_promoted_to_defect(self):
        state={};ledger=EvidenceLedger(state,Settings());measure=Mock(side_effect=OperationError('missing_evidence','no axis'))
        a=ledger.acquire({'kind':'fit'},{'scene_sha256':'s'},measure)
        b=ledger.acquire({'kind':'fit'},{'scene_sha256':'s'},measure)
        self.assertEqual(a['status'],'invalid');self.assertFalse(a['information_gain']);measure.assert_called_once()
        report=merge({'scene_sha256':'s','rows':[]},[a,b]);self.assertEqual(report['rows'],[])

    def test_no_gain_is_bounded_and_persisted(self):
        state={};settings=replace(Settings(),diagnostic_stalls=2);ledger=EvidenceLedger(state,settings)
        measure=lambda:{'status':'valid','observations':{'distance':1},'coverage':{'same_surface':True}}
        first=ledger.acquire({'a':1},{},measure);ledger.finish_batch({},[first])
        second=ledger.acquire({'a':2},{},measure);self.assertFalse(second['information_gain'])
        self.assertFalse(ledger.finish_batch({},[second])['exhausted'])
        self.assertTrue(EvidenceLedger(state,settings).finish_batch({},[second])['exhausted'])

    def test_interrupted_probe_retains_reservation_and_is_not_retried(self):
        state={};ledger=EvidenceLedger(state,Settings())
        with self.assertRaises(KeyboardInterrupt):ledger.acquire({'kind':'probe'},{},lambda:(_ for _ in ()).throw(KeyboardInterrupt()),sim_seconds=2)
        measure=Mock();row=EvidenceLedger(state,Settings()).acquire({'kind':'probe'},{},measure,sim_seconds=2)
        self.assertEqual(row['status'],'interrupted');measure.assert_not_called();self.assertEqual(state['sim_seconds_reserved'],2)

    def test_unknown_tool_and_unsafe_probe_are_rejected(self):
        for q in ({'kind':'invented_sensor'},{'kind':'probe','node':'n','parameter':'allowed_contacts','value':[]},
                  {'kind':'contacts','execute':'arbitrary code'}):
            with self.assertRaises(OperationError):normalize(q)


class MeasurementTests(unittest.TestCase):
    def setup_scene(self,root):
        source=fixture(root/'source')
        settings=replace(Settings(),settle_seconds=.1,local_retries=0)
        result=prepare(source,root/'dynamic',identity(source),settings)
        return root/'dynamic',result['scene_sha256'],settings

    def test_local_geometry_reuses_only_unchanged_dependencies_across_scenes(self):
        import shutil
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,sha,settings=self.setup_scene(root);state={};query={'kind':'geometry','geom':'object__geom'}
            first=collect(DiagnosticContext(scene,sha,settings,evidence_root=root/'diagnostics'),state,[query])[0][0]
            self.assertIn('local_geometry_sha256',first['binding'])
            shutil.copytree(root/'source',root/'source2')
            xml=ET.parse(root/'source2/model.xml');xml.find('.//light').set('diffuse','.2 .3 .4');xml.write(root/'source2/model.xml')
            version2=prepare(root/'source2',root/'dynamic2',identity(root/'source2'),settings)
            ctx=DiagnosticContext(root/'dynamic2',version2['scene_sha256'],settings,evidence_root=root/'diagnostics')
            second=collect(ctx,state,[query])[0][0]
            self.assertEqual(second['delivery'],'local_dependency_reused')
            self.assertEqual(second['rebound_from'],first['id']);self.assertNotEqual(second['binding']['scene_sha256'],sha)
            detail=collect(ctx,state,[{'kind':'evidence_detail','evidence_id':second['id'],'path':['observations','geom']}])[0][0]
            self.assertEqual(detail['result']['observations']['data'],'object__geom')
            shutil.copytree(root/'source2',root/'source3')
            xml=ET.parse(root/'source3/model.xml');xml.find('.//geom[@name="object__geom"]').set('friction','.8 .01 .001');xml.write(root/'source3/model.xml')
            version3=prepare(root/'source3',root/'dynamic3',identity(root/'source3'),settings)
            third=collect(DiagnosticContext(root/'dynamic3',version3['scene_sha256'],settings,evidence_root=root/'diagnostics'),state,[query])[0][0]
            self.assertEqual(third['delivery'],'measured');self.assertNotEqual(third['binding']['local_geometry_sha256'],first['binding']['local_geometry_sha256'])
            self.assertEqual(identity(scene),sha)

    def test_full_evidence_is_archived_paged_and_integrity_checked_on_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,sha,settings=self.setup_scene(root);state={};evidence_root=root/'diagnostics'
            ledger=EvidenceLedger(state,settings,archive_root=evidence_root/'evidence')
            query=normalize({'kind':'trace','window':10})
            result={'status':'valid','observations':{'window':[{'step':i} for i in range(21)]},'rows':[]}
            record=ledger.acquire(query,{'scene_sha256':sha},lambda:result)
            self.assertEqual(len(record['result']['observations']['window']),6)
            self.assertEqual(len(read(record['result_reference']['path'])['observations']['window']),21)
            ctx=DiagnosticContext(scene,sha,settings,evidence_root=evidence_root)
            page=collect(ctx,state,[{'kind':'evidence_detail','evidence_id':record['id'],'path':['observations','window'],'offset':8,'limit':2}])[0][0]
            self.assertEqual(page['result']['observations']['data'],[{'step':8},{'step':9}])
            self.assertEqual(page['result']['observations']['next_offset'],10)
            Path(record['result_reference']['path']).write_text('{}',encoding='utf-8')
            with self.assertRaises(OperationError) as caught:ledger.acquire(query,{'scene_sha256':sha},lambda:result)
            self.assertEqual(caught.exception.code,'stale_evidence')

    def test_generic_geometry_pose_and_joints_are_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,sha,settings=self.setup_scene(Path(tmp));ctx=DiagnosticContext(scene,sha,settings)
            records,_=collect(ctx,{},[
                {'kind':'geometry','geom':'object__geom'},
                {'kind':'relative_pose','site_a':'object__site','site_b':'robot__left_tcp','axis':[0,0,1]},
                {'kind':'joint_state','joints':['robot__left_x']},
                {'kind':'distance','geoms':['object__geom','robot__chassis'],'limit':.5},
                {'kind':'contacts','bodies':['object__body']}])
            self.assertTrue(all(r['status']=='valid' for r in records),records)
            self.assertIn('lateral_m',records[1]['result']['observations'])
            self.assertEqual(identity(scene),sha)

    def test_cached_measurement_rejects_scene_changed_after_context_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,sha,settings=self.setup_scene(Path(tmp));ctx=DiagnosticContext(scene,sha,settings);ledger={}
            query={'kind':'geometry','geom':'object__geom'}
            collect(ctx,ledger,[query])
            path=scene/'model.xml';path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaises(OperationError) as caught:collect(ctx,ledger,[query])
            self.assertEqual(caught.exception.code,'stale_scene');self.assertEqual(len(ledger['records']),1)

    def test_fixture_repair_changes_geometry_preserves_task_and_replays(self):
        import shutil
        from robot_operation.diagnostic_service import inspect_with_ledger
        from robot_operation.recording import replay
        from robot_operation.joint_state import compare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=fixture(root/'source')
            xml=(source/'model.xml').read_text(encoding='utf-8')
            post='<body name="holder__body" pos=".3 .2 0"><geom name="holder__post" type="box" pos="0 0 .25" size=".025 .06 .25"/></body>'
            (source/'model.xml').write_text(xml.replace('</worldbody>',post+'</worldbody>'),encoding='utf-8')
            meta=read(source/'scene.json');meta['plan']['instances'].append({'id':'holder','job':'holder_job','placement':'floor'})
            meta['plan']['jobs']=[{'id':'holder_job','role':'fixture'},{'id':'asset','role':'asset'}];atomic_json(source/'scene.json',meta)
            settings=replace(Settings(),settle_seconds=.1,rrt_iterations=100,local_retries=0)
            original_sha=identity(source);prepared=prepare(source,root/'before',original_sha,settings);sha=prepared['scene_sha256']
            task,graph=task_graph(sha);goal={'kind':'joint_range','joint':'robot__left_x','range':[.32,.34]}
            task['stages'][0]['goal']=goal;task['final']=goal;graph['task_sha256']=digest(task)
            graph['nodes'][0].update(operation='joint_move',parameters={'arms':['left'],'positions':{'robot__left_x':.33}},completion=goal,timeout=3)
            failed=run(root/'before',sha,task,graph,root/'failed',settings,mode='trial',images=False)
            self.assertEqual(failed['status'],'failed',failed)
            query={'kind':'configuration','positions':{'robot__left_x':.33},'bodies':['holder__body']}
            ledger={};archive=root/'diagnostics'
            report=inspect_with_ledger(root/'before',sha,settings,ledger,run_root=root/'failed',evidence_root=archive,queries=[query])
            evidence=next(r for r in report['evidence'] if r['query']['kind']=='configuration')
            ticket=repair_ticket(report,{'reason':'开发者预置返修：降低夹具支柱高度，保持安装位置','instances':['holder'],
                'evidence_ids':[evidence['id']],'checks':[query]},'fixture',sha)
            ticket['baseline_checks']=[evidence]
            # 模拟开发者编写的资产作者响应；只在新产物修改夹具，原产物保持只读。
            shutil.copytree(source,root/'repaired_source')
            path=root/'repaired_source/model.xml'
            path.write_text(path.read_text(encoding='utf-8').replace('pos="0 0 .25" size=".025 .06 .25"','pos="0 0 .17" size=".025 .06 .17"'),encoding='utf-8')
            candidate=prepare(root/'repaired_source',root/'after',identity(root/'repaired_source'),settings);new_sha=candidate['scene_sha256']
            new_graph={**deepcopy(graph),'scene_sha256':new_sha}
            completed=run(root/'after',new_sha,task,new_graph,root/'completed',settings,mode='final',images=False)
            self.assertEqual(completed['status'],'success',completed)
            new_report=inspect_with_ledger(root/'after',new_sha,settings,ledger,run_root=root/'completed',evidence_root=archive)
            records,_=collect(DiagnosticContext(root/'after',new_sha,settings,evidence_root=archive),ledger,ticket['checks'],count_stall=False)
            new_report=merge(new_report,records);verification=check_obligations(new_report,ticket,records)
            self.assertTrue(verification['original_problem_resolved'],verification)
            self.assertFalse(verification['regressions']);self.assertEqual(records[0]['delivery'],'measured')
            new_report['repair_verification']=verification
            self.assertEqual(compare({'diagnosis':report,'result':failed},{'diagnosis':new_report,'result':completed}),'improved')
            self.assertEqual(replay(root/'completed')['max_state_error'],0)
            self.assertEqual(identity(source),original_sha)
            self.assertEqual(digest(read(root/'failed/task.json')),digest(read(root/'completed/task.json')))

    def test_missing_failure_state_is_invalid_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,sha,settings=self.setup_scene(Path(tmp));ctx=DiagnosticContext(scene,sha,settings);state={}
            q={'kind':'contacts','state':'failure'}
            first,_=collect(ctx,state,[q]);second,_=collect(ctx,state,[q])
            self.assertEqual(first[0]['status'],'invalid');self.assertEqual(second[0]['delivery'],'reused')

    def test_trace_reuses_real_recording_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,sha,settings=self.setup_scene(root);task,graph=task_graph(sha)
            run(scene,sha,task,graph,root/'run',settings,mode='trial',images=False)
            ctx=DiagnosticContext(scene,sha,settings,run_root=root/'run')
            condition=collect(ctx,{},[{'kind':'condition','node':'one','condition':'precondition','state':'node_start'}])[0][0]
            self.assertEqual(condition['status'],'valid',condition)
            self.assertTrue(condition['result']['observations']['passed'])
            record=collect(ctx,{},[{'kind':'trace','node':'one','bodies':['object__body']}])[0][0]
            self.assertEqual(record['status'],'valid');self.assertGreater(record['result']['observations']['steps'],0)
            self.assertFalse(any(r['repair_authorized'] for r in record['result']['rows']))
            (root/'run/node_start_state.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(OperationError):DiagnosticContext(scene,sha,settings,run_root=root/'run')
            with self.assertRaises(OperationError):collect(ctx,{},[{'kind':'contacts'}])

    def test_paired_actuator_probe_and_no_repeated_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,sha,settings=self.setup_scene(root);task,graph=task_graph(sha)
            node=graph['nodes'][0];node.update(operation='joint_move',parameters={'arms':['left'],'positions':{'robot__left_x':.15}},timeout=2)
            result=run(scene,sha,task,graph,root/'run',settings,mode='trial',images=False)
            self.assertEqual(result['status'],'failed',result)
            ctx=DiagnosticContext(scene,sha,settings,run_root=root/'run',evidence_root=root/'probes');state={}
            query={'kind':'probe','node':'one','parameter':'positions','value':{'robot__left_x':0},'seconds':2}
            records,_=collect(ctx,state,[query]);record=records[0]
            self.assertEqual(record['status'],'valid',record)
            self.assertTrue(record['result']['observations']['variant_improved'],record)
            before=len(list((root/'probes').rglob('steps.jsonl.gz')))
            again,_=collect(ctx,state,[query]);self.assertEqual(again[0]['delivery'],'reused')
            self.assertEqual(before,2);self.assertEqual(len(list((root/'probes').rglob('steps.jsonl.gz'))),2)
            self.assertEqual(identity(scene),sha)

    def test_unknown_check_does_not_certify_repair(self):
        before={'query':{'kind':'mechanism'},'result':{'rows':[]}}
        ticket={'scene_sha256':'old','checks':[before['query']],'baseline_checks':[before]}
        after={**before,'id':'e','status':'unknown'}
        report=check_obligations({'scene_sha256':'new'},ticket,[after])
        self.assertFalse(report['coverage_complete'])

    def test_unsupported_measurement_is_cached_as_capability_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,sha,settings=self.setup_scene(Path(tmp));ctx=DiagnosticContext(scene,sha,settings);state={}
            query={'kind':'unavailable_deformation_sensor','body':'object__body'}
            a,_=collect(ctx,state,[query]);b,_=collect(ctx,state,[{**query,'question':'换一种问法'}])
            self.assertEqual(a[0]['status'],'invalid');self.assertEqual(a[0]['result']['code'],'capability_missing')
            self.assertEqual(b[0]['delivery'],'reused')

    def test_standalone_query_resume_is_readonly_and_reuses_ledger(self):
        from robot_operation.diagnostic_service import query_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,sha,settings=self.setup_scene(root);task,graph=task_graph(sha)
            run(scene,sha,task,graph,root/'run',settings,mode='trial',images=False)
            before=identity(root/'run');query=[{'kind':'contacts'}]
            first=query_run(root/'run',root/'analysis',query)
            count=len(read(root/'analysis/ledger.json')['records'])
            second=query_run(root/'run',root/'analysis',query,resume=True)
            self.assertEqual(first['status'],'diagnosed');self.assertEqual(second['status'],'diagnosed')
            self.assertEqual(len(read(root/'analysis/ledger.json')['records']),count)
            self.assertEqual(identity(root/'run'),before)
            with self.assertRaises(OperationError):query_run(root/'run',root/'run/child',query)

    def test_no_trace_matches_is_unknown_not_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,sha,settings=self.setup_scene(root);task,graph=task_graph(sha)
            second=deepcopy(graph['nodes'][0]);second['id']='later';graph['nodes'].append(second)
            graph['nodes'][0].update(operation='check',parameters={},on_success='later')
            run(scene,sha,task,graph,root/'run',settings,mode='trial',images=False)
            ctx=DiagnosticContext(scene,sha,settings,run_root=root/'run')
            record=collect(ctx,{},[{'kind':'trace','node':'one'}])[0][0]
            self.assertEqual(record['status'],'unknown')


class CoordinatorTests(unittest.TestCase):
    def test_failure_probe_revision_and_continuous_final_execution(self):
        from test_joint_upgrade import JointTests
        from robot_operation.recording import replay
        with tempfile.TemporaryDirectory() as tmp:
            helper=JointTests();root=Path(tmp);source,taskroot,profile=helper.inputs(root)
            coordinator=helper.create(root,source,taskroot,profile);coordinator.advance()
            sha=coordinator.state['current']['scene_sha256'];task,graph=task_graph(sha)
            task.update(input_sha256=coordinator.bundle['input_sha256'],public_profile_sha256=coordinator.bundle['public_profile_sha256'])
            helper.reply(coordinator,{'task':task,'unverified':[]});coordinator.advance()
            graph['task_sha256']=digest(task)
            graph['nodes'][0].update(operation='joint_move',parameters={'arms':['left'],'positions':{'robot__left_x':.15}},timeout=2)
            helper.reply(coordinator,{'graph':graph});coordinator.advance()
            self.assertEqual(coordinator.state['phase'],'route');self.assertEqual(coordinator.state['last_result']['status'],'failed')
            helper.reply(coordinator,{'diagnose':{'queries':[{'kind':'probe','node':'one','parameter':'positions','value':{'robot__left_x':0},'seconds':2}]}})
            coordinator.advance()
            self.assertTrue(any(c['target']=='trajectory' for c in coordinator.state['diagnosis']['attribution']['candidates']))
            helper.reply(coordinator,{'operation':{'reason':'开发者样例：局部对照支持修正关节目标'}});coordinator.advance()
            graph['nodes'][0]['parameters']['positions']['robot__left_x']=0
            helper.reply(coordinator,{'graph':graph});result=coordinator.advance()
            self.assertEqual(result['status'],'verified_task',result)
            self.assertEqual(replay(coordinator.root/'final/1')['max_state_error'],0)
            self.assertNotIn('checkpoint_restore',(coordinator.root/'final/1/events.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(result['consumption']['model_calls'],0)
            self.assertEqual(identity(coordinator.state['current']['scene']),sha)

    def test_offline_route_reuses_evidence_and_bounds_stalled_requests(self):
        from test_joint_upgrade import JointTests
        with tempfile.TemporaryDirectory() as tmp:
            helper=JointTests();root=Path(tmp);source,taskroot,profile=helper.inputs(root)
            coordinator=helper.create(root,source,taskroot,profile)
            coordinator.advance();task,graph=task_graph(coordinator.state['current']['scene_sha256'])
            task.update(input_sha256=coordinator.bundle['input_sha256'],public_profile_sha256=coordinator.bundle['public_profile_sha256'])
            helper.reply(coordinator,{'task':task,'unverified':[]});coordinator.advance()
            graph['task_sha256']=digest(task);helper.reply(coordinator,{'graph':graph});coordinator.advance()
            coordinator.state.update(phase='route',status='running')
            coordinator.ask=lambda *args:({'diagnose':{'queries':[{'kind':'contacts'}]}},{})
            before=len(coordinator.state['diagnostic_ledger']['records'])
            for _ in range(coordinator.settings.diagnostic_stalls):coordinator.route()
            self.assertEqual(len(coordinator.state['diagnostic_ledger']['records']),before)
            self.assertEqual(coordinator.state['phase'],'deferred')
            self.assertEqual(coordinator.store.totals()['model_calls'],0)


class RoutingTests(unittest.TestCase):
    def report(self):
        return {'scene_sha256':'s','failure':{'scope':'operation'},'instance_roles':{'cup':'asset','holder':'fixture'},
            'evidence':[{'id':'e','status':'valid','binding':{'scene_sha256':'s'}}],
            'rows':[{'kind':'trace_contact','evidence_id':'e','instances':['cup','holder'],'repair_authorized':False,
                     'observation':{'limit_exceeded':True}}]}
    def request(self):
        return {'instances':['holder'],'evidence_ids':['e'],'reason':'夹具与入口干涉',
                'checks':[{'kind':'contacts','bodies':['holder__body']}]}
    def test_same_observation_allows_fixture_trial_but_not_instrument_redesign(self):
        report=self.report();request=self.request()
        self.assertEqual(repair_ticket(report,request,'fixture','s')['scope'],'fixture_design_trial')
        request['instances']=['cup']
        with self.assertRaises(OperationError):repair_ticket(report,request,'asset','s')
    def test_normal_contact_stale_wrong_object_and_engineering_not_repair_evidence(self):
        for mutate in (lambda r:r['rows'][0]['observation'].update(limit_exceeded=False),
                       lambda r:r['evidence'][0]['binding'].update(scene_sha256='old'),
                       lambda r:r['rows'][0].update(instances=['cup']),
                       lambda r:r['failure'].update(scope='engineering')):
            report=self.report();mutate(report)
            with self.assertRaises(OperationError):repair_ticket(report,self.request(),'fixture','s')
    def test_new_interference_blocks_candidate_adoption(self):
        from robot_operation.joint_state import compare
        before={'id':'old','status':'valid','query':{'kind':'contacts'},'result':{'rows':[]}}
        after={**deepcopy(before),'id':'new','result':{'rows':[{'kind':'collision','geoms':['a','b']}]}}
        ticket={'scene_sha256':'old','checks':[before['query']],'baseline_checks':[before]}
        verification=check_obligations({'scene_sha256':'new'},ticket,[after])
        self.assertTrue(verification['regressions'])
        baseline={'diagnosis':{'rows':[],'coverage_gaps':[]},'result':{'status':'failed','completed_stages':[]}}
        candidate=deepcopy(baseline);candidate['result']['status']='success';candidate['diagnosis']['repair_verification']=verification
        self.assertEqual(compare(baseline,candidate),'exploration')

    def test_original_problem_remaining_is_not_reported_resolved(self):
        before={'id':'old','status':'valid','query':{'kind':'contacts'},'result':{'rows':[{'kind':'collision','geoms':['a','b']}]}}
        ticket={'scene_sha256':'old','checks':[before['query']],'baseline_checks':[before],'original_pairs':[['a','b']]}
        after={**deepcopy(before),'id':'new'}
        verification=check_obligations({'scene_sha256':'new'},ticket,[after])
        self.assertTrue(verification['coverage_complete']);self.assertFalse(verification['original_problem_resolved'])

    def test_repair_checks_must_cover_selected_instance(self):
        request=self.request();request['checks']=[{'kind':'geometry','geom':'cup__geom'}]
        with self.assertRaises(OperationError):repair_ticket(self.report(),request,'fixture','s')

    def test_repair_cannot_replace_original_configuration_check_with_geometry_read(self):
        report=self.report();request=self.request()
        original=normalize({'kind':'configuration','positions':{'robot__left_x':.3},'bodies':['holder__body']})
        report['evidence'][0]['query']=original
        request['checks']=[{'kind':'geometry','geom':'holder__post'}]
        with self.assertRaises(OperationError):repair_ticket(report,request,'fixture','s')
        request['checks']=[original]
        self.assertEqual(repair_ticket(report,request,'fixture','s')['checks'],[original])

    def test_normal_fixture_contact_does_not_become_a_supported_cause(self):
        from robot_operation.diagnostic_service import attribution
        report=self.report();report['rows'][0]['observation']['limit_exceeded']=False
        self.assertEqual(attribution(report)['status'],'unresolved')
