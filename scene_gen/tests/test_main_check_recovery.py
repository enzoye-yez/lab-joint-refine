"""开发者合成故障注入；不请求模型服务，不修改正式实验产物。"""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from asset_generation.distance import DistanceQuery, MeasurementError, observe
from asset_generation.pipeline import snapshot
from scene_generation.geometry import collision_scan
from scene_generation.checks import upward_resultant
from scene_generation.recovery import decide, report_status
from scene_generation.edge_mounts import boundary_evidence
from scene_generation.contracts import SceneConfig
from test_distance_evidence import compiled
import test_scene_pipeline as pipeline_tests
from test_scene_pipeline import REVIEW
from scene_fixtures import SceneSandbox


def uncertain():
    return {'status':'partial','checks':[{'id':'scene_collisions','status':'partial','instances':[],
        'coverage_gaps':[{'geom1':'environment__table_top','geom2':'station__base_geom',
                          'instances':['environment','station'],'error_kind':'measurement_error'}]}],
        'coverage':[],'renders':[]}


class RecoveringBackend(SceneSandbox):
    def __init__(self, outcome='fail'):
        super().__init__()
        self.rechecks=0
        self.outcome=outcome
        self.sources=[]

    def assemble_and_check(self,*args):
        report,renders=super().assemble_and_check(*args)
        if self.scene_calls == 1 or self.outcome=='partial':
            report={**uncertain(),'renders':report['renders']}
        return report,renders

    def recheck_scene(self,artifact,destination,limits,contract):
        self.rechecks+=1
        before=snapshot(artifact)
        self.sources.append((artifact,before))
        if self.outcome=='partial':
            return uncertain()
        # 对合成场景进行真实几何查询，先确认原部件的接触，再注入确定的修复需求。
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(artifact/'model.xml'));d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        query=DistanceQuery(m,d,recovery=True)
        measured=query.measure(m.geom('environment__table_top').id,m.geom('station__base_geom').id)
        self.last_measurement=measured
        assert snapshot(artifact)==before
        return {'status':'fail','checks':[{'id':'scene_collisions','status':'fail',
                'instances':['station'],'detail':'开发者注入确定的安装位置缺陷','distance_evidence':measured}],
                'coverage':[]}


class MainCheckRecoveryTests(unittest.TestCase):
    setUp=pipeline_tests.ScenePipelineTests.setUp
    run_loop=pipeline_tests.ScenePipelineTests.run_loop

    def test_unknown_recheck_repair_and_recheck_reaches_candidate(self):
        backend=RecoveringBackend()
        repair={'jobs':[],'placements':[{'id':'station','position':[.6,0,.8],'yaw':0}],
                'reason':'开发者合成布局修复'}
        result,transport=self.run_loop(self.base_replies+[repair,REVIEW,REVIEW],sandbox=backend)
        self.assertEqual(result['status'],'candidate_ready')
        self.assertEqual(backend.rechecks,1)
        self.assertEqual(backend.scene_calls,2)
        self.assertEqual(backend.build_calls,2)
        self.assertIsNotNone(backend.last_measurement['distance_m'])
        context=json.loads(transport.requests[5]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['feedback']['checks'][0]['id'],'scene_collisions')
        self.assertEqual(context['allowed_instances'],['station'])
        for root,before in backend.sources:
            self.assertEqual(snapshot(root),before)
        self.assertEqual(result['check_recovery'][0]['status'],'repair_ready')

    def test_persistent_unknown_cannot_pass_or_repeat_same_recovery_on_resume(self):
        backend=RecoveringBackend('partial')
        result,transport=self.run_loop(self.base_replies,sandbox=backend)
        self.assertEqual(result['status'],'needs_check_support')
        self.assertEqual(backend.rechecks,1)
        self.assertEqual(len(transport.requests),5)
        result,transport=self.run_loop([],resume=True,sandbox=backend)
        self.assertEqual(result['status'],'needs_check_support')
        self.assertEqual(backend.rechecks,1)
        self.assertEqual(transport.requests,[])
        self.assertTrue(result['stop_reason'])
        self.assertTrue(any('scene_collisions' in s for s in result['unverified']))

    def test_generation_receives_authoritative_layout(self):
        result,transport=self.run_loop(self.base_replies+[REVIEW])
        for index in (3,4):
            ctx=json.loads(transport.requests[index]['messages'][1]['content'][0]['text'])
            self.assertEqual(ctx['scene_layout']['instances'],self.base_replies[0]['instances'])
            self.assertEqual(ctx['dependencies']['instrument']['instances'],[self.base_replies[0]['instances'][0]])

    def test_interrupted_readonly_recheck_resumes_without_reassembly(self):
        backend=RecoveringBackend('partial')
        real=backend.recheck_scene
        with patch.object(backend,'recheck_scene',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            self.run_loop(self.base_replies,sandbox=backend)
        with patch.object(backend,'assemble_and_check',side_effect=AssertionError('must reuse compiled scene')):
            result,transport=self.run_loop([],resume=True,sandbox=backend)
        self.assertEqual(result['status'],'needs_check_support')
        self.assertEqual(backend.rechecks,1)
        self.assertEqual(transport.requests,[])
        self.assertEqual(result['check_recovery'][0]['attempts'],2)

    def test_real_local_recheck_preserves_compiled_scene_and_contract(self):
        from asset_generation.local import LocalBackend
        from scene_generation.assembly import assemble
        from scene_generation.contracts import numerical_contract
        from scene_fixtures import write_component
        assets={'instrument':write_component(self.root/'asset'),'tool':write_component(self.root/'tool',True)}
        artifact=assemble(self.base_replies[0],self.env,assets,self.root/'robot',self.root/'compiled')
        before=snapshot(artifact)
        backend=LocalBackend(self.generation,self.project)
        contract=numerical_contract({})
        expected=copy.deepcopy(contract)
        report=backend.recheck_scene(artifact,self.root/'supplement',SceneConfig(validation_mode='static'),contract)
        self.assertEqual(report['status'],'pass',report)
        self.assertEqual(contract,expected)
        self.assertEqual(snapshot(artifact),before)
        # 检查器异常必须保留编译产物并与生成缺陷区分。
        invalid={**contract,'fits':[{'outer':['missing','missing'],'inner':['missing','missing']}]}
        report=backend.recheck_scene(artifact,self.root/'broken_check',SceneConfig(validation_mode='static'),invalid)
        self.assertEqual(report['status'],'error')
        self.assertEqual(report['error_kind'],'check_execution')
        self.assertEqual(report['execution_failure']['kind'],'program')
        self.assertEqual(report['execution_failure']['components'],['scene_generation.geometry'])
        self.assertEqual(snapshot(artifact),before)


class GeometryRecoveryTests(unittest.TestCase):
    def test_malformed_mount_request_is_validation_feedback_not_program_crash(self):
        from scene_generation.edge_mounts import validate_mounts
        from asset_prep.errors import ValidationError
        plan={'jobs':[{'id':'fixture','role':'fixture'}],
              'instances':[{'id':'f','job':'fixture','placement':'table'}]}
        for field in ('instance','edge'):
            row={'instance':'f','edge':'x+','geoms':['clamp'],'reason':'synthetic'}
            row[field]=[]
            with self.subTest(field=field),self.assertRaises(ValidationError):
                validate_mounts([row],plan)

    def test_unknown_pair_preserves_contact_fact_but_not_complete_support(self):
        from scene_generation.checks import support_evidence
        from scene_generation.geometry import bounds
        m,d=compiled('<mujoco><worldbody><body name="rack__body"><geom name="rack__top" type="box" size=".1 .1 .1"/></body>'
                     '<body name="tube__body"><geom name="tube__base" type="box" size=".01 .01 .01" pos="0 0 .11"/>'
                     '<geom name="tube__unknown" type="box" size=".01 .01 .01" pos=".04 0 .11"/></body></worldbody></mujoco>')
        geometry={g:bounds(m,d,g) for g in range(m.ngeom)}
        query=DistanceQuery(m,d);real=query.measure
        def injected(a,b,limit):
            if b==2:
                raise MeasurementError('synthetic unsupported pair')
            return real(a,b,limit)
        with patch.object(query,'measure',side_effect=injected):
            result=support_evidence(m,d,{'id':'tube','support_instance':'rack'},.003,.0005,geometry,query)
        self.assertTrue(result['supported'])
        self.assertEqual(result['contact_count'],1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['measurement_error_count'],1)
    def test_one_unknown_pair_does_not_erase_other_collision(self):
        m,d=compiled('<mujoco><worldbody><geom name="a__g" type="sphere" size=".1"/>'
                     '<geom name="b__g" type="sphere" size=".1" pos=".1 0 0"/>'
                     '<geom name="c__g" type="sphere" size=".1" pos="-.1 0 0"/></worldbody></mujoco>')
        real=DistanceQuery(m,d).measure
        def injected(a,b,limit):
            if (a,b)==(0,1):
                raise MeasurementError('开发者注入单对失败')
            return real(a,b,limit)
        query=DistanceQuery(m,d)
        with patch.object(query,'measure',side_effect=injected):
            scan=collision_scan(m,d,.0005,query=query)
        self.assertTrue(scan['collisions'])
        self.assertEqual(scan['coverage_gaps'][0]['instances'],['a','b'])
        self.assertEqual(scan['examined_pairs'],3)
        row={'id':'scene_collisions','status':'fail',**scan}
        self.assertEqual(decide({'status':report_status([row]),'checks':[row]}),'repair')

    def test_pair_limit_retains_hits_and_names_remaining_scope(self):
        m,d=compiled('<mujoco><worldbody>'+''.join(f'<geom name="p{i}__g" type="sphere" size=".1" pos="{i*.01} 0 0"/>' for i in range(3))+'</worldbody></mujoco>')
        scan=collision_scan(m,d,.0005,max_pairs=1)
        self.assertTrue(scan['collisions'])
        self.assertEqual(scan['unexamined_pairs'],2)
        self.assertEqual(scan['coverage_gaps'][-1]['error_kind'],'resource_limit')
        self.assertFalse(scan['complete'])

    def test_overlap_certificate_is_lower_bound_and_never_invents_exact_distance(self):
        m,d=compiled('<mujoco><worldbody><geom name="table" type="box" size="1 .8 .03" pos="0 0 .77"/>'
                     '<geom name="wheel" type="cylinder" size=".012 .003" pos="-.43 .287 .81" euler="90 0 0"/></worldbody></mujoco>')
        query=DistanceQuery(m,d,recovery=True)
        with patch('mujoco.mj_geomDistance',return_value=0),patch.object(query,'convex_distance',return_value=None):
            evidence=query.measure(0,1)
        self.assertEqual(evidence['method'],'common_interior_ball')
        self.assertIsNone(evidence['distance_m'])
        self.assertGreater(evidence['penetration_lower_bound_m'],.0005)
        self.assertLessEqual(evidence['penetration_lower_bound_m'],.002)
        # 分离后不能沿用旧的相交证据。
        m.geom_pos[1,2]+=.02
        import mujoco
        mujoco.mj_forward(m,d)
        self.assertIsNone(query.overlap_certificate(0,1))

    def test_programming_error_is_not_an_unknown_measurement(self):
        class Broken:
            def measure(self,*args):
                raise TypeError('developer bug')
        with self.assertRaises(TypeError):
            observe(Broken(),0,1)

    def test_cone_normals_combine_but_single_sloped_contact_does_not(self):
        normals=[{'normal':[np.cos(a)*.944,np.sin(a)*.944,.33]} for a in np.linspace(0,2*np.pi,8,endpoint=False)]
        self.assertTrue(upward_resultant(normals))
        self.assertFalse(upward_resultant(normals[:1]))
        self.assertFalse(upward_resultant([]))
        self.assertTrue(upward_resultant([{'normal':[0,0,1]}]))

    def test_edge_mount_only_exempts_named_local_strip(self):
        from scene_generation.geometry import bounds
        m,d=compiled('<mujoco><worldbody><geom name="fixture__clamp" type="box" size=".02 .01 .01" pos="1.001 0 .8"/>'
                     '<geom name="fixture__body" type="box" size=".02 .01 .01" pos=".5 0 .81"/></worldbody></mujoco>')
        geometry={g:bounds(m,d,g) for g in range(m.ngeom)}
        item={'id':'fixture'};env={'table':{'size':[2,1.6,.06],'height':.8}}
        self.assertFalse(boundary_evidence(m,geometry,item,env,[])['valid'])
        entries=[{'instance':'fixture','geoms':['clamp'],'edge':'x+','reason':'合成桌边夹持'}]
        self.assertTrue(boundary_evidence(m,geometry,item,env,entries)['valid'])
        geometry[1]=(np.array([1.1,0,.8]),np.array([1.2,.1,.9]))
        self.assertFalse(boundary_evidence(m,geometry,item,env,entries)['valid'])
