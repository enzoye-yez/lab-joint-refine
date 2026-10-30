"""开发者合成组件与量测故障；不调用模型，不手改实验资产。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np

from asset_generation.distance import DistanceQuery, MeasurementError
from scene_generation.checks import support_evidence, scene_checks
from scene_generation.geometry import bounds
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.assembly import assemble
from scene_generation.capabilities import SupportCertificates, merge, proposal, confirmed_names
from scene_generation.progress import evidence_progress
from asset_generation.check_report import CheckContractError, validate_rows
from scene_generation.recovery import decide
from scene_generation.dispatch import recover
from scene_fixtures import environment, plan, write_component


BASE = '<body name="station__base"><geom name="station__base_g" type="box" size=".1 .1 .05" pos="0 0 .05"/>{child}</body>'


def body(name, z, *, free=False, visual=False):
    return (f'<body name="station__{name}" pos="0 0 {z}">'
            + ('<freejoint/>' if free else '')
            + f'<geom name="station__{name}_g" type="box" size=".05 .05 .025"'
            + (' contype="0" conaffinity="0"' if visual else '') + '/></body>')


def case(extra='', *, child='', weld='', base=True):
    xml = ('<mujoco><compiler angle="radian"/><worldbody>'
           '<geom name="environment__table_top" type="box" size="1 1 .05" pos="0 0 -.05"/>'
           + (BASE.format(child=child) if base else '') + extra + '</worldbody>' + weld + '</mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    geometry = {g:bounds(model,data,g) for g in range(model.ngeom)
                if model.geom_contype[g] or model.geom_conaffinity[g]}
    return model,data,geometry


def check(model,data,geometry,**kwargs):
    return support_evidence(model,data,{'id':'station','placement':'table'},.003,.0005,geometry,**kwargs)


class ComponentSupportTests(unittest.TestCase):
    def test_grounded_base_cannot_mask_independent_floating_body(self):
        result = check(*case(body('floating',.3)))
        self.assertTrue(result['vertical_resultant_available'])
        self.assertFalse(result['supported'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['unsupported_components'][0]['body'],'station__floating')

    def test_free_component_resting_on_same_instance_is_supported(self):
        result = check(*case(body('cap',.125,free=True)))
        self.assertTrue(result['supported'],result)
        self.assertEqual(result['component_contact_count'],1)
        self.assertEqual(len(result['component_support']),2)

    def test_multilevel_support_propagates_independent_of_body_order(self):
        result = check(*case(body('top',.175,free=True)+body('middle',.125,free=True)))
        self.assertTrue(result['supported'],result)
        self.assertEqual(len(result['component_support']),3)

    def test_unanchored_contact_chain_does_not_support_itself(self):
        result = check(*case(body('low',.3)+body('top',.35),base=False))
        self.assertFalse(result['supported'])
        self.assertEqual(len(result['unsupported_components']),2)

    def test_positive_gap_is_not_internal_support(self):
        result = check(*case(body('cap',.126,free=True)))
        self.assertFalse(result['supported'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['component_contact_count'],0)

    def test_active_weld_connects_component_but_disabled_weld_does_not(self):
        weld='<equality><weld name="lock" body1="station__base" body2="station__cap"/></equality>'
        model,data,geometry=case(body('cap',.3,free=True),weld=weld)
        self.assertTrue(check(model,data,geometry)['supported'])
        data.eq_active[0]=False
        result=check(model,data,geometry)
        self.assertFalse(result['supported'])
        self.assertEqual(len(result['component_support']),2)

    def test_real_joint_link_is_retained_without_demanding_bottom_contact(self):
        child='<body name="station__lid" pos="0 0 .2"><joint name="hinge" axis="0 1 0"/><geom name="station__lid_g" type="box" size=".05 .05 .01"/></body>'
        result=check(*case(child=child))
        self.assertTrue(result['supported'])
        self.assertEqual(len(result['component_support']),1)

    def test_unknown_internal_measurement_retains_contact_and_does_not_authorize_repair(self):
        model,data,geometry=case(body('cap',.125,free=True))
        query=DistanceQuery(model,data)
        original=query.measure
        def measure(a,b,limit):
            if {model.geom(a).name,model.geom(b).name}=={'station__base_g','station__cap_g'}:
                raise MeasurementError('developer injected unknown internal pair')
            return original(a,b,limit)
        with patch.object(query,'measure',side_effect=measure):
            result=check(model,data,geometry,query=query)
        self.assertFalse(result['complete'])
        self.assertFalse(result['supported'])
        self.assertTrue(result['support_contacts'])
        self.assertFalse(result['unsupported_components'])
        self.assertEqual(result['component_support'][1]['status'],'unknown')

    def test_internal_budget_is_shared_with_external_contacts(self):
        result=check(*case(body('cap',.125,free=True)),max_pairs=1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['component_examined_pairs'],0)
        self.assertEqual(result['component_unexamined_pairs'],1)
        self.assertFalse(result['unsupported_components'])

    def test_unknown_chain_does_not_erase_separate_confirmed_floating_component(self):
        model,data,geometry=case(body('uncertain_cap',.125,free=True)+body('floating',.5))
        query=DistanceQuery(model,data)
        original=query.measure
        def measure(a,b,limit):
            if model.geom(b).name=='station__uncertain_cap_g':
                raise MeasurementError('developer injected uncertainty')
            return original(a,b,limit)
        with patch.object(query,'measure',side_effect=measure):
            result=check(model,data,geometry,query=query)
        self.assertFalse(result['complete'])
        self.assertEqual([x['body'] for x in result['unsupported_components']],['station__floating'])
        statuses={x['body']:x['status'] for x in result['component_support']}
        self.assertEqual(statuses['station__uncertain_cap'],'unknown')
        self.assertEqual(statuses['station__floating'],'unsupported')

    def test_unknown_external_support_propagates_only_to_its_dependents(self):
        model,data,geometry=case(body('cap',.125,free=True)+body('floating',.5))
        query=DistanceQuery(model,data)
        original=query.measure
        def measure(a,b,limit):
            if model.geom(a).name=='environment__table_top':
                raise MeasurementError('developer injected external uncertainty')
            return original(a,b,limit)
        with patch.object(query,'measure',side_effect=measure):
            result=check(model,data,geometry,query=query)
        statuses={x['body']:x['status'] for x in result['component_support']}
        self.assertEqual(statuses['station__base'],'unknown')
        self.assertEqual(statuses['station__cap'],'unknown')
        self.assertEqual(statuses['station__floating'],'unsupported')

    def test_unowned_world_geometry_is_unknown_instead_of_crashing_or_passing(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<geom name="environment__table_top" type="box" size="1 1 .05" pos="0 0 -.05"/>'
            '<geom name="station__g" type="box" size=".05 .05 .05" pos="0 0 .05"/>'
            '</worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        result=check(model,data,{g:bounds(model,data,g) for g in range(model.ngeom)})
        self.assertFalse(result['complete'])
        self.assertFalse(result['supported'])
        self.assertTrue(result['component_errors'])

    def test_independent_visual_only_component_is_unknown_not_silently_omitted(self):
        result=check(*case(body('visual',.3,visual=True)))
        self.assertFalse(result['complete'])
        self.assertFalse(result['supported'])
        self.assertEqual(result['component_error_count'],1)
        self.assertFalse(result['unsupported_components'])

    def test_selected_geoms_do_not_claim_component_coverage(self):
        result=check(*case(body('floating',.3)),object_geoms=['station__base_g'])
        self.assertTrue(result['supported'])
        self.assertEqual(result['component_scope'],'selected_geoms')
        self.assertNotIn('component_support',result)

    def test_internal_reverse_orientation_preserves_witness_convention(self):
        model,data,geometry=case(body('top',.175)+body('middle',.125))
        before=data.qpos.copy()
        with patch('mujoco.mj_step',side_effect=AssertionError('static only')):
            result=check(model,data,geometry)
        np.testing.assert_array_equal(data.qpos,before)
        for row in result['component_contacts']:
            self.assertEqual(row['geom1'],row['support_geom'])
            self.assertEqual(row['geom2'],row['object_geom'])
            self.assertGreater(row['normal'][2],0)


class ComponentRecoveryTests(unittest.TestCase):
    def fixture(self, root, *, floating=False):
        env=environment(root/'robot')
        instrument=write_component(root/'instrument')
        xml=(instrument/'model.xml').read_text(encoding='utf-8')
        xml=xml.replace('type="box" size=".05 .05 .05" pos="0 0 .05"','type="sphere" size=".1" pos="0 0 .1"')
        extra='<body name="cap" pos="0 0 .25"><freejoint/><geom name="cap_g" type="sphere" size=".05"/></body>'
        if floating:
            extra+='<body name="floating" pos=".3 0 .4"><geom name="floating_g" type="sphere" size=".02"/></body>'
        xml=xml.replace('</worldbody>',extra+'</worldbody>')
        (instrument/'model.xml').write_text(xml,encoding='utf-8')
        return assemble(plan(),env,{'instrument':instrument,'tool':write_component(root/'tool',True)},root/'robot',root/'scene')

    def inject_internal_unknown(self, original):
        def measure(query,a,b,limit):
            if {query.model.geom(a).name,query.model.geom(b).name}=={'station__base_geom','station__cap_g'}:
                raise MeasurementError('developer injected main internal-contact failure')
            return original(query,a,b,limit)
        return measure

    def test_main_report_keeps_confirmed_component_defect_beside_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact=self.fixture(Path(temporary),floating=True)
            with patch.object(DistanceQuery,'measure',new=self.inject_internal_unknown(DistanceQuery.measure)):
                report=scene_checks(artifact,SceneConfig(validation_mode='static'),numerical_contract({}))
            row=next(r for r in report['checks'] if r['id']=='support:station')
            self.assertEqual(row['status'],'fail')
            self.assertFalse(row['complete'])
            self.assertEqual(row['measurement_error_count'],1)
            self.assertEqual([c['body'] for c in row['unsupported_components']],['station__floating'])
            self.assertEqual(decide(report),'repair')

    def test_independent_capability_resolves_internal_contact_after_full_recheck_is_ineffective(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            artifact=self.fixture(root)
            limits=SceneConfig(validation_mode='static');contract=numerical_contract({})
            with patch.object(DistanceQuery,'measure',new=self.inject_internal_unknown(DistanceQuery.measure)):
                report=scene_checks(artifact,limits,contract)
                row=next(r for r in report['checks'] if r['id']=='support:station')
                self.assertEqual(row['status'],'partial')
                calls=[]
                def recheck(*args):
                    calls.append('full')
                    return report
                def recovery(artifact,destination,limits,contract,capability,request,input_key):
                    calls.append(capability)
                    from scene_generation.capabilities import REGISTRY
                    return {**REGISTRY[capability].inspect(artifact,limits,contract,request),'input_key':input_key}
                backend=SimpleNamespace(recheck_scene=recheck,recover_scene=recovery)
                run=SimpleNamespace(root=root,state={'plan':plan()},sandbox=backend,limits=limits,contract=contract,
                                    check_pause=lambda:None,store=SimpleNamespace(save=lambda:None))
                updated=recover(run,report,artifact,root/'recovery')
            self.assertEqual(calls[:2],['full','support_certificates_v1'])
            self.assertIn('material_certificates_v1',calls)
            self.assertEqual(next(r for r in updated['checks'] if r['id']=='material_integrity:station')['status'],'pass')
            row=next(r for r in updated['checks'] if r['id']=='support:station')
            self.assertEqual(row['status'],'pass')
            self.assertEqual(len(row['component_support']),2)
            self.assertTrue(row['component_complete'])
            actions=next(iter(run.state['check_recovery'].values()))['actions']
            self.assertFalse(actions[0]['effective'])
            self.assertTrue(actions[1]['effective'])

    def test_partial_selection_cannot_overwrite_whole_support_check(self):
        old={'checks':[{'id':'support:station','status':'partial','complete':False}]}
        result={'scope':'checks','input_key':'v','checks':[{'id':'support:station','status':'pass','component_scope':'selected_geoms'}]}
        with self.assertRaises(CheckContractError):
            merge(old,result,proposal(['support:station']),'v')

    def test_alternative_failure_cannot_erase_previously_confirmed_components(self):
        old={'checks':[{'id':'support:station','status':'fail','unsupported_components':[{'body':'station__a'}]}]}
        result={'scope':'checks','input_key':'v','checks':[{'id':'support:station','status':'fail','unsupported_components':[{'body':'station__b'}]}]}
        request=proposal(['support:station'])
        merged=merge(old,result,request,'v')
        self.assertEqual({c['body'] for c in merged['checks'][0]['unsupported_components']},{'station__a','station__b'})
        progress=evidence_progress(old,merged,SupportCertificates(),request,status='completed',result=result)
        self.assertEqual(progress['confirmed_defects'],[{'check':'support:station','confirmed_defects':1}])
        self.assertTrue(progress['evidence_changed'])
        repeated=evidence_progress(merged,merged,SupportCertificates(),request,status='completed',result=result)
        self.assertFalse(repeated['evidence_changed'])

    def test_components_cannot_be_declared_failed_and_pass_at_once(self):
        with self.assertRaises(CheckContractError):
            validate_rows([{'id':'support:station','status':'pass','unsupported_components':[{'body':'station__a'}]}])

    def test_component_summary_and_scope_are_part_of_report_contract(self):
        variants=[{'component_complete':False}, {'component_complete':'true'},
                  {'component_scope':[]}, {'all_components_supported':1},
                  {'component_support':[{'body':'station__a','status':'unknown'}]},
                  {'component_support':None}]
        for variant in variants:
            with self.subTest(variant=variant),self.assertRaises(CheckContractError):
                validate_rows([{'id':'support:station','status':'pass',**variant}])

    def test_real_main_and_recovery_checks_feed_same_problem_to_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            env=environment(root/'robot')
            instrument=write_component(root/'instrument')
            text=(instrument/'model.xml').read_text(encoding='utf-8')
            text=text.replace('</worldbody>', '<body name="floating" pos=".25 0 .3"><geom name="floating_g" type="box" size=".025 .025 .025"/></body></worldbody>')
            (instrument/'model.xml').write_text(text,encoding='utf-8')
            artifact=assemble(plan(),env,{'instrument':instrument,'tool':write_component(root/'tool',True)},root/'robot',root/'scene')
            limits=SceneConfig(validation_mode='static')
            contract=numerical_contract({})
            report=scene_checks(artifact,limits,contract)
            row=next(r for r in report['checks'] if r['id']=='support:station')
            self.assertEqual(row['status'],'fail',row)
            self.assertEqual(decide(report),'repair')
            self.assertIn('station__floating',confirmed_names(row))
            request=proposal(['support:station'],instance='station',max_pairs=800000)
            result=SupportCertificates().inspect(artifact,limits,contract,request)
            result['input_key']='current'
            # 同问题补查必须保留组件缺陷，不能退回只检查底座的旧算法。
            unrelated={'id':'unrelated','status':'partial','coverage_gaps':[{'error_kind':'measurement_error'}]}
            report['checks'].append(unrelated)
            merged=merge(report,result,request,'current',mode='static')
            new=next(r for r in merged['checks'] if r['id']=='support:station')
            self.assertEqual(new['status'],'fail')
            self.assertEqual(new['unsupported_components'],row['unsupported_components'])
            self.assertIn(unrelated,merged['checks'])

