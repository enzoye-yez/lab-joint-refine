"""开发者合成几何与故障注入，不调用模型或修改正式实验产物。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from asset_generation.distance import DistanceQuery, MeasurementError
from asset_generation.evidence import inspect_artifact
from asset_generation.pipeline import snapshot
from asset_prep.errors import ValidationError
from asset_prep.storage import atomic_json
from scene_generation.assembly import assemble
from scene_generation.checks import support_evidence
from scene_generation.geometry import bounds, collisions
from scene_generation.inspection import mount, perimeter
from scene_fixtures import environment, plan, write_component


def compiled(xml):
    m=mujoco.MjModel.from_xml_string(xml)
    d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    return m,d


class DistanceEvidenceTests(unittest.TestCase):
    def test_bad_zero_witness_is_recovered_with_separation_certificate(self):
        m,d=compiled('<mujoco><worldbody><geom name="a" type="box" size="1 1 .1"/>'
                     '<geom name="b" type="box" size=".1 .1 .1" pos="0 0 .202"/></worldbody></mujoco>')
        original=int(m.opt.disableflags)
        qpos=d.qpos.copy()
        def bad(*args):
            args[-1][:]=[0,0,.1,0,0,.6]
            return 0
        with patch('mujoco.mj_geomDistance',side_effect=bad):
            result=DistanceQuery(m,d).measure(0,1)
        self.assertEqual(result['method'],'analytic_face')
        self.assertAlmostEqual(result['distance_m'],.002)
        self.assertAlmostEqual(np.linalg.norm(np.array(result['points'][3:])-result['points'][:3]),.002)
        self.assertEqual(int(m.opt.disableflags),original)
        np.testing.assert_array_equal(d.qpos,qpos)

    def test_two_unreliable_engines_never_count_as_contact(self):
        m,d=compiled('<mujoco><worldbody><geom name="a" type="sphere" size=".1"/>'
                     '<geom name="b" type="sphere" size=".1" pos=".3 0 0"/></worldbody></mujoco>')
        original=int(m.opt.disableflags)
        with patch('mujoco.mj_geomDistance',return_value=0),patch.object(DistanceQuery,'convex_distance',return_value=None),self.assertRaises(MeasurementError):
            DistanceQuery(m,d).measure(0,1)
        self.assertEqual(int(m.opt.disableflags),original)

    def test_independent_convex_distance_recovers_two_bad_engine_results(self):
        m,d=compiled('<mujoco><worldbody><geom name="a" type="sphere" size=".1"/>'
                     '<geom name="b" type="sphere" size=".1" pos=".3 .4 0"/></worldbody></mujoco>')
        with patch('mujoco.mj_geomDistance',return_value=0):
            result=DistanceQuery(m,d).measure(0,1,.5)
        self.assertEqual(result['method'],'convex_distance')
        self.assertAlmostEqual(result['distance_m'],.3,places=7)
        np.testing.assert_allclose(result['normal'],[.6,.8,0],atol=1e-7)

    def test_independent_distance_does_not_turn_penetration_into_positive_gap(self):
        m,d=compiled('<mujoco><worldbody><geom name="a" type="sphere" size=".1"/>'
                     '<geom name="b" type="sphere" size=".1" pos=".1 0 0"/></worldbody></mujoco>')
        self.assertIsNone(DistanceQuery(m,d).convex_distance(0,1,.5))

    def test_flush_face_contact_is_not_a_full_thickness_penetration(self):
        # MuJoCo 3.3.7 returns -5.95 mm for this tube tip seated flush on a rack floor.
        plate=('<mesh name="plate" vertex="-.051 -.032 0 .051 -.032 0 .051 .032 0 -.051 .032 0 '
               '-.051 -.032 .0053 .051 -.032 .0053 .051 .032 .0053 -.051 .032 .0053"/>')
        for z,expected in ((.005625,0.),(.005325,-.0003)):
            m,d=compiled(f'<mujoco><asset>{plate}</asset><worldbody><geom name="a" type="mesh" mesh="plate"/>'
                         f'<geom name="b" type="cylinder" size=".0009 .000325" pos="0 -.0178 {z}"/></worldbody></mujoco>')
            self.assertAlmostEqual(DistanceQuery(m,d).measure(0,1)['distance_m'],expected,places=7)

    def test_unresolved_contact_cannot_silently_pass_collision_checker(self):
        m,d=compiled('<mujoco><worldbody><geom name="a__g" type="sphere" size=".1"/>'
                     '<geom name="b__g" type="sphere" size=".1" pos=".1 0 0"/></worldbody></mujoco>')
        with patch.object(DistanceQuery,'measure',side_effect=MeasurementError('synthetic unavailable witness')):
            with self.assertRaises(MeasurementError):
                collisions(m,d,.0005)

    def test_valid_legacy_is_checked_and_flags_restored(self):
        m,d=compiled('<mujoco><worldbody><geom name="a" type="sphere" size=".1"/>'
                     '<geom name="b" type="sphere" size=".1" pos=".3 0 0"/></worldbody></mujoco>')
        calls=[]
        native_off=int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        def probe(*args):
            calls.append(int(m.opt.disableflags))
            # Native CCD (tried in both argument orders) returns an inconsistent zero.
            if not calls[-1]&native_off:
                return 0
            args[-1][:]=[.1,0,0,.2,0,0]
            return .1
        with patch('mujoco.mj_geomDistance',side_effect=probe):
            result=DistanceQuery(m,d).measure(0,1,.2)
        self.assertEqual(result['method'],'legacy_checked')
        self.assertEqual(int(m.opt.disableflags),0)
        self.assertNotEqual(calls[0],calls[-1])

    def test_positive_gap_is_not_support_and_selector_is_scoped(self):
        m,d=compiled('<mujoco><worldbody><body name="rack__body"><geom name="rack__top" type="box" size=".1 .1 .1"/></body>'
                     '<body name="tube__body"><geom name="tube__base" type="box" size=".01 .01 .01" pos="0 0 .112"/></body></worldbody></mujoco>')
        geometry={g:bounds(m,d,g) for g in range(m.ngeom)}
        item={'id':'tube','support_instance':'rack'}
        result=support_evidence(m,d,item,.003,.0005,geometry)
        self.assertFalse(result['supported'])
        self.assertAlmostEqual(result['nearby_noncontacts'][0]['distance_m'],.002)
        with self.assertRaises(ValidationError):
            support_evidence(m,d,item,.003,.0005,geometry,object_geoms=['rack__top'])


class SceneInspectionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)

    def perimeter_case(self,x):
        xml=f'''<mujoco><worldbody><body name="device__body"><geom name="device__g" type="box" size=".1 .1 .1" pos="0 0 .2"/></body>
        <body name="support__body"><geom name="support__g" type="box" size=".15 .15 .05" pos="0 0 .05"/></body>
        <body name="fixture__body"><geom name="fixture__g" type="box" size=".01 .01 .01" pos="{x} 0 .2" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
        (self.root/'model.xml').write_text(xml,encoding='utf-8')
        m,d=compiled(xml)
        meta={'plan':{'instances':[{'id':'device'}]}}
        before=snapshot(self.root)
        with patch('mujoco.mj_step',side_effect=AssertionError('no dynamics')):
            result=perimeter(self.root,m,d,meta,{'kind':'perimeter','instance':'device','distance_m':.05})
        self.assertEqual(before,snapshot(self.root))
        return result

    def test_perimeter_excludes_bottom_support(self):
        self.assertTrue(self.perimeter_case(.3)['region_clear'])

    def test_perimeter_detects_filtered_fixture_inside_band(self):
        result=self.perimeter_case(.125)
        self.assertFalse(result['region_clear'])
        self.assertTrue(all(x['instance']=='fixture' for x in result['intrusions']))

    def test_mount_reports_real_transform_without_inventing_old_metadata(self):
        env=environment(self.root/'robot')
        assets={'instrument':write_component(self.root/'asset'),'tool':write_component(self.root/'tool',True)}
        output=assemble(plan(),env,assets,self.root/'robot',self.root/'scene')
        meta=json.loads((output/'scene.json').read_text(encoding='utf-8'))
        m=mujoco.MjModel.from_xml_path(str(output/'model.xml'));d=mujoco.MjData(m)
        d.qpos[:]=meta['home_qpos'];mujoco.mj_forward(m,d)
        req={'kind':'mount','instance':'gripper','distance_m':.02}
        result=mount(m,d,meta,req)
        self.assertTrue(result['attachment_transform_matches'],result)
        meta.pop('mounts')
        result=mount(m,d,meta,req)
        self.assertFalse(result['expected_transform_available'])
        self.assertFalse(result['attachment_transform_matches'])

    def test_pair_measurement_failure_is_explicit_and_artifact_unchanged(self):
        xml='<mujoco><compiler angle="radian"/><worldbody><geom name="a" type="sphere" size=".1"/><geom name="b" type="sphere" size=".1" pos=".3 0 0"/></worldbody></mujoco>'
        (self.root/'model.xml').write_text(xml,encoding='utf-8')
        atomic_json(self.root/'scene.json',{'home_qpos':[],'home_ctrl':[]})
        before=snapshot(self.root)
        with patch('mujoco.mj_geomDistance',return_value=0),patch.object(DistanceQuery,'convex_distance',return_value=None):
            report=inspect_artifact(self.root,'scene',[{'kind':'pairs','pairs':[['a','b']],'distance_m':.2}],self.root/'out')
        self.assertEqual(report['status'],'incomplete')
        self.assertEqual(report['results'][0]['error_kind'],'measurement_error')
        self.assertEqual(before,snapshot(self.root))
