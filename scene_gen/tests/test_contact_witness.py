"""开发者合成几何：接触区域、退化与不变性；不读取或修改正式资产。"""

import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from asset_generation.convex import closest_witness
from asset_generation.distance import DistanceQuery, MeasurementError
from scene_generation.checks import support_evidence
from scene_generation.geometry import bounds, collision_scan
from test_distance_evidence import compiled


MESH = '<asset><mesh name="slab" vertex="-2 -1.5 -.125 -2 -1.5 .125 -2 1.5 -.125 -2 1.5 .125 2 -1.5 -.125 2 -1.5 .125 2 1.5 -.125 2 1.5 .125"/></asset>'


def on_slab(kind='box', gap=0., x=.3125, frame='', rotation=''):
    shape={'box':'type="box" size=".25 .0625 .125"',
           'cylinder':'type="cylinder" size=".125 .125"',
           'sphere':'type="sphere" size=".125"',
           'ellipsoid':'type="ellipsoid" size=".2 .1 .125"',
           'capsule':'type="capsule" size=".0625 .0625"'}[kind]
    return compiled(f'<mujoco><compiler angle="radian"/>{MESH}<worldbody><body {frame}>'
                    '<body name="platform__body"><geom name="platform__g" type="mesh" mesh="slab"/></body>'
                    f'<body name="object__body"><geom name="object__g" {shape} {rotation} pos="{x} .1875 {.25+gap:.17g}"/></body>'
                    '</body></worldbody></mujoco>')


class ContactWitnessTests(unittest.TestCase):
    def assert_witness(self, result, query, first, second, expected, atol=1e-8):
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['distance_m'],expected,delta=atol)
        points=np.array(result['points']);normal=np.array(result['normal'])
        np.testing.assert_allclose(points[3:]-points[:3],expected*normal,atol=atol)
        self.assertLessEqual(max(result['surface_errors_m']),2e-6)
        # 凸区域见证必须仍在两个原实体表面，而非平移求交的临时位置。
        for geom,point,n in ((first,points[:3],normal),(second,points[3:],-normal)):
            self.assertLessEqual(query.surface_error(geom,point,n),2e-6)

    def test_mesh_surface_interior_contact_for_multiple_shape_families(self):
        for kind in ('box','cylinder','sphere','ellipsoid','capsule'):
            for gap in (0.,2e-10,1e-5,.002):
                with self.subTest(kind=kind,gap=gap):
                    m,d=on_slab(kind,gap)
                    q=DistanceQuery(m,d)
                    with patch('mujoco.mj_geomDistance',return_value=0):
                        result=q.measure(0,1)
                    self.assert_witness(result,q,0,1,gap)
                    self.assertEqual(result['method'],'analytic_face')

    def test_circular_face_can_overlap_without_its_center_over_the_other_face(self):
        # 圆柱中心在板外，圆盘边缘仍有真实接触；中心点特判无法解决。
        m,d=on_slab('cylinder',x=2.0625)
        q=DistanceQuery(m,d)
        self.assert_witness(q.analytic_face(0,1,.05),q,0,1,0.)

    def test_crossed_faces_have_contact_without_any_overlapping_original_vertex(self):
        for penetration in (0.,.001,.02):
            with self.subTest(penetration=penetration):
                m,d=compiled('<mujoco><worldbody><geom type="box" size="1 .1 .1"/>'
                             f'<geom type="box" size=".1 1 .1" pos="0 0 {.2-penetration:.17g}"/></worldbody></mujoco>')
                q=DistanceQuery(m,d)
                with patch('mujoco.mj_geomDistance',return_value=0):
                    result=q.measure(0,1)
                self.assert_witness(result,q,0,1,-penetration)

    def test_pair_order_and_common_rigid_transform_preserve_distance(self):
        rng=np.random.default_rng(481)
        for trial in range(12):
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            angle=float(rng.uniform(-2,2))
            quat=np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
            pos=rng.uniform(-3,3,3)
            frame='pos="'+ ' '.join(map(str,pos))+'" quat="'+' '.join(map(str,quat))+'"'
            m,d=on_slab('box',.002,frame=frame,rotation='euler="0 0 .7"')
            q=DistanceQuery(m,d)
            with self.subTest(trial=trial):
                result=q.analytic_face(0,1,.05);reverse=q.analytic_face(1,0,.05)
                self.assert_witness(result,q,0,1,.002)
                self.assert_witness(reverse,q,1,0,.002)
                np.testing.assert_allclose(result['normal'],-np.array(reverse['normal']),atol=1e-8)

    def test_zero_and_positive_gap_are_not_conflated_by_support_checker(self):
        for gap,expected in ((0.,True),(.002,False)):
            m,d=on_slab('cylinder',gap)
            q=DistanceQuery(m,d)
            geometry={g:bounds(m,d,g) for g in range(m.ngeom)}
            with patch('mujoco.mj_geomDistance',return_value=0):
                result=support_evidence(m,d,{'id':'object','support_instance':'platform'},.003,.0005,geometry,q)
            self.assertTrue(result['complete'])
            self.assertEqual(result['supported'],expected)

    def test_mesh_penetration_is_not_falsely_certified_as_zero_contact(self):
        for depth in (1e-7,.01):
            with self.subTest(depth=depth):
                m,d=on_slab('box',-depth);q=DistanceQuery(m,d)
                self.assertIsNone(q.analytic_face(0,1,.05))
                with patch('mujoco.mj_geomDistance',return_value=0),self.assertRaises(MeasurementError):
                    q.measure(0,1)

    def test_disjoint_coplanar_support_regions_do_not_make_a_zero_distance(self):
        m,d=on_slab('cylinder',x=2.25);q=DistanceQuery(m,d)
        normal=np.array([0.,0.,1.])
        self.assertIsNone(q.support_witness(0,1,normal,0.,.05))
        with patch('mujoco.mj_geomDistance',return_value=0):
            result=q.measure(0,1,.5)
        self.assertGreater(result['distance_m'],.1)

    def test_box_containment_retains_escape_depth(self):
        m,d=compiled('<mujoco><worldbody><geom type="box" size="1 1 1"/>'
                     '<geom type="box" size=".1 .1 .1" pos=".2 0 0"/></worldbody></mujoco>')
        q=DistanceQuery(m,d)
        self.assert_witness(q.box_overlap(0,1,2.),q,0,1,-.9)

    def test_negative_gap_requires_authoritative_penetration_algorithm(self):
        # 任意支撑平面的负投影不等于最小穿透；解析面入口必须拒绝。
        m,d=compiled('<mujoco><worldbody><geom type="box" size="1 1 1"/>'
                     '<geom type="box" size="1 1 1" pos="1.9 .2 .3"/></worldbody></mujoco>')
        q=DistanceQuery(m,d)
        self.assertIsNone(q.analytic_face(0,1,.05))
        self.assert_witness(q.box_overlap(0,1,.05),q,0,1,-.1)

    def test_new_solver_does_not_discharge_shared_program_fault(self):
        from scene_generation.faults import independent
        from scene_generation.capabilities import REGISTRY
        fault={'kind':'program','components':['asset_generation.convex']}
        for key in ('full_recheck_v2','collision_certificates_v1','support_certificates_v1'):
            self.assertFalse(independent(fault,REGISTRY[key]))

    def test_closest_solver_keeps_zero_witness_without_asserting_signed_distance(self):
        a=np.array([[-1.,0,0],[1.,0,0]])
        b=np.array([[0.,-1,0],[0.,1,0]])
        result=closest_witness(lambda n:(a[np.argmax(a@n)],b[np.argmax(b@(-n))]),[1.,1.,0.])
        self.assertLess(result['upper'],1e-10)
        np.testing.assert_allclose(result['a'],result['b'],atol=1e-10)
        self.assertNotIn('distance_m',result)

    def test_fixed_checker_accepts_real_contacts_but_still_detects_penetration(self):
        m,d=compiled('<mujoco><worldbody><geom name="a__g" type="box" size="1 .1 .1"/>'
                     '<geom name="b__g" type="box" size=".1 1 .1" pos="0 0 .199"/></worldbody></mujoco>')
        with patch('mujoco.mj_geomDistance',return_value=0):
            result=collision_scan(m,d,.0005)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['collisions']),1)
        self.assertAlmostEqual(result['collisions'][0]['distance_m'],-.001)
