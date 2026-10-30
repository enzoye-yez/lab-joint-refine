"""真实失败路径：无效见证、备用能力、持续未知、共享故障。开发者合成几何。"""

import copy
import unittest
from unittest.mock import patch

import mujoco

from asset_generation.distance import DistanceQuery, MeasurementError
from asset_generation.pipeline import snapshot
from scene_generation.capabilities import REGISTRY
from scene_generation.faults import independent
from scene_generation.recovery import decide
import test_recovery_dispatch as fixtures


def compiles(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class InvalidWitnessTests(unittest.TestCase):
    def test_rejected_engine_witness_falls_back_to_analytic_face(self):
        """引擎给出不自洽见证时，必须拒绝并以真实支撑区域求交成功。"""
        model, data = compiles(
            '<mujoco><compiler angle="radian"/><asset><mesh name="slab" '
            'vertex="-2 -1.5 -.125 -2 -1.5 .125 -2 1.5 -.125 -2 1.5 .125 '
            '2 -1.5 -.125 2 -1.5 .125 2 1.5 -.125 2 1.5 .125"/></asset><worldbody>'
            '<geom name="deck__g" type="mesh" mesh="slab"/>'
            '<geom name="block__g" type="box" size=".25 .0625 .125" pos=".3125 .1875 .25"/>'
            '</worldbody></mujoco>')
        query = DistanceQuery(model, data)

        def inconsistent(_model, _data, g1, g2, limit, segment):
            segment[:] = [0, 0, 0, 0.15, 0.15, 0.15]   # 距离零却相隔 26 厘米
            return 0.0

        with patch('mujoco.mj_geomDistance', side_effect=inconsistent):
            result = query.measure(0, 1)
        self.assertEqual(result['method'], 'analytic_face')
        self.assertAlmostEqual(result['distance_m'], 0.0, delta=1e-8)
        points = result['points']
        # 见证必须落在两个实体表面，而不是平移求交的临时位置。
        self.assertLessEqual(max(result['surface_errors_m']), 2e-6)
        normal = result['normal']
        self.assertLessEqual(query.surface_error(0, points[:3], normal), 2e-6)
        self.assertLessEqual(query.surface_error(1, points[3:], [-v for v in normal]), 2e-6)

    def test_unresolvable_pair_stays_unknown_and_never_becomes_a_defect(self):
        model, data = compiles(
            '<mujoco><worldbody><geom name="a__g" type="sphere" size=".1"/>'
            '<geom name="b__g" type="sphere" size=".1" pos=".1 0 0"/></worldbody></mujoco>')
        from scene_generation.geometry import collision_scan
        with patch.object(DistanceQuery, 'measure', side_effect=MeasurementError('无可靠见证')) as measure:
            scan = collision_scan(model, data, .0005)
        self.assertGreater(measure.call_count, 0)
        report = {'status':'partial','checks':[{'id':'scene_collisions','status':'partial',**scan}]}
        self.assertEqual(report['checks'][0]['status'], 'partial')
        self.assertEqual(report['checks'][0]['coverage_gaps'][0]['error_kind'], 'measurement_error')
        self.assertNotEqual(decide(report), 'repair')
        self.assertNotEqual(decide(report), 'candidate_ready')


class SharedFaultIndependenceTests(unittest.TestCase):
    def test_numeric_core_fault_cannot_be_discharged_by_any_registered_capability(self):
        """共享数值核心的故障，不能被同源能力包装成已解除。"""
        fault = {'kind': 'program', 'components': ['asset_generation.distance']}
        for key in ('full_recheck_v2', 'collision_certificates_v1', 'support_certificates_v1'):
            with self.subTest(capability=key):
                self.assertFalse(independent(fault, REGISTRY[key]))

    def test_unrelated_capability_can_discharge_isolated_fault(self):
        # locate_defects 只读取编译场景与父体归属，不依赖数值距离核心。
        fault = {'kind': 'program', 'components': ['asset_generation.distance']}
        self.assertTrue(independent(fault, REGISTRY['locate_defects_v1']))
        # 但任何能力都声明共享 check_report 契约，因此这条故障不能被它解除。
        shared = {'kind': 'program', 'components': ['asset_generation.check_report']}
        self.assertFalse(independent(shared, REGISTRY['locate_defects_v1']))
