import unittest
import numpy as np
import mujoco

from scene_generation.geometry import collisions


class BroadphaseTests(unittest.TestCase):
    def test_conservative_filter_matches_exhaustive_actual_distances(self):
        rng = np.random.default_rng(123)
        geometries = ['<geom name="floor__plane" type="plane" size="1 1 .1"/>']
        for i in range(24):
            kind,size = [('box','.07 .02 .03'),('sphere','.045'),('capsule','.02 .08'),
                         ('cylinder','.04 .05'),('ellipsoid','.03 .06 .04')][i % 5]
            pos = rng.uniform(-.12,.12,3)
            angles = rng.uniform(-2,2,3)
            geometries.append(f'<geom name="part{i}__geom" type="{kind}" size="{size}" pos="{" ".join(map(str,pos))}" euler="{" ".join(map(str,angles))}"/>')
        model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>'+''.join(geometries)+'</worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        # 本测试只隔离验证粗筛不会漏掉引擎返回的碰撞对。
        # 精确距离的见证点校验另由 test_distance_evidence 覆盖；随机穿透样例
        # 的原生距离本身也可能缺少可靠见证，不能把它当成校验器的真值。
        class ReferenceQuery:
            def measure(self,i,j,limit):
                return {'geom1':model.geom(i).name,'geom2':model.geom(j).name,
                        'distance_m':float(mujoco.mj_geomDistance(model,data,i,j,limit,None))}
        actual = {frozenset((r['geom1'],r['geom2'])) for r in collisions(model,data,.0005,query=ReferenceQuery())}
        expected = {frozenset((model.geom(i).name,model.geom(j).name))
                    for i in range(model.ngeom) for j in range(i+1,model.ngeom)
                    if mujoco.mj_geomDistance(model,data,i,j,.05,None) < -.0005}
        self.assertTrue(expected)
        self.assertEqual(actual,expected)

    def test_separated_long_boxes_do_not_exhaust_exact_pair_budget(self):
        shapes = ''.join(f'<geom name="part{i}__geom" type="box" size="10 .001 .001" pos="0 {i*.1} 0"/>' for i in range(30))
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+shapes+'</worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        self.assertEqual(collisions(model,data,.0005,max_pairs=1),[])
