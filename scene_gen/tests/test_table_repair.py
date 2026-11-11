import copy
import tempfile
import unittest
from pathlib import Path

import mujoco
from asset_prep.errors import ValidationError
from scene_generation.table import validate_cuts, table_cells
from scene_generation.contracts import SceneConfig, repair_action, numerical_contract
from scene_generation.assembly import assemble
from scene_generation.checks import scene_checks, support_evidence
from scene_generation.geometry import bounds
from scene_fixtures import environment,plan,write_component


class TableRepairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.env=environment(self.root/'robot')
        self.plan=plan()
        self.cut={'id':'slot','instance':'station','center':[.5,0],'size':[.12,.12],'depth':.01,'reason':'synthetic mounting pocket'}

    def test_robot_edge_depth_and_scope_rejected(self):
        for change in ({'center':[0,0]}, {'center':[.99,0]}, {'depth':.055}, {'center':[.89,.69],'size':[.04,.04]}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                validate_cuts([{**self.cut,**change}],self.plan,self.env)
        with self.assertRaises(ValidationError):
            repair_action({'jobs':[],'placements':[],'reason':'synthetic','table_modifications':[self.cut]},
                          self.plan,[],[],SceneConfig(),self.env)

    def test_other_instances_existing_cuts_cannot_be_removed(self):
        self.plan['instances'].append({'id':'other','job':'instrument','placement':'table','position':[-.5,0,.8],'yaw':0})
        self.plan['table_modifications']=[{**self.cut,'instance':'other','center':[-.5,0]}]
        with self.assertRaises(ValidationError):
            repair_action({'jobs':[],'placements':[],'reason':'synthetic','table_modifications':[]},
                          self.plan,[],['station'],SceneConfig(),self.env)

    def test_pocket_preserves_exact_remaining_volume(self):
        cells=table_cells(self.env['table'],[self.cut])
        volume=sum(8*s[0]*s[1]*s[2] for p,s in cells)
        self.assertAlmostEqual(volume,2*1.6*.06-.12*.12*.01)

    def test_real_pocket_support_and_through_hole_has_no_fake_support(self):
        assets={'instrument':write_component(self.root/'instrument'),'tool':write_component(self.root/'tool',True)}
        for depth,z,expected in ((.01,.79,'pass'),(.06,.8,'fail')):
            p=copy.deepcopy(self.plan)
            p['table_modifications']=[{**self.cut,'depth':depth}]
            p['instances'][0]['position'][2]=z
            output=assemble(p,self.env,assets,self.root/'robot',self.root/str(depth))
            report=scene_checks(output,SceneConfig(validation_mode='static'),numerical_contract({}))
            support=next(c for c in report['checks'] if c['id']=='support:station')
            self.assertEqual(support['status'],expected,support)
            if expected=='pass':
                self.assertEqual(report['status'],'pass',report)

    def test_flange_support_detected_without_bottom_contact(self):
        xml='''<mujoco><worldbody><body name="rack__base">
        <geom name="rack__left" type="box" pos="-.03 0 .03" size=".01 .04 .03"/>
        <geom name="rack__right" type="box" pos=".03 0 .03" size=".01 .04 .03"/></body>
        <body name="tube__body"><geom name="tube__shaft" type="cylinder" size=".008 .025" pos="0 0 .035"/>
        <geom name="tube__flange" type="box" size=".04 .035 .005" pos="0 0 .065"/></body></worldbody></mujoco>'''
        model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
        geometry={g:bounds(model,data,g) for g in range(model.ngeom)}
        evidence=support_evidence(model,data,{'id':'tube','support_instance':'rack'},.003,.0005,geometry)
        self.assertTrue(evidence['supported'],evidence)
        self.assertTrue(any(c['object_geom']=='tube__flange' for c in evidence['support_contacts']))
        side_only=xml.replace('<geom name="tube__flange" type="box" size=".04 .035 .005" pos="0 0 .065"/>','').replace('size=".008 .025"','size=".02 .025"')
        model=mujoco.MjModel.from_xml_string(side_only); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
        geometry={g:bounds(model,data,g) for g in range(model.ngeom)}
        evidence=support_evidence(model,data,{'id':'tube','support_instance':'rack'},.003,.0005,geometry)
        self.assertFalse(evidence['supported'],evidence)
