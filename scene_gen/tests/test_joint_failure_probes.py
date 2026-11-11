"""Developer failure injections, real MuJoCo geometry, no generated success labels."""
import tempfile,unittest
from pathlib import Path
import mujoco,numpy as np
from asset_prep.storage import atomic_json
from asset_prep.errors import ValidationError
from asset_generation.operation_interface import validate_operation
from robot_operation.common import Settings,OperationError,identity,read
from robot_operation.scene import prepare,load
from robot_operation.runtime import Session
from robot_operation.diagnosis import inspect_scene
from operation_fixtures import fixture,task_graph

class FailureProbes(unittest.TestCase):
    def session(self,root,grasp=False,change=None):
        source=fixture(root/'source',grasp_ready=grasp)
        if change:
            p=source/'model.xml';p.write_text(change(p.read_text()),encoding='utf-8')
        result=prepare(source,root/'dynamic',identity(source),Settings(settle_seconds=.1))
        m,d,meta,manifest=load(root/'dynamic',result['scene_sha256']);task,_=task_graph(result['scene_sha256'])
        return Session(m,d,meta,manifest,task,Settings()),root/'dynamic',result['scene_sha256']
    def test_hinge_axis_misalignment_is_not_accepted_as_operable(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody><body name="door"><joint name="hinge" axis="0 1 0" range="0 1"/><geom size=".1"/><site name="handle"/></body></worldbody></mujoco>')
        basis={k:{'kind':'assumption','reference':'developer hinge injection'} for k in ('axis','joint','range')}
        interface={'id':'open','body':'door','site':'handle','operation':{'status':'declared','axis':[0,0,1],'joint':'hinge','range':[0,1],'basis':basis}}
        with self.assertRaises(ValidationError):validate_operation(interface,m,{'open'})
        interface['operation']['axis']=[0,1,0];validate_operation(interface,m,{'open'})
    def test_other_arm_and_local_obstacle_are_in_collision_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,_,_=self.session(Path(tmp));before=s.data.qpos.copy()
            _,_,valid=s.query(['robot__left_y','robot__right_y'])
            self.assertFalse(valid(np.array([-.2,.2])));np.testing.assert_array_equal(s.data.qpos,before)
        with tempfile.TemporaryDirectory() as tmp:
            def obstruct(xml):return xml.replace('</worldbody>','<body name="obstacle"><geom pos=".1 .2 .5" size=".04 .08 .05" type="box"/></body></worldbody>')
            s,_,_=self.session(Path(tmp),change=obstruct);before=s.data.qpos.copy()
            _,_,valid=s.query(['robot__left_x']);self.assertFalse(valid(np.array([.1])));np.testing.assert_array_equal(s.data.qpos,before)
    def test_blocked_insertion_has_evidence_without_automatic_asset_blame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def block(xml):return xml.replace('pos=".6 0 .03"','pos=".6 0 .3"').replace('</worldbody>','<body name="receiver__cover"><geom name="receiver__blocked_slot" type="box" pos=".6 0 .2" size=".08 .08 .01"/><site name="receiver__entry" pos=".6 0 .3"/></body></worldbody>')
            source=fixture(root/'source');p=source/'model.xml';p.write_text(block(p.read_text()),encoding='utf-8')
            meta=read(source/'scene.json');m=mujoco.MjModel.from_xml_path(str(p));meta['home_qpos']=mujoco.MjData(m).qpos.tolist();meta['plan']['instances'].append({'id':'receiver','job':'receiver','placement':'floor'});atomic_json(source/'scene.json',meta)
            manifest=read(source/'interfaces.json');manifest['interfaces'].append({'id':'receiver__slot','body':'receiver__cover','site':'receiver__entry','kind':'insert','operation':{'axis':[0,0,-1]}});atomic_json(source/'interfaces.json',manifest)
            result=prepare(source,root/'dynamic',identity(source),Settings(settle_seconds=.1));sha=result['scene_sha256']
            report=inspect_scene(root/'dynamic',sha,queries=[{'kind':'fit','interface':'receiver__slot','body':'object__body','distance':.2,'samples':9}]);row=report['rows'][-1]
            self.assertEqual(row['status'],'confirmed_observation');self.assertFalse(row['repair_authorized']);self.assertTrue(any(x['collisions'] for x in row['samples']));self.assertEqual(identity(root/'dynamic'),sha)
    def test_release_without_designated_support_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            def no_support(xml):return xml.replace('name="support" pos="0 .2 .4"','name="support" pos="2 2 .4"')
            s,_,_=self.session(Path(tmp),True,no_support);s.deadline=3
            s.allowed=[['robot__left_finger_body1','object__body'],['robot__left_finger_body2','object__body']]
            s.grasp({'arm':'left','body':'object__body','controls':{'robot__left_grip1':.016,'robot__left_grip2':.016}})
            with self.assertRaises(OperationError) as error:s.place({'arm':'left','body':'object__body','support':'support','controls':{'robot__left_grip1':0,'robot__left_grip2':0}})
            self.assertIn(error.exception.code,{'timeout','collision'});self.assertFalse(s.evidence.evaluate({'kind':'supported','body':'object__body','support':'support'}))
