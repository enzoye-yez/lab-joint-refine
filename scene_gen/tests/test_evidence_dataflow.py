"""Developer-authored regression fixtures; no real provider requests or asset edits."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch,MagicMock
import subprocess
from asset_prep.storage import Store,encoded,digest
from asset_prep.config import Config
from asset_prep.errors import ContextLimitError,ProviderError,PrepError,ValidationError
from asset_generation.contracts import GenerationConfig
from asset_generation.model import GenerationModel
from asset_generation.transport import GenerationTransport
from scene_generation.context import ContextArchive
from scene_generation.evidence_state import session,merge_checks,action_key
from scene_generation.evidence import review_with_evidence
from scene_generation.recovery import decide
from scene_generation.review import scoped_review

ACCEPT={"verdict":"accept","findings":[],"unverified":[]}
UNCERTAIN={"verdict":"uncertain","findings":[],"unverified":[],"blocking":["support is not measured"]}
SUPPORT={"kind":"support","instance":"device"}
UNKNOWN={"id":"support:device","status":"partial","complete":False,"measurement_error_count":1}
PASS={"id":"support:device","status":"pass","complete":True,"component_scope":"whole_instance"}

class DataflowTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.artifact=self.root/"artifact";self.artifact.mkdir()
        (self.artifact/"geometry.txt").write_text("developer fixture")
        self.calls=[];self.inspections=[]
    def run_review(self,replies,report,*,model="A",row=PASS,extra=None):
        outer=self
        class Model:
            def ask(inner,instruction,context,images,phase):
                outer.calls.append(deepcopy(context));value=replies.pop(0)
                if isinstance(value,BaseException):raise value
                return value,{"synthetic":True}
        class Backend:
            def inspect_artifact(inner,artifact,destination,scope,requests):
                outer.inspections.append(deepcopy(requests))
                destination.mkdir(parents=True,exist_ok=True)
                if requests[0]["kind"]=="inventory":return {"status":"pass","inventory":{"bodies":["base"]}},None
                evidence={"status":"pass" if row["complete"] else "incomplete","results":[{"request":requests[0],"checks":[deepcopy(row)],"complete":row["complete"]}]}
                return evidence,None
        state={}
        run=SimpleNamespace(root=self.root,state=state,store=SimpleNamespace(state={"identity":{"implementation":"synthetic","model":model}},save=lambda:None),
            model=Model(),sandbox=Backend(),generation=GenerationConfig(),check_pause=lambda:None,
            limits=SimpleNamespace(max_evidence_rounds=3,max_evidence_plan_attempts=2),images=lambda *args:[])
        context={"checks":report,**(extra or {})}
        review,call=review_with_evidence(run,"synthetic",context,[],"scene_review",self.root,
            lambda v:scoped_review(v,"scene",instance_ids={"device"}),self.artifact,"scene")
        return review,run

    def test_full_support_evidence_updates_main_decision(self):
        report={"status":"partial","checks":[deepcopy(UNKNOWN)]}
        review,_=self.run_review([UNCERTAIN,{"requests":[SUPPORT]},ACCEPT],report)
        self.assertEqual(report["status"],"pass")
        self.assertEqual(decide(report,review),"candidate_ready")

    def test_local_evidence_cannot_clear_other_check_or_component(self):
        report={"status":"partial","checks":[deepcopy(UNKNOWN),{"id":"other","status":"partial","complete":False}]}
        review,_=self.run_review([UNCERTAIN,{"requests":[SUPPORT]},ACCEPT],report)
        self.assertEqual(report["checks"][0]["status"],"pass")
        self.assertEqual(report["checks"][1]["status"],"partial")
        self.assertEqual(decide(report,review),"needs_check_support")
        subset={"checks":[{**PASS,"component_scope":"selected_geoms"}],"input_key":"current"}
        self.assertEqual(merge_checks({"checks":[UNKNOWN]},subset,scope="scene",input_key="current")["checks"],[UNKNOWN])
        with self.assertRaises(ValidationError):merge_checks({"checks":[UNKNOWN]},subset,scope="scene",input_key="changed")

    def test_resume_and_model_switch_reuse_numeric_without_review_reuse(self):
        report={"status":"partial","checks":[deepcopy(UNKNOWN)]}
        with self.assertRaises(KeyboardInterrupt):
            self.run_review([UNCERTAIN,{"requests":[SUPPORT]},KeyboardInterrupt()],report)
        before=deepcopy(self.inspections)
        restored={"status":"partial","checks":[deepcopy(UNKNOWN)]}
        review,_=self.run_review([ACCEPT],restored,model="B")
        self.assertEqual(self.inspections,before)
        self.assertEqual(restored["status"],"pass")
        self.assertEqual(self.calls[-1]["checks"]["status"],"pass")
        self.assertEqual(review["verdict"],"accept")

    def test_completed_unknown_not_repeated_by_reason_rewrite(self):
        report={"status":"partial","checks":[deepcopy(UNKNOWN)]}
        requests=[UNCERTAIN,{"requests":[SUPPORT]},UNCERTAIN,{"requests":[SUPPORT]},
            {"requests":[],"blocked":{"kind":"unsupported_check","detail":"no applicable remaining method"}}]
        review,run=self.run_review(requests,report,row=UNKNOWN)
        self.assertEqual(len(self.inspections),2)
        self.assertEqual(review["verdict"],"uncertain")
        self.assertNotEqual(report["status"],"pass")
        self.assertEqual(action_key({**SUPPORT,"question":"old"}),action_key({**SUPPORT,"question":"new"}))
        self.assertNotEqual(action_key(SUPPORT),action_key({**SUPPORT,"object_geoms":["flange"]}))

    def test_changed_reviewer_cannot_clear_persisted_unknown_by_accepting(self):
        report={"status":"partial","checks":[deepcopy(UNKNOWN)]}
        with self.assertRaises(KeyboardInterrupt):
            self.run_review([UNCERTAIN,{"requests":[SUPPORT]},KeyboardInterrupt()],report,row=UNKNOWN)
        # Three invalid accepts become uncertain; interrupted review also cannot
        # accept the saved unknown; explicit capability stop preserves it.
        blocked={"requests":[],"blocked":{"kind":"unsupported_check","detail":"no supported alternative"}}
        review,run=self.run_review([ACCEPT,ACCEPT,ACCEPT,UNCERTAIN,blocked],report,model="B",row=UNKNOWN)
        self.assertEqual(review["verdict"],"uncertain")
        self.assertEqual(report["status"],"partial")
        self.assertEqual(len(self.inspections),2)

    def test_order_changes_keep_tool_identity_but_geometry_changes_do_not(self):
        run=SimpleNamespace(root=self.root,store=SimpleNamespace(state={"identity":{"implementation":"same"}}))
        a,_=session(run,self.artifact,"p","scene",{"checks":{"checks":[{"id":"a"},{"id":"b"}]}},self.root)
        b,_=session(run,self.artifact,"p","scene",{"checks":{"checks":[{"id":"b"},{"id":"a"}]}},self.root)
        self.assertEqual(a["input_key"],b["input_key"])
        (self.artifact/"geometry.txt").write_text("changed pose")
        c,_=session(run,self.artifact,"p","scene",{},self.root)
        self.assertNotEqual(a["input_key"],c["input_key"])

    def test_view_or_context_query_cannot_replace_numerical_checks(self):
        report={"status":"partial","checks":[UNKNOWN]}
        value=merge_checks(report,{"kind":"details","input_key":"current","complete":True},scope="scene",input_key="current")
        self.assertEqual(value["status"],"partial")

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def test_full_details_are_paginated_without_erasing_unknowns(self):
        archive=ContextArchive(self.root,"binding",12000)
        rows=[{"geom1":f"a{i}","geom2":"b","detail":"unknown"*50} for i in range(1500)]
        ref=archive.reference(rows)
        projection=archive.project({"checks":{"status":"partial","coverage_gaps":rows},"inventory":{"geoms":["x"*100]*10000}})
        self.assertLessEqual(len(encoded(projection)),12000)
        self.assertEqual(projection["checks"]["status"],"partial")
        recovered=[];offset=0
        while True:
            result=archive.inspect({"kind":"details","reference":ref,"offset":offset,"limit":32})
            recovered.extend(result["data"])
            if result["next_offset"] is None:break
            offset=result["next_offset"]
        self.assertEqual(recovered,rows)
        with self.assertRaises(ValidationError):ContextArchive(self.root,"different").inspect({"kind":"details","reference":ref})
        with self.assertRaises(ValidationError):archive.inspect({"kind":"details","reference":"../state"})
    def test_all_check_identities_remain_indexed_with_large_reports(self):
        archive=ContextArchive(self.root,"binding",120000)
        rows=[{"id":f"check:{i}","status":"partial","complete":False,"coverage_gaps":[{"detail":"unknown"*200}]*100} for i in range(133)]
        result=archive.project({"checks":{"status":"partial","checks":rows,"metadata":{"a":"x"*10000}},"plan":{"source":"z"*100000}})
        self.assertEqual({r["id"] for r in result["check_index"]},{r["id"] for r in rows})
        self.assertLessEqual(len(encoded(result)),120000)
        for r in result["check_index"]:self.assertTrue((archive.root/(r["evidence_ref"]+".json")).is_file())

    def test_nested_large_item_remains_retrievable(self):
        archive=ContextArchive(self.root,"binding",8000)
        value={"nested":{"text":"large"*20000}}
        ref=archive.reference(value)
        page=archive.inspect({"kind":"details","reference":ref})
        self.assertTrue(page["data"][0]["value"]["text"]["omitted"])
        self.assertTrue(page["complete"])
    def test_multi_round_context_stays_bounded(self):
        archive=ContextArchive(self.root,"binding",120000)
        for count in (1,4,20):
            context={"checks":{"status":"partial","checks":[{"id":"unknown","status":"partial","coverage_gaps":[{"detail":"x"*500}]*2000}]},
                "history":[{"results":[{"detail":"y"*10000}]*20}]*count}
            self.assertLessEqual(len(encoded(archive.project(context))),120000)

class RequestBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(Path(self.tmp.name),{},Config(max_model_calls=20,max_reserved_tokens=10000000))
    def make(self,transport):
        return GenerationModel(Config(),GenerationConfig(),{"HARNESS_MODEL_API_KEY":"synthetic"},self.store,transport)
    def test_context_limit_is_not_an_asset_error_and_is_never_sent(self):
        transport=MagicMock();model=self.make(transport)
        model.generation=replace(model.generation,max_context_tokens=50000)
        with self.assertRaises(ContextLimitError):model.ask("test",{"payload":"x"*100000},[],"scene_review")
        transport.post.assert_not_called();self.assertEqual(self.store.state["calls"],[])
    def test_exact_observed_limit_error_is_never_retried(self):
        transport=MagicMock();transport.post.side_effect=ProviderError("rejected",diagnostics={"http_status":400,"provider_error":{
            "message":"Input tokens exceed the configured limit of 922000 tokens. Your messages resulted in 1614803 tokens."}})
        model=self.make(transport);model.generation=replace(model.generation,max_unknown_http_retries=2,max_transport_retries=30)
        with patch("asset_generation.model.time.sleep") as sleep,self.assertRaises(ProviderError):model.ask("test",{},[],"scene_review")
        with self.assertRaises(ContextLimitError):model.ask("test",{},[],"scene_review")
        self.assertEqual(transport.post.call_count,1);sleep.assert_not_called()
        self.assertEqual(self.store.state["calls"][0]["failure_kind"],"context_limit")
    def test_pause_during_cooldown_cannot_send_second_request(self):
        transport=MagicMock();transport.post.side_effect=ProviderError("unknown",uncertain=True,diagnostics={"failure":"timeout"})
        model=self.make(transport);paused=[False]
        def guard():
            if paused[0]:raise PrepError("paused")
        model.execution_guard=guard
        with patch("asset_generation.model.time.sleep",side_effect=lambda _:paused.__setitem__(0,True)),self.assertRaises(PrepError):
            model.ask("test",{},[],"scene_review")
        self.assertEqual(transport.post.call_count,1)
        self.assertEqual(self.store.state["calls"][0]["status"],"pending")
    def test_inflight_pause_terminates_owned_child(self):
        child=MagicMock();child.args=[]
        child.communicate.side_effect=subprocess.TimeoutExpired([],1)
        transport=GenerationTransport();count=[0]
        def guard():
            count[0]+=1
            if count[0]==3:raise PrepError("paused")
        transport.check_cancel=guard
        with patch("asset_generation.transport.subprocess.Popen",return_value=child),patch("asset_generation.process_tree.ProcessTree") as tree,self.assertRaises(PrepError):
            transport.post("https://example.invalid",{},"synthetic",10)
        tree.return_value.attach.assert_called_once_with(child)
        tree.return_value.stop.assert_called_once_with(child)
        self.assertEqual(child.communicate.call_count,1)

class SurfaceCertificateTests(unittest.TestCase):
    def compiled(self,x=0,z=.01):
        import mujoco,trimesh
        from asset_generation.static_state import geometry_state
        ring=trimesh.creation.annulus(r_min=.02,r_max=.04,height=.01,sections=24)
        vertices=" ".join(map(str,ring.vertices.reshape(-1)))
        faces=" ".join(map(str,ring.faces.reshape(-1)))
        xml=f'<mujoco><asset><mesh name="ring" vertex="{vertices}" face="{faces}"/></asset><worldbody><body name="rack__base"><geom name="rack__ring" type="mesh" mesh="ring"/></body><body name="tube__body"><geom name="tube__base" type="box" size=".003 .003 .005" pos="{x} 0 {z}"/></body></worldbody></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);geometry_state(m,d)
        return m,d
    def test_triangle_surface_distance_is_scale_invariant_including_tiny_facets(self):
        import numpy as np
        from asset_generation.distance import triangle_surface_distance
        triangle=np.array([[[0.,0,0],[1.,0,0],[0.,1.,0]]])
        for scale in (1e-6,1e-4,.1,10.):
            for point,expected in (([.2,.3,0],0.),([.2,.3,.02],.02),([.8,.8,0],.6/np.sqrt(2))):
                measured=triangle_surface_distance(triangle*scale,np.array(point)*scale)
                self.assertAlmostEqual(measured/scale,expected,places=12)
        collapsed=np.array([[[0.,0,0],[0.,0,0],[1.,0,0]]])
        self.assertAlmostEqual(triangle_surface_distance(collapsed,np.array([.5,.2,0])),.2)

    def test_compiled_tiny_mesh_face_interior_has_zero_surface_error(self):
        import mujoco,trimesh,numpy as np
        from asset_generation.distance import DistanceQuery
        from asset_generation.static_state import geometry_state
        mesh=trimesh.creation.box(extents=[.0001,.0002,.0001])
        vertices=" ".join(map(str,mesh.vertices.reshape(-1)));faces=" ".join(map(str,mesh.faces.reshape(-1)))
        model=mujoco.MjModel.from_xml_string(f'<mujoco><asset><mesh name="tiny" vertex="{vertices}" face="{faces}"/></asset><worldbody><geom type="mesh" mesh="tiny" pos=".3 .4 .8"/></worldbody></mujoco>')
        data=mujoco.MjData(model);geometry_state(model,data);q=DistanceQuery(model,data)
        for triangle in q.vertices(0)[1]:
            point=triangle.mean(axis=0)@data.geom_xmat[0].reshape(3,3).T+data.geom_xpos[0]
            self.assertLess(q.surface_error(0,point),1e-12)

    def test_hull_touch_inside_ring_hole_is_not_actual_support(self):
        from asset_generation.distance import DistanceQuery
        from asset_generation.surface_contact import contact
        m,d=self.compiled();q=DistanceQuery(m,d)
        self.assertIsNone(contact(q,0,1,[0,0,1]))
    def test_original_ring_patch_proves_contact_at_unchanged_tolerance(self):
        from asset_generation.distance import DistanceQuery
        from asset_generation.surface_contact import contact
        from asset_generation.material import visible_witness
        import numpy as np
        m,d=self.compiled(x=.03);q=DistanceQuery(m,d)
        result=contact(q,0,1,[0,0,1])
        self.assertIsNotNone(result)
        self.assertEqual(result["method"],"original_surface_patches")
        self.assertLessEqual(abs(result["distance_m"]),1e-5)
        self.assertIsNotNone(visible_witness(m,d,q,0,np.array(result["points"][:3])))
        self.assertIsNotNone(visible_witness(m,d,q,1,np.array(result["points"][3:])))
    def test_actual_gap_cannot_be_converted_into_contact(self):
        from asset_generation.distance import DistanceQuery
        from asset_generation.surface_contact import contact
        m,d=self.compiled(x=.03,z=.01003)
        self.assertIsNone(contact(DistanceQuery(m,d),0,1,[0,0,1]))
    def test_open_or_concave_mesh_cannot_supply_interior_ball(self):
        from asset_generation.distance import DistanceQuery
        import numpy as np
        m,d=self.compiled();self.assertIsNone(DistanceQuery(m,d).interior_radius(0,np.zeros(3)))
    def union_case(self,z=.005,separate=False):
        import mujoco,trimesh
        from asset_generation.distance import DistanceQuery,MeasurementError
        from asset_generation.static_state import geometry_state
        from asset_generation.material import integrity
        mesh=trimesh.creation.annulus(r_min=.02,r_max=.04,height=.01,sections=16)
        vertices=" ".join(map(str,mesh.vertices.reshape(-1)));faces=" ".join(map(str,mesh.faces.reshape(-1)))
        roof=f'<geom name="roof" type="cylinder" size=".05 .001" pos="0 0 {z}"/>'
        if separate:roof='<body name="separate">'+roof+'</body>'
        m=mujoco.MjModel.from_xml_string(f'<mujoco><asset><mesh name="ring" vertex="{vertices}" face="{faces}"/></asset><worldbody><body name="base"><geom name="ring" type="mesh" mesh="ring"/>{roof}</body></worldbody></mujoco>')
        d=mujoco.MjData(m);geometry_state(m,d);q=DistanceQuery(m,d,recovery=True)
        with patch.object(q,"measure",side_effect=MeasurementError("synthetic unavailable distance")):
            return integrity(m,d,query=q)
    def test_actual_original_surface_intersection_proves_same_body_union(self):
        result=self.union_case()
        self.assertEqual(result["status"],"pass",result)
        self.assertTrue(result["connection_witnesses"])
    def test_union_witness_cannot_clear_cross_body_collision_unknown(self):
        result=self.union_case(separate=True)
        self.assertEqual(result["status"],"partial")
        self.assertTrue(result["coverage_gaps"])
    def test_union_witness_cannot_bridge_positive_gap(self):
        result=self.union_case(z=.008)
        self.assertNotEqual(result["status"],"pass")
        self.assertFalse(result["connection_witnesses"])

    def test_convex_closed_mesh_can_certify_real_intersection(self):
        import mujoco,numpy as np,trimesh
        from asset_generation.distance import DistanceQuery
        from asset_generation.static_state import geometry_state
        box=trimesh.creation.box(extents=[.1,.1,.1])
        vertices=" ".join(map(str,box.vertices.reshape(-1)));faces=" ".join(map(str,box.faces.reshape(-1)))
        m=mujoco.MjModel.from_xml_string(f'<mujoco><asset><mesh name="box" vertex="{vertices}" face="{faces}"/></asset><worldbody><geom type="mesh" mesh="box"/><geom type="sphere" size=".02"/></worldbody></mujoco>')
        d=mujoco.MjData(m);geometry_state(m,d);q=DistanceQuery(m,d,recovery=True)
        result=q.overlap_certificate(0,1)
        self.assertIsNotNone(result)
        self.assertGreater(result["penetration_lower_bound_m"],.0005)
        self.assertLess(q.interior_radius(0,np.array([.1,0,0])),0)

if __name__=="__main__":unittest.main()
