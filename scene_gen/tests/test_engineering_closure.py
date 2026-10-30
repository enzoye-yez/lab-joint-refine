"""Developer-authored counterexamples; no provider requests or generated-asset edits."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mujoco
from asset_generation.static_state import geometry_state
from asset_generation.material import integrity
from asset_generation.checks import validate_artifact
from asset_generation.contracts import check_profile
from asset_generation.evidence import inspect_artifact,requests_valid
from asset_generation.configurations import selection,catalogue
from asset_generation.pipeline import snapshot
from asset_prep.errors import ValidationError
from scene_generation.evidence_state import action_key,coverage_key,merge_checks
from scene_generation.contracts import validate_plan,SceneConfig
from scene_generation.assembly import assemble
from scene_fixtures import plan,environment,write_component

def model_xml(bodies,equality=""):
    return '<mujoco><compiler angle="radian"/><worldbody>'+bodies+'</worldbody>'+equality+'</mujoco>'

def probe(bodies,equality=""):
    model=mujoco.MjModel.from_xml_string(model_xml(bodies,equality))
    data=mujoco.MjData(model);geometry_state(model,data)
    return integrity(model,data)

def manifest(body="base"):
    return {"schema_version":"1.0","asset_ids":["A01"],"parts":[{"id":body,"body":body}],
            "interfaces":[],"assumptions":["developer authored"],"approximations":[],"buttons":[],"connections":[]}

BASE='<body name="base"><geom name="base_g" type="box" size=".1 .1 .05" pos="0 0 .05"/>{child}</body>'
FLOAT='<geom name="floating" type="box" size=".02 .02 .02" pos="0 0 .3"/>'

class MaterialTests(unittest.TestCase):
    def test_same_body_floating_part_is_not_supported_by_membership(self):
        r=probe(BASE.format(child=FLOAT))
        self.assertEqual(r["status"],"fail")
        self.assertEqual(r["disconnected_count"],2)
    def test_body_rename_does_not_change_verdict(self):
        x=BASE.format(child=FLOAT)
        self.assertEqual(probe(x)["status"],probe(x.replace("base","renamed"))["status"])
    def test_parent_frame_cannot_bridge_empty_space(self):
        r=probe(BASE.format(child='<body name="child" pos="0 0 .3"><geom name="child_g" type="box" size=".02 .02 .02"/></body>'))
        self.assertTrue(r["disconnected_components"])
    def test_weld_cannot_bridge_empty_space(self):
        r=probe(BASE.format(child="")+'<body name="cap" pos="0 0 .3"><freejoint/><geom name="cap_g" type="box" size=".02 .02 .02"/></body>',
                '<equality><weld body1="base" body2="cap"/></equality>')
        self.assertTrue(r["disconnected_components"])
    def test_actual_joint_contact_is_valid(self):
        r=probe(BASE.format(child='<body name="lid" pos="0 0 .125"><joint axis="0 1 0"/><geom name="lid_g" type="box" size=".05 .05 .025"/></body>'))
        self.assertEqual(r["status"],"pass",r)
    def test_same_body_constructive_union_is_not_collision(self):
        r=probe(BASE.format(child='<geom name="union" type="box" size=".03 .03 .04" pos="0 0 .11"/>'))
        self.assertEqual(r["status"],"pass",r)
    def test_filtered_distinct_body_intersection_is_found(self):
        r=probe(BASE.format(child='<body name="intersect"><geom name="intersect_g" type="box" size=".03 .03 .02" pos="0 0 .05" contype="0" conaffinity="0"/></body>'))
        self.assertGreater(r["collision_count"],0,r)
    def test_invisible_collision_bridge_does_not_hide_visual_gap(self):
        r=probe(BASE.format(child=FLOAT+'<geom name="bridge" type="box" size=".01 .01 .15" pos="0 0 .15" rgba="1 1 1 0"/>'))
        self.assertTrue(r["disconnected_components"])
    def test_collision_protrudes_below_visible_material(self):
        r=probe('<body name="b"><geom name="visible" type="box" size=".1 .1 .05" pos="0 0 .15" contype="0" conaffinity="0"/><geom name="hidden" type="box" size=".1 .1 .1" pos="0 0 .1" rgba="1 1 1 0"/></body>')
        self.assertTrue(r["surface_mismatches"])
    def test_budget_unknown_never_passes(self):
        r=probe(BASE.format(child='<geom name="touch" type="box" size=".03 .03 .03" pos="0 0 .13"/>'))
        m=mujoco.MjModel.from_xml_string(model_xml(BASE.format(child='<geom name="touch" type="box" size=".03 .03 .03" pos="0 0 .13"/>')))
        d=mujoco.MjData(m);geometry_state(m,d)
        self.assertEqual(integrity(m,d,max_pairs=0)["status"],"partial")

class VariantObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.artifact=self.root/"asset";self.artifact.mkdir()
        (self.artifact/"model.xml").write_text(model_xml(BASE.format(child="")),encoding="utf-8")
        (self.artifact/"interfaces.json").write_text(json.dumps(manifest()),encoding="utf-8")
    def view(self,**kwargs):
        return {"kind":"view","joints":{},"hide_bodies":[],"focus_body":"base","azimuth":45,"elevation":90,**kwargs}
    def variants(self):
        m=manifest();m.update(configurations={},selected_configuration="small")
        for name,size in (("small",.1),("large",.2)):
            (self.artifact/(name+".xml")).write_text(model_xml(BASE.format(child="").replace('size=".1 .1 .05"',f'size="{size} .1 .05"')),encoding="utf-8")
            (self.artifact/(name+".json")).write_text(json.dumps(manifest()),encoding="utf-8")
            m["configurations"][name]={"file":name+".xml","interfaces_file":name+".json"}
        (self.artifact/"interfaces.json").write_text(json.dumps(m),encoding="utf-8")
    def test_bottom_view_and_markers_are_supported(self):
        before=snapshot(self.artifact)
        with patch("mujoco.mj_forward",side_effect=AssertionError("global dynamics forbidden")),patch("mujoco.mj_step",side_effect=AssertionError("no time step")):
            r=inspect_artifact(self.artifact,"asset",[self.view(hide_sites=True,zoom=1)],self.root/"view")
        self.assertTrue(r["results"][0]["quality"]["observable"],r)
        self.assertEqual(before,snapshot(self.artifact))
    def test_occluded_target_cannot_count_as_observed(self):
        (self.artifact/"model.xml").write_text(model_xml(BASE.format(child='<geom name="inside" type="sphere" size=".005" pos="0 0 .05"/>')),encoding="utf-8")
        r=inspect_artifact(self.artifact,"asset",[self.view(focus_geoms=["inside"])],self.root/"hidden")
        self.assertEqual(r["status"],"incomplete")
        self.assertFalse(r["results"][0]["quality"]["observable"])
    def test_all_variants_are_checked_and_selected_explicitly(self):
        self.variants()
        r=validate_artifact(self.artifact,["A01"],check_profile({"validation_mode":"static"}))
        self.assertEqual(set(r["configurations"]),{"small","large"})
        self.assertEqual(r["status"],"pass",r)
        with self.assertRaises(ValidationError):selection(self.artifact)
        _,filename,_=selection(self.artifact,"large");self.assertEqual(filename,"large.xml")
        r=inspect_artifact(self.artifact,"asset",[{"kind":"inventory"}],self.root/"inventory")
        self.assertEqual(set(r["inventory"]["configurations"]),{"small","large"})
    def test_variant_views_use_the_requested_file(self):
        self.variants()
        r=inspect_artifact(self.artifact,"asset",[self.view(configuration="large")],self.root/"large")
        self.assertEqual(r["status"],"pass",r)
        self.assertEqual(r["results"][0]["request"]["configuration"],"large")
    def test_variant_path_escape_is_rejected(self):
        self.variants()
        m=json.loads((self.artifact/"interfaces.json").read_text())
        m["configurations"]["small"]["file"]="../model.xml"
        (self.artifact/"interfaces.json").write_text(json.dumps(m))
        with self.assertRaises(ValidationError):catalogue(self.artifact)
    def test_static_asset_validation_does_not_call_dynamics(self):
        with patch("mujoco.mj_forward",side_effect=AssertionError("no dynamics")):
            r=validate_artifact(self.artifact,["A01"],check_profile({"validation_mode":"static"}))
        self.assertEqual(r["status"],"pass",r)

class CoverageTests(unittest.TestCase):
    def test_action_dedup_normalizes_equivalent_cameras(self):
        a={"kind":"view","joints":{},"hide_bodies":[],"focus_body":"b","azimuth":0,"elevation":90}
        self.assertEqual(action_key(a),action_key({**a,"azimuth":360,"question":"rewritten_reason","zoom":1,"hide_sites":True,"diagnostic_light":True}))
        self.assertNotEqual(action_key(a),action_key({**a,"elevation":30}))
    def test_alternative_observation_covers_only_same_question_and_target(self):
        a={"kind":"view","question":"issue_1","focus_body":"base","azimuth":0}
        self.assertEqual(coverage_key(a),coverage_key({**a,"azimuth":90}))
        self.assertNotEqual(coverage_key(a),coverage_key({**a,"focus_body":"lid"}))
    def test_partial_result_cannot_wash_other_checks_or_known_defect(self):
        report={"validation_mode":"static","checks":[{"id":"a","status":"partial","complete":False},
                {"id":"b","status":"partial","complete":False},{"id":"c","status":"fail"}]}
        r=merge_checks(report,{"checks":[{"id":"a","status":"pass","complete":True},{"id":"c","status":"pass","complete":True}]})
        self.assertEqual([x["status"] for x in r["checks"]],["pass","partial","fail"])
    def test_unrelated_result_is_rejected(self):
        with self.assertRaises(ValidationError):
            merge_checks({"checks":[{"id":"a","status":"partial"}]},{"checks":[{"id":"other","status":"pass","complete":True}]})


class ClosureBoundaryTests(unittest.TestCase):
    def test_joint_clearance_is_unknown_not_a_geometric_defect(self):
        r=probe(BASE.format(child='<body name="lid" pos="0 0 .126"><joint axis="0 1 0"/><geom name="lid_g" type="box" size=".05 .05 .025"/></body>'))
        self.assertEqual(r["status"],"partial",r)
        self.assertEqual(r["disconnected_count"],0)
        self.assertFalse(r["complete"])

    def test_hidden_collision_proxy_intersection_is_not_filtered_out(self):
        r=probe(BASE.format(child='<body name="child"><geom name="child_visible" type="box" size=".02 .02 .02" pos="0 0 .12"/><geom name="hidden_proxy" type="box" size=".02 .02 .02" pos="0 0 .1" rgba="1 1 1 0"/></body>'))
        self.assertTrue(any("hidden_proxy" in (c["geom1"],c["geom2"]) for c in r["collisions"]),r)

    def test_closed_mesh_certificate_rejects_open_or_concave_surface(self):
        import numpy as np
        from asset_generation.material import convex_solid
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        self.assertTrue(convex_solid(vertices,faces))
        self.assertFalse(convex_solid(vertices,faces[:-1]))
        # A dent in one tetrahedron face gives a closed, nonconvex surface.
        vertices=np.vstack([vertices,[.2,.2,.1]])
        dent=np.vstack([faces[1:],[[0,2,4],[2,1,4],[1,0,4]]])
        self.assertFalse(convex_solid(vertices,dent))

    def test_unknown_null_bound_survives_as_unknown(self):
        from asset_generation.distance import DistanceQuery,MeasurementError
        m=mujoco.MjModel.from_xml_string(model_xml(BASE.format(child='<geom name="touch" type="box" size=".03 .03 .03" pos="0 0 .13"/>')))
        d=mujoco.MjData(m);geometry_state(m,d)
        with patch.object(DistanceQuery,"measure",side_effect=MeasurementError("synthetic")):
            r=integrity(m,d)
        self.assertEqual(r["status"],"partial",r)
        self.assertEqual(r["measurement_error_count"],1)

    def test_partial_confirmed_defect_is_consumed_without_clearing_other_gap(self):
        r=merge_checks({"checks":[{"id":"a","status":"partial","complete":False},{"id":"b","status":"partial"}]},
                       {"checks":[{"id":"a","status":"fail","complete":False,"collisions":[{"geom1":"a","geom2":"b"}]}]})
        self.assertEqual(r["status"],"fail")
        self.assertEqual(r["checks"][1]["status"],"partial")

    def test_supplement_cannot_change_acceptance_tolerance(self):
        with self.assertRaises(ValidationError):
            merge_checks({"checks":[{"id":"a","status":"partial","tolerance_m":.0001}]},
                         {"checks":[{"id":"a","status":"pass","complete":True,"tolerance_m":.001}]})

    def test_material_diagnostics_cannot_be_hidden_in_pass(self):
        from asset_generation.check_report import validate_rows
        for key,value in (("disconnected_components",[{"geom1":"g"}]),("surface_mismatch_count",1)):
            with self.subTest(key=key),self.assertRaises(ValidationError):
                validate_rows([{"id":"material_integrity","status":"pass",key:value}])

class VariantAssemblyTests(VariantObservationTests):
    def test_different_selected_files_reach_actual_assembly(self):
        self.variants()
        p=plan();p["jobs"][0]["configurations"]=["small","large"]
        p["instances"][0]["configuration"]="small"
        p["instances"].append({**p["instances"][0],"id":"station_large","configuration":"large","position":[-.5,0,.8]})
        validate_plan(p,["A01"],SceneConfig(validation_mode="static"))
        env=environment(self.root/"robot")
        output=assemble(p,env,{"instrument":self.artifact,"tool":write_component(self.root/"tool",True)},self.root/"robot",self.root/"scene")
        from asset_generation.checks import load_model
        m=load_model(output,scope="scene")
        self.assertAlmostEqual(m.geom("station__base_g").size[0],.1)
        self.assertAlmostEqual(m.geom("station_large__base_g").size[0],.2)
        meta=json.loads((output/"scene.json").read_text(encoding="utf-8"))
        self.assertEqual({x["configuration"] for x in meta["component_versions"]},{"small","large","default"})
        with patch("mujoco.mj_forward",side_effect=AssertionError("no dynamics")):
            from scene_generation.checks import render_details
            (self.root/"details").mkdir()
            self.assertEqual(len(render_details(m,meta["home_qpos"],p,self.root/"details")),3)

    def test_missing_or_unknown_instance_selection_is_rejected(self):
        p=plan();p["jobs"][0]["configurations"]=["small","large"]
        with self.assertRaises(ValidationError):validate_plan(p,["A01"],SceneConfig())
        p["instances"][0]["configuration"]="missing"
        with self.assertRaises(ValidationError):validate_plan(p,["A01"],SceneConfig())

class AssetRecoveryTests(unittest.TestCase):
    def test_real_second_capability_changes_decision_and_receipts_prevent_repeat(self):
        from asset_generation.check_recovery import recover_checks
        from asset_generation.inspection_registry import REGISTRY,MaterialInspection
        from asset_generation.distance import DistanceQuery,MeasurementError
        class Ineffective(MaterialInspection):
            id="synthetic_ineffective"
            def inspect(self,model,data,request,root,scope,output,number):
                row=integrity(model,data)
                return {"kind":self.id,"request":request,"checks":[row],"complete":False}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);asset=root/"asset";asset.mkdir()
            (asset/"model.xml").write_text(model_xml('<body name="base"><geom name="ball" type="sphere" size=".1"/><body name="cap" pos="0 0 .15"><geom name="cap_ball" type="sphere" size=".05"/></body></body>'),encoding="utf-8")
            (asset/"interfaces.json").write_text(json.dumps(manifest()),encoding="utf-8")
            m=mujoco.MjModel.from_xml_path(str(asset/"model.xml"));d=mujoco.MjData(m);geometry_state(m,d)
            calls=[]
            class Backend:
                def inspect_artifact(self,a,destination,scope,requests):
                    calls.append(requests[0]["kind"])
                    return inspect_artifact(a,scope,requests,destination),None
            with patch.dict(REGISTRY,{"synthetic_ineffective":Ineffective(),"material":MaterialInspection()},clear=True), \
                 patch.object(DistanceQuery,"measure",side_effect=MeasurementError("synthetic primary failure")):
                report={"status":"partial","validation_mode":"static","checks":[integrity(m,d)]}
                result=recover_checks(Backend(),asset,report,root/"recovery",implementation="test")
                self.assertEqual(result["status"],"pass",result)
                self.assertEqual(calls,["synthetic_ineffective","material"])
                replay=recover_checks(Backend(),asset,report,root/"recovery",implementation="test")
                self.assertEqual(replay["status"],"pass")
                self.assertEqual(len(calls),2)
                # A new input version cannot reuse the old success receipt.
                with (asset/"model.xml").open("a",encoding="utf-8") as f:f.write("\n<!-- new version -->")
                recover_checks(Backend(),asset,report,root/"recovery",implementation="test")
                self.assertEqual(len(calls),4)

    def test_interruption_retains_completed_action_before_next_decision(self):
        from asset_generation.check_recovery import recover_checks
        from asset_generation.inspection_registry import REGISTRY,MaterialInspection
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);asset=root/"asset";asset.mkdir();(asset/"marker").write_text("same input")
            row={"id":"material_integrity","status":"partial","complete":False,"tolerance_m":.0005}
            calls=[]
            class Backend:
                def inspect_artifact(self,a,destination,scope,requests):
                    calls.append(requests[0])
                    return {"status":"incomplete","results":[{"request":requests[0],"checks":[row],"complete":False}],"renders":[]},None
            n=[0]
            def pause():
                n[0]+=1
                if n[0]==2:raise KeyboardInterrupt()
            with patch.dict(REGISTRY,{"material":MaterialInspection()},clear=True):
                with self.assertRaises(KeyboardInterrupt):
                    recover_checks(Backend(),asset,{"checks":[row]},root,implementation="test",check_pause=pause)
                result=recover_checks(Backend(),asset,{"checks":[row]},root,implementation="test")
            self.assertEqual(len(calls),1)
            self.assertEqual(result["status"],"partial")
            self.assertEqual(result["numerical_recovery"]["stop_reason"],"no_executable_action")


class RecoveryBoundaryTests(unittest.TestCase):
    def test_program_error_is_not_asset_feedback(self):
        from asset_generation.check_recovery import recover_checks
        from asset_generation.inspection_registry import REGISTRY,MaterialInspection
        class Broken:
            def inspect_artifact(self,*args):raise TypeError("synthetic checker bug")
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);asset=root/"asset";asset.mkdir()
            row={"id":"material_integrity","status":"partial","complete":False,"tolerance_m":.0005}
            with patch.dict(REGISTRY,{"material":MaterialInspection()},clear=True):
                result=recover_checks(Broken(),asset,{"status":"partial","checks":[row]},root)
            self.assertEqual(result["status"],"error")
            self.assertEqual(result["execution_failure"]["kind"],"program")
            self.assertEqual(result["checks"][0]["status"],"partial")

    def test_redundant_unknown_union_pair_is_resolved_by_measured_connection_path(self):
        from asset_generation.distance import DistanceQuery,MeasurementError
        m=mujoco.MjModel.from_xml_string(model_xml('<body name="b"><geom name="a" type="box" size=".1 .1 .1"/><geom name="b" type="box" size=".1 .1 .1" pos=".09 0 0"/><geom name="bridge" type="box" size=".08 .1 .1" pos=".045 0 0"/></body>'))
        d=mujoco.MjData(m);geometry_state(m,d)
        class Query(DistanceQuery):
            def measure(self,a,b,limit=.05):
                if {self.model.geom(a).name,self.model.geom(b).name}=={"a","b"}:
                    raise MeasurementError("synthetic redundant pair")
                return super().measure(a,b,limit)
        result=integrity(m,d,query=Query(m,d,recovery=True))
        self.assertEqual(result["status"],"pass",result)
        self.assertEqual(result["measurement_error_count"],0)

class UnknownAssetResumeTests(unittest.TestCase):
    from test_scene_pipeline import ScenePipelineTests as _Cases
    setUp=_Cases.setUp
    run_loop=_Cases.run_loop
    def test_resume_unknown_keeps_asset_ledger_and_never_calls_author(self):
        from scene_fixtures import SceneSandbox
        class Unknown(SceneSandbox):
            def build_and_check(self,*args):
                report,renders=super().build_and_check(*args)
                report.update(status="partial",checks=[{"id":"material_integrity","status":"partial","complete":False,"tolerance_m":.0005}])
                return report,renders
        result,_=self.run_loop(self.base_replies[:2],sandbox=Unknown())
        self.assertEqual(result["status"],"needs_check_support")
        before=json.loads((self.root/"run/state.json").read_text(encoding="utf-8"))
        artifact=self.root/"run"/before["scene"]["jobs"]["instrument"]["artifact"]
        geometry=snapshot(artifact)
        sandbox=Unknown()
        result,transport=self.run_loop([],resume=True,sandbox=sandbox)
        after=json.loads((self.root/"run/state.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"],"needs_check_support")
        self.assertEqual(sandbox.build_calls,0)
        self.assertEqual(transport.requests,[])
        self.assertEqual(after["calls"],before["calls"])
        self.assertEqual(snapshot(artifact),geometry)


class StaticBoundaryTests(unittest.TestCase):
    def test_static_scene_does_not_expand_or_limit_operation_targets(self):
        from scene_generation.checks import scene_checks
        from scene_generation.contracts import numerical_contract
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);env=environment(root/"robot")
            artifact=assemble(plan(),env,{"instrument":write_component(root/"asset"),"tool":write_component(root/"tool",True)},root/"robot",root/"scene")
            path=artifact/"interfaces.json";value=json.loads(path.read_text(encoding="utf-8"))
            value["interfaces"]=[{**value["interfaces"][0],"id":"operation_"+str(n)} for n in range(300)]
            path.write_text(json.dumps(value),encoding="utf-8")
            with patch("mujoco.mj_forward",side_effect=AssertionError("no dynamics")),patch("mujoco.mj_step",side_effect=AssertionError("no stepping")):
                result=scene_checks(artifact,SceneConfig(validation_mode="static"),numerical_contract({}))
            self.assertEqual(result["status"],"pass",result)
            self.assertEqual(next(r for r in result["checks"] if r["id"]=="robot_reachability")["status"],"skipped")

    def test_support_surface_index_invalidates_after_pose_change(self):
        import numpy as np
        from asset_generation.material import visible_witness
        from asset_generation.distance import DistanceQuery
        m=mujoco.MjModel.from_xml_string(model_xml('<body name="b"><freejoint/><geom name="g" type="sphere" size=".1"/></body>'))
        d=mujoco.MjData(m);geometry_state(m,d);q=DistanceQuery(m,d)
        self.assertIsNotNone(visible_witness(m,d,q,0,np.array([.1,0,0])))
        d.qpos[0]=1;geometry_state(m,d)
        self.assertIsNone(visible_witness(m,d,q,0,np.array([.1,0,0])))
        self.assertIsNotNone(visible_witness(m,d,q,0,np.array([1.1,0,0])))

class StandaloneUnknownTests(unittest.TestCase):
    from test_generation_pipeline import GenerationPipelineTests as _Cases
    setUp=_Cases.setUp
    run_loop=_Cases.run_loop
    def test_standalone_unknown_resume_does_not_author_or_modify_original(self):
        from generation_fixtures import FixtureSandbox,SOURCE
        class Unknown(FixtureSandbox):
            def build_and_check(self,*args):
                report,renders=super().build_and_check(*args)
                row=next(r for r in report["checks"] if r["id"]=="material_integrity")
                row.update(status="partial",complete=False)
                report["status"]="partial"
                return report,renders
        result,_=self.run_loop([{"model_py":SOURCE}],sandbox=Unknown())
        self.assertEqual(result["status"],"needs_check_support")
        before=snapshot(self.run/"revisions/0001")
        sandbox=Unknown()
        result,transport=self.run_loop([],resume=True,sandbox=sandbox)
        self.assertEqual(result["status"],"needs_check_support")
        self.assertEqual(transport.requests,[])
        self.assertEqual(sandbox.calls,0)
        self.assertEqual(before,snapshot(self.run/"revisions/0001"))

if __name__=="__main__":unittest.main()
