"""Developer-authored closure regressions; zero provider calls."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
from generation_fixtures import XML, MANIFEST, write_artifact
from asset_generation.checks import validate_manifest
from asset_generation.runtime import InteractionRuntime
from asset_prep.errors import ValidationError
from scene_generation.candidate import assembly_manifest, inspect_candidate
from scene_generation.assembly import namespace_manifest
from scene_generation.layout_metrics import union_area, measure
from scene_generation.progress import repair_comparison
from scene_generation.layout_planning import refine

class CompactClosureTests(unittest.TestCase):
    def test_footprints_use_union_not_sum_and_missing_is_explicit(self):
        self.assertEqual(union_area([(0,0,2,2),(1,0,3,2)]),6)
        table={"id":"t","size":[4,2,.05],"position":[0,0],"yaw":0,"reason":"shared"}
        plan={"layout":{"tables":[table],"room":{"size":[8,8]}},
              "instances":[{"id":"a","table_id":"t"},{"id":"b","table_id":"t"}]}
        result=measure(plan,{"a":[([-1,-1,0],[1,1,1])]})
        self.assertEqual(result["tables"][0]["occupancy_upper_bound"],.5)
        self.assertEqual(result["tables"][0]["missing_geometry"],["b"])
        self.assertEqual(result["tables"][0]["operation_clearance"],"not_quantified")

    def mechanical(self):
        value=copy.deepcopy(MANIFEST)
        value["buttons"]=[]
        value["mechanical_events"]=[{"id":"release","interface":"press","action":"release"}]
        return value

    def test_mechanical_declaration_does_not_release_or_become_software(self):
        model=mujoco.MjModel.from_xml_string(XML)
        value=validate_manifest(self.mechanical(),model,["A01"])
        runtime=InteractionRuntime(model,value)
        data=mujoco.MjData(model)
        self.assertEqual(runtime.update(data),[])
        self.assertTrue(data.eq_active[0])
        with self.assertRaises(ValueError): runtime.update(data,["release"])
        prefixed=namespace_manifest(value,"item__")
        self.assertEqual(prefixed["mechanical_events"][0]["interface"],"item__press")
        self.assertEqual(prefixed["connections"][0]["release_event"],"item__release")

    def test_mechanical_probe_is_unknown_without_inventing_release(self):
        from asset_generation.probes import interaction_probes
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"model.xml").write_text(XML,encoding="utf-8")
            rows=interaction_probes(root,self.mechanical())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["status"],"partial")
        self.assertFalse(rows[0]["complete"])

    def test_mechanical_invalid_reference_and_action_rejected(self):
        model=mujoco.MjModel.from_xml_string(XML)
        for field,bad in (("interface","missing"),("action","attach")):
            value=self.mechanical();value["mechanical_events"][0][field]=bad
            with self.subTest(field=field),self.assertRaises(ValidationError):validate_manifest(value,model,["A01"])

    def test_invalid_interaction_keeps_diagnostic_geometry_but_not_quality(self):
        model=mujoco.MjModel.from_xml_string(XML)
        value=copy.deepcopy(MANIFEST);value["connections"][0]["release_event"]="missing"
        original=copy.deepcopy(value)
        diagnostic,issues=assembly_manifest(value,model,["A01"])
        self.assertEqual(value,original)
        self.assertEqual(issues[0]["status"],"fail")
        self.assertFalse(diagnostic["connections"])
        with self.assertRaises(ValidationError):validate_manifest(value,model,["A01"])
        with tempfile.TemporaryDirectory() as temp:
            write_artifact(temp,manifest=value)
            report=inspect_candidate(temp,["A01"],{})
            self.assertTrue(report["assembly_readiness"]["loadable"])
            self.assertFalse(report["assembly_readiness"]["interfaces_valid"])
        with self.assertRaises(ValidationError):assembly_manifest(value,model,["A02"])

    def test_unchanged_candidate_does_not_replace_verified_baseline(self):
        report={"checks":[{"id":"support","status":"fail","complete":True}]}
        self.assertEqual(repair_comparison(report,report)["outcome"],"ineffective")
        self.assertFalse(repair_comparison(report,report)["adopt"])

    def test_compactness_cannot_hide_lost_target_or_added_defect(self):
        before={"checks":[{"id":"support","status":"pass","complete":True}],
                "layout_assessment":{"home_clear":True,"compactness":{"table_count":5,"table_area_m2":20,"room_area_m2":120},
                    "instances":[{"instance":"a","targets":[{"target":"grasp","status":"plausible"}]}]}}
        after=copy.deepcopy(before);after["layout_assessment"]["compactness"]["table_count"]=3
        self.assertEqual(repair_comparison(before,after)["outcome"],"improved_layout")
        after["checks"][0]["status"]="fail"
        self.assertFalse(repair_comparison(before,after)["adopt"])
        after=copy.deepcopy(before);after["layout_assessment"]["compactness"]["table_count"]=3
        after["layout_assessment"]["instances"]=[]
        self.assertFalse(repair_comparison(before,after)["adopt"])

    def test_partial_assets_refine_once_per_real_input_without_budget_reset(self):
        jobs=[{"id":"a","role":"asset"},{"id":"b","role":"asset"}]
        plan={"jobs":jobs,"instances":[],"layout":{"tables":[],"room":{"size":[8,8]}}}
        state={"plan":plan,"jobs":{"a":{"assembly_readiness":{"loadable":True,"artifact_sha256":"one"}}}}
        calls=[]
        def ask(instruction,context,images,phase):
            calls.append(context);return {"keep_layout":"developer fixture"},{}
        run=SimpleNamespace(state=state,env={"robot":{}},limits=SimpleNamespace(max_repair_plan_attempts=3),robot_joints={},
                            check_pause=lambda:None,store=SimpleNamespace(save=lambda:None),model=SimpleNamespace(ask=ask))
        with patch("scene_generation.layout_planning.usable",side_effect=lambda r,v:bool(v)),patch("scene_generation.layout_planning.table_policy",return_value={}):
            refine(run);refine(run)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]["missing_assets"][0]["job"],"b")
            state["jobs"]["b"]={"assembly_readiness":{"loadable":True,"artifact_sha256":"two"}}
            refine(run)
            self.assertEqual(len(calls),2)
            self.assertEqual(len(state["layout_refinement"]["attempts"]),2)

    def test_segment_obstruction_is_geometry_based(self):
        from scene_generation.mobile import segment_box
        self.assertTrue(segment_box([0,0,0],[2,0,0],[.9,-.1,-.1],[1.1,.1,.1]))
        self.assertFalse(segment_box([0,0,2],[2,0,2],[.9,-.1,-.1],[1.1,.1,.1]))

    def test_context_defer_has_a_validated_explicit_contract(self):
        from asset_generation.contracts import author_result
        from asset_prep.errors import ContextLimitError
        with self.assertRaises(ContextLimitError):
            author_result({"deferred":{"reason":"missing source", "missing":["dimensions"]}},None,1000)
        with self.assertRaises(ValidationError):author_result({"deferred":{}},None,1000)

    def test_deferred_job_gets_one_recovery_and_keeps_attempt_ledger(self):
        from scene_generation.scene_first import collect
        state={"plan":{"jobs":[{"id":"a","role":"asset","depends_on":[]}],"instances":[]},
               "order":["a"],"jobs":{},"waiting_on":{}}
        def generate(job,feedback,draft):
            state["jobs"]["a"]={"stage":"synthetic-stage","task_deferred":"no new evidence","feedback":{},"assembly_readiness":{"loadable":False}}
            return False
        run=SimpleNamespace(state=state,generation=SimpleNamespace(max_iterations=60),
             implementation_guard=SimpleNamespace(expected="implementation"),
             check_pause=lambda:None,store=SimpleNamespace(save=lambda:None),generate_job=generate)
        with patch("scene_generation.scene_first.publish",return_value={}),patch("scene_generation.scene_first.usable",return_value=False):
            collect(run);collect(run)
            self.assertEqual(len(state["candidate_attempts"]["a"]),2)
            self.assertEqual(state["blocked_jobs"]["a"]["kind"],"task_deferred")
            self.assertEqual(len(state["task_context_recoveries"]),1)
            state["repair_directive"]={"reason":"new interface evidence"}
            collect(run)
            self.assertEqual(len(state["candidate_attempts"]["a"]),4)
            self.assertEqual(len(state["job_reactivations"]),1)


class SupportCandidateIndexTests(unittest.TestCase):
    def test_index_matches_exhaustive_order_with_planes_and_budget_prefix(self):
        import numpy as np
        from scene_generation.geometry import support_candidates
        rng=np.random.default_rng(27)
        for count in (0,1,70):
            geometry={}
            for g in range(count+20):
                lo=rng.uniform(-2,2,3);geometry[g]=(lo,lo+rng.uniform(.001,.8,3))
            objects=list(reversed(range(count)))
            parents=list(range(count,count+20))+[-1]
            for tolerance in (0,.001,.2):
                expected=[(a,b) for a in parents for b in objects if a not in geometry or not (
                    np.any(geometry[a][1]+tolerance<geometry[b][0]) or
                    np.any(geometry[b][1]+tolerance<geometry[a][0]))]
                actual=list(support_candidates(geometry,parents,objects,tolerance))
                self.assertEqual(actual,expected)
                self.assertEqual(actual[:3],expected[:3])
                self.assertEqual(len(actual[3:]),len(expected[3:]))

    def test_boundary_contact_and_negative_world_coordinates_are_preserved(self):
        import numpy as np
        from scene_generation.geometry import support_candidates
        geometry={0:(np.array([-2.,-1.,-1.]),np.array([-1.,1.,1.])),
                  1:(np.array([-1.,-.5,-.5]),np.array([0.,.5,.5])),
                  2:(np.array([.1,-.5,-.5]),np.array([1.,.5,.5]))}
        self.assertEqual(list(support_candidates(geometry,[0],[2,1],0)),[(0,1)])
        self.assertEqual(list(support_candidates(geometry,[0],[2,1],1.1)),[(0,2),(0,1)])
