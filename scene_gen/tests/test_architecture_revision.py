"""Developer synthetic regressions for the seven architecture audit findings."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from asset_generation.contracts import GenerationConfig
from asset_generation.request_context import RequestContext, unpack_tables
from asset_generation.model import GenerationModel
from asset_prep.config import Config
from asset_prep.storage import Store
from asset_prep.errors import ValidationError, BudgetExceeded
from scene_generation.contracts import SceneConfig
from scene_generation.pipeline import SceneRun
from scene_generation.progress import repair_comparison
from scene_generation.recovery import decide, repairable_rows, isolate_asset_failures
from scene_generation.review import scoped_review
from scene_generation.layout import review_targets, author_layout_policy, table_policy
from test_mobile_layout import mobile_fixture
from test_scene_evidence import UNCERTAIN, VIEW
from scene_generation.evidence import review_with_evidence

ACCEPT = {"verdict":"accept", "motion_coverage": [],"findings":[],"unverified":[]}


class RevisionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env, self.plan = mobile_fixture(self.root)

    def route(self, action, review=None, report=None):
        plan = copy.deepcopy(self.plan)
        plan["jobs"].insert(1, {"id":"broken","asset_ids":["A02"],"role":"asset",
                               "depends_on":[],"requirements":"developer fixture"})
        plan["jobs"][-1]["depends_on"].append("broken")
        plan["jobs"][-1]["asset_ids"].append("A02")
        plan["instances"].append({"id":"missing","job":"broken","placement":"floor",
                                  "position":[3,0,0],"yaw":0})
        (self.root/"source.py").write_text("# developer fixture",encoding="utf-8")
        report = report or {"status":"fail","checks":[{"id":"support:station","status":"fail","instances":["station"]}]}
        commits, requests = [], []
        def ask(instruction, context, images, phase):
            requests.append(copy.deepcopy(context))
            return copy.deepcopy(action), {}
        run = SimpleNamespace(root=self.root,env=self.env,contract={},generation=GenerationConfig(),
            before_repair=lambda action:None,robot_joints={},catalogue={"asset_ids":["A01","A02"]},
            limits=SceneConfig(validation_mode="static"),check_pause=lambda:None,repair_input_key=lambda:"fixture",
            progress=lambda phase,event:None,
            store=SimpleNamespace(state={"calls":[]},save=lambda:None),model=SimpleNamespace(ask=ask),
            begin=lambda kind:self.root,commit=lambda kind,record:commits.append(record),
            state={"plan":plan,"order":["instrument","broken","tool"],"feedback":report,"rounds":[],
                   "latest":{"directory":"synthetic_scene","report":report,"review":review},
                   "jobs":{"instrument":{"source":"source.py","stage":"asset",
                                         "assembly_readiness":{"loadable":True}},
                           "broken":{"source":None,"stage":"failed","assembly_readiness":{"loadable":False}}}})
        self.assertTrue(SceneRun.route_repair(run))
        return run, commits, requests

    def test_partial_scene_layout_repair_leaves_ungenerated_tool_for_scheduler(self):
        run, commits, _ = self.route({"jobs":[],"placements":[{"id":"station","position":[.6,0,.8],"yaw":0}],
                                     "reason":"repair actual support"})
        self.assertNotIn("tool",run.state["jobs"])
        self.assertEqual(commits[-1]["pending_generation_jobs"],["tool"])
        self.assertEqual(run.state["plan"]["instances"][0]["position"],[.6,0,.8])
        self.assertEqual(commits[-1]["dirty_jobs"],[])

    def test_partial_baseline_rollback_preserves_unfinished_plan_and_budget(self):
        for directory in ("scene","asset"):
            (self.root/directory).mkdir()
        partial=copy.deepcopy(self.plan)
        partial["jobs"]=[partial["jobs"][0]]
        partial["instances"]=[partial["instances"][0]]
        old={"directory":"scene","kind":"scene","files":{},"plan":partial,
             "requested_plan":self.plan,"report":{"checks":[]},"missing_jobs":["tool"]}
        asset={"directory":"asset","kind":"asset","files":{},"job":"instrument","status":"assembly_ready"}
        entry={"status":"evaluated","baseline":{"scene":"scene","jobs":{"instrument":"asset"}},
               "comparison":{"adopt":False}}
        run=SimpleNamespace(root=self.root,env=self.env,catalogue={"asset_ids":["A01"]},limits=SceneConfig(),
                            store=SimpleNamespace(state={"calls":[{"reserved_tokens":999}]},save=lambda:None),
                            state={"repair_history":[entry],"stages":[old,asset],"order":["instrument","tool"]})
        self.assertTrue(SceneRun.select_repair_candidate(run))
        self.assertEqual(run.state["plan"],self.plan)
        self.assertEqual(run.state["order"],["instrument","tool"])
        self.assertEqual(run.store.state["calls"],[{"reserved_tokens":999}])

    def fault_report(self, kind="temporary"):
        return {"status":"fail","checks":[
            {"id":"asset:instrument:execution","status":"error","error_kind":"backend_execution",
             "failure_scope":"asset_job","asset_job":"instrument","instances":["station"],
             "execution_failure":{"kind":kind,"retryable":kind=="temporary"}},
            {"id":"support:independent","status":"fail","instances":["independent"]},
            {"id":"support:tool","status":"fail","instances":["gripper"]}]}

    def test_local_fault_preserves_independent_repair_but_blocks_dependents(self):
        report = isolate_asset_failures(self.fault_report(),self.plan)
        self.assertEqual(decide(report),"repair")
        self.assertEqual([r["id"] for r in repairable_rows(report)],["support:independent"])
        self.assertIn("gripper",report["checks"][0]["instances"])
        self.assertEqual(report,isolate_asset_failures(report,self.plan))

    def test_program_or_unclassified_fault_cannot_be_laundered_as_local(self):
        for kind in ("program","unknown"):
            self.assertEqual(decide(isolate_asset_failures(self.fault_report(kind),self.plan)),"needs_environment")
        report = isolate_asset_failures(self.fault_report(),self.plan)
        report["checks"].append({"id":"global","status":"error","error_kind":"check_execution"})
        self.assertEqual(decide(report),"needs_environment")
        report=self.fault_report()
        report.update(status="error",error_kind="check_execution")
        self.assertEqual(decide(isolate_asset_failures(report,self.plan)),"needs_environment")

    def test_table_to_floor_obligation_needs_actual_plan_and_complete_support(self):
        before = {"checks":[{"id":"table_boundary:station","status":"pass"},
                            {"id":"support:station","status":"fail"}]}
        after = {"checks":[{"id":"support:station","status":"pass"}]}
        changed = copy.deepcopy(self.plan)
        changed["instances"][0]["placement"]="floor"
        changed["instances"][0].pop("table_id")
        result = repair_comparison(before,after,old_plan=self.plan,new_plan=changed)
        self.assertEqual(result["outcome"],"improved")
        self.assertEqual(result["superseded_checks"][0]["check"],"table_boundary:station")
        self.assertNotIn("table_boundary:station",result["resolved_checks"])
        for old,new in ((None,None),(self.plan,self.plan)):
            self.assertFalse(repair_comparison(before,after,old_plan=old,new_plan=new)["adopt"])
        after["checks"][0]["status"]="partial"
        self.assertFalse(repair_comparison(before,after,old_plan=self.plan,new_plan=changed)["adopt"])

    def test_missing_jobs_check_stays_present_when_completed(self):
        before={"checks":[{"id":"assembly_missing_jobs","status":"partial","complete":False,"missing_jobs":["tool"]}]}
        after={"checks":[{"id":"assembly_missing_jobs","status":"pass","complete":True,"missing_jobs":[]}]}
        self.assertTrue(repair_comparison(before,after)["adopt"])
        self.assertFalse(repair_comparison(before,{"checks":[]})["adopt"])

    def test_robot_layout_review_routes_without_geometry_edit_rights(self):
        review={"verdict":"repair","findings":["home outside room"],"unverified":[],
                "affected_layout":["robot_home"]}
        scoped_review(review,"scene",instance_ids={"station","gripper"},layout_targets=review_targets(self.plan))
        layout=copy.deepcopy(self.plan["layout"]);layout["robot_home"]=[1,-2,0]
        run,_,requests=self.route({"jobs":[],"placements":[],"layout":layout,"reason":"correct home"},
            review=review,report={"status":"pass","checks":[]})
        self.assertEqual(requests[0]["allowed_jobs"],[])
        self.assertEqual(requests[0]["repair_tasks"][0]["layout_targets"],["robot_home"])
        self.assertNotIn("robot",run.state["jobs"])
        for targets in (["table:invented"],["robot_home","robot_home"]):
            with self.assertRaises(ValidationError):
                scoped_review({**review,"affected_layout":targets},"scene",layout_targets=review_targets(self.plan))

    def test_author_mobile_policy_matches_table_transform_and_legacy_stays_fixed(self):
        self.assertEqual(table_policy(self.env)["robot_protected_half_size_m"],0)
        prompt=author_layout_policy(self.env)
        self.assertIn("local coordinates of the worktable",prompt)
        self.assertNotIn("mounting zone are fixed",prompt)
        self.assertIn("mounting zone are fixed",author_layout_policy({"robot":{}}))

    def model(self, transport, calls=100, generation=None):
        store=Store(self.root/"run",{"implementation":"developer_fixture"},
                    Config(max_model_calls=calls,max_reserved_tokens=50000000))
        return GenerationModel(Config(),generation or GenerationConfig(),{"HARNESS_MODEL_API_KEY":"synthetic"},store,transport)

    def test_oversized_task_source_delivered_verbatim_in_bounded_pages_and_resumes(self):
        text="requirement unique 中文\n"*24000
        requests=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload["messages"][1]["content"][0]["text"]);requests.append(c)
                d=c["context_delivery"]
                value=({"context_notes":[{"reference":d["source_ref"],"text":"developer fixture summary only"}]}
                       if d["remaining_batches"] else {"result":"fixture"})
                return {"choices":[{"message":{"content":json.dumps(value)}}]}
        model=self.model(Transport())
        value,_=model.ask("fixture",{"input":{"source":text}},[],"author")
        self.assertEqual(value,{"result":"fixture"})
        pages=[p for c in requests for p in c["context_delivery"]["material_pages"]]
        self.assertEqual("".join(p["data"] for p in pages),text)
        self.assertTrue(all(c["context_budget"]["required"]<=922000 for c in model.store.state["calls"]))
        count=len(requests)
        self.assertEqual(model.ask("fixture",{"input":{"source":text}},[],"author")[0],value)
        self.assertEqual(len(requests),count)

    def test_reference_images_are_paged_all_sent_and_earlier_defects_survive(self):
        requests=[]
        images=[{"path":str(i),"sha256":str(i),"data_url":"data:"+str(i)} for i in range(55)]
        class Transport:
            def post(inner,url,payload,key,timeout):
                parts=payload["messages"][1]["content"]
                requests.append(parts)
                value=({"verdict":"repair","findings":["visible floating cap"],"unverified":[],
                        "affected_instances":["station"]} if len(requests)==1 else UNCERTAIN)
                return {"choices":[{"message":{"content":json.dumps(value)}}]}
        model=self.model(Transport())
        value,_=model.ask("fixture",{},images,"scene_review")
        self.assertEqual(len(requests),3)
        self.assertEqual(value["verdict"],"repair")
        self.assertIn("visible floating cap",value["findings"])
        sent=[p["image_url"]["url"] for r in requests for p in r if p["type"]=="image_url"]
        self.assertEqual(sent,[i["data_url"] for i in images])
        self.assertTrue(all(len([p for p in r if p["type"]=="image_url"])<=24 for r in requests))

    def test_batch_defect_merge_does_not_grant_unknown_layout_rights_or_hide_bad_format(self):
        from asset_generation.review import retain_batch_defects
        earlier={**ACCEPT,"verdict":"repair","findings":["fixture defect"],"affected_layout":["table:invented"]}
        with self.assertRaises(ValidationError):
            scoped_review(retain_batch_defects([earlier],{**ACCEPT,"verdict":"repair","findings":["later"]})[0],"scene",layout_targets=review_targets(self.plan))
        # The final page saw the earlier findings; its accept stands and the overruled ones are kept for the receipt.
        self.assertEqual(retain_batch_defects([earlier],ACCEPT),(ACCEPT,["fixture defect"]))
        with self.assertRaises(ValidationError):
            retain_batch_defects([earlier],{**UNCERTAIN,"blocking":"invalid string"})

    def test_delivery_cannot_bypass_budget_or_cancel(self):
        class Transport:
            def post(inner,*args):
                return {"choices":[{"message":{"content":json.dumps(ACCEPT)}}]}
        model=self.model(Transport(),calls=1)
        images=[{"path":str(i),"sha256":str(i),"data_url":"data:"+str(i)} for i in range(25)]
        with self.assertRaises(BudgetExceeded):
            model.ask("fixture",{},images,"scene_review")
        self.assertEqual(len(model.store.state["calls"]),1)

    def test_many_dependencies_remain_addressable_below_budget(self):
        deps={str(i):{"manifest":{"interfaces":[{"id":str(j),"description":"x"*2000} for j in range(30)]},
                     "instances":[{"id":str(i),"position":[1,2,3]}]} for i in range(21)}
        view=RequestContext(self.root,{},"fixture",{"dependencies":deps},[],"asset:tool",GenerationConfig())
        self.assertLess(view.after["required"],922000)
        self.assertTrue(view.materials)
        self.assertEqual(unpack_tables(view.original["dependencies"]),deps)
        projected=unpack_tables(view.context["dependencies"])
        self.assertEqual(projected["0"]["instances"],deps["0"]["instances"])
        reference=projected["0"]["manifest"]["interfaces"]["required_material"]
        text="".join(p["data"] for p in view.material_pages if p["reference"]==reference)
        self.assertEqual(json.loads(text),deps["0"]["manifest"]["interfaces"])

    def test_task_source_keeps_shared_footer_and_excludes_unrelated_asset_body(self):
        from scene_generation.inputs import catalog, task_bundle
        package=self.root/"package.md"
        package.write_text("# Shared header\nall materials inert\n### A01 | one\nselected requirement\n"
                           "### A02 | two\nunrelated instrument detail\n## Common footer\nno scaling\n",encoding="utf-8")
        bundle=task_bundle(catalog(package,GenerationConfig()),{"asset_ids":["A01"],"id":"one"})
        source=bundle["source"]
        self.assertIn("all materials inert",source["global_requirements"])
        self.assertIn("no scaling",source["global_requirements"])
        self.assertIn("selected requirement",source["selected_sections"])
        self.assertNotIn("unrelated instrument detail",json.dumps(source))

    def test_large_plan_requirements_page_without_hiding_instance_positions(self):
        plan=copy.deepcopy(self.plan)
        plan["jobs"][0]["requirements"]="exact physical requirement "*40000
        view=RequestContext(self.root,{},"fixture",{"plan":plan},[],"scene_plan",GenerationConfig())
        self.assertEqual(view.context["plan"]["instances"],plan["instances"])
        self.assertEqual(view.context["plan"]["jobs"][0]["id"],plan["jobs"][0]["id"])
        self.assertTrue(view.material_pages)
        self.assertEqual("".join(p["data"] for p in view.material_pages),plan["jobs"][0]["requirements"])

    def test_short_dependency_documents_share_delivery_without_merging_identities(self):
        deps={str(i):{"source_requirements":str(i)+"x"*5000,"instances":[{"id":str(i),"position":[i,0,0]}]}
              for i in range(30)}
        view=RequestContext(self.root,{},"fixture",{"dependencies":deps},[],"asset:tool",GenerationConfig())
        self.assertLess(view.delivery_count,len(view.material_pages))
        self.assertEqual(len(view.material_pages),30)
        self.assertEqual(len({p["reference"] for b in view.material_batches for p in b}),30)

    def test_image_cross_page_comparison_and_interrupted_resume_use_same_ledger(self):
        images=[{"path":str(i),"sha256":str(i),"data_url":"data:"+str(i)} for i in range(25)]
        requests=[]
        class Transport:
            def post(inner,url,payload,key,timeout):
                c=json.loads(payload["messages"][1]["content"][0]["text"])
                requests.append(c)
                if len(requests)==2:raise KeyboardInterrupt()
                value=ACCEPT
                if c.get("context_delivery",{}).get("batch")==1 and "context_image_selection" not in c:
                    value={"context_images":[0,24]}
                return {"choices":[{"message":{"content":json.dumps(value)}}]}
        model=self.model(Transport())
        with self.assertRaises(KeyboardInterrupt):
            model.ask("fixture",{},images,"scene_review")
        # An interrupted sent request is unknown. Explicit retry permission is
        # needed; replaying page one cannot erase its reservation or this request.
        from asset_prep.errors import UncertainRequest
        with self.assertRaises(UncertainRequest):
            model.ask("fixture",{},images,"scene_review")
        model.store.retry_uncertain=True
        value,_=model.ask("fixture",{},images,"scene_review")
        self.assertEqual(value,ACCEPT)
        self.assertEqual(len(requests),4)
        self.assertEqual(requests[-1]["context_image_selection"],[0,24])
        self.assertTrue(any(c.get("outcome_unknown") for c in model.store.state["calls"]))

    def test_single_image_raw_transport_size_is_checked_before_any_paid_request(self):
        from asset_prep.errors import ContextLimitError
        transport=SimpleNamespace(post=lambda *a:self.fail("must not send oversized image"))
        model=self.model(transport,generation=replace(GenerationConfig(),max_request_bytes=100000))
        with self.assertRaises(ContextLimitError):
            model.ask("fixture",{},[{"path":"big","sha256":"fixture","data_url":"x"*100000}],"review")
        self.assertFalse(model.store.state["calls"])

    def test_numerical_and_new_view_queries_execute_even_with_full_image_archive(self):
        for request in (VIEW,{"kind":"clearance","instance":"station","distance_m":.05}):
            with self.subTest(request=request):
                destination=self.root/request["kind"];destination.mkdir()
                replies=[UNCERTAIN,{"requests":[request]},ACCEPT]
                inspections=[]
                def inspect(artifact,path,scope,requests):
                    inspections.append(requests)
                    if requests[0]["kind"]=="inventory":
                        return {"status":"pass","inventory":{"bodies":["base","lid"],"instances":["station"]}},None
                    row={"request":requests[0]}
                    if request["kind"]=="view":row["image"]="view.png"
                    return {"status":"pass","results":[row],"renders":["view.png"]},None
                run=SimpleNamespace(root=destination,state={},check_pause=lambda:None,generation=GenerationConfig(),
                    limits=SceneConfig(validation_mode="static"),store=SimpleNamespace(state={"identity":{}},save=lambda:None),
                    model=SimpleNamespace(ask=lambda *a:(copy.deepcopy(replies.pop(0)),{})),
                    sandbox=SimpleNamespace(inspect_artifact=inspect))
                review,_=review_with_evidence(run,"fixture",{},
                    [{"path":str(i),"sha256":str(i)} for i in range(25)],"test",destination,
                    lambda v:scoped_review(v,"scene",instance_ids={"station"}),self.root/"artifact","scene")
                self.assertEqual(review["verdict"],"accept")
                self.assertEqual(len(inspections),2)


if __name__=="__main__":
    unittest.main()
