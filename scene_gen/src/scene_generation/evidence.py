"""When a review is uncertain, gather actual evidence first, then re-review; never change geometry and never decide a pass on the reviewer's behalf."""

import hashlib
import re

from asset_generation.contracts import require
from asset_generation.check_report import inspection_results, incomplete
from asset_generation.evidence import requests_valid
from asset_generation.review import ask_review
from asset_prep.errors import ValidationError,StaleEvidence,ContextLimitError
from asset_prep.storage import atomic_json, digest


PLAN = '''You are a static geometry inspection planner. Choose read-only inspections from the review's blocking reasons and the actually compiled inventory.
Output only {"requests":[...]}, 1 to 4 items per round; do not change geometry or requirements, do not execute Python.
Tools:
1. {"kind":"view","joints":{"actual joint name":value},"hide_bodies":["actual part name"],"focus_body":"actual part name or null","azimuth":45,"elevation":-35}
   Camera angles are in degrees, azimuth=-360..360, elevation=-90..90; joint angles are in radians, displacements in metres; only actual bounded hinge/slide joints may be set; the initial pose is the baseline, and each view sets all the joints it needs on its own.
   hide_bodies hides parts and their descendants for internal cutaway views; this is only an inspection view and must not be claimed as a successful actual disassembly.
   Prefer opening doors/lids; hide_bodies may be used to remove occluders; if the motion relationship is unclear, gather evidence with an explicitly labelled cutaway.
   focus_body must actually exist, or be JSON null for the whole. Use elevation=-90 for a top view and +90 for a bottom view. view also supports focus_geoms:[actual geom names] for precise framing,
   hide_sites:true to hide interface markers, zoom:0.25..4, diagnostic_light:true for diagnostic lighting; by default markers are hidden and dark surfaces are enhanced.
   quality.observable=false means the target was not observed and does not count as completed evidence. inventory contains actual coordinates and extents; choose framing by target.
2. scene stage only: {"kind":"clearance","instance":"actual instance id","distance_m":0.05}
   Measures the current 3D geometric distance from the target to other instruments, the robot and fixtures, excluding the tabletop and the floor, covering visual and collision geometry, unaffected by collision filtering.
   Within the query distance it gives the minimum distance, otherwise only a lower bound. Meshes use the engine's convex hulls; this does not replace operation or heat-transfer performance verification.
Use existing evidence to avoid repeating identical requests. Measurement thresholds must come from the original requirements; tools never declare acceptance passed by themselves.
3. scene only: {"kind":"pairs","pairs":[["actual geom name 1","actual geom name 2"]],"distance_m":0.01}, at most 32 pairs; outputs signed distance, nearest points, normal and a consistency check; used to re-measure the originally specified collision pairs; an overall pass cannot substitute for it.
4. scene only: {"kind":"support","instance":"actual instance id"}, finds upward-facing contact from the real support object declared by the scene; object_geoms and support_geoms lists of actual geom names may be added to check a flange against its support ledge precisely, avoiding unrelated nearby geometry such as lid edges. A positive gap never counts as contact.
5. scene only: {"kind":"perimeter","instance":"actual instance id","distance_m":0.05}, checks the rectangular perimeter zone outside the actual outer envelope, between the object's lowest and highest heights, distinguishing support inside/under the envelope from perimeter intrusion; it is not an overall minimum distance and does not verify heat dissipation.
6. scene only: {"kind":"mount","instance":"instance id of the tool on the robot arm","distance_m":0.02}, gives the actual flange transform, the tool root and nearby robot geometry pairs; it cannot replace unmodelled cabling or manufacturing specifications that were not given.
For asset, only when inventory.configurations exists, configuration is required and must be chosen from it; an ordinary single-configuration inventory omits configuration or uses default. Geometric requests in the same batch use the same configuration.\ninventory.geoms and sites give the real names. Specific collision pairs in the current check report can be re-measured directly. For support anomalies use support/pairs first, perimeter for perimeter clearance, mount for mounting; do not substitute images alone for this numeric evidence.
unresolved_checks lists inspections still incomplete; each is cleared only by successfully retrying the corresponding request; other images or different requests never cancel a measurement error. If it cannot be completed, report blocked explicitly.
validation_feedback contains the history of empty plans, tool errors or invalid requests; adjust your requests accordingly. Empty requests without a valid blocked trigger bounded replanning.
If truly unsupported, return {"requests":[],"blocked":{"kind":"unsupported_check|missing_requirement|measurement_error","detail":"what specifically is missing and the methods already tried"}}; never describe an inspection tool error as an asset geometry defect. Do not invent joint or part names that do not exist.
'''


def evidence_plan(value, scope):
    require(isinstance(value,dict) and {'requests'}<=set(value)<={'requests','blocked'},'Invalid evidence plan fields')
    if 'blocked' in value:
        blocked=value['blocked']
        require(isinstance(blocked,dict) and set(blocked)=={'kind','detail'} and isinstance(blocked['kind'],str)
                and blocked['kind'] in {'unsupported_check','missing_requirement','measurement_error'}
                and isinstance(blocked['detail'],str) and blocked['detail'].strip(), 'Invalid evidence blocked classification')
    if value['requests']==[] and 'blocked' in value:
        return []
    require(value['requests']!=[], 'Empty evidence plan; replan using actual geom pairs, support, perimeter or mount inspections, and state why they cannot be executed')
    from .context import ContextArchive
    requests=value['requests']
    require(isinstance(requests,list) and 1<=len(requests)<=4,"invalid evidence request count")
    for request in requests:
        require(isinstance(request,dict),"invalid evidence request")
        if request.get("kind")=="details":ContextArchive.validate(request)
        else:requests_valid([request],scope)
    require(len({digest(r) for r in requests})==len(requests),"duplicate evidence requests")
    return requests
def working_state(state, unresolved, questions, limit=24):
    """Bounded, reusable working state for a new logical request.

    Re-sending every earlier evidence round made the planning request grow with each
    round, so it eventually had to be folded into paged reads: the paid context was
    rebuilt instead of carried. This keeps what a decision actually needs — open
    questions, already-attempted actions with their outcomes, and confirmed facts —
    and leaves the full records addressable by reference.
    """
    tried = []
    for key, action in list(state.get("actions", {}).items())[-limit:]:
        row = {"action": key, "status": action.get("status")}
        if action.get("effective") is False:
            row["outcome"] = "failed"
            row["failure"] = str(action.get("failure") or "")[:300]
        elif action.get("effective") is True:
            row["outcome"] = "observed"
        if action.get("retryable"):
            row["retryable"] = True
        tried.append(row)
    open_rows = []
    for key, item in list(unresolved.items())[:limit]:
        open_rows.append({"coverage": key, "request": item.get("request"),
                          "error": str(item.get("error") or "")[:200]})
    return {
        "open_questions": open_rows,
        "questions": dict(list(questions.items())[-limit:]),
        "attempted_actions": tried,
        "capability_gaps": state.get("capability_gaps", {}),
        "confirmed_facts": state.get("confirmed_facts", []),
        "rule": "Choose the next batch of actions from these facts only; when the original records are needed, read them page by page via evidence_ref; do not rebuild the background.",
    }


def remember_facts(state, evidence, limit=32):
    """Keep positive findings small and reusable; they are not a second acceptance path."""
    facts = state.setdefault("confirmed_facts", [])
    for result in evidence.get("results", []):
        if not isinstance(result, dict):
            continue
        quality = result.get("quality") or {}
        if result.get("error") or result.get("error_kind") or not quality:
            continue
        if not quality.get("observable"):
            continue
        fact = {"targets": result.get("targets", [])[:8], "kind": result.get("kind"),
                "camera": result.get("camera"), "observed": True}
        if fact not in facts:
            facts.append(fact)
    del facts[:-limit]


def semantic_text(text):
    """Drop wording that carries no problem identity: punctuation, spaces, tool names."""
    return re.sub(r"[\s，。；：、,.;:!?（）()\[\]\"'“”‘’]+", "", str(text)).lower()


def evidence_progress(state, rounds, window=3):
    """Progress means the open problem set shrank, not that a tool ran successfully.

    A rendered image or a completed measurement only proves the tool executed; it says
    nothing about whether the blocking question was answered. Only fewer open coverage
    entries, or a newly answered one, count. Repeated failures of the same question
    never count, however many successful renders accompany them.
    """
    def opened(entry):
        return {k: digest(v) for k, v in (entry.get("unresolved") or {}).items()}

    current = opened(rounds[-1]) if rounds else {}
    previous = opened(rounds[-2]) if len(rounds) > 1 else {}
    answered = set(previous) - set(current)
    added = set(current) - set(previous)
    recent = rounds[-window:]
    attempts = sum(len(e.get("requests") or []) for e in recent)
    failed = sum(1 for e in recent for result in (e.get("evidence") or {}).get("results", [])
                 if result.get("error") or result.get("error_kind") or result.get("complete") is False)
    return {"open": len(current), "answered": len(answered), "added": len(added),
            "attempts": attempts, "failed": failed,
            "progress": bool(answered) and len(added) <= len(answered),
            "saturated": attempts > 0 and failed == attempts}


def question_identity(reason):
    """Canonical identity for a reviewer question; keeps the historical scheme stable."""
    return "issue_" + digest(" ".join(str(reason).split()))[:16]


def question_alias(reason):
    """Equivalence alias: wording that differs only in spacing or punctuation is one question."""
    return "issue_" + hashlib.sha256(semantic_text(reason).encode("utf-8")).hexdigest()[:16]



def review_with_evidence(run, instruction, context, images, phase, destination, validator, artifact, scope):
    if not hasattr(run,"state"):run.state={}
    try:
        review,call=_review_with_evidence(run,instruction,context,images,phase,destination,validator,artifact,scope)
    except ContextLimitError as exc:
        reason=str(exc)
        run.state.setdefault("evidence_outcomes",{})[phase]={"reason":"task_deferred","detail":reason}
        review,call={"verdict":"uncertain","findings":[],"unverified":[reason],"blocking":[reason]},None
    if call and call.get("task_deferred"):
        run.state.setdefault("evidence_outcomes",{})[phase]={"reason":"task_deferred","detail":call["task_deferred"]}
    pending=[d for d in context.get("deferred_checks",[]) if d.get("stage")=="scene"]
    format_failed=any(r.startswith("Visual review format invalid three times in a row") for r in review.get("blocking",[]))
    # Shrink an unsuccessfully answered set, not the evidence or the acceptance contract.
    if scope=="scene" and len(pending)>1 and (format_failed or
            run.state.get("evidence_outcomes",{}).get(phase,{}).get("reason")=="task_deferred"):
        return _review_with_evidence(run,instruction,{**context,"review_batch_limit":1},images,
            phase+":items",destination,validator,artifact,scope)
    return review,call


def _review_with_evidence(run, instruction, context, images, phase, destination, validator, artifact, scope):
    pending=[d for d in context.get("deferred_checks",[]) if d.get("stage")=="scene"]
    batch_size=context.get("review_batch_limit",16)
    if scope=="scene" and len(pending)>batch_size:
        from .review import scoped_review
        from .layout import review_targets
        batches=[];last_call=None
        for offset in range(0,len(pending),batch_size):
            run.check_pause()
            subset=pending[offset:offset+batch_size]
            batch_context={**context,"deferred_checks":subset,"review_batch":{"offset":offset,"total":len(pending),"size":len(subset)}}
            value,last_call=review_with_evidence(run,instruction,batch_context,images,phase+":batch:"+str(offset//batch_size+1),
                destination,lambda v:scoped_review(v,"scene",subset,
                    preview=context.get("review_scope",{}).get("validation_mode")=="static",
                    instance_ids={i["id"] for i in context.get("plan",{}).get("instances",[])},
                    layout_targets=review_targets(context.get("plan",{}))),artifact,scope)
            if value["verdict"]=="uncertain":
                answered={r["id"]:r for r in value.get("deferred_results",[])}
                value={**value,"deferred_results":[answered.get(d["id"],{"id":d["id"],
                    "status":"unresolved","evidence":"framework record: this batch's review is suspended; other batches continue"}) for d in subset]}
            batches.append(value)
        verdict="repair" if any(v["verdict"]=="repair" for v in batches) else "uncertain" if any(v["verdict"]=="uncertain" for v in batches) else "accept"
        combined={"verdict":verdict,"findings":list(dict.fromkeys(x for v in batches for x in v["findings"])),
                  "unverified":list(dict.fromkeys(x for v in batches for x in v["unverified"])),
                  "deferred_results":[r for v in batches for r in v.get("deferred_results",[])]}
        blockers=[x for v in batches for x in v.get("blocking",[])]
        if verdict=="uncertain":combined["blocking"]=list(dict.fromkeys(blockers))
        else:combined["unverified"]+=blockers
        if verdict=="repair":
            combined["affected_instances"]=sorted({i for v in batches for i in v.get("affected_instances",[])})
            combined["affected_layout"]=sorted({i for v in batches for i in v.get("affected_layout",[])})
        return validator(combined),last_call
    if not hasattr(run,'state'):
        run.state={}
    outcomes=run.state.setdefault('evidence_outcomes',{})
    outcomes.pop(phase,None)
    from .evidence_state import session,action_key,coverage_key
    from asset_generation.inspection_registry import REGISTRY
    from asset_generation.pipeline import snapshot
    if not hasattr(run.sandbox,"inspect_artifact"):
        return ask_review(run.model,instruction,context,images,phase,destination,validator=validator)
    state,save=session(run,artifact,phase,scope,context,destination)
    from .evidence_state import recover_pending_execution
    if recover_pending_execution(state):save()
    artifact_files=snapshot(artifact)
    from .context import ContextArchive,ProjectedModel
    from copy import deepcopy
    archive=ContextArchive(getattr(run,"root",destination),state["input_key"],run.generation.review_context_bytes)
    model=ProjectedModel(run.model,archive)
    identity=getattr(getattr(run,"store",None),"state",{}).get("identity",{})
    review_key=digest({"model":identity.get("model"),"provider":identity.get("provider"),"instruction":instruction,
                       "images":[{k:v for k,v in im.items() if k!="path"} for im in images]})
    if state.get("review_key") not in {None,review_key}:
        state.setdefault("prior_reviews",[]).append({"review_key":state["review_key"],"initial":state.get("initial"),
            "rounds":[{"review":e.get("review"),"call":e.get("call")} for e in state["rounds"]]})
        for field in ("initial","finished","outcome"):state.pop(field,None)
        for entry in state["rounds"]:
            entry.pop("review",None);entry.pop("call",None)
    state["review_key"]=review_key;save()
    def ask(instruction, context, images, phase, destination, validator):
        binding=digest({"review_key":review_key,"instruction":instruction,"context":context,
                        "images":[{k:v for k,v in im.items() if k!="path"} for im in images]})
        receipt=state.setdefault("reviews",{}).setdefault(binding,{})
        return ask_review(model,instruction,context,images,phase,destination,validator=validator,session=receipt,save=save)
    authoritative=context.get("checks")
    def persist_report(updated):
        # One authoritative report is shared by review, checkpoint and the scheduler.
        if authoritative is not None:
            authoritative.clear();authoritative.update(updated)
            pending=run.state.get("pending",{}).get("review_checkpoint")
            if pending:
                if pending["kind"]=="scene":pending["record"]["report"]=deepcopy(updated)
                elif pending["kind"]=="asset":pending["record"].setdefault("feedback",{})["numerical"]=deepcopy(updated)
            if scope=="scene" and run.state.get("published_scene"):
                published=run.state["published_scene"]
                run.state["current_assessment"]={"assembly_key":published.get("assembly_key"),
                    "artifact_sha256":published.get("artifact_sha256"),"report":deepcopy(updated)}
            if hasattr(run,"store"):run.store.save()
    def compact_history():
        # Inventory is indexed once; a tool response's repeated copy adds no evidence.
        return [{k:v for k,v in e.items() if k!="inventory"} for e in history]
    from .evidence_state import merge_checks
    def merge_numeric(report,result):
        try:
            updated=merge_checks(report,result,result["request"].get("configuration"),scope=scope,input_key=state["input_key"])
            persist_report(updated)
            return authoritative if authoritative is not None else updated
        except ValidationError as exc:
            state["numeric_contract_failure"]=str(exc)
            save()
            return report
    for evidence in [e["evidence"] for e in state["rounds"]]:
        for result in evidence.get("results",[]):
            if result.get("checks") and context.get("checks"):
                context={**context,"checks":merge_numeric(context["checks"],result)}
    def initial_checked(value):
        result=validator(value)
        pending_checks=state["rounds"][-1].get("unresolved") if state["rounds"] else None
        defects=any(r.get("status")=="fail" and r.get("component_scope")!="selected_geoms"
            for e in state["rounds"] for observation in e["evidence"].get("results",[]) for r in observation.get("checks",[]))
        require(not (pending_checks or defects) or result["verdict"] not in {"accept","defer"},
                "saved unresolved checks or defects cannot be cleared by changing reviewer")
        return result
    if state.get("numeric_contract_failure"):
        review,call={"verdict":"uncertain","findings":[],"unverified":[],"blocking":[state["numeric_contract_failure"]]},None
    elif "initial" in state and state["initial"]["review"]["verdict"]=="uncertain":
        review,call=state["initial"]["review"],state["initial"]["call"]
    else:
        review,call=ask(instruction,context,images,phase,destination,validator=initial_checked)
        state["initial"]={"review":review,"call":call};save()
    if call and call.get("task_deferred"):
        outcomes[phase]={"reason":"task_deferred","detail":call["task_deferred"]}
        return review,call
    if any(reason.startswith("Visual review format invalid three times in a row") for reason in review.get("blocking",[])):
        outcomes[phase]={"reason":"review_format_exhausted","detail":"The review contract was not restored; this cannot be treated as a geometry evidence question"}
        save()
        return review,call
    if review["verdict"] != "uncertain" or not hasattr(run.sandbox,"inspect_artifact"):
        if snapshot(artifact)!=artifact_files:raise StaleEvidence("review input artifact changed")
        return review,call
    from .recovery import repairable_rows
    if scope=="scene" and repairable_rows(context.get("checks") or {}):
        outcomes[phase]={"reason":"confirmed_defects_first","detail":"Repair defects with existing localizing evidence first; other reviews and gaps are kept and re-verified after the change"}
        save()
        return review,call
    try:
        if "inventory" in state:
            inventory=state["inventory"]
        else:
            inventory, _ = run.sandbox.inspect_artifact(artifact,destination/"evidence_inventory",scope,[{"kind":"inventory"}])
            state["inventory"]=inventory;save()
        inspection_results(inventory,[{'kind':'inventory'}])
        require(inventory['status']=='pass', 'The actual geometry inventory inspection is incomplete')
    except (ValidationError,OSError) as exc:
        inventory={'status':'incomplete','error':str(exc)}
    history,used,feedback,unresolved=[],set(),[],{}
    # Questions outlive individual reviewers. A new description must not orphan
    # an unresolved observation whose alternative view still targets the old ID.
    questions=state.setdefault("questions",{})
    aliases=state.setdefault("question_aliases",{})
    def remember_questions(value):
        for reason in value.get("blocking",[]):
            canonical=question_identity(reason)
            questions[canonical]=reason
            alias=question_alias(reason)
            if alias!=canonical:
                # Reworded punctuation or spacing names the same gap; keep one canonical id.
                aliases[alias]=canonical
        save()
    remember_questions(review)
    for entry in state["rounds"]:
        history.append(entry["evidence"])
        feedback=entry.get("feedback",feedback)
        unresolved=entry.get("unresolved",unresolved)
        if entry.get("renders"):
            require(snapshot(entry["renders"])==entry.get("render_hashes"),"stored observation was modified")
            images=images+run.images(entry["renders"],entry["evidence"])
        if entry.get("review") is not None:review,call=entry["review"],entry["call"]
    numerical_defects=any(row.get("status")=="fail" for evidence in history
        for result in evidence.get("results",[]) for row in result.get("checks",[]) if row.get("component_scope")!="selected_geoms")
    used={k for k,v in state["actions"].items() if v.get("status")=="completed"
          and (not v.get("retryable") or v.get("attempts",1)>=2)}
    if history and state.get("finished") in {"resolved","geometry_repair_required","capability_blocked","evidence_no_progress","review_format_exhausted","confirmed_defects_first"}:
        outcomes[phase]=state["outcome"]
        return review,call
    limits=getattr(run,'limits',None)
    rounds=getattr(limits,'max_evidence_rounds',6)
    attempts=getattr(limits,'max_evidence_plan_attempts',5)
    def finish(reason, **extra):
        if snapshot(artifact)!=artifact_files:raise StaleEvidence("review input artifact changed")
        outcome={'reason':reason,'completed_rounds':len(history),'max_evidence_rounds':rounds,
                 'max_evidence_plan_attempts':attempts,'planning_feedback':feedback,
                 'unresolved_checks':list(unresolved.values()),
                 'capability_gaps':state.get('capability_gaps',{}),**extra}
        outcomes[phase]=outcome
        state['finished']=reason;state['outcome']=outcome;save()
        atomic_json(destination/'evidence_outcome.json',outcome)
        return review,call
    if inventory.get("status")!="pass" or not isinstance(inventory.get("inventory"),dict):
        return finish("inspection_unavailable",detail=inventory.get("error","The actual geometry inventory is not yet available"))
    if state.get("numeric_contract_failure"):
        review={**review,"verdict":"uncertain","findings":[],"blocking":[state["numeric_contract_failure"]]}
        return finish("inspection_unavailable",detail=state["numeric_contract_failure"])
    # A completed tool execution whose review was interrupted is consumed without rerunning tools.
    if state["rounds"] and state["rounds"][-1].get("review") is None:
        entry=state["rounds"][-1]
        def resume_checked(value):
            result=validator(value)
            require(not unresolved or result["verdict"] not in {"accept","defer"},"unresolved checks cannot pass")
            require(not numerical_defects or result["verdict"] not in {"accept","defer"},"confirmed numerical defects cannot be accepted")
            return result
        review,call=ask(instruction,{**context,"previous_review":review,"supplemental_evidence":compact_history(),
            "unresolved_checks":list(unresolved.values())},images,phase,destination,validator=resume_checked)
        entry.update(review=review,call=call);save()
        if any(reason.startswith("Visual review format invalid three times in a row") for reason in review.get("blocking",[])):
            return finish("review_format_exhausted")
        if scope=="scene" and review["verdict"]=="uncertain" and repairable_rows(context.get("checks") or {}):
            return finish("confirmed_defects_first")
        if review["verdict"]!="uncertain":
            return finish("resolved" if review["verdict"] in {"accept","defer"} else "geometry_repair_required")
    # A round window is a progress checkpoint, not an unconditional stop, and a
    # successful tool call is not progress. Only a shrinking problem set continues.
    attempt=len(history)
    while True:
        if attempt>=rounds:
            progress=evidence_progress(state,state["rounds"])
            latest=state["rounds"][-1]["evidence"].get("results",[]) if state["rounds"] else []
            fresh=any(not r.get("reused") and not r.get("error") and not r.get("error_kind")
                      and not incomplete(r) for r in latest)
            if progress["progress"]:
                state["grace_used"]=False
            elif fresh and not progress["saturated"] and not state.get("grace_used"):
                # A new usable observation earns exactly one more round; without an
                # answered question after that, the session stops instead of chaining.
                state["grace_used"]=True;save()
            else:
                return finish("evidence_no_progress",progress=progress)
        run.check_pause()
        remember_questions(review)
        if not inventory.get("inventory"):
            return finish('inspection_unavailable',detail=inventory.get('error','No actual geometry inventory'))
        work = destination/f"evidence_{attempt+1}"
        work.mkdir(exist_ok=True)
        plan_state=state.setdefault("planning",{}).setdefault(str(attempt),{"attempts":0})
        feedback=plan_state.get("feedback",feedback)
        for planning_attempt in range(plan_state["attempts"],attempts):
            run.check_pause()
            value=None
            try:
                value, planning_call = model.ask(PLAN,{"scope":scope,"review":review,
                    "inventory":inventory["inventory"],"working_state":working_state(state,unresolved,questions),
                    "previous_evidence":compact_history()[-4:],"current_checks":context.get('checks'),
                    "source_requirements":context.get("input",context.get("source",context.get("protocol"))),
                    "fixed_contract":context.get("fixed_contract"),"validation_feedback":feedback[-6:],
                    "unresolved_checks":list(unresolved.values())[:24],"questions":dict(list(questions.items())[-24:]),
                    "capabilities":[{"kind":"details","description":"Read complete archived evidence by evidence_ref with pagination; never changes check status"}]+[{"kind":c.id,"description":c.description} for c in REGISTRY.values()],
                    "round":attempt,"planning_attempt":planning_attempt,
                    "image_delivery":{"per_request_limit":run.generation.max_images,
                        "archived_images":len(images),"policy":"Images are read page by page; the number of existing images does not limit numeric inspections or new observations"}},[],phase+":evidence_plan")
                atomic_json(work/("plan.json" if planning_attempt==0 else f"plan_{planning_attempt+1}.json"),{"response":value,"call":planning_call})
                requests=evidence_plan(value,scope)
                if not requests:
                    # A model's inability is a proposal, not a scheduler instruction.
                    alternatives=[]
                    for capability in REGISTRY.values():
                        propose=getattr(capability,"propose",None)
                        if propose:
                            try:
                                for option in propose(context.get("checks") or {},scope):
                                    requests_valid([option],scope)
                                    if action_key(option) not in used:
                                        alternatives.append(option)
                            except (ValidationError,OSError) as exc:
                                feedback.append({"capability":capability.id,"error":str(exc),
                                    "origin":"framework_proposal"})
                    requests=alternatives[:1]
                    if not requests:
                        # A claimed blocker is a proposal, not a scheduler instruction.
                        # Repeating it (even reworded) must not buy another attempt.
                        signature=digest({"kind":value["blocked"]["kind"],
                                          "detail":semantic_text(value["blocked"]["detail"])})
                        previous=plan_state.get("claimed_blocker")
                        plan_state["claimed_blocker"]=signature;save()
                        if previous==signature:
                            return finish("capability_blocked",blocked=value["blocked"],framework_alternatives=0)
                        raise ValidationError("The framework found no numeric alternative action; check the view/details capabilities in the current inventory, or state the specific reason it cannot be executed again")
                for request in requests:
                    if request.get("question") in aliases:
                        request["question"]=aliases[request["question"]]
                for request in requests:
                    require("question" not in request or request["question"] in questions,"unknown question identity")
                for request in requests:
                    if request["kind"]=="material":
                        check_id="material_integrity"+(":"+request["instance"] if scope=="scene" else "")
                        if request.get("configuration"):check_id=request["configuration"]+":"+check_id
                        row=next((r for r in (context.get("checks") or {}).get("checks",[]) if r["id"]==check_id),None)
                        require(row is not None,"material target outside current check coverage")
                        trusted=row.get("tolerance_m",.0005)
                        require("tolerance_m" not in request or request["tolerance_m"]==trusted,"inspection cannot change fixed tolerance")
                        request["tolerance_m"]=trusted
                    if request["kind"]=="support":
                        for field,default in (("support_tolerance_m",.003),("collision_tolerance_m",.0005)):
                            trusted=getattr(getattr(run,"limits",None),field,default)
                            require(field not in request or request[field]==trusted,"inspection cannot change fixed tolerance")
                            request[field]=trusted
                validate_inventory(requests,inventory["inventory"],scope)
                keys=[action_key(r) for r in requests]
                cached={k:state["actions"][k]["result"] for k in keys if k in used and state["actions"][k].get("effective") and state["actions"][k].get("result")}
                require(not any(k in used and k not in cached for k in keys),
                        'This action was already executed but the evidence is still incomplete, or its transient retries are exhausted; see previous_evidence for existing results; propose a different method or state an explicit capability blocker')
                if len(cached)==len(keys) and state.get("last_reused_keys")==keys:
                    return finish("evidence_no_progress",detail="Existing evidence was repeated consecutively; no new action that could remove the uncertainty was proposed")
                plan_state["attempts"]=planning_attempt+1;save()
                break
            except (ValidationError,ContextLimitError) as exc:
                plan_state["attempts"]=planning_attempt+1
                feedback.append({'round':attempt,'attempt':planning_attempt,'response':value,'error':str(exc)})
                plan_state['feedback']=feedback;save()
                atomic_json(work/'planning_feedback.json',feedback)
                print(f'Evidence plan replanning: {exc} ({planning_attempt+1}/{attempts})',flush=True)
        else:
            return finish('evidence_planning_exhausted')
        state["last_reused_keys"]=keys if len(cached)==len(keys) else None
        for key,item in zip(keys,requests):
            if key in cached:continue
            count=state["actions"].get(key,{}).get("attempts",0)+1
            state["actions"][key]={"status":"executing","request":item,"input_key":state["input_key"],"attempts":count}
        save()
        try:
            geometric=[r for r,k in zip(requests,keys) if r["kind"]!="details" and k not in cached]
            before=snapshot(artifact)
            if before!=artifact_files:raise StaleEvidence("inspection input changed since session binding")
            if geometric:
                evidence, renders = run.sandbox.inspect_artifact(artifact,work/"inspection",scope,geometric)
                inspection_results(evidence,geometric)
            else:evidence,renders={"status":"pass","results":[]},None
            if snapshot(artifact)!=before:raise StaleEvidence("read-only inspection modified artifact")
            evidence={k:v for k,v in evidence.items() if k!="inventory"}
            evidence["results"]=evidence.get("results",[])+[archive.inspect(r) for r,k in zip(requests,keys) if r["kind"]=="details" and k not in cached]
            evidence["results"] += [{**deepcopy(cached[k]),"request":r,"reused":True} for r,k in zip(requests,keys) if k in cached]
            for result in evidence["results"]:result["input_key"]=state["input_key"]
            for result in evidence.get("results",[]):
                key=action_key(result["request"])
                effective=not result.get("error") and not result.get("error_kind") and not incomplete(result)
                retryable=bool(result.get("error")) and result.get("error_kind") in {None,"temporary"}
                state["actions"][key].update(status="completed",effective=effective,retryable=retryable,result=result)
                if not retryable or state["actions"][key]["attempts"]>=2:used.add(key)
            state["pending_evidence"]={"evidence":evidence,"requests":requests,"renders":str(renders) if renders else None,
                "render_hashes":snapshot(renders) if renders else None}
            save()
        except (ValidationError,OSError) as exc:
            from .faults import exception_failure
            classified=exception_failure(exc)
            retryable=classified["retryable"]
            evidence={"status":"incomplete","error":str(exc),"results":[{
                "request":r,"complete":False,"error":str(exc),"error_kind":classified["kind"]} if k not in cached else {**deepcopy(cached[k]),"request":r,"reused":True} for r,k in zip(requests,keys)]}
            for key in keys:
                if key in cached:continue
                state["actions"][key].update(status="completed",effective=False,retryable=retryable,failure=str(exc))
                if not retryable or state["actions"][key]["attempts"]>=2:used.add(key)
            save()
            renders=None
        try:
            images=images+(run.images(renders,evidence) if renders else [])
        except (ValidationError,OSError) as exc:
            # A broken image cannot erase a completed numerical measurement.
            evidence.update(status="incomplete",render_error=str(exc))
            for result in evidence.get("results",[]):
                if result["request"]["kind"]=="view" and not result.get("reused"):
                    result.update(complete=False,error=str(exc),error_kind="program")
                    state["actions"][action_key(result["request"])].update(effective=False,retryable=False)
        history.append(evidence)
        # A later successful image must not mask an earlier incomplete numeric inspection.
        results={digest(r['request']):r for r in evidence.get('results',[]) if 'request' in r}
        for item in requests:
            item_key=coverage_key(item)
            result=results.get(digest(item),{})
            failed=result.get('error') or result.get('error_kind') or incomplete(result)
            if failed or not result:
                unresolved[item_key]={'request':item,'error':failed or evidence.get('error','inspection incomplete')}
            else:
                unresolved.pop(item_key,None)
        if evidence.get('status')!='pass':
            feedback.append({'round':attempt,'requests':requests,'error':'Inspection incomplete; choose other verifiable evidence that addresses the specific measurement error','details_ref':archive.reference(evidence)})
        # A question whose alternative observations keep failing does not earn unlimited
        # rounds: it becomes an explicit capability gap and stops holding the round open.
        blocked_now = {item: info for item, info in unresolved.items()
                       if str(info.get("error")) == "target is obscured, too small or too dark"}
        stalled = {item: count for item, count in state.setdefault("observation_failures", {}).items()
                   if item in blocked_now}
        for item in blocked_now:
            stalled[item] = stalled.get(item, 0) + 1
        state["observation_failures"] = stalled
        exhausted = sorted(item for item, count in stalled.items() if count >= 3)
        if exhausted:
            feedback.append({"round": attempt, "observations_unavailable": exhausted,
                             "error": "Three consecutive observation batches for the same question failed to capture a readable target; keep it unknown and move on to other executable work",
                             "details_ref": archive.reference(evidence)})
            unresolved = {k: v for k, v in unresolved.items() if k not in exhausted}
            state.setdefault("capability_gaps", {}).update(
                {item: {"kind": "observation_incomplete", "round": attempt,
                        "request": blocked_now[item]["request"]} for item in exhausted})
        remember_facts(state,evidence)
        atomic_json(work/"evidence.json",evidence)
        entry={"evidence":evidence,"requests":requests,"renders":str(renders) if renders else None,
               "render_hashes":snapshot(renders) if renders else None,"review":None,
               "unresolved":dict(unresolved),"feedback":list(feedback)}
        state["rounds"].append(entry);state.pop("pending_evidence",None);save()
        from .evidence_state import merge_checks
        for result in evidence.get("results",[]):
            if result.get("checks") and context.get("checks"):
                context={**context,"checks":merge_numeric(context["checks"],result)}
        if state.get("numeric_contract_failure"):
            review={**review,"verdict":"uncertain","findings":[],"blocking":[state["numeric_contract_failure"]]}
            return finish("inspection_unavailable",detail=state["numeric_contract_failure"])
        numerical_defects=any(row.get("status")=="fail" for entry in history
            for result in entry.get("results",[]) for row in result.get("checks",[]) if row.get("component_scope")!="selected_geoms")
        request = {**context,"previous_review":review,"supplemental_evidence":compact_history(),
                   "inspection_inventory":inventory["inventory"],"unresolved_checks":list(unresolved.values())}
        def checked(value):
            result = validator(value)
            require(not numerical_defects or result["verdict"] not in {"accept","defer"},"confirmed numerical defects cannot be accepted")
            require((evidence.get("status") == "pass" and not unresolved) or result["verdict"] not in {"accept","defer"},
                    "Evidence inspections are still incomplete; cannot pass or defer automatically; retry the corresponding inspections, keep uncertain, or name the known defects")
            return result
        review, call = ask(instruction+'''
Supplemental images are static inspection poses rendered by the framework from the same actual geometry. Hidden parts and set joints are recorded in supplemental_evidence;
do not treat cutaway hiding as parts missing from the original model, and do not treat a changed inspection pose as a modification of the delivered initial state or as a performed disassembly.
Supplemental measurements are only evidence and never override the original requirements. Use repair for confirmed actual defects; keep uncertain only if the evidence cannot be completed.
''',request,images,phase,work,validator=checked)
        atomic_json(work/"review.json",{"review":review,"call":call})
        entry.update(review=review,call=call);save()
        if any(reason.startswith("Visual review format invalid three times in a row") for reason in review.get("blocking",[])):
            return finish("review_format_exhausted")
        if scope=="scene" and review["verdict"]=="uncertain" and repairable_rows(context.get("checks") or {}):
            return finish("confirmed_defects_first")
        if review["verdict"] != "uncertain":
            return finish('resolved' if review['verdict'] in {'accept','defer'} else 'geometry_repair_required')
        attempt+=1
        # A question whose alternative observations keep failing does not earn unlimited
        # rounds: it is recorded as an explicit capability gap and left unresolved, not
        # retried with the same evidence class until the whole protocol budget is spent.
        blocked_now = {item: error for item, error in unresolved.items()
                       if str(error.get("error")) == "target is obscured, too small or too dark"}
        stalled = {item: count for item, count in state.setdefault("observation_failures", {}).items()
                   if item in blocked_now}
        for item in blocked_now:
            stalled[item] = stalled.get(item, 0) + 1
        state["observation_failures"] = stalled
        exhausted = sorted(item for item, count in stalled.items() if count >= 3)
        if exhausted:
            feedback.append({"round": attempt, "observations_unavailable": exhausted,
                             "error": "Three consecutive observation batches for the same question failed to capture a readable target; keep it unknown and move on to other executable work",
                             "details_ref": archive.reference(evidence)})
            # Do not let the same failed observation class keep the round open.
            unresolved = {k: v for k, v in unresolved.items() if k not in exhausted}
            state.setdefault("capability_gaps", {}).update(
                {item: {"kind": "observation_incomplete", "round": attempt,
                        "request": blocked_now[item]["request"]} for item in exhausted})
            save()


PLAN += """
Each request may carry question, referencing a stable question id in questions; do not invent new acceptance requirements.
working_state is this session's reusable state: actions already tried and their results, still-unresolved questions, and confirmed observable facts.
Choose the next batch of actions directly from these facts; read details page by page only when the original records are needed, and do not re-read the same content to rebuild the background.
A failed observation of the same question and target may be replaced by another angle; images of other targets cannot clear the original gap.
Items in capability_gaps have been judged unobservable; do not propose the same kind of observation for them again.
capabilities lists additional read-only tools; material supports asset or scene (scene requires instance) and checks solid connectivity and internal overlap.
A single action that already succeeded cannot be repeated by regrouping request batches or rewording the reason. After a failed observation, change the target geometry, angle or occlusion, or use the corresponding numeric capability.
"""


def validate_inventory(requests, inventory, scope):
    """Reject impossible requests before starting the inspection worker."""
    geometric=[r for r in requests if r["kind"] not in {"details","inventory"}]
    require(len({r.get("configuration") for r in geometric})<=1,"Geometric requests in the same batch must use the same configuration")
    for request in geometric:
        current=inventory
        if scope=="asset":
            options=inventory.get("configurations")
            name=request.get("configuration")
            if options is not None:
                require(name in options,"Missing or unknown configuration; available configuration="+str(sorted(options)))
                current=options[name]
            else:require(name in (None,"default"),"A single-configuration asset only allows omitting configuration or default")
        fields={"bodies":request.get("hide_bodies",[])+([request["focus_body"]] if request.get("focus_body") else []),
                "geoms":request.get("focus_geoms",[])+request.get("object_geoms",[])+request.get("support_geoms",[])+[n for pair in request.get("pairs",[]) for n in pair],
                "instances":[request["instance"]] if "instance" in request else []}
        for field,names in fields.items():
            if field in current:
                unknown=sorted(set(names)-set(current[field]))
                require(not unknown,field+" references unknown objects="+str(unknown)+"; valid names are in the current inventory."+field)
        if "joints" in current:
            joints={r["name"]:r for r in current["joints"]}
            for name,value in request.get("joints",{}).items():
                require(name in joints,"unknown joint="+name+"; valid joints are in the current inventory.joints")
                joint=joints[name]
                require(joint["limited"] and joint["type"] in (2,3) and joint["range"][0]<=value<=joint["range"][1],
                        "Joint must be a bounded slide/hinge with a value within range: "+str(joint))
