"""审查不确定时先补充实际证据，再复审；不改几何、不替审查器判定通过。"""

import hashlib
import re

from asset_generation.contracts import require
from asset_generation.check_report import inspection_results, incomplete
from asset_generation.evidence import requests_valid
from asset_generation.review import ask_review
from asset_prep.errors import ValidationError,StaleEvidence,ContextLimitError
from asset_prep.storage import atomic_json, digest


PLAN = '''你是静态几何检查规划器。根据审查阻断原因及实际编译的inventory选择只读检查。
只输出 {"requests":[...]}，每轮1至4项；不修改几何或要求，不执行Python。
工具：
1. {"kind":"view","joints":{"实际关节名":数值},"hide_bodies":["实际部件名"],"focus_body":"实际部件名或null","azimuth":45,"elevation":-35}
   相机角度为度，范围azimuth=-360..360、elevation=-90..90；关节角度为弧度，位移米；只能设置实际有界hinge/slide关节；初始姿态为基准，每个视图单独设置全部所需关节。
   hide_bodies隐藏部件及其后代用于内部剖视；只作为检查视图，不能声称实际拆卸成功。
   优先开门/开盖，移除遮挡可用hide_bodies；若不清楚运动关系，用明确标注的剖视补证据。
   focus_body必须实际存在，或JSON null表示整体。可用elevation=-90俯视、+90底视。view还支持focus_geoms:[实际几何名]精确取景，
   hide_sites:true隐藏接口标记、zoom:0.25..4、diagnostic_light:true诊断照明；默认隐藏标记及增强暗表面。
   quality.observable=false说明未观察到目标，不能算补齐证据。inventory含实际坐标和范围，按目标选取景。
2. 仅scene阶段：{"kind":"clearance","instance":"实际实例编号","distance_m":0.05}
   量测目标对其他仪器、机器人及夹具的当前三维几何距离，排除台面、地面，涵盖视觉及碰撞几何，不受碰撞过滤影响。
   在查询距离以内给最小距离，否则只给距离下界。网格采用引擎凸包；不替代操作或传热性能验证。
根据已有证据避免重复相同请求。量测阈值应来自原始要求；工具不会自动宣布验收通过。
3. 仅scene：{"kind":"pairs","pairs":[["实际几何名1","实际几何名2"]],"distance_m":0.01}，最多32对，输出有符号距离、最近点、法向及一致性校验；用于原指定碰撞对复测，不能用总pass替代。
4. 仅scene：{"kind":"support","instance":"实际实例编号"}，从场景声明的真实承托对象查找朝上接触；可增加object_geoms和support_geoms实际几何名列表，精确检查凸缘与承托台，避免盖边等无关邻近几何。正间隙不冒充接触。
5. 仅scene：{"kind":"perimeter","instance":"实际实例编号","distance_m":0.05}，检查实际外包络之外、物体最低至最高高度之间的矩形周边区，区分包络内部/底部支撑与周边侵入；不是整体最小距离，不验证散热性能。
6. 仅scene：{"kind":"mount","instance":"机械臂上工具的实例编号","distance_m":0.02}，实际法兰变换、工具根部、近邻机器人几何对；不能替代未建模的布线或未给出的制造规格。
asset仅当inventory.configurations存在时必须加configuration并从其中选择；普通单构型清单省略configuration或用default。同批几何请求使用同一构型。\ninventory.geoms和sites提供真实名称。当前检查报告中的具体碰撞对可直接复测。承托异常先用support/pairs，周边净空用perimeter，安装用mount；不要只用图片替代这些数值证据。
unresolved_checks列出仍未完成的检查，必须重试对应请求成功才能清除；其他图片或不同请求不会自动抵消量测错误。无法完成时明确报告blocked。
validation_feedback包含空方案、工具异常或无效请求的历史，必须据此调整请求。没有合法blocked的空requests会触发有界重新规划。
若确实不支持，可返回 {"requests":[],"blocked":{"kind":"unsupported_check|missing_requirement|measurement_error","detail":"具体缺失及已尝试方法"}}，不得把检查工具异常描述成资产几何缺陷。不要伪造不存在的关节或部件名。
'''


def evidence_plan(value, scope):
    require(isinstance(value,dict) and {'requests'}<=set(value)<={'requests','blocked'},'补证据计划字段无效')
    if 'blocked' in value:
        blocked=value['blocked']
        require(isinstance(blocked,dict) and set(blocked)=={'kind','detail'} and isinstance(blocked['kind'],str)
                and blocked['kind'] in {'unsupported_check','missing_requirement','measurement_error'}
                and isinstance(blocked['detail'],str) and blocked['detail'].strip(), '补证阻塞分类无效')
    if value['requests']==[] and 'blocked' in value:
        return []
    require(value['requests']!=[], '补证方案为空；请利用实际几何对、承托、周边区或安装检查重新规划，并说明不可执行的原因')
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
        "rule": "只用这里的事实选择下一批动作；需要原始记录时按 evidence_ref 分页读取，不要重建背景。",
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
    format_failed=any(r.startswith("视觉审查连续三次格式无效") for r in review.get("blocking",[]))
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
                    "status":"unresolved","evidence":"框架记录：该批审查暂挂；其他批次继续"}) for d in subset]}
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
    if any(reason.startswith("视觉审查连续三次格式无效") for reason in review.get("blocking",[])):
        outcomes[phase]={"reason":"review_format_exhausted","detail":"审查约定未恢复，不能作为几何补查问题"}
        save()
        return review,call
    if review["verdict"] != "uncertain" or not hasattr(run.sandbox,"inspect_artifact"):
        if snapshot(artifact)!=artifact_files:raise StaleEvidence("review input artifact changed")
        return review,call
    from .recovery import repairable_rows
    if scope=="scene" and repairable_rows(context.get("checks") or {}):
        outcomes[phase]={"reason":"confirmed_defects_first","detail":"先修复已有定位证据的缺陷；其他审查与缺口保留，修改后重新验证"}
        save()
        return review,call
    try:
        if "inventory" in state:
            inventory=state["inventory"]
        else:
            inventory, _ = run.sandbox.inspect_artifact(artifact,destination/"evidence_inventory",scope,[{"kind":"inventory"}])
            state["inventory"]=inventory;save()
        inspection_results(inventory,[{'kind':'inventory'}])
        require(inventory['status']=='pass', '实际几何清单检查未完成')
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
        return finish("inspection_unavailable",detail=inventory.get("error","实际几何清单尚不可用"))
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
        if any(reason.startswith("视觉审查连续三次格式无效") for reason in review.get("blocking",[])):
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
            return finish('inspection_unavailable',detail=inventory.get('error','没有实际几何清单'))
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
                        "archived_images":len(images),"policy":"图片分页读取；已有图片数不限制数值检查或新增观察"}},[],phase+":evidence_plan")
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
                        raise ValidationError("框架没有找到数值替代动作；请核对当前清单中的视图/详情能力，或再次说明具体不可执行原因")
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
                        '该动作已执行但证据仍不完整，或临时重试耗尽；已有结果见previous_evidence，须说明不同方法或明确能力阻塞')
                if len(cached)==len(keys) and state.get("last_reused_keys")==keys:
                    return finish("evidence_no_progress",detail="连续重复已有证据，未提出可消除不确定性的新动作")
                plan_state["attempts"]=planning_attempt+1;save()
                break
            except (ValidationError,ContextLimitError) as exc:
                plan_state["attempts"]=planning_attempt+1
                feedback.append({'round':attempt,'attempt':planning_attempt,'response':value,'error':str(exc)})
                plan_state['feedback']=feedback;save()
                atomic_json(work/'planning_feedback.json',feedback)
                print(f'补证方案重新规划：{exc}（{planning_attempt+1}/{attempts}）',flush=True)
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
        # 不允许后续一张成功图片掩盖之前未完成的数值检查。
        results={digest(r['request']):r for r in evidence.get('results',[]) if 'request' in r}
        for item in requests:
            item_key=coverage_key(item)
            result=results.get(digest(item),{})
            failed=result.get('error') or result.get('error_kind') or incomplete(result)
            if failed or not result:
                unresolved[item_key]={'request':item,'error':failed or evidence.get('error','检查未完成')}
            else:
                unresolved.pop(item_key,None)
        if evidence.get('status')!='pass':
            feedback.append({'round':attempt,'requests':requests,'error':'检查未完成；请针对具体量测异常选择其他可验证证据','details_ref':archive.reference(evidence)})
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
                             "error": "同一问题连续三批观察未能取到可判读目标；保留未知并转入可执行的其他工作",
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
                    "仍有未完成的补证检查，不能自动通过或延后；须重试对应检查、保留不确定或指出已知缺陷")
            return result
        review, call = ask(instruction+'''
补充图像是框架从同一实际几何渲染的静态检查姿态。被隐藏的部件及设置的关节写在supplemental_evidence；
不能把剖视隐藏当作原模型缺件，不能把改变检查姿态当作修改已交付初态或执行过拆卸操作。
补充量测只是证据，不自动覆盖原要求。证实实际缺陷用repair；无法补齐才保持uncertain。
''',request,images,phase,work,validator=checked)
        atomic_json(work/"review.json",{"review":review,"call":call})
        entry.update(review=review,call=call);save()
        if any(reason.startswith("视觉审查连续三次格式无效") for reason in review.get("blocking",[])):
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
                             "error": "同一问题连续三批观察未能取到可判读目标；保留未知并转入可执行的其他工作",
                             "details_ref": archive.reference(evidence)})
            # Do not let the same failed observation class keep the round open.
            unresolved = {k: v for k, v in unresolved.items() if k not in exhausted}
            state.setdefault("capability_gaps", {}).update(
                {item: {"kind": "observation_incomplete", "round": attempt,
                        "request": blocked_now[item]["request"]} for item in exhausted})
            save()


PLAN += """
每个请求可附question，引用questions中的稳定问题编号，不能新造验收要求。
working_state是本会话的可复用状态：已经尝试过的动作及结果、仍未解决的问题、已确认的可观察事实。
用这些事实直接选择下一批动作；需要原始记录时才按details分页读取，不要为了重建背景而重复读同一内容。
同一问题、同一目标的失败观察可用另一角度替代；其他目标的图片不能清除原缺口。
capability_gaps中的项目已判定为不可观测，不要再为它们提出同一类观察。
capabilities列出新增只读工具；material支持asset或scene（scene须instance），检查实体连接和内部交叠。
成功执行过的单项动作不能靠重组请求批次或改写原因重复。观察失败后应更换目标几何、角度、遮挡或采用对应数值能力。
"""


def validate_inventory(requests, inventory, scope):
    """Reject impossible requests before starting the inspection worker."""
    geometric=[r for r in requests if r["kind"] not in {"details","inventory"}]
    require(len({r.get("configuration") for r in geometric})<=1,"同批几何请求须使用同一构型")
    for request in geometric:
        current=inventory
        if scope=="asset":
            options=inventory.get("configurations")
            name=request.get("configuration")
            if options is not None:
                require(name in options,"缺少或未知构型；可用configuration="+str(sorted(options)))
                current=options[name]
            else:require(name in (None,"default"),"单构型资产仅允许省略configuration或default")
        fields={"bodies":request.get("hide_bodies",[])+([request["focus_body"]] if request.get("focus_body") else []),
                "geoms":request.get("focus_geoms",[])+request.get("object_geoms",[])+request.get("support_geoms",[])+[n for pair in request.get("pairs",[]) for n in pair],
                "instances":[request["instance"]] if "instance" in request else []}
        for field,names in fields.items():
            if field in current:
                unknown=sorted(set(names)-set(current[field]))
                require(not unknown,field+"引用未知对象="+str(unknown)+"；合法名称见当前inventory."+field)
        if "joints" in current:
            joints={r["name"]:r for r in current["joints"]}
            for name,value in request.get("joints",{}).items():
                require(name in joints,"未知关节="+name+"；合法关节见当前inventory.joints")
                joint=joints[name]
                require(joint["limited"] and joint["type"] in (2,3) and joint["range"][0]<=value<=joint["range"][1],
                        "关节必须为有界滑动/转动且值在范围内："+str(joint))
