"""Local numerical recovery using the same registered inspection capabilities.

No model requests; each completed action has an immutable-input receipt. Unknown
results exhaust applicable capabilities, never enter authoring as defects.
"""
from pathlib import Path
from copy import deepcopy
from asset_prep.storage import atomic_json,digest
from asset_prep.errors import StaleEvidence
from .contracts import read_json
from .check_report import inspection_results,incomplete,CheckContractError
from .inspection_registry import REGISTRY

def recover_checks(backend,artifact,report,destination,**options):
    try:
        return _recover_checks(backend,artifact,report,destination,**options)
    except StaleEvidence:raise
    except Exception as exc:
        from .check_report import failed_report
        failure=failed_report(exc)
        return {**report,**failure,"checks":deepcopy(report.get("checks",[])),"renders":report.get("renders",[])}

def _recover_checks(backend,artifact,report,destination,*,implementation=None,contract=None,check_pause=lambda:None,max_actions=16,max_retries=2):
    from .pipeline import snapshot
    from scene_generation.evidence_state import action_key,merge_checks
    original=snapshot(artifact)
    identity=digest({"artifact":original,"implementation":implementation,"contract":contract,
        "mode":report.get("validation_mode"),
        "checks":[{k:r[k] for k in ("id","tolerance_m","configuration") if k in r} for r in report.get("checks",[])]})
    directory=Path(destination)/"numerical_recovery"/identity
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"actions.json"
    state=read_json(path,32000000) if path.exists() else {"input_key":identity,"actions":[]}
    if state["input_key"]!=identity:raise StaleEvidence("asset recovery input identity changed")
    current=deepcopy(report)
    for action in state["actions"]:
        if action.get("result"):
            inspection_results(action["result"],[action["request"]])
            for result in action["result"].get("results",[]):
                current=merge_checks(current,result,action["request"].get("configuration"))
    while hasattr(backend,"inspect_artifact"):
        check_pause()
        if snapshot(artifact)!=original:raise StaleEvidence("read-only recovery changed asset")
        candidates=[]
        for capability in REGISTRY.values():
            propose=getattr(capability,"propose",None)
            if not propose:continue
            if capability.id in state.get("proposal_failures",{}):continue
            try:
                for request in propose(deepcopy(current),"asset"):
                    capability.validate(request,"asset")
                    key=action_key(request)
                    previous=next((a for a in state["actions"] if a["key"]==key),None)
                    if (previous is None or previous["status"]=="started" and previous.get("attempts",1)<2
                            or previous["status"]=="failed" and previous.get("failure",{}).get("retryable")
                            and previous.get("attempts",1)<1+max_retries):
                        candidates.append((previous is not None,key,request,previous))
            except Exception as exc:
                from scene_generation.faults import exception_failure
                state.setdefault("proposal_failures",{})[capability.id]=exception_failure(exc)
                atomic_json(path,state)
        if not candidates:
            state["stop_reason"]="no_executable_action";break
        if sum(a.get("attempts",1) for a in state["actions"])>=max_actions:
            state["stop_reason"]="action_budget";break
        _,key,request,action=min(candidates,key=lambda c:c[0])
        if action is None:
            action={"key":key,"request":request,"status":"started","attempts":0}
            state["actions"].append(action)
        action["attempts"]+=1
        atomic_json(path,state)
        try:
            result,_=backend.inspect_artifact(artifact,directory/(key+"_"+str(action["attempts"])),"asset",[request])
            if snapshot(artifact)!=original:raise StaleEvidence("read-only recovery changed asset")
            inspection_results(result,[request])
            action.update(status="completed",result=result)
            if "failure" in action:
                action.setdefault("prior_failures",[]).append(action.pop("failure"))
            atomic_json(path,state)  # receipt before consumption/review
            from scene_generation.capabilities import semantic
            before=digest(semantic(current.get("checks",[])))
            remaining_before={r["id"] for r in current.get("checks",[]) if incomplete(r)}
            for item in result.get("results",[]):
                current=merge_checks(current,item,request.get("configuration"))
            action["effective"]=before!=digest(semantic(current.get("checks",[])))
            action["resolved_checks"]=sorted(remaining_before-{r["id"] for r in current.get("checks",[]) if incomplete(r)})
        except StaleEvidence:raise
        except Exception as exc:
            from scene_generation.faults import exception_failure
            action.update(status="failed",failure=exception_failure(exc))
        atomic_json(path,state)
    else:state["stop_reason"]="inspection_unavailable"
    if snapshot(artifact)!=original:raise StaleEvidence("read-only recovery changed asset")
    state["remaining_checks"]=[r["id"] for r in current.get("checks",[]) if incomplete(r)]
    atomic_json(path,state)
    failures=list(state.get("proposal_failures",{}).values())+[a["failure"] for a in state["actions"] if a.get("failure")]
    if failures:
        current["numerical_recovery_failures"]=failures
        current["execution_failure"]=failures[0]
        current["status"]="error"
    current["numerical_recovery"]={"input_key":identity,"stop_reason":state["stop_reason"],
        "actions":[{k:v for k,v in a.items() if k!="result"} for a in state["actions"]],
        "remaining_checks":state["remaining_checks"]}
    return current
