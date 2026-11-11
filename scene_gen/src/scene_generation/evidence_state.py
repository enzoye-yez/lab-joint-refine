"""Persistent read-only actions bound to source bytes, requirements and implementation."""
from pathlib import Path
from copy import deepcopy
from asset_prep.storage import digest,atomic_json
from asset_generation.contracts import read_json,require
from asset_generation.pipeline import snapshot

def action_key(request):
    value=deepcopy(request)
    value.pop("question",None)
    for field in ("object_geoms","support_geoms","pairs"):
        if field in value:value[field]=sorted(value[field],key=digest)
    if value.get("kind")=="details":
        for field,default in (("offset",0),("limit",16),("contains","")):value.setdefault(field,default)
    if value.get("kind")=="material":value.setdefault("tolerance_m",.0005)
    if value.get("kind")=="support":
        value.setdefault("support_tolerance_m",.003);value.setdefault("collision_tolerance_m",.0005)
    if value.get("kind")=="view":
        value["azimuth"]=float(value["azimuth"])%360
        value["elevation"]=float(value["elevation"])
        for field in ("hide_bodies","focus_geoms"):
            if field in value:value[field]=sorted(set(value[field]))
        for field,default in (("hide_sites",True),("diagnostic_light",True),("zoom",1)):
            value.setdefault(field,default)
    return digest(value)

def coverage_key(request):
    # Camera alternatives can resolve the same explicitly identified observation,
    # but never an unrelated metric or a different part/configuration.
    if request.get("kind")=="view" and request.get("question"):
        return digest({k:request.get(k) for k in ("question","kind","configuration","focus_body","focus_geoms")})
    return action_key(request)

def session_path(run,artifact,phase,scope,context,destination):
    identity=getattr(getattr(run,"store",None),"state",{}).get("identity",{})
    key=digest({"files":snapshot(artifact),"scope":scope,"phase":phase,
                "implementation":identity.get("implementation"),
                "requirements":{k:context.get(k) for k in ("input","source","protocol","deferred_checks")},
                "contract":context.get("fixed_contract"),
                "layout":{k:context.get(k) for k in ("plan","scene_layout","instance_contract")}})
    directory=Path(run.root)/"evidence_sessions" if hasattr(run,"root") else Path(destination)
    return key,directory/("evidence_session_"+key+".json")

def session(run,artifact,phase,scope,context,destination):
    key,path=session_path(run,artifact,phase,scope,context,destination)
    state=read_json(path,32000000) if path.exists() else {"input_key":key,"rounds":[],"actions":{}}
    require(state.get("input_key")==key,"stale evidence session")
    return state,lambda:atomic_json(path,state)

def merge_checks(report, result, configuration=None, *, scope="asset", input_key=None):
    """Only exact, complete check identities may replace unknown evidence."""
    from asset_generation.check_report import validate_rows,report_status
    if input_key is not None:
        require(result.get("input_key")==input_key,"supplemental evidence version mismatch")
    updated=deepcopy(report)
    rows={row["id"]:row for row in updated.get("checks",[])}
    for raw in result.get("checks",[]):
        row=deepcopy(raw)
        if configuration:
            row["id"]=configuration+":"+row["id"];row["configuration"]=configuration
        require(row["id"] in rows,"supplemental result outside existing check coverage")
        validate_rows([row],domain=scope,mode=report.get("validation_mode","static"))
        if row.get("component_scope")=="selected_geoms":
            # Local contacts do not discharge a whole-instance support obligation.
            continue
        old=rows[row["id"]]
        require(row.get("tolerance_m")==old.get("tolerance_m"),"supplemental result changed fixed tolerance")
        for field in ("support_tolerance_m","collision_tolerance_m"):
            if field in old:require(row.get(field)==old[field],"supplemental support changed fixed tolerance")
        if old["status"]=="fail":continue
        if row.get("complete") is True:
            rows[row["id"]]=row
        elif row["status"]=="fail":
            row["complete"]=False
            row["coverage_gaps"]=old.get("coverage_gaps",[])+row.get("coverage_gaps",[])
            rows[row["id"]]=row
    updated["checks"]=list(rows.values())
    updated["status"]=report_status(updated["checks"])
    if report.get("status")=="error" and report.get("error_kind")!="measurement_error":
        updated["status"]="error"
    return updated


def recover_pending_execution(state):
    """Consume a saved tool receipt even if observation loading was interrupted."""
    receipt=state.pop("pending_evidence",None)
    if receipt is None:return False
    from asset_generation.check_report import incomplete
    last=state["rounds"][-1] if state["rounds"] else {}
    unresolved=deepcopy(last.get("unresolved",{}))
    results={digest(r["request"]):r for r in receipt["evidence"].get("results",[])}
    for request in receipt["requests"]:
        result=results.get(digest(request),{})
        failure=result.get("error") or result.get("error_kind") or incomplete(result)
        if failure or not result:unresolved[coverage_key(request)]={"request":request,"error":failure or "inspection incomplete"}
        else:unresolved.pop(coverage_key(request),None)
    state["rounds"].append({**receipt,"review":None,"unresolved":unresolved,"feedback":deepcopy(last.get("feedback",[]))})
    return True
