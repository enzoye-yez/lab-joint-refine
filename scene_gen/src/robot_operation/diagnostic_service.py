"""Fixed adapter layer between the existing joint run and the generic evidence tools."""
from copy import deepcopy
from .common import need, digest, OperationError
from .diagnostic_tools import DiagnosticContext, normalize, CATALOG
from .diagnostic_ledger import EvidenceLedger


def collect(context, state, queries, save=lambda: None, *, count_stall=True):
    need(isinstance(queries, list) and 0 < len(queries) <= context.settings.diagnostic_batch, "Diagnostic batch too large")
    context.verify_sources()
    ledger = EvidenceLedger(state, context.settings, save,
                            context.evidence_root/"evidence" if context.evidence_root else None)
    context.evidence_records=ledger.state["records"]
    records = []
    for raw in queries:
        try:
            query = normalize(raw, context.settings)
        except (OperationError,ValueError,TypeError) as exc:
            query={k:v for k,v in raw.items() if k not in {"question","hypotheses"}} if isinstance(raw,dict) else {"malformed":raw}
            query.setdefault("state","initial")
            binding={"scene_sha256":context.scene_hash,"invalid_request":True,"tool_version":1}
            result={"status":"invalid","code":getattr(exc,"code","invalid_query"),"detail":str(exc),
                    "repair_authorized":False}
            records.append(ledger.acquire(query,binding,lambda r=result:r))
            continue
        seconds = 2*query["seconds"] if query["kind"] == "probe" else 0
        record = ledger.acquire(query, context.binding(query), lambda q=query: context.measure(q), sim_seconds=seconds)
        records.append(record)
    context.verify_sources()
    receipt = ledger.finish_batch({"scene_sha256": context.scene_hash, "run_sha256": digest(context.files)}, records) if count_stall else None
    return records, receipt


def merge(report, records):
    report = deepcopy(report)
    evidence = {r["id"]: r for r in report.get("evidence", [])}
    rows = {digest(row): row for row in report.get("rows", [])}
    for record in records:
        need(record["binding"]["scene_sha256"] == report["scene_sha256"], "Evidence from an older version cannot enter the current diagnosis", "stale_evidence")
        evidence[record["id"]] = record
        if record["status"] != "valid": continue
        for row in record["result"].get("rows", []):
            entry = {**row, "evidence_id": record["id"], "scene_sha256": report["scene_sha256"],
                     "evidence_state": record["query"]["state"]}
            rows[digest(entry)] = entry
    report.update(rows=list(rows.values()), evidence=list(evidence.values()))
    report["diagnostic_tools"] = CATALOG
    return report


def inspect_with_ledger(scene, scene_hash, settings, state, *, run_root=None, evidence_root=None,
                        queries=(), save=lambda: None, guard=lambda: None, cancel_guard=None):
    from .diagnosis import diagnose_run, inspect_scene
    context = DiagnosticContext(scene, scene_hash, settings, run_root=run_root, evidence_root=evidence_root, guard=guard,cancel_guard=cancel_guard)
    report = diagnose_run(run_root) if run_root else inspect_scene(scene, scene_hash, settings)
    # Source initial-state issues keep the existing fixed-check results; execution-failure observations are never promoted to initial-state defects.
    if run_root:
        initial = inspect_scene(scene, scene_hash, settings)
        report["rows"].extend(initial["rows"])
        report["coverage_gaps"] = sorted(set(report["coverage_gaps"]+initial["coverage_gaps"]))
    source = "failure" if context.failure else "initial"
    automatic = [{"kind": "contacts", "state": "initial"}]
    if source != "initial": automatic.append({"kind":"contacts","state":source})
    if context.record and (context.run_root/"steps.jsonl.gz").exists(): automatic.append({"kind": "trace", "state": source})
    records, _ = collect(context, state, automatic, save, count_stall=False)
    report = merge(report, records)
    if queries:
        records, receipt = collect(context, state, list(queries), save)
        report = merge(report, records)
        report["last_diagnostic_batch"] = receipt
    report["instance_roles"] = context.roles()
    report["attribution"] = attribution(report)
    return report


def attribution(report):
    """Offer evidence-backed candidate sources; never interpret an observation automatically as the unique root cause."""
    if report.get("failure", {}).get("scope") == "engineering":
        return {"status": "engineering", "candidates": [], "reason": "Execution or simulation fault: resolve the engineering issue first"}
    candidates = []
    for row in report.get("rows", []):
        roles = report.get("instance_roles", {})
        assets=[i for i in row.get("instances", []) if roles.get(i) == "asset"]
        if row.get("repair_authorized") and assets:
            candidates.append({"target": "asset", "instances": assets,
                               "basis": "confirmed_defect", "evidence_id": row.get("evidence_id", "legacy:"+digest(row))})
        fixtures = [i for i in row.get("instances", []) if roles.get(i) == "fixture"]
        observation=row.get("observation",{})
        if fixtures and (row.get("repair_authorized") or observation.get("limit_exceeded") or observation.get("retention_failure_observed")):
            candidates.append({"target": "fixture", "instances": fixtures, "basis": "design_hypothesis",
                               "evidence_id": row.get("evidence_id", "legacy:"+digest(row))})
        if row.get("kind") == "trajectory_probe" and row.get("observation", {}).get("variant_improved"):
            candidates.append({"target": "trajectory", "basis": "paired_local_improvement", "evidence_id": row["evidence_id"]})
    return {"status": "supported_candidates" if candidates else "unresolved", "candidates": candidates,
            "limitations": "Several causes can coexist; a local success does not prove a unique cause, and a failed search does not prove an asset defect"}


def repair_ticket(report, request, target, scene_hash):
    need(target in {"asset", "fixture", "scene"}, "Unknown repair category")
    need(report["scene_sha256"] == scene_hash, "Repair evidence belongs to an older scene", "stale_evidence")
    need(report.get("failure", {}).get("scope") != "engineering", "An engineering fault does not authorize asset changes")
    need(isinstance(request, dict) and set(request) <= {"reason", "instances", "evidence_ids", "checks"}, "Invalid repair request fields")
    need(isinstance(request.get("reason"), str) and request["reason"].strip(), "A repair requires a stated reason")
    selected = request.get("instances", [])
    checks = request.get("checks", [])
    ids = request.get("evidence_ids", [])
    need(isinstance(selected, list) and selected and len(set(selected)) == len(selected), "A repair must name its instances explicitly")
    need(isinstance(ids, list) and ids, "A repair must cite actual evidence ids")
    need(isinstance(checks, list) and 0 < len(checks) <= 16, "A repair must declare bounded re-checks")
    roles = report.get("instance_roles", {})
    need(all(i in roles for i in selected), "Repair target does not exist")
    if target != "scene": need(all(roles[i] == target for i in selected), "Repair category does not match the target's actual role")
    records = {r["id"]: r for r in report.get("evidence", [])}
    need(all(i in records and records[i]["status"] == "valid" and records[i]["binding"]["scene_sha256"] == scene_hash for i in ids),
         "Repair cites invalid or stale evidence")
    rows = [r for r in report["rows"] if r.get("evidence_id") in ids]
    for instance in selected:
        relevant = [r for r in rows if instance in r.get("instances", [])]
        need(relevant, "Evidence does not involve the instance to be modified: "+instance)
        if target != "fixture": need(any(r.get("repair_authorized") for r in relevant), "Observations are insufficient to authorize modifying the real instrument or scene")
        else:
            # Normal contact without anomalies cannot become grounds for a design change on the strength of model text.
            need(any(r.get("repair_authorized") or r.get("observation", {}).get("limit_exceeded") or
                     r.get("observation", {}).get("retention_failure_observed") for r in relevant),
                 "A fixture design trial requires evidence of interference or holding failure; ordinary contact is not enough to establish a design problem")
    normalized = [normalize(q) for q in checks]
    need(all(q["kind"] != "probe" and q["state"] == "initial" for q in normalized),
         "Cross-version re-checks use initial-state measurements; actual operation is verified by a full execution after re-planning")
    from .diagnostic_ledger import covers
    for evidence_id in ids:
        original=records[evidence_id].get("query",{})
        if original.get("state")=="initial" and original.get("kind") in {"contacts","configuration","mechanism","fit"}:
            need(any(q==original or covers(q,original) for q in normalized),
                 "Re-checks must cover the pose, object or mechanism travel where the problem was originally found; reading geometry parameters cannot replace the original measurement")
    def referenced(query,instance):
        if query["kind"]=="contacts" and not query["bodies"]:return True
        def strings(v):
            if isinstance(v,str):return [v]
            if isinstance(v,list):return [s for x in v for s in strings(x)]
            if isinstance(v,dict):return [s for x in v.values() for s in strings(x)]
            return []
        return any(s.startswith(instance+"__") for s in strings(query))
    need(all(any(referenced(q,i) for q in normalized) for i in selected),"Re-checks do not cover all instances to be modified")
    original_pairs=set()
    retention=False
    for row in rows:
        observation=row.get("observation",{})
        retention |= bool(observation.get("retention_failure_observed"))
        if row.get("kind")=="collision" or observation.get("limit_exceeded"):
            pair=row.get("geoms",observation.get("geoms",[]))
            if pair:original_pairs.add(tuple(sorted(pair)))
        for sample in row.get("samples",[]):
            original_pairs.update(tuple(sorted(c["geoms"])) for c in sample.get("collisions",[]))
    return {"version": 1, "target": target, "instances": selected, "reason": request["reason"],
            "scene_sha256": scene_hash, "evidence_ids": ids, "checks": normalized,
            "original_pairs":[list(p) for p in sorted(original_pairs)],"retention_failure":retention,
            "scope": "fixture_design_trial" if target == "fixture" else "confirmed_repair",
            "preserve": ["frozen_task", "instrument_specifications", "robot_parameters", "collision_checks"],
            "acceptance": "original_checks_and_full_continuous_execution"}


def check_obligations(report, ticket, records):
    # If scene names changed and no valid query remains, return a coverage gap; never transplant the old report onto the new version.
    results = [{"query": r["query"], "evidence_id": r["id"], "status": r["status"]} for r in records]
    def defects(record):
        result=set()
        for row in (record.get("result") or {}).get("rows",[]):
            if row.get("kind")=="collision": result.add(tuple(sorted(row.get("geoms",[]))))
            for sample in row.get("samples",[]):
                result.update(tuple(sorted(c["geoms"])) for c in sample.get("collisions",[]))
        return result
    regressions=[]
    comparisons=[]
    for before,after in zip(ticket.get("baseline_checks",[]),records):
        old,new=defects(before),defects(after)
        comparisons.append({"query":after["query"],"resolved_pairs":[list(p) for p in sorted(old-new)],
                            "remaining_pairs":[list(p) for p in sorted(old&new)]})
        regressions.extend({"query":after["query"],"new_pair":list(p)} for p in sorted(new-old))
    active_pairs=set()
    for row in report.get("rows",[]):
        observation=row.get("observation",{})
        if row.get("kind")=="collision" or observation.get("limit_exceeded"):
            pair=row.get("geoms",observation.get("geoms",[]))
            if pair:active_pairs.add(tuple(sorted(pair)))
    for record in records:active_pairs |= defects(record)
    remaining=active_pairs & {tuple(p) for p in ticket.get("original_pairs",[])}
    retention_remains=ticket.get("retention_failure",False) and report.get("failure",{}).get("code")=="grasp_lost"
    return {"ticket_sha256": digest(ticket), "source_scene_sha256": ticket["scene_sha256"],
            "candidate_scene_sha256": report["scene_sha256"], "checks": results,
            "coverage_complete": len(results) == len(ticket["checks"]) and all(r["status"] == "valid" for r in results),
            "regressions":regressions,"comparisons":comparisons,
            "remaining_original_pairs":[list(p) for p in sorted(remaining)],
            "original_problem_resolved":not remaining and not retention_remains,
            "full_execution_required": True}


def query_run(run_root, output, queries, *, resume=False):
    """Standalone offline follow-up query entry point; evidence goes to a new directory and the original execution record stays read-only."""
    from pathlib import Path
    from dataclasses import asdict
    from asset_prep.storage import atomic_json,run_lock
    from asset_prep.implementation import source_manifest,implementation_digest
    from .common import read,Settings,new_directory
    from .diagnostic_tools import VERSION
    run_root,output=Path(run_root).resolve(),Path(output).resolve()
    record=read(run_root/"recording.json")
    settings=Settings(**record["settings"])
    project=Path(__file__).resolve().parents[2]
    configuration={"version":VERSION,"run_root":str(run_root),"scene_sha256":record["scene_sha256"],
                   "source_receipt":digest(read(run_root/"files.json")),"settings":asdict(settings),
                   "implementation":implementation_digest(source_manifest(project))}
    need(not output.is_relative_to(run_root) and not run_root.is_relative_to(output),"Diagnostic output must not overwrite or nest with the original execution record")
    if not resume:new_directory(output,[run_root,record["scene_path"]])
    else:need(output.is_dir(),"Diagnosis to resume does not exist")
    with run_lock(output):
        if resume:need(read(output/"configuration.json")==configuration,"Diagnostic inputs or implementation changed; create a new diagnosis version")
        else:atomic_json(output/"configuration.json",configuration)
        state=read(output/"ledger.json") if (output/"ledger.json").exists() else {}
        def save():atomic_json(output/"ledger.json",state)
        report=inspect_with_ledger(record["scene_path"],record["scene_sha256"],settings,state,
            run_root=run_root,evidence_root=output/"trials",queries=queries,save=save)
        atomic_json(output/"report.json",report)
        return {"status":"needs_input" if report.get("last_diagnostic_batch",{}).get("exhausted") else "diagnosed",
                "report":str(output/"report.json"),"evidence":report.get("evidence",[]),
                "last_batch":report.get("last_diagnostic_batch"),"model_api_calls":0}
