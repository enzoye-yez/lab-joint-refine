"""现有联合运行与通用证据工具之间的固定接入层。"""
from copy import deepcopy
from .common import need, digest, OperationError
from .diagnostic_tools import DiagnosticContext, normalize, CATALOG
from .diagnostic_ledger import EvidenceLedger


def collect(context, state, queries, save=lambda: None, *, count_stall=True):
    need(isinstance(queries, list) and 0 < len(queries) <= context.settings.diagnostic_batch, "诊断批次超限")
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
        need(record["binding"]["scene_sha256"] == report["scene_sha256"], "旧版本证据不可进入当前诊断", "stale_evidence")
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
    # 来源初态问题仍保留现有固定检查结果；执行失败观测不会被提升为初态缺陷。
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
    """提供证据支持的候选来源，不把观测自动解释成唯一根因。"""
    if report.get("failure", {}).get("scope") == "engineering":
        return {"status": "engineering", "candidates": [], "reason": "执行或仿真异常先处理工程问题"}
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
            "limitations": "多个原因可并存；局部成功不证明唯一原因，搜索失败不证明资产缺陷"}


def repair_ticket(report, request, target, scene_hash):
    need(target in {"asset", "fixture", "scene"}, "未知返修类别")
    need(report["scene_sha256"] == scene_hash, "返修依据属于旧场景", "stale_evidence")
    need(report.get("failure", {}).get("scope") != "engineering", "工程故障不授权改资产")
    need(isinstance(request, dict) and set(request) <= {"reason", "instances", "evidence_ids", "checks"}, "返修请求字段无效")
    need(isinstance(request.get("reason"), str) and request["reason"].strip(), "返修需要原因说明")
    selected = request.get("instances", [])
    checks = request.get("checks", [])
    ids = request.get("evidence_ids", [])
    need(isinstance(selected, list) and selected and len(set(selected)) == len(selected), "返修须明确实例")
    need(isinstance(ids, list) and ids, "返修须引用实际证据编号")
    need(isinstance(checks, list) and 0 < len(checks) <= 16, "返修须声明有界复查")
    roles = report.get("instance_roles", {})
    need(all(i in roles for i in selected), "返修对象不存在")
    if target != "scene": need(all(roles[i] == target for i in selected), "返修类别与实际对象角色不符")
    records = {r["id"]: r for r in report.get("evidence", [])}
    need(all(i in records and records[i]["status"] == "valid" and records[i]["binding"]["scene_sha256"] == scene_hash for i in ids),
         "返修引用无效或过期证据")
    rows = [r for r in report["rows"] if r.get("evidence_id") in ids]
    for instance in selected:
        relevant = [r for r in rows if instance in r.get("instances", [])]
        need(relevant, "证据未涉及待修改实例："+instance)
        if target != "fixture": need(any(r.get("repair_authorized") for r in relevant), "观测不足以授权修改真实仪器或场景")
        else:
            # 无异常的正常接触不能凭模型文字变成设计改良依据。
            need(any(r.get("repair_authorized") or r.get("observation", {}).get("limit_exceeded") or
                     r.get("observation", {}).get("retention_failure_observed") for r in relevant),
                 "夹具设计试验须有干涉或保持失败证据；普通接触不足以判定设计问题")
    normalized = [normalize(q) for q in checks]
    need(all(q["kind"] != "probe" and q["state"] == "initial" for q in normalized),
         "跨版本复查使用初态量测；实际操作由重新规划后的完整执行验证")
    from .diagnostic_ledger import covers
    for evidence_id in ids:
        original=records[evidence_id].get("query",{})
        if original.get("state")=="initial" and original.get("kind") in {"contacts","configuration","mechanism","fit"}:
            need(any(q==original or covers(q,original) for q in normalized),
                 "复查须覆盖原先发现问题的姿态、对象或机构行程；读取几何参数不能代替原量测")
    def referenced(query,instance):
        if query["kind"]=="contacts" and not query["bodies"]:return True
        def strings(v):
            if isinstance(v,str):return [v]
            if isinstance(v,list):return [s for x in v for s in strings(x)]
            if isinstance(v,dict):return [s for x in v.values() for s in strings(x)]
            return []
        return any(s.startswith(instance+"__") for s in strings(query))
    need(all(any(referenced(q,i) for q in normalized) for i in selected),"复查没有覆盖全部待修改实例")
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
    # 新场景名变化且无有效查询时返回覆盖缺口，不把旧报告移植到新版本。
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
    """独立离线补查入口；证据写在新目录，原执行记录保持只读。"""
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
    need(not output.is_relative_to(run_root) and not run_root.is_relative_to(output),"诊断输出不能覆盖或嵌套原执行记录")
    if not resume:new_directory(output,[run_root,record["scene_path"]])
    else:need(output.is_dir(),"待恢复诊断不存在")
    with run_lock(output):
        if resume:need(read(output/"configuration.json")==configuration,"诊断输入或实现变化，须新建诊断版本")
        else:atomic_json(output/"configuration.json",configuration)
        state=read(output/"ledger.json") if (output/"ledger.json").exists() else {}
        def save():atomic_json(output/"ledger.json",state)
        report=inspect_with_ledger(record["scene_path"],record["scene_sha256"],settings,state,
            run_root=run_root,evidence_root=output/"trials",queries=queries,save=save)
        atomic_json(output/"report.json",report)
        return {"status":"needs_input" if report.get("last_diagnostic_batch",{}).get("exhausted") else "diagnosed",
                "report":str(output/"report.json"),"evidence":report.get("evidence",[]),
                "last_batch":report.get("last_diagnostic_batch"),"model_api_calls":0}
