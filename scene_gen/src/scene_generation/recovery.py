"""检查事实与恢复决策分离；未知不可作为通过，也不可要求作者猜测几何。"""

from asset_prep.storage import digest
from asset_generation.check_report import report_status, incomplete


def isolate_asset_failures(report, plan):
    """Map trusted per-asset worker failures to the same fault contract as rechecks.

    Only classified environmental/transient failures are local. A shared checker
    bug, malformed report or unspecified exception still needs engineering repair.
    Descendants and every copy of a shared asset remain blocked.
    """
    from copy import deepcopy
    result = deepcopy(report)
    faults = result.setdefault("recovery_failures", [])
    if (report.get("status")=="error" and report.get("error_kind")!="measurement_error"
            and not faults):
        # Top-level scene worker failure has no proven object scope.
        return result
    for row in result.get("checks", []):
        failure = row.get("execution_failure", {})
        if (row.get("status") != "error" or row.get("failure_scope") != "asset_job"
                or failure.get("kind") not in {"temporary", "environment"}):
            continue
        blocked = {row["asset_job"]}
        while True:
            expanded = blocked | {j["id"] for j in plan["jobs"] if blocked.intersection(j["depends_on"])}
            if expanded == blocked:
                break
            blocked = expanded
        row["instances"] = [i["id"] for i in plan["instances"] if i["job"] in blocked]
        identity = digest({"asset_execution": row})
        if not any(f["id"] == identity for f in faults):
            faults.append({"id":identity, "status":"open", "scope":"checks", "kind":failure["kind"],
                "remaining_checks":[row["id"]], "remaining_pairs":[], "affected_checks":[row["id"]],
                "capability":"asset_worker", "diagnostic":failure, "resolutions":[],
                "components":failure.get("components", ["*"]), "retryable":failure.get("retryable", False)})
    return result


def gaps(report):
    affected = {c for f in report.get('recovery_failures',[]) if f['status']=='open'
                for c in f.get('remaining_checks',[]) if f.get('scope')!='localization'}
    return [r for r in report.get('checks',[]) if (incomplete(r) and r.get('status') != 'error')
            or r['id'] in affected
            or r.get('status') == 'partial'
            or r.get('status') == 'error' and r.get('error_kind',report.get('error_kind')) == 'measurement_error']


def decide(report, review=None):
    """补查在场景构建内执行后，按剩余证据选择动作。"""
    # 检查器程序异常不交给几何作者修复，即使同份报告还含有缺陷。
    if any(f['status']=='open' for f in report.get('recovery_failures',[])):
        return 'repair' if repairable_rows(report) else 'needs_environment'
    if report.get('status')=='error' and report.get('error_kind')!='measurement_error':
        return 'needs_environment'
    if report.get('error_kind') in {'check_execution','backend_execution'} or any(
            r.get('status') == 'error' and r.get('error_kind',report.get('error_kind')) != 'measurement_error'
            for r in report.get('checks', [])):
        return 'needs_environment'
    defects = [r for r in report.get('checks',[]) if r.get('status') == 'fail']
    if defects or review and review.get('verdict') == 'repair':
        return 'repair'
    if gaps(report):
        return 'needs_check_support'
    if report.get('status') in {'error','partial'}:
        return 'needs_environment'
    if report.get('status') == 'pass' and review and review.get('verdict') == 'accept':
        return 'candidate_ready'
    return 'needs_review'


def evidence_key(plan, assets):
    # 版本目录名变化但源文件内容相同时，不应获得无穷次相同补查。
    return digest({'plan':plan,'assets':assets,'method':'main_check_recovery_v1'})


def repairable_rows(report):
    """Only explicitly local transient/environment faults can leave independent repairs available."""
    rows=report.get("checks",[])
    defects=[r for r in rows if r.get("status")=="fail"]
    faults=[f for f in report.get("recovery_failures",[]) if f["status"]=="open"]
    if not faults:
        if report.get("status")=="error" and report.get("error_kind")!="measurement_error":return []
        if any(r.get("status")=="error" and r.get("error_kind",report.get("error_kind"))!="measurement_error" for r in rows):return []
        return defects
    if any(f.get("scope")!="checks" or f.get("kind")=="program" or not f.get("remaining_checks") for f in faults):return []
    blocked={c for f in faults for c in f["remaining_checks"]}
    # A localized fault cannot hide another unclassified failure.
    if any(r.get("status")=="error" and r["id"] not in blocked
           and r.get("error_kind",report.get("error_kind"))!="measurement_error" for r in rows):return []
    by_id={r["id"]:r for r in rows}
    if any(c not in by_id or not by_id[c].get("instances") for c in blocked):return []
    instances={i for c in blocked for i in by_id[c]["instances"]}
    return [r for r in defects if r["id"] not in blocked and r.get("instances") and not instances.intersection(r["instances"])]


def located_repairs(report, plan):
    """Confirmed, editable defects can proceed while unrelated evidence stays unknown."""
    instances = {i['id'] for i in plan['instances']}
    layout = 'layout' in plan
    return [row for row in repairable_rows(report)
            if instances.intersection(row.get('instances', []))
            or layout and (row['id'].startswith('table_')
                           or {'robot', 'environment'}.intersection(row.get('instances', [])))]
