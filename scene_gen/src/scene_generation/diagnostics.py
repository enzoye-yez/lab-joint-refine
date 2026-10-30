"""Bounded factual triage; unknown measurements are never reclassified as defects."""
from collections import Counter
from asset_generation.check_report import incomplete


def triage(report):
    rows=[]
    for row in report.get("checks",[]):
        if row.get("status") not in {"fail","partial","error"} and not incomplete(row):continue
        # Some producers expose the same list through both aliases; do not double count.
        gaps=row.get("coverage_gaps") or row.get("measurement_errors") or []
        kinds=Counter(g.get("error_kind","unspecified") for g in gaps)
        if row.get("unexamined_pairs") and "resource_limit" not in kinds:kinds["resource_limit"]=1
        rows.append({"check":row["id"],"confirmed_defect":row.get("status")=="fail",
            "evidence_incomplete":incomplete(row),"gap_kinds":dict(kinds),
            "execution_error":row.get("status")=="error",
            "next_step":"diagnose_checker" if row.get("status")=="error" else
                        "repair_confirmed_and_measure_unknown" if row.get("status")=="fail" and incomplete(row) else
                        "repair_confirmed" if row.get("status")=="fail" else "obtain_evidence"})
    return {"counts":dict(Counter(r["next_step"] for r in rows)),"checks":rows[:64],"total_checks":len(rows),"truncated":len(rows)>64,
            "detail_reference":"feedback.checks[id]",
            "rule":"错误类别来自检查记录；未知不等于资产缺陷，缺陷与证据缺口可同时存在"}


def localization(row):
    names=set()
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in {"body","geom","geom1","geom2","object_geom","support_geom"} and isinstance(item,str):names.add(item)
                elif key in {"bodies","geoms"} and isinstance(item,list):names.update(x for x in item if isinstance(x,str))
                elif isinstance(item,(dict,list)):visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for field in ("unsupported_components","disconnected_components","surface_mismatches","collisions"):
        visit(row.get(field,[]))
    return {"model_entities":sorted(names)[:32],"total_entities":len(names),"truncated":len(names)>32,
            "source":"actual check records; names are not editable permission grants"}
