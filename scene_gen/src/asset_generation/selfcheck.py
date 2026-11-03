"""Local self-check tool for the coding agent: builds model.py in the work directory and runs the single-asset checks.

Only quick feedback for the author's own testing, not acceptance: after submission the framework still independently rebuilds, checks and reviews.
Usage (in the agent work directory): ./check  ->  python -m asset_generation.selfcheck <work directory>
"""

import json
from pathlib import Path
import sys
import time

from asset_generation.check_report import report_status
from asset_generation.contracts import GenerationConfig, check_profile
from asset_generation.local import LocalBackend

DIAGNOSTICS = ("collisions", "disconnected_components", "surface_mismatches", "unsupported_components",
               "floating_articulations")


def compact(row, limit=8):
    """One check row reduced to what an author can act on."""
    item = {k: row[k] for k in ("id", "status", "detail") if k in row}
    for key in DIAGNOSTICS:
        values = row.get(key) or []
        if values:
            item[key] = [{k: v for k, v in value.items() if k in
                          ("geom1", "geom2", "geoms", "body", "distance_m", "excess_m", "repair_hint", "detail",
                           "joint", "parent_body", "free_directions", "penetration_lower_bound_m", "depth_method")}
                         for value in values[:limit]]
            item[key + "_total"] = len(values)
    for key in ("problems", "drifting_bodies", "high_blocking_contacts", "low_blocking_contacts", "reached",
                "lock_holds", "moved_when_active_mm", "moved_when_released_mm", "trapped_by", "unswept_joints"):
        if row.get(key) not in (None, [], {}):
            item[key] = row[key]
    if row.get("coverage_gaps"):
        item["coverage_gaps_total"] = len(row["coverage_gaps"])
        item["coverage_gap_examples"] = [g.get("detail", "")[:200] for g in row["coverage_gaps"][:3]]
    if row.get("engagement_count"):
        item["designed_engagements"] = row["engagement_count"]
    return item


def run(workspace):
    workspace = Path(workspace).resolve()
    job = json.loads((workspace / "selfcheck.json").read_text(encoding="utf-8"))
    runs = workspace / "checks"
    runs.mkdir(exist_ok=True)
    destination = runs / f"{len(list(runs.iterdir())) + 1:03d}"
    destination.mkdir()
    project = Path(__file__).resolve().parents[2]
    config = GenerationConfig()
    backend = LocalBackend(config, project)
    started = time.monotonic()
    report, renders = backend.build_and_check(workspace / "model.py", destination, job["asset_ids"],
                                              check_profile(job["profile"]))
    rows = report.get("checks", [])
    if job.get("role") == "tool" and (destination / "artifact").is_dir():
        from scene_generation.tool_contract import printability_check
        rows = rows + [printability_check(destination / "artifact")]
        report["status"] = report_status(rows)
    summary = {"status": report.get("status"),
               "errors": [compact(r) for r in rows if r.get("status") == "error"] or
                         ([report["error"]] if report.get("error") else []),
               "failing": [compact(r) for r in rows if r.get("status") == "fail"],
               "incomplete": [compact(r, 3) for r in rows if r.get("status") == "partial"],
               "passed": [r["id"] for r in rows if r.get("status") == "pass"],
               "renders": sorted(str(p) for p in Path(renders).glob("**/*.png")) if renders else [],
               "seconds": round(time.monotonic() - started, 1),
               "framework_budget_seconds": config.build_timeout_seconds,
               "note": "partial only means a measurement was unreliable or no probe exists; it is not a defect. "
                       "Every fail must be fixed. The framework recheck runs the same checks under the budget above "
                       "(per configuration); a model that takes longer is rejected as a timeout."}
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if summary["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "."))
