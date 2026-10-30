"""编码 agent 的本地自检工具：构建工作目录中的 model.py 并运行单资产检查。

只为作者自测提供快速反馈，不是验收：框架提交后仍独立重建、检查和审查。
用法（在 agent 工作目录中）：./check  →  python -m asset_generation.selfcheck <工作目录>
"""

import json
from pathlib import Path
import sys

from asset_generation.contracts import GenerationConfig, check_profile
from asset_generation.local import LocalBackend

DIAGNOSTICS = ("collisions", "disconnected_components", "surface_mismatches", "unsupported_components")


def compact(row, limit=8):
    """One check row reduced to what an author can act on."""
    item = {k: row[k] for k in ("id", "status", "detail") if k in row}
    for key in DIAGNOSTICS:
        values = row.get(key) or []
        if values:
            item[key] = [{k: v for k, v in value.items() if k in
                          ("geom1", "geom2", "geoms", "body", "distance_m", "excess_m", "repair_hint", "detail")}
                         for value in values[:limit]]
            item[key + "_total"] = len(values)
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
    backend = LocalBackend(GenerationConfig(), project)
    report, renders = backend.build_and_check(workspace / "model.py", destination, job["asset_ids"],
                                              check_profile(job["profile"]))
    rows = report.get("checks", [])
    summary = {"status": report.get("status"),
               "errors": [compact(r) for r in rows if r.get("status") == "error"] or
                         ([report["error"]] if report.get("error") else []),
               "failing": [compact(r) for r in rows if r.get("status") == "fail"],
               "incomplete": [compact(r, 3) for r in rows if r.get("status") == "partial"],
               "passed": [r["id"] for r in rows if r.get("status") == "pass"],
               "renders": sorted(str(p) for p in Path(renders).glob("**/*.png")) if renders else [],
               "note": "partial 仅因量测不可靠或缺少探测器时不是缺陷；fail 必须修复。"}
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if summary["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "."))
