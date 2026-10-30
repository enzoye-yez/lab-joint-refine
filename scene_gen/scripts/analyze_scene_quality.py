"""只读汇总一次场景运行的质量与修复情况；不改运行记录、不调用模型。

用法：python scripts/analyze_scene_quality.py <运行目录> [--json 输出文件]
运行目录需含 result.json 与 stages/（runs/<运行名> 或导出包的 output/）。
"""

import argparse
import collections
import json
from pathlib import Path
import re
import statistics


def latest_scene_record(root):
    records = sorted(root.glob("stages/*_scene/record.json"))
    return json.loads(records[-1].read_text(encoding="utf-8")) if records else None


def check_summary(checks):
    family = lambda check: re.split(r"[:/]", check["id"])[0]
    by_status = collections.Counter(check["status"] for check in checks)
    failing = collections.Counter(family(c) for c in checks if c["status"] == "fail")
    incomplete = collections.Counter(family(c) for c in checks if c["status"] in {"partial", "unknown", "error"})
    return {"by_status": dict(by_status), "failing_families": dict(failing), "incomplete_families": dict(incomplete)}


def material_summary(checks):
    rows, depths = [], []
    for check in checks:
        if not check["id"].startswith("material_integrity"):
            continue
        pens = [-c["distance_m"] for c in check.get("collisions", []) if c.get("distance_m") is not None]
        gaps = check.get("coverage_gaps", [])
        depths += pens
        rows.append({"instance": check["id"].split(":", 1)[-1], "status": check["status"],
                     "collisions": check.get("collision_count", len(pens)),
                     "max_penetration_mm": round(max(pens) * 1000, 1) if pens else 0.0,
                     "disconnected": len(check.get("disconnected_components", [])),
                     "measurement_gaps": sum(g.get("error_kind") == "measurement_error" for g in gaps),
                     "other_gaps": sum(g.get("error_kind") != "measurement_error" for g in gaps)})
    rows.sort(key=lambda row: -row["collisions"])
    depths.sort()
    share = lambda limit: round(sum(d <= limit + 1e-9 for d in depths) / len(depths), 3) if depths else None
    return {"instances": len(rows), "status": dict(collections.Counter(r["status"] for r in rows)),
            "collision_pairs": len(depths),
            "penetration_mm": {"median": round(statistics.median(depths) * 1000, 2) if depths else None,
                               "p90": round(depths[int(0.9 * len(depths))] * 1000, 2) if depths else None,
                               "share_le_1mm": share(0.001), "share_le_3mm": share(0.003)},
            "worst": rows[:15]}


def repair_summary(result):
    progress = result.get("repair_progress") or []
    targets = collections.Counter(t for p in progress for t in (p.get("target_checks") or [])[:3])
    resolved = collections.Counter(t for p in progress for t in (p.get("comparison") or {}).get("resolved_checks", []))
    regressions = collections.Counter(n["check"] for p in progress
                                      for n in (p.get("comparison") or {}).get("new_defects", []))
    return {"rounds": len(progress),
            "outcomes": dict(collections.Counter(p.get("outcome") for p in progress)),
            "selection": dict(collections.Counter(p.get("selection") for p in progress)),
            "top_targets": targets.most_common(10), "resolved": dict(resolved),
            "regressions_by_check": dict(regressions)}


def stage_number(stage):
    return int(Path(str(stage)).name.split("_", 1)[0])


def asset_versions(root, result, record):
    """场景实际使用的每个任务版本；早于首个场景修复轮的版本视为首稿（导出包可能只含被使用的阶段）。"""
    rounds = [p.get("cycle") or p.get("scene") for p in result.get("repair_progress") or []]
    scenes = [stage_number(s) for s in rounds if s] or [stage_number(p.parent.name)
                                                      for p in root.glob("stages/*_scene/record.json")]
    first_scene = min(scenes) if scenes else None
    used = (record or {}).get("job_versions") or {}
    rows = {job: {"stage": stage, "first_draft": first_scene is not None and stage_number(stage) < first_scene}
            for job, stage in used.items()}
    return {"jobs": len(rows), "still_first_draft": sum(r["first_draft"] for r in rows.values()), "versions": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    root = args.run
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    record = latest_scene_record(root)
    checks = (record or {}).get("report", {}).get("checks", [])
    summary = {"status": result.get("status"), "budget": result.get("budget"),
               "requests": result.get("requests"),
               "job_status": dict(collections.Counter(v if isinstance(v, str) else v.get("status")
                                                      for v in (result.get("jobs") or {}).values())),
               "checks": check_summary(checks), "material_integrity": material_summary(checks),
               "repairs": repair_summary(result), "assets": asset_versions(root, result, record)}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
