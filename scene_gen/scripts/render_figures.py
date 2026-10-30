"""Presentation figures for a scene run: per-asset studio stills, mechanism end poses and a
turntable video that plays every mechanism; plus lit overviews of the assembled scene.

CPU rendering only (Mesa EGL). Output goes under reports/<run>/figures and never into
the run's evidence tree.

  .venv/bin/python scripts/render_figures.py --run runs/<name> [--jobs asset06 asset11] [--no-video]
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))

from asset_prep.state_archive import read_state  # noqa: E402
from asset_generation import studio  # noqa: E402

SCENE_VIEWS = {"overview_a": (135, -38, .62), "overview_b": (45, -38, .62), "overview_c": (225, -34, .62),
               "top": (90, -89.5, .55), "low_robot": (160, -18, .45)}


def latest_jobs(run):
    state = read_state(run/"state.json")["scene"]
    plan = {j["id"]: j for j in state["plan"]["jobs"]}
    out = {}
    for key, job in state["jobs"].items():
        if job.get("artifact"):
            out[key] = {"artifact": run/job["artifact"], "status": job.get("status"),
                        "asset_ids": plan.get(key, {}).get("asset_ids", []), "role": plan.get(key, {}).get("role")}
    return state, out


def render_job(key, job, out, video=True):
    artifact = job["artifact"]
    manifest = json.loads((artifact/"interfaces.json").read_text(encoding="utf-8"))
    mechanisms = manifest.get("mechanisms") or []
    target = out/key
    stills = studio.still_views(artifact/"model.xml", target)
    poses = studio.mechanism_stills(artifact/"model.xml", mechanisms, target)
    record = {"job": key, "status": job["status"], "asset_ids": job["asset_ids"],
              "stills": [str(p.relative_to(out.parent)) for p in stills],
              "mechanism_stills": {k: [str(p.relative_to(out.parent)) for p in v] for k, v in poses.items()},
              "mechanisms": [{k: m.get(k) for k in ("id", "requirement", "joints", "lock", "actuation")} for m in mechanisms]}
    if video:
        clip = studio.turntable(artifact/"model.xml", mechanisms, target/"turntable.mp4")
        record["video"] = str(Path(clip["video"]).relative_to(out.parent))
        record["video_slots"] = clip["slots"]
    (target/"figure.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-scene", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.output or PROJECT/"reports"/run.name/"figures").resolve()
    out.mkdir(parents=True, exist_ok=True)
    state, jobs = latest_jobs(run)
    records = []
    for key in sorted(jobs):
        if args.jobs and key not in args.jobs:
            continue
        print("asset", key, flush=True)
        records.append(render_job(key, jobs[key], out, video=not args.no_video))
    scene = None
    published = state.get("published_scene")
    if published and not args.no_scene:
        artifact = run/published["directory"]/"scene"/"artifact"
        home = json.loads((artifact/"scene.json").read_text(encoding="utf-8"))["home_qpos"]
        print("scene", artifact, flush=True)
        paths = studio.scene_lit(artifact/"model.xml", home, out/"scene", SCENE_VIEWS)
        scene = {"directory": published["directory"], "views": [str(p.relative_to(out.parent)) for p in paths]}
    (out/"figures.json").write_text(json.dumps({"run": run.name, "assets": records, "scene": scene},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out/"figures.json")


if __name__ == "__main__":
    main()
