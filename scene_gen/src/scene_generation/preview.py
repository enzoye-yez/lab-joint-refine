"""View an exported scene; does not call the model, run modeling scripts, or advance simulation time."""

import argparse
import html
import hashlib
import json
import time
from pathlib import Path

from asset_generation.checks import load_model, load_scene_model, render_views
from asset_generation.contracts import read_json, require
from asset_generation.pipeline import snapshot
from asset_prep.errors import PrepError
from asset_prep.storage import atomic_json, digest, safe_path


def preview_summary(result):
    """Only navigation and qualification metadata; never copy diagnostic rows."""
    require(isinstance(result, dict), "result report must be an object")
    require(isinstance(result.get("assembly", {}), dict), "result assembly status must be an object")
    view = {k: result.get(k) for k in ("status", "provenance", "validation_mode", "qualification",
                                      "scene_artifact", "scene_sha256", "verification_scope", "quality_progress")}
    view["assembly"] = {k: result.get("assembly", {}).get(k) for k in
                        ("available", "complete", "missing_jobs", "quality_accepted", "scene_version")}
    view["unverified_count"] = len(result.get("unverified", []))
    view["report"] = "result.json"
    return view


def report_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_preview_index(root, result):
    atomic_json(Path(root)/"preview.json", {"schema_version": 1,
        "result_sha256": report_hash(Path(root)/"result.json"), "view": preview_summary(result)})


def resolve_artifact(project, run):
    root = safe_path(Path(project)/"runs", run)
    report = safe_path(root, "result.json")
    require(report.is_file(), "missing result file result.json; no scene result has been published yet")
    index = safe_path(root, "preview.json")
    if index.is_file():
        record = read_json(index)
        require(isinstance(record, dict) and record.get("schema_version") == 1 and isinstance(record.get("view"), dict), "invalid preview index format")
        require(record.get("result_sha256") == report_hash(report), "preview index does not match the result version; republish the result")
        result = record["view"]
    else:
        # Legacy runs predate the small index. Bounded compatibility read, no archive mutation.
        require(report.stat().st_size <= 64000000, "legacy result report exceeds the 64 MB compatibility read limit; migrate the preview index")
        result = preview_summary(read_json(report, 64000000))
    require(isinstance(result.get("scene_artifact"), str), "no published scene yet; partial scenes can also be previewed, check the job status")
    artifact = safe_path(root, result["scene_artifact"])
    require((artifact/"model.xml").is_file(), "model.xml file of the published scene is missing")
    require(digest(snapshot(artifact)) == result["scene_sha256"], "scene files changed; the original result evidence cannot be reused")
    return root, artifact, result


def resolve_asset(project, run, job):
    root = safe_path(Path(project)/"runs", run)
    from asset_prep.state_archive import read_state
    state = read_state(root/"state.json")
    record = state["scene"]["jobs"].get(job)
    require(record and record.get("artifact"), "the specified job has no viewable asset yet")
    artifact = safe_path(root, record["artifact"])
    stage = next(s for s in state["scene"]["stages"]
                 if s["directory"] == record.get("artifact_stage", record["stage"]))
    stage_root = safe_path(root, stage["directory"])
    require(artifact.is_relative_to(stage_root), "asset does not belong to the declared archive")
    prefix = artifact.relative_to(stage_root).as_posix() + "/"
    expected = {k.replace("\\", "/")[len(prefix):]:v for k,v in stage["files"].items()
                if k.replace("\\", "/").startswith(prefix)}
    require(expected and snapshot(artifact) == expected, "asset archive digest mismatch")
    return artifact, record["status"]


def export_preview(artifact, destination, result):
    destination = Path(destination)
    require(not destination.exists(), "preview directory already exists; choose another directory to preserve evidence")
    model = load_scene_model(artifact)
    meta = read_json(Path(artifact)/"scene.json",8000000)
    paths = render_views(model,[("scene",meta["home_qpos"])],destination,width=1280,height=960)
    title = "Static scene preview (no operation simulation run)"
    details = json.dumps({k:result.get(k) for k in ("status","provenance","validation_mode","qualification","unverified_count","report","assembly")},ensure_ascii=False,indent=2)
    figures = "".join(f'<figure><img src="{html.escape(p,quote=True)}"><figcaption>{html.escape(p)}</figcaption></figure>' for p in paths)
    (destination/"index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<title>{title}</title><style>body{{max-width:1200px;margin:32px auto;font:16px system-ui;background:#f2f4f7;color:#16202a}}'
        'figure{display:inline-block;margin:8px;max-width:100%}img{width:560px;max-width:100%}pre{white-space:pre-wrap;background:white;padding:20px}</style>'
        f'<h1>{title}</h1><p>Images show the initial pose only. Engineering samples or results that failed acceptance cannot serve as evidence of formal experimental success.</p>{figures}<pre>{html.escape(details)}</pre></html>',encoding="utf-8")
    return destination/"index.html"


def show_static(artifact, *, asset=False):
    import mujoco
    import mujoco.viewer
    model = load_model(artifact) if asset else load_scene_model(artifact)
    data = mujoco.MjData(model)
    meta = ({"home_qpos":data.qpos.copy(), "home_ctrl":data.ctrl.copy()} if asset else
            read_json(Path(artifact)/"scene.json",8000000))
    data.qpos[:] = meta["home_qpos"]
    data.ctrl[:] = meta["home_ctrl"]
    from asset_generation.static_state import geometry_state
    geometry_state(model,data)
    frozen_qpos = data.qpos.copy()
    with mujoco.viewer.launch_passive(model,data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                data.qpos[:] = frozen_qpos
                data.qvel[:] = 0
                data.ctrl[:] = meta["home_ctrl"]
                data.time = 0
                geometry_state(model,data)
            viewer.sync()
            time.sleep(1/30)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Statically view a published full or partial scene with mouse rotate/zoom; no operation simulation is run")
    parser.add_argument("--run",required=True,help="run name under the runs (run results) directory")
    parser.add_argument("--job",help="view a specific single-asset job, e.g. asset_A06; candidates pending review can be viewed")
    parser.add_argument("--render-only",action="store_true",help="only export the three views and the web page; do not open an interactive window")
    parser.add_argument("--output",type=Path,help="new preview output directory, default runs/<run name>/preview")
    args = parser.parse_args(argv)
    try:
        project = Path(__file__).resolve().parents[2]
        if args.job:
            require(not args.render_only and args.output is None, "single-asset viewing does not use --render-only or --output")
            artifact, status = resolve_asset(project,args.run,args.job)
            print(f"Single-asset static view: {args.job}; review status {status}; having a model does not mean acceptance passed.")
            show_static(artifact, asset=True)
            return 0
        root, artifact, result = resolve_artifact(project,args.run)
        if args.render_only:
            print(export_preview(artifact,args.output or root/"preview",result))
        else:
            require(args.output is None,"--output is only for --render-only")
            print("Static view: "+result["status"]+"; close the window to exit; physics time is not advanced.")
            show_static(artifact)
        return 0
    except (PrepError,OSError,ValueError,KeyError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
