"""Presentation figures for a scene run, one set per asset configuration: studio stills,
mechanism end poses and a turntable video that plays every mechanism; lit overviews of the
assembled scene; and a MuJoCo package per asset for opening locally.

CPU rendering only (Mesa EGL). Output goes under reports/<run>/ and never into the run's
evidence tree.

  .venv/bin/python scripts/render_figures.py --run runs/<name> [--jobs asset06 asset11] [--no-video] [--workers 12]
"""
import argparse
import json
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))

from asset_prep.state_archive import read_state  # noqa: E402

SCENE_VIEWS = {"overview_a": (135, -38, .62), "overview_b": (45, -38, .62), "overview_c": (225, -34, .62),
               "top": (90, -89.5, .55), "low_robot": (160, -18, .45)}

PLAYER = '''"""Open one asset configuration in the MuJoCo viewer and play its mechanisms.

  python play_mechanisms.py <configuration>_poses.xml

Keyframe 0 is the home pose; each further keyframe moves one mechanism group to its far
limit. The script blends home -> key -> home for each in turn, forever. The same keyframes
are also listed in the viewer's own "Key" panel.
"""
import math
import sys
import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
home = model.key_qpos[0].copy()
with mujoco.viewer.launch_passive(model, data) as viewer:
    k = 1
    while viewer.is_running():
        if model.nkey < 2:
            data.qpos[:] = home; mujoco.mj_forward(model, data); viewer.sync(); time.sleep(.05); continue
        print("mechanism:", model.key(k).name)
        for step in range(120):
            amount = .5 - .5*math.cos(2*math.pi*step/120)
            data.qpos[:] = home + amount*(model.key_qpos[k]-home)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1/60)
            if not viewer.is_running():
                break
        k = k+1 if k+1 < model.nkey else 1
'''


def latest_jobs(run):
    state = read_state(run/"state.json")["scene"]
    plan = {j["id"]: j for j in state["plan"]["jobs"]}
    out = {}
    for key, job in state["jobs"].items():
        if job.get("artifact"):
            out[key] = {"artifact": str(run/job["artifact"]), "status": job.get("status"),
                        "asset_ids": plan.get(key, {}).get("asset_ids", []), "role": plan.get(key, {}).get("role")}
    return state, out


def poses(model_path, mechanisms):
    """Home qpos and one far-limit qpos per mechanism group, welded parts carried along."""
    import mujoco
    from asset_generation import studio
    from asset_generation.motion import _welded_followers, _follow
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    home = data.qpos.copy()
    followers = _welded_followers(model, data)
    out = [("home", home.copy())]
    for ids, ranges, direction, held in studio.mechanism_schedule(model, mechanisms, max_slots=64, data=data):
        if direction < 0:
            continue
        data.qpos[:] = studio.far_pose(model, studio._pose(model, home, held), ranges)
        _follow(model, data, followers)
        out.append(("+".join(ids)[:60], data.qpos.copy()))
    return out


def export_mujoco(model_file, mechanisms, target, name):
    """<name>_poses.xml next to the copied artifact: the model with a keyframe per mechanism pose."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(target/model_file))
    for label, qpos in poses(target/model_file, mechanisms):
        spec.add_key(name=label, qpos=qpos.tolist())
    spec.compile()
    path = target/f"{name}_poses.xml"
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


def configurations(job):
    from asset_generation.configurations import catalogue
    return sorted(catalogue(Path(job["artifact"])))


def render_configuration(args):
    """One configuration of one asset: stills, mechanism poses, turntable, MuJoCo export."""
    key, job, name, out, mujoco_root, video = args
    from asset_generation import studio
    from asset_generation.configurations import selection
    artifact = Path(job["artifact"])
    single = configurations(job) == ["default"]
    _, model_file, manifest = selection(artifact, None if single else name)
    mechanisms = manifest.get("mechanisms") or []
    target = out/key if single else out/key/name
    model_path = artifact/model_file
    stills = studio.still_views(model_path, target)
    mech = studio.mechanism_stills(model_path, mechanisms, target)
    entry = {"name": name, "model_file": model_file,
             "stills": [str(p.relative_to(out.parent)) for p in stills],
             "mechanism_stills": {k: [str(p.relative_to(out.parent)) for p in v] for k, v in mech.items()},
             "mechanisms": [{k: m.get(k) for k in ("id", "requirement", "joints", "lock", "actuation")} for m in mechanisms]}
    if video:
        clip = studio.turntable(model_path, mechanisms, target/"turntable.mp4", seconds=10., fps=24, width=960, height=720)
        entry["video"] = str(Path(clip["video"]).relative_to(out.parent))
        entry["video_slots"] = clip["slots"]
    exported = export_mujoco(model_file, mechanisms, mujoco_root/key, name)
    entry["mujoco"] = str(exported.relative_to(mujoco_root))
    print("rendered", key, name, flush=True)
    return key, entry


def render_assembled(args):
    """One job's scene instances together (what the per-configuration stills show apart)."""
    key, xml, home, instances, target = args
    from asset_generation import studio
    paths = studio.assembled_views(xml, home, instances, target)
    print("assembled", key, flush=True)
    return key, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--scene-candidate", type=Path, help="unpublished scene artifact to show instead (snapshot of a running batch)")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run = args.run.resolve()
    report = (args.output or PROJECT/"reports"/run.name).resolve()
    out, mujoco_root = report/"figures", report/"mujoco"
    out.mkdir(parents=True, exist_ok=True); mujoco_root.mkdir(parents=True, exist_ok=True)
    state, jobs = latest_jobs(run)
    selected = [key for key in sorted(jobs) if not args.jobs or key in args.jobs]
    for key in selected:
        # One copy of each artifact; every configuration adds its own <name>_poses.xml to it.
        target = mujoco_root/key
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(jobs[key]["artifact"], target, ignore=shutil.ignore_patterns("*.log"))
        (target/"play_mechanisms.py").write_text(PLAYER, encoding="utf-8")
    work = [(key, jobs[key], name, out, mujoco_root, not args.no_video) for key in selected for name in configurations(jobs[key])]
    # Largest models first so the long renders start early.
    work.sort(key=lambda w: -Path(w[1]["artifact"]).stat().st_size)
    scene_artifact = None
    published = state.get("published_scene")
    if args.scene_candidate and not args.no_scene:
        scene_artifact = args.scene_candidate.resolve()
    elif published and not args.no_scene:
        scene_artifact = run/published["directory"]/"scene"/"artifact"
        if not scene_artifact.is_dir():
            scene_artifact = run/published["artifact"]
    assembled = []
    if scene_artifact is not None:
        scene_json = json.loads((scene_artifact/"scene.json").read_text(encoding="utf-8"))
        for key in selected:
            instances = [c["instance"] for c in scene_json.get("component_versions", []) if c.get("job") == key]
            if instances:
                assembled.append((key, str(scene_artifact/"model.xml"), scene_json["home_qpos"], instances, out/key/"assembled"))
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(render_configuration, work))
        views = dict(pool.map(render_assembled, assembled))
    records = []
    for key in selected:
        record = {"job": key, "status": jobs[key]["status"], "asset_ids": jobs[key]["asset_ids"],
                  "configurations": sorted((e for k, e in done if k == key), key=lambda e: e["name"]),
                  "assembled": [str(p.relative_to(out.parent)) for p in views.get(key, [])]}
        (out/key).mkdir(parents=True, exist_ok=True)
        (out/key/"figure.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        records.append(record)
    archive = report/"mujoco_assets.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(mujoco_root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(mujoco_root))
    scene = None
    if scene_artifact is not None:
        from asset_generation import studio
        artifact = scene_artifact
        home = json.loads((artifact/"scene.json").read_text(encoding="utf-8"))["home_qpos"]
        print("scene", artifact, flush=True)
        paths = studio.scene_lit(artifact/"model.xml", home, out/"scene", SCENE_VIEWS)
        scene = {"directory": str(artifact.relative_to(run)) if artifact.is_relative_to(run) else str(artifact),
                 "candidate": bool(args.scene_candidate), "views": [str(p.relative_to(report)) for p in paths]}
    (out/"figures.json").write_text(json.dumps({"run": run.name, "assets": records, "scene": scene,
                                                "mujoco_archive": archive.name}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out/"figures.json", "and", archive)


if __name__ == "__main__":
    main()
