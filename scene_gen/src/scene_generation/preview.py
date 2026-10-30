"""查看已导出场景；不调用模型、不执行建模脚本、不推进仿真时间。"""

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
    require(isinstance(result, dict), "结果报告必须为对象")
    require(isinstance(result.get("assembly", {}), dict), "结果装配状态必须为对象")
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
    require(report.is_file(), "缺少结果文件 result.json；尚未发布场景结果")
    index = safe_path(root, "preview.json")
    if index.is_file():
        record = read_json(index)
        require(isinstance(record, dict) and record.get("schema_version") == 1 and isinstance(record.get("view"), dict), "预览索引格式无效")
        require(record.get("result_sha256") == report_hash(report), "预览索引与结果版本不一致；请重新发布结果")
        result = record["view"]
    else:
        # Legacy runs predate the small index. Bounded compatibility read, no archive mutation.
        require(report.stat().st_size <= 64000000, "旧版结果报告超过 64 MB 兼容读取上限；需迁移预览索引")
        result = preview_summary(read_json(report, 64000000))
    require(isinstance(result.get("scene_artifact"), str), "尚无已发布场景；部分场景也可预览，请查看任务状态")
    artifact = safe_path(root, result["scene_artifact"])
    require((artifact/"model.xml").is_file(), "已发布场景的 model.xml 文件缺失")
    require(digest(snapshot(artifact)) == result["scene_sha256"], "场景文件已变化，不能复用原结果证据")
    return root, artifact, result


def resolve_asset(project, run, job):
    root = safe_path(Path(project)/"runs", run)
    from asset_prep.state_archive import read_state
    state = read_state(root/"state.json")
    record = state["scene"]["jobs"].get(job)
    require(record and record.get("artifact"), "指定任务尚无可查看的资产")
    artifact = safe_path(root, record["artifact"])
    stage = next(s for s in state["scene"]["stages"]
                 if s["directory"] == record.get("artifact_stage", record["stage"]))
    stage_root = safe_path(root, stage["directory"])
    require(artifact.is_relative_to(stage_root), "资产不属于声明的归档")
    prefix = artifact.relative_to(stage_root).as_posix() + "/"
    expected = {k.replace("\\", "/")[len(prefix):]:v for k,v in stage["files"].items()
                if k.replace("\\", "/").startswith(prefix)}
    require(expected and snapshot(artifact) == expected, "资产归档摘要不一致")
    return artifact, record["status"]


def export_preview(artifact, destination, result):
    destination = Path(destination)
    require(not destination.exists(), "预览目录已存在，请换一个目录以保留证据")
    model = load_scene_model(artifact)
    meta = read_json(Path(artifact)/"scene.json",8000000)
    paths = render_views(model,[("scene",meta["home_qpos"])],destination,width=1280,height=960)
    title = "静态场景预览（未执行操作仿真）"
    details = json.dumps({k:result.get(k) for k in ("status","provenance","validation_mode","qualification","unverified_count","report","assembly")},ensure_ascii=False,indent=2)
    figures = "".join(f'<figure><img src="{html.escape(p,quote=True)}"><figcaption>{html.escape(p)}</figcaption></figure>' for p in paths)
    (destination/"index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<title>{title}</title><style>body{{max-width:1200px;margin:32px auto;font:16px system-ui;background:#f2f4f7;color:#16202a}}'
        'figure{display:inline-block;margin:8px;max-width:100%}img{width:560px;max-width:100%}pre{white-space:pre-wrap;background:white;padding:20px}</style>'
        f'<h1>{title}</h1><p>图片只显示初始姿态。工程样例或未通过验收的结果不能作为正式实验成功证据。</p>{figures}<pre>{html.escape(details)}</pre></html>',encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="静态查看已发布的完整或部分场景，鼠标旋转/缩放，不运行操作仿真")
    parser.add_argument("--run",required=True,help="runs（运行结果）目录下的运行名称")
    parser.add_argument("--job",help="查看指定单资产任务，例如 asset_A06；可查看待审查候选")
    parser.add_argument("--render-only",action="store_true",help="只导出三视图和网页，不打开交互窗口")
    parser.add_argument("--output",type=Path,help="新的预览输出目录，默认 runs/运行名称/preview")
    args = parser.parse_args(argv)
    try:
        project = Path(__file__).resolve().parents[2]
        if args.job:
            require(not args.render_only and args.output is None, "单资产查看不使用 --render-only 或 --output")
            artifact, status = resolve_asset(project,args.run,args.job)
            print(f"单资产静态查看：{args.job}；审查状态 {status}；有模型不代表通过验收。")
            show_static(artifact, asset=True)
            return 0
        root, artifact, result = resolve_artifact(project,args.run)
        if args.render_only:
            print(export_preview(artifact,args.output or root/"preview",result))
        else:
            require(args.output is None,"--output 仅用于 --render-only")
            print("静态查看："+result["status"]+"；关闭窗口结束，不推进物理时间。")
            show_static(artifact)
        return 0
    except (PrepError,OSError,ValueError,KeyError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
