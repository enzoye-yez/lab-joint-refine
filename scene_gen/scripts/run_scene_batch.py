"""用户启动的顺序静态实验队列；请求故障暂停队列，几何失败可继续下一项。"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from contextlib import contextmanager
from urllib.parse import quote

from asset_generation.contracts import GenerationConfig
from asset_generation.model import request_summary
from asset_prep.config import Config
from asset_prep.errors import PrepError
from asset_prep.implementation import implementation_digest, PACKAGES
from asset_prep.storage import atomic_json, run_lock, safe_path
from scene_generation.contracts import SceneConfig, environment
from scene_generation.gallery import export as export_assets

# Per-run limits: twice the original 800 calls / 240M tokens, pending measured runs.
BUDGET = {"max_model_calls": 1600, "max_reserved_tokens": 480000000}


READY = {"static_preview_ready", "static_candidate_ready"}
TERMINAL = {"completed", "failed", "preview_failed", "blocked_unknown_request", "diagnostic_available"}


@contextmanager
def awake_while_running():
    """仅在当前队列期间请求系统保持唤醒，不改永久电源计划，不阻止手动休眠。"""
    kernel = None
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        if not kernel.SetThreadExecutionState(0x80000001):
            print("未能请求系统保持唤醒，请检查电脑休眠设置。",flush=True)
    try:
        yield
    finally:
        if kernel is not None:
            kernel.SetThreadExecutionState(0x80000000)


def read(path):
    if Path(path).name == 'state.json':
        from asset_prep.state_archive import read_state
        return read_state(path)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes(paths):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def code_files(project):
    return [path for name in PACKAGES
            for path in (project / "src" / name).rglob("*.py")] + list((project/'src').glob('runtime_bootstrap.py'))


def prepare(project, root, args):
    protocols = args.protocols
    if len(protocols) != len(set(protocols)) or any(not re.fullmatch(r"NP-\d{3}", p) for p in protocols):
        raise ValueError("规程编号须为不重复的 NP-001 等形式")
    generation = read(project / (getattr(args, "generation_config", None) or "config/generation.json"))
    generation["max_iterations"] = args.asset_iterations
    generation["max_images"] = max(generation["max_images"], 128)
    GenerationConfig(**generation)
    provider = read(project / (getattr(args, "provider_config", None) or "config/default.json"))
    if getattr(args,"model",None):provider["model"]=args.model
    Config(**provider)  # 只允许非敏感配置字段；不复制凭据。
    scene = read(project / (getattr(args, "scene_config", None) or "config/scene.json"))
    scene.update(max_iterations=args.scene_iterations, max_model_calls=args.max_model_calls,
                 max_reserved_tokens=args.max_reserved_tokens, validation_mode="static")
    SceneConfig(**scene)
    env = environment(read(project / (getattr(args,"environment",None) or "config/scene_environment.json")))
    entries, inputs = [], []
    for protocol in protocols:
        source = args.input_root.resolve() / protocol
        pdf, package = source / "source.pdf", source / "asset_package.md"
        if not package.is_file():
            raise ValueError(f"缺少 {protocol} 的资料包：{source}")
        from scene_generation.benchmark_input import load_public, bind_environment, PUBLIC_PATH
        public = load_public(package)
        use_pdf = not public or 'source.pdf' in public['identity']['files']
        if use_pdf and not pdf.is_file():
            raise ValueError(f"缺少 {protocol} 的原文：{source}")
        bind_environment({'benchmark_public':public}, env, project / env['robot']['directory'])
        run = args.batch + "_" + protocol.lower().replace("-", "")
        if (project / "runs" / run).exists():
            raise ValueError(f"运行目录已存在，不能覆盖：{run}")
        if public:
            inputs.extend(source / name for name in public['identity']['files'])
            inputs.extend([source / 'input_manifest.json', (source / PUBLIC_PATH).resolve()])
        else:
            inputs.extend([pdf, package])
            inputs.extend(path for path in (source / "images").rglob("*") if path.is_file())
        entries.append(dict(protocol=protocol, run=run, pdf=str(pdf) if use_pdf else None, package=str(package), status="queued"))
    configs = {"generation.json": generation, "provider.json": provider,
               "scene.json": scene, "environment.json": env}
    for name, value in configs.items():
        atomic_json(root / "config" / name, value)
    frozen = hashes(code_files(project) + inputs + [root / "config" / name for name in configs])
    return dict(version=1, batch=args.batch, created_at=now(), status="prepared",
                budgets=dict(model_calls=args.max_model_calls, reserved_tokens=args.max_reserved_tokens,
                             scene_iterations=args.scene_iterations, asset_iterations=args.asset_iterations),
                frozen=frozen, entries=entries)


def verify(manifest, project):
    frozen = manifest["frozen"]
    if any(str(path.resolve()) not in frozen for path in code_files(project)):
        raise ValueError("生成框架新增了源文件，请建立新批次")
    for name, expected in frozen.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"批次使用的代码、输入或配置已变化：{name}；请建立新批次")


def save(root, manifest, project):
    manifest["updated_at"] = now()
    atomic_json(root / "batch.json", manifest)
    render_index(root, manifest, project)


def refresh_assets(project, run):
    if not (project / "runs" / run / "state.json").is_file():
        return None
    try:
        return export_assets(project, run, refresh=True, allow_empty=True)
    except (OSError, ValueError, KeyError) as exc:
        # 展示失败不能中断生成；保留原有图片与页面供检查。
        print(f"资产预览暂未刷新：{run}；{exc}", flush=True)
        return None


def render_index(root, manifest, project):
    """只刷新派生页面，不修改实验账本、资产或旧批次状态。"""
    def href(path):
        return html.escape(quote(os.path.relpath(path, root).replace("\\", "/"), safe="/"), quote=True)
    rows = []
    for entry in manifest["entries"]:
        folder = project / "runs" / entry["run"]
        gallery = refresh_assets(project, entry["run"])
        asset_link = f'<a href="{href(gallery)}">查看已有资产图片</a>' if gallery else "尚未生成图片"
        if gallery:
            sources = read(gallery.parent / "sources.json")
            thumbnails = []
            for asset in sources["assets"]:
                first = next(iter(asset["images"]), None)
                if first:
                    thumbnails.append(f'<a href="{href(gallery)}"><img width="150" loading="lazy" '
                                      f'src="{href(folder / first)}" alt="{html.escape(asset["job"],quote=True)}"></a>')
            asset_link += '<div>' + ''.join(thumbnails) + '</div>'
        preview = entry.get("preview")
        link = f'<a href="{href(Path(preview))}">完整场景预览</a>' if preview else "尚无完整场景"
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            entry["protocol"], entry["run"], entry["status"], entry.get("result_status", ""),
            entry.get("model_calls", ""), entry.get("reported_total_tokens", ""),
            entry.get("request_failure", {}).get("diagnostics", {}).get("http_status", ""),
            entry.get("request_failure", {}).get("provider_error", {}).get("message", ""))) +
            f'<td>{asset_link}</td><td>{link}</td><td><a href="{href(folder)}">实验目录</a></td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>顺序实验汇总</title>'
            '<style>body{font-family:system-ui;margin:30px}td,th{padding:10px;border:1px solid #ccc}'
            'table{border-collapse:collapse}</style><h1>静态场景实验汇总</h1>'
            '<p>成功表示交付静态候选或布局预览；不表示完成规程操作或全部适配验证。刷新页面可更新。</p>'
            '<table><tr><th>输入</th><th>运行名称</th><th>队列状态</th><th>生成结果</th>'
            '<th>请求尝试数</th><th>服务报告令牌</th><th>响应状态码</th><th>服务错误摘要</th><th>单资产图片</th><th>完整场景</th><th>产物</th></tr>' +
            ''.join(rows) + '</table>')
    (root / "index.html").write_text(page, encoding="utf-8")


def progress(run_root):
    try:
        state = read(run_root / "state.json")
        calls = state.get("calls", [])
        scene = state.get("scene", {})
        jobs = scene.get("jobs", {})
        ready = sum(j.get("assembly_readiness",{}).get("loadable") or j.get("status") in {"candidate_ready", "assembly_ready"} for j in jobs.values())
        counts = request_summary(calls)
        return (f"请求尝试 {counts['attempts']}，成功响应 {counts['successful_responses']}，重试 {counts['retry_attempts']}；资产就绪 {ready}/{len(jobs)}；"
                f"场景轮次 {len(scene.get('rounds', []))}；状态 {scene.get('status', '准备中')}；"
                f"阶段 {scene.get('activity',{}).get('phase','待记录')} / {scene.get('activity',{}).get('event','')}")
    except (OSError, ValueError):
        return "正在准备输入或等待账本写入"


def run_child(command, log, project, run_root):
    if command[1:3] == ['-m', 'scene_generation']:
        command = [command[0], '-B', '-P', str(project/'src/runtime_bootstrap.py'),
                   '--module', 'scene_generation', '--', *command[3:]]
    env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    with log.open("ab") as stream:
        stream.write(("\n--- " + now() + " ---\n").encode("utf-8"))
        stream.flush()
        from asset_generation.process_tree import ProcessTree
        tree=ProcessTree();child=None
        try:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=os.name!="nt")
            tree.attach(child)
            while True:
                try:
                    return child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    refresh_assets(project, run_root.name)
                    print("  " + progress(run_root), flush=True)
        finally:
            if child is not None:tree.stop(child)
            else:tree.close()



def execute(manifest, root, project, runner=run_child):
    for index, entry in enumerate(manifest["entries"], 1):
        if entry["status"] in TERMINAL:
            continue
        verify(manifest, project)
        run_root = project / "runs" / entry["run"]
        config = root / "config"
        # 使用队列准备时冻结的身份，不能启动时重新计算并自行认可新源码。
        expected = implementation_digest({Path(path).relative_to(project/'src').as_posix():sha
            for path,sha in manifest['frozen'].items() if Path(path).is_relative_to(project/'src')})
        command = [sys.executable, "-m", "scene_generation", "--backend", "local", "--static-only",
                   '--expected-implementation', expected,
                   "--execute", "--run", entry["run"], "--package", entry["package"],
                   "--config", str(config / "scene.json"), "--generation-config", str(config / "generation.json"),
                   "--provider-config", str(config / "provider.json"), "--environment", str(config / "environment.json")]
        if entry.get('pdf'):
            command.extend(['--protocol-pdf', entry['pdf']])
        if (run_root / "state.json").exists():
            command.append("--resume")
        if entry.pop("retry_uncertain", False):
            command.append("--retry-uncertain")
        rendering_resume = entry["status"] == "rendering"
        entry.update(status="running", started_at=now())
        manifest["status"] = "running"
        save(root, manifest, project)
        print(f"[{index}/{len(manifest['entries'])}] 开始 {entry['protocol']} → {entry['run']}", flush=True)
        print(f"  日志：{root / (entry['run'] + '.log')}", flush=True)
        code = (0 if rendering_resume else
                runner(command, root / (entry["run"] + ".log"), project, run_root))
        entry["exit_code"] = code
        if code in {130, -1073741510, 3221225786}:  # 控制台中断：停止整个队列。
            raise KeyboardInterrupt
        try:
            result = read(run_root / "result.json")
        except (OSError, ValueError):
            result = {}
        entry["result_status"] = result.get("status", "no_result")
        for key in ("model_calls", "reported_total_tokens", "accounted_tokens"):
            entry[key] = result.get("budget", {}).get(key)
        entry["status"] = "failed"
        failure = result.get("request_failure")
        if result.get("scene_artifact") and not (run_root/"pause.request").exists():
            # 每次导出使用新目录，保留中断时的部分渲染。
            destination = run_root / "preview"
            suffix = 1
            while destination.exists():
                destination = run_root / f"preview_batch_{suffix}"
                suffix += 1
            entry["status"] = "rendering"
            save(root, manifest, project)
            render = [sys.executable, "-m", "scene_generation.preview", "--run", entry["run"],
                      "--render-only", "--output", str(destination)]
            render_code = runner(render, root / (entry["run"] + "_preview.log"), project, run_root)
            if render_code in {130, -1073741510, 3221225786}:
                raise KeyboardInterrupt
            entry["preview_exit_code"] = render_code
            entry["status"] = "preview_failed"
            if render_code == 0 and (destination / "index.html").is_file():
                entry.update(status="completed" if code == 0 and result.get("status") in READY else "diagnostic_available", preview=str(destination / "index.html"))
        if (run_root/"pause.request").exists():
            entry.update(status="paused_user",paused_at=now())
            manifest["status"]="paused_user"
            save(root,manifest,project)
            return 130
        if code != 0 and failure:
            entry.update(status="paused_provider", request_failure=failure, paused_at=now())
            manifest["status"] = "paused_provider"
            save(root, manifest, project)
            print(f"请求故障，队列已暂停于 {entry['protocol']}；后续输入保持排队。"
                  f" 响应状态码：{failure.get('diagnostics', {}).get('http_status')}；"
                  f"说明：{failure.get('provider_error', {}).get('message', '未提供错误详情')}。", flush=True)
            return 3
        entry.pop("request_failure", None)
        entry["finished_at"] = now()
        save(root, manifest, project)
        print(f"[{index}/{len(manifest['entries'])}] {entry['protocol']}：{entry['status']}；"
              f"生成结果 {entry['result_status']}", flush=True)
    manifest["status"] = ("completed" if all(e["status"] == "completed" for e in manifest["entries"])
                          else "completed_with_failures")
    save(root, manifest, project)
    return 0 if manifest["status"] == "completed" else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description="按顺序运行静态场景实验；默认仅准备，--execute 才调用模型")
    parser.add_argument("--batch", required=True, help="新批次名称；使用 --resume 接续未完成队列")
    parser.add_argument("--protocols", nargs="+", default=[f"NP-{i:03d}" for i in range(1, 6)])
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark")
    parser.add_argument("--scene-config", type=Path, help="场景配置；相对路径以项目根目录为准，仅用于新批次")
    parser.add_argument("--generation-config", type=Path, help="资产生成配置；相对路径以项目根目录为准，仅用于新批次")
    parser.add_argument("--provider-config", type=Path, help="显式服务配置；仅用于新批次，随后冻结并传给运行入口")
    parser.add_argument("--environment",type=Path,help="机器人及工作台环境配置；仅用于新批次，随后冻结")
    parser.add_argument("--max-model-calls", type=int, default=BUDGET["max_model_calls"])
    parser.add_argument("--max-reserved-tokens", type=int, default=BUDGET["max_reserved_tokens"])
    parser.add_argument("--scene-iterations", type=int, default=80)
    parser.add_argument("--asset-iterations", type=int, default=60)
    parser.add_argument("--model", help="New batch model override, saved in frozen provider configuration")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true",
                        help="接续因结果不明请求而阻断的实验，并允许重发该请求；先确认原请求已终止")
    parser.add_argument("--resume", action="store_true", help="使用已存清单及预算；跳过已结束实验，不重试失败项")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", args.batch):
        parser.error("批次名称须为 1–40 个字母、数字、下划线或连字符，首位为字母或数字")
    project = Path(__file__).resolve().parents[1]
    root = safe_path(project / "runs" / "batches", args.batch)
    manifest = None
    try:
        with run_lock(root):
            path = root / "batch.json"
            if path.exists():
                if not args.resume:
                    raise ValueError("批次已存在；换新名称，或 --resume 接续原队列")
                manifest = read(path)
                verify(manifest, project)
            else:
                if args.resume:
                    raise ValueError("没有可恢复的批次")
                manifest = prepare(project, root, args)
                save(root, manifest, project)
            print(f"批次 {args.batch}：{len(manifest['entries'])} 个实验，严格顺序执行。", flush=True)
            print(f"每个实验预算：{manifest['budgets']}；恢复时使用此已保存预算。", flush=True)
            print(f"汇总：{root / 'index.html'}", flush=True)
            if args.retry_uncertain:
                for entry in manifest["entries"]:
                    if entry["status"] == "blocked_unknown_request":
                        entry.update(status="queued", retry_uncertain=True)
                save(root, manifest, project)
            if not args.execute:
                print("仅准备清单，未调用模型。启动请在同一命令加 --resume --execute。")
                return 0
            try:
                with awake_while_running():
                    return execute(manifest, root, project)
            except KeyboardInterrupt:
                manifest["status"] = "interrupted"
                save(root, manifest, project)
                print("队列已中断；--resume --execute 接续。结果不明的请求不会自动重发。")
                return 130
    except (PrepError, OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
