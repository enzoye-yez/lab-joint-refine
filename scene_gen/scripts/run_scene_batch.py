"""User-launched sequential static experiment queue; a request fault pauses the queue, a geometry failure moves on to the next entry."""

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

# Per-run limits: NP-002 with gpt-6.1-sol used 411M tokens over 400 calls before its repair
# rounds converged (about 25M tokens a round), so tokens get room for about ten more rounds.
BUDGET = {"max_model_calls": 1600, "max_reserved_tokens": 720000000}


READY = {"static_preview_ready", "static_candidate_ready"}
TERMINAL = {"completed", "failed", "preview_failed", "blocked_unknown_request", "diagnostic_available"}


@contextmanager
def awake_while_running():
    """Request that the system stay awake only while this queue runs; does not change the permanent power plan or block manual sleep."""
    kernel = None
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        if not kernel.SetThreadExecutionState(0x80000001):
            print("Could not request the system to stay awake; check the computer sleep settings.",flush=True)
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
        raise ValueError("protocol ids must be unique and of the form NP-001")
    generation = read(project / (getattr(args, "generation_config", None) or "config/generation.json"))
    generation["max_iterations"] = args.asset_iterations
    generation["max_images"] = max(generation["max_images"], 128)
    GenerationConfig(**generation)
    provider = read(project / (getattr(args, "provider_config", None) or "config/default.json"))
    if getattr(args,"model",None):provider["model"]=args.model
    Config(**provider)  # Only non-sensitive config fields are allowed; credentials are not copied.
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
            raise ValueError(f"missing material package for {protocol}: {source}")
        from scene_generation.benchmark_input import load_public, bind_environment, PUBLIC_PATH
        public = load_public(package)
        use_pdf = not public or 'source.pdf' in public['identity']['files']
        if use_pdf and not pdf.is_file():
            raise ValueError(f"missing source text for {protocol}: {source}")
        bind_environment({'benchmark_public':public}, env, project / env['robot']['directory'])
        run = args.batch + "_" + protocol.lower().replace("-", "")
        if (project / "runs" / run).exists():
            raise ValueError(f"run directory already exists, cannot overwrite: {run}")
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
        raise ValueError("the generation framework gained new source files; create a new batch")
    for name, expected in frozen.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"code, input or config used by the batch has changed: {name}; create a new batch")


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
        # A display failure must not interrupt generation; keep the existing images and pages for inspection.
        print(f"asset preview not refreshed yet: {run}; {exc}", flush=True)
        return None


def render_index(root, manifest, project):
    """Refresh only derived pages; do not modify the experiment ledger, assets or old batch state."""
    def href(path):
        return html.escape(quote(os.path.relpath(path, root).replace("\\", "/"), safe="/"), quote=True)
    rows = []
    for entry in manifest["entries"]:
        folder = project / "runs" / entry["run"]
        gallery = refresh_assets(project, entry["run"])
        asset_link = f'<a href="{href(gallery)}">View existing asset images</a>' if gallery else "No images generated yet"
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
        link = f'<a href="{href(Path(preview))}">Full scene preview</a>' if preview else "No full scene yet"
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            entry["protocol"], entry["run"], entry["status"], entry.get("result_status", ""),
            entry.get("model_calls", ""), entry.get("reported_total_tokens", ""),
            entry.get("request_failure", {}).get("diagnostics", {}).get("http_status", ""),
            entry.get("request_failure", {}).get("provider_error", {}).get("message", ""))) +
            f'<td>{asset_link}</td><td>{link}</td><td><a href="{href(folder)}">Experiment folder</a></td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>Sequential experiment summary</title>'
            '<style>body{font-family:system-ui;margin:30px}td,th{padding:10px;border:1px solid #ccc}'
            'table{border-collapse:collapse}</style><h1>Static scene experiment summary</h1>'
            '<p>Success means a static candidate or layout preview was delivered; it does not mean the protocol operations or full adaptation validation were completed. Refresh the page to update.</p>'
            '<table><tr><th>Input</th><th>Run name</th><th>Queue status</th><th>Generation result</th>'
            '<th>Request attempts</th><th>Provider-reported tokens</th><th>Response status code</th><th>Provider error summary</th><th>Per-asset images</th><th>Full scene</th><th>Artifacts</th></tr>' +
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
        return (f"request attempts {counts['attempts']}, successful responses {counts['successful_responses']}, retries {counts['retry_attempts']}; assets ready {ready}/{len(jobs)}; "
                f"scene rounds {len(scene.get('rounds', []))}; status {scene.get('status', 'preparing')}; "
                f"phase {scene.get('activity',{}).get('phase','not yet recorded')} / {scene.get('activity',{}).get('event','')}")
    except (OSError, ValueError):
        return "preparing input or waiting for the ledger to be written"


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
        # Use the identity frozen when the queue was prepared; do not recompute it at launch and self-approve new source.
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
        print(f"[{index}/{len(manifest['entries'])}] start {entry['protocol']} → {entry['run']}", flush=True)
        print(f"  log: {root / (entry['run'] + '.log')}", flush=True)
        code = (0 if rendering_resume else
                runner(command, root / (entry["run"] + ".log"), project, run_root))
        entry["exit_code"] = code
        if code in {130, -1073741510, 3221225786}:  # Console interrupt: stop the whole queue.
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
            # Each export uses a new directory, keeping partial renders from interruptions.
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
            print(f"Request fault; queue paused at {entry['protocol']}; remaining inputs stay queued."
                  f" Response status code: {failure.get('diagnostics', {}).get('http_status')}；"
                  f"Details: {failure.get('provider_error', {}).get('message', 'no error details provided')}.", flush=True)
            return 3
        entry.pop("request_failure", None)
        entry["finished_at"] = now()
        save(root, manifest, project)
        print(f"[{index}/{len(manifest['entries'])}] {entry['protocol']}：{entry['status']}；"
              f"generation result {entry['result_status']}", flush=True)
    manifest["status"] = ("completed" if all(e["status"] == "completed" for e in manifest["entries"])
                          else "completed_with_failures")
    save(root, manifest, project)
    return 0 if manifest["status"] == "completed" else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run static scene experiments sequentially; by default only prepares, --execute calls the model")
    parser.add_argument("--batch", required=True, help="new batch name; use --resume to continue an unfinished queue")
    parser.add_argument("--protocols", nargs="+", default=[f"NP-{i:03d}" for i in range(1, 6)])
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark")
    parser.add_argument("--scene-config", type=Path, help="scene config; relative paths resolve against the project root; only for new batches")
    parser.add_argument("--generation-config", type=Path, help="asset generation config; relative paths resolve against the project root; only for new batches")
    parser.add_argument("--provider-config", type=Path, help="explicit provider config; only for new batches, then frozen and passed to the run entry point")
    parser.add_argument("--environment",type=Path,help="robot and workbench environment config; only for new batches, then frozen")
    parser.add_argument("--max-model-calls", type=int, default=BUDGET["max_model_calls"])
    parser.add_argument("--max-reserved-tokens", type=int, default=BUDGET["max_reserved_tokens"])
    parser.add_argument("--scene-iterations", type=int, default=80)
    parser.add_argument("--asset-iterations", type=int, default=60)
    parser.add_argument("--model", help="New batch model override, saved in frozen provider configuration")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true",
                        help="continue an experiment blocked by a request with unknown outcome and allow resending it; first confirm the original request has terminated")
    parser.add_argument("--resume", action="store_true", help="use the saved manifest and budgets; skip finished experiments, do not retry failed ones")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", args.batch):
        parser.error("batch name must be 1-40 letters, digits, underscores or hyphens, starting with a letter or digit")
    project = Path(__file__).resolve().parents[1]
    root = safe_path(project / "runs" / "batches", args.batch)
    manifest = None
    try:
        with run_lock(root):
            path = root / "batch.json"
            if path.exists():
                if not args.resume:
                    raise ValueError("batch already exists; choose a new name, or use --resume to continue the existing queue")
                manifest = read(path)
                verify(manifest, project)
            else:
                if args.resume:
                    raise ValueError("no batch to resume")
                manifest = prepare(project, root, args)
                save(root, manifest, project)
            print(f"batch {args.batch}: {len(manifest['entries'])} experiments, executed strictly in order.", flush=True)
            print(f"per-experiment budget: {manifest['budgets']}; resuming uses this saved budget.", flush=True)
            print(f"summary: {root / 'index.html'}", flush=True)
            if args.retry_uncertain:
                for entry in manifest["entries"]:
                    if entry["status"] == "blocked_unknown_request":
                        entry.update(status="queued", retry_uncertain=True)
                save(root, manifest, project)
            if not args.execute:
                print("Prepared the manifest only; the model was not called. To start, rerun the same command with --resume --execute.")
                return 0
            try:
                with awake_while_running():
                    return execute(manifest, root, project)
            except KeyboardInterrupt:
                manifest["status"] = "interrupted"
                save(root, manifest, project)
                print("Queue interrupted; continue with --resume --execute. Requests with unknown outcome are not resent automatically.")
                return 130
    except (PrepError, OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
