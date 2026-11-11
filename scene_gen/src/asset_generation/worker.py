"""Standalone worker process launched by the backend; container and explicit local mode share the same build-and-check entry point."""

import argparse
import contextlib
import json
import runpy
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "validate", "scene", "inspect"))
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--implementation", default=None)
    parser.add_argument("--checkpoints", type=Path, default=None)
    args = parser.parse_args()
    mode, source, output = args.mode, args.input, args.output
    # The worker trusts only an implementation identity passed explicitly and verified against disk; it never infers it from environment variables or caches.
    from asset_prep import implementation as binding
    expected = binding.require_identity(args.implementation, where="worker startup")
    binding.verify(expected, base=Path(__file__).resolve().parents[1], where="worker startup")
    if mode != 'build':
        job = json.loads((source/'job.json').read_text(encoding='utf-8'))
        if job.get('implementation') != expected:
            raise binding.ImplementationDrift('Worker job does not match the implementation version actually executing')
    if mode == "build":
        # Container logs are limited by the tmpfs quota; local logs are collected with bounds by the parent process.
        with (open("/tmp/model.log", "w", encoding="utf-8") if source == Path("/input")
              else contextlib.nullcontext(sys.stdout)) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    namespace = runpy.run_path(str(source/"model.py"), run_name="asset_model")
                    namespace["build"](output)
                except Exception:
                    (output/"build_error.json").write_text(json.dumps({"detail": traceback.format_exc()[-5000:]}), encoding="utf-8")
                    raise
    elif mode == "inspect":
        from .evidence import inspect_artifact
        report = inspect_artifact(source/"artifact",job["scope"],job["requests"],output)
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    elif mode == "validate":
        from .checks import validate_artifact
        if job.get('scope') == 'scene':
            from scene_generation.checks import scene_checks
            from scene_generation.contracts import SceneConfig
            try:
                if job.get('capability'):
                    from scene_generation.capabilities import REGISTRY
                    report = REGISTRY[job['capability']].inspect(source/'artifact',SceneConfig(**job['limits']),
                                                                job['contract'],job['request'],checkpoint_dir=args.checkpoints)
                    report['input_key'] = job['input_key']
                else:
                    report = scene_checks(source/'artifact',SceneConfig(**job['limits']),job['contract'],
                                          render_dir=output/'renders' if job.get('render') else None,
                                          recovery=job.get('recovery',False),checkpoint_dir=args.checkpoints)
            except Exception as exc:
                from .check_report import failed_report
                report = failed_report(exc)
                report['detail'] = traceback.format_exc()[-5000:]
                if job.get('capability'):
                    report = {'input_key':job['input_key'],'scope':'checks','checks':[],
                              'failure':report['execution_failure']}
        else:
            try:
                if job.get("admission"):
                    from scene_generation.candidate import inspect_candidate
                    report = inspect_candidate(source/"artifact",job["asset_ids"],job["profile"],output/"renders" if job.get("candidate_render") else None)
                else:
                    report = validate_artifact(source/"artifact", job["asset_ids"], job["profile"], output/"renders")
            except Exception as exc:
                from .check_report import failed_report
                from asset_prep.errors import ValidationError
                if job.get("admission") and isinstance(exc,(ValidationError,ValueError)):
                    report={"status":"fail","checks":[{"id":"assembly_compile","status":"fail","detail":str(exc)}],
                            "renders":[],"manifest":None,"coverage":["Not loadable for assembly"]}
                else:
                    report = failed_report(exc)
        (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    elif mode == "scene":
        from .timing import stage
        from scene_generation.assembly import assemble
        from scene_generation.checks import scene_checks
        from scene_generation.contracts import SceneConfig
        try:
            with stage("scene_assembly"):
                artifact = assemble(job["plan"],job["environment"],
                    {j["id"]:source/"components"/j["id"] for j in job["plan"]["jobs"]},
                    source/"robot",output/"artifact")
            try:
                if job.get("admission"):
                    from .checks import measure_model
                    from dataclasses import replace
                    from scene_generation.checkpoints import Checkpoints
                    with stage("scene_admission_load"):
                        limits=replace(SceneConfig(**job['limits']),validation_mode='static')
                        cache=Checkpoints(args.checkpoints,artifact,limits,job['contract'],False)
                        cache.load_model(lambda:measure_model(artifact,scope='scene',static=True))
                    report={"status":"partial","checks":[{"id":"scene_quality_pending","status":"partial",
                        "detail":"Compiled diagnostic scene; quality assessment pending"}],"renders":[],
                        "assembly_readiness":{"loadable":True},"coverage":["Diagnostic assembly only"]}
                else:
                    with stage("scene_assessment"):
                        report = scene_checks(artifact,SceneConfig(**job["limits"]),job["contract"],output/"renders",
                                              checkpoint_dir=args.checkpoints)
            except Exception as exc:
                from .check_report import failed_report
                report = failed_report(exc)
                report['detail'] = traceback.format_exc()[-5000:]
            (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        except Exception:
            (output/"build_error.json").write_text(json.dumps({"detail":traceback.format_exc()[-5000:]}),encoding="utf-8")
            raise
    else:
        raise ValueError("Invalid worker mode")
    binding.verify(expected, base=Path(__file__).resolve().parents[1], where='worker submission')


if __name__ == "__main__":
    main()
