"""Full-scene entry point. Offline preflight by default; formal model execution requires an explicit user choice."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from asset_generation.contracts import GenerationConfig, read_json
from asset_generation.sandbox import DockerSandbox, SandboxUnavailable
from asset_generation.local import LocalBackend
from asset_generation.checks import load_model
from asset_prep.config import Config, credentials
from asset_prep.errors import PrepError, BudgetExceeded, UncertainRequest, ProviderLimitError
from asset_generation.transport import transport_for
from asset_prep.storage import run_directory, run_lock, atomic_json, safe_path
from asset_prep import implementation
from .contracts import SceneConfig, environment, numerical_contract
from .inputs import catalog
from .pipeline import SceneRun


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full-scene generation and bounded repair; evaluation skips asset extraction and search")
    parser.add_argument("--prepare-robot",action="store_true",help="verify whole-robot resources per environment; legacy environments download the fixed FR3; does not call the model")
    parser.add_argument("--package",type=Path)
    parser.add_argument("--protocol-pdf",type=Path,help="source PDF: fully parsed page by page as scene context; asset extraction is not called")
    parser.add_argument("--run")
    parser.add_argument("--environment",type=Path)
    parser.add_argument("--checks",type=Path,help="user-fixed numeric/fit/operation-pose acceptance checks")
    parser.add_argument("--config",type=Path,help="scene budget and check limits")
    parser.add_argument("--generation-config",type=Path)
    parser.add_argument("--provider-config",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--expected-implementation", help="formal execution must specify the implementation digest from the offline verification manifest")
    parser.add_argument("--backend", choices=("docker", "local"), default="docker",
                        help="execution backend; local is an explicit local-process mode that does not isolate host files or network")
    parser.add_argument("--static-only",action="store_true",help="accept the static scene only; skip dynamic interaction, gravity stepping and operation-pose solving")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-uncertain",action="store_true")
    parser.add_argument("--feedback",type=Path,help="planner failure feedback bound to the scene digest")
    for key in ("max-model-calls","max-reserved-tokens","max-iterations","max-plan-attempts","max-repair-plan-attempts","max-evidence-rounds","max-evidence-plan-attempts"):
        parser.add_argument("--"+key,type=int)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    try:
        if args.execute:
            active = getattr(sys, '_harness_runtime', None)
            if not active or not args.expected_implementation:
                raise PrepError('formal execution must use run.ps1 and provide --expected-implementation with the verified source digest')
            implementation.verify(args.expected_implementation, project=project, where='formal execution start')
        if args.prepare_robot:
            if args.execute or args.package or args.protocol_pdf or args.run or args.resume or args.feedback:
                parser.error("robot preparation must run on its own")
            from .robot import prepare_environment
            env = environment(read_json(args.environment or project/"config/scene_environment.json"))
            print(json.dumps({"robot_directory":str(prepare_environment(project,env))},ensure_ascii=False))
            return 0
        if not args.package or not args.run:
            parser.error("--package and --run are required")
        if (args.resume or args.retry_uncertain or args.feedback) and not args.execute:
            parser.error("resume and feedback require formal execution")
        if args.feedback and not args.resume:
            parser.error("planner feedback must resume an existing scene")
        provider = Config.load(args.provider_config or project/"config/default.json")
        generation = GenerationConfig.load(args.generation_config or project/"config/generation.json")
        limits = SceneConfig(**read_json(args.config or project/"config/scene.json"))
        limits = replace(limits,**{k:getattr(args,k) for k in ("max_model_calls","max_reserved_tokens","max_iterations","max_plan_attempts","max_repair_plan_attempts","max_evidence_rounds","max_evidence_plan_attempts") if getattr(args,k) is not None})
        if args.static_only:
            limits = replace(limits,validation_mode="static")
        if args.feedback and limits.validation_mode == "static":
            parser.error("static preview does not accept operation-planning feedback; use a new run directory for full verification")
        env = environment(read_json(args.environment or project/"config/scene_environment.json"))
        contract = numerical_contract(read_json(args.checks) if args.checks else {})
        catalogue = catalog(args.package,generation,args.protocol_pdf,provider.max_document_chars)
        robot = safe_path(project,env["robot"]["directory"])
        from .benchmark_input import bind_environment
        bind_environment(catalogue, env, robot)
        sandbox = (LocalBackend if args.backend == "local" else DockerSandbox)(generation,project)
        root = run_directory(project,args.run)
        with run_lock(root):
            if not args.execute:
                if any(p.name != ".lock" for p in root.iterdir()):
                    raise PrepError("preflight directory already has outputs; choose a new name")
                try:
                    backend = sandbox.preflight()
                except SandboxUnavailable as exc:
                    backend = {"available":False,"reason":str(exc)}
                robot_ready = False
                if robot.is_dir():
                    load_model(robot)
                    robot_ready = True
                report = {"status":"input_checked","asset_ids":catalogue["asset_ids"],"identity":catalogue["identity"],
                          "implementation":implementation.runtime_record(project, inputs={}, configuration={})['implementation'],
                          "robot_ready":robot_ready,"sandbox":backend,"provider":provider.provider,"model":provider.model,
                          "model_calls":0,"extraction_calls":0,"search_calls":0,"environment":env,
                          "validation_mode":limits.validation_mode,
                          "protocol":({"files":catalogue["protocol"]["files"],
                                       "pages":len(catalogue["protocol"]["sources"]),
                                       "characters":catalogue["protocol"]["parsing"]["total_characters"],
                                       "warnings":catalogue["protocol"]["warnings"]} if catalogue["protocol"] else None),
                          "note":"read-only preflight does not certify generation quality; no credentials were read and no model was called"}
                atomic_json(root/"preflight.json",report)
                print(json.dumps({k:report[k] for k in ("status","asset_ids","protocol","validation_mode","robot_ready","sandbox","model_calls")},ensure_ascii=False))
                return 0
            sandbox.preflight()
            load_model(robot)
            run = SceneRun(catalogue,env,contract,provider,generation,limits,credentials(project),root,transport_for(provider,root),
                           sandbox,robot,project,args.resume,args.retry_uncertain,
                           expected_implementation=args.expected_implementation)
            result = run.run(read_json(args.feedback) if args.feedback else None)
            print(json.dumps(result,ensure_ascii=False))
            return 0 if result["status"] in {"candidate_ready", "static_candidate_ready", "static_preview_ready"} else 3
    except (BudgetExceeded,UncertainRequest,ProviderLimitError,SandboxUnavailable) as exc:
        print(str(exc),file=sys.stderr)
        return 3
    except (PrepError,OSError,ValueError,KeyError,TypeError) as exc:
        print(str(exc),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted by user; call ledger and archived versions are kept",file=sys.stderr)
        return 130
