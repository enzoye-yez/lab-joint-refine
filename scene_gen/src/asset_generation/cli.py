"""Invoked via run.ps1 generate; offline preflight by default, real calls must be selected explicitly."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from asset_prep.config import Config, credentials
from asset_prep.errors import PrepError, BudgetExceeded, UncertainRequest, ProviderLimitError
from .transport import transport_for
from asset_prep.storage import atomic_json, run_directory, run_lock
from .contracts import GenerationConfig, check_profile, read_json
from .inputs import load_package, public_bundle
from .pipeline import execute
from .sandbox import DockerSandbox, SandboxUnavailable
from .local import LocalBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive asset generation; does not invoke asset extraction and does not run real robot experiments")
    parser.add_argument("--package", type=Path, required=True, help="Revised package Markdown or asset extraction JSON output")
    parser.add_argument("--asset", action="append", required=True, help="Asset id; repeat to specify a group of closely fitting assets")
    parser.add_argument("--run", required=True, help="Independent run name")
    parser.add_argument("--checks", type=Path, help="Optional read-only numeric acceptance config")
    parser.add_argument("--config", type=Path, help="Generation config")
    parser.add_argument("--provider-config", type=Path, help="Reuse the asset extraction provider config")
    parser.add_argument("--execute", action="store_true", help="Real model calls, run by the user; offline preflight only by default")
    parser.add_argument("--backend", choices=("docker", "local"), default="docker",
                        help="Execution backend; local is the explicit host-process mode and does not isolate host files or network")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly retry after confirming you accept possible duplicate requests")
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-reserved-tokens", type=int)
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args(argv)
    if (args.resume or args.retry_uncertain) and not args.execute:
        parser.error("Resume and unknown-request retry are only for real execution")
    project = Path(__file__).resolve().parents[2]
    try:
        generation = GenerationConfig.load(args.config or project/"config/generation.json")
        changes = {name: getattr(args, name) for name in ("max_model_calls", "max_reserved_tokens", "max_iterations")
                   if getattr(args, name) is not None}
        generation = replace(generation, **changes)
        config = Config.load(args.provider_config or project/"config/default.json")
        from scene_generation.benchmark_input import load_public, attach_bundle
        public = load_public(args.package)
        bundle = attach_bundle(load_package(args.package, args.asset, generation), public)
        profile = check_profile(read_json(args.checks) if args.checks else {})
        root = run_directory(project, args.run)
        sandbox = (LocalBackend if args.backend == "local" else DockerSandbox)(generation, project)
        with run_lock(root):
            if not args.execute:
                if any(p.name != ".lock" for p in root.iterdir()):
                    raise PrepError("Preflight directory already has artifacts; use a new run name")
                try:
                    backend = sandbox.preflight()
                except SandboxUnavailable as exc:
                    backend = {"available": False, "reason": str(exc)}
                report = {"status": "input_checked", "input": public_bundle(bundle), "profile": profile,
                          "sandbox": backend, "model_calls": 0, "max_model_calls": generation.max_model_calls,
                          "max_iterations": generation.max_iterations,
                          "note": "Does not prove the asset is semantically complete or of good quality; no credentials read and no model called"}
                atomic_json(root/"preflight.json", report)
                print(json.dumps({"status": report["status"], "output": str(root/"preflight.json"),
                                  "images": len(bundle["images"]), "model_calls": 0, "sandbox": backend}, ensure_ascii=False))
                return 0
            sandbox.preflight()
            result = execute(bundle, profile, config, generation, credentials(project), root, transport_for(config, root),
                             sandbox, project, resume=args.resume, retry_uncertain=args.retry_uncertain)
            print(json.dumps({k: result[k] for k in ("status", "iterations", "latest_revision", "budget")}, ensure_ascii=False))
            return 0 if result["status"] == "candidate_ready" else 3
    except (BudgetExceeded, UncertainRequest, ProviderLimitError, SandboxUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (PrepError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted by user; completed requests and versions are kept", file=sys.stderr)
        return 130
