"""Command-line entry point; imports no code or configuration from the old project."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config, credentials
from .documents import load_documents
from .errors import BudgetExceeded, PrepError, UncertainRequest, ProviderLimitError
from .network import Transport
from .pipeline import execute, run_identity
from .preflight import inspect_inputs
from .storage import atomic_json, run_directory, run_lock, safe_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract assets from an experimental protocol and search for reference material")
    parser.add_argument("--pdf", type=Path, help="primary source of facts")
    parser.add_argument("--xml", type=Path, help="optional backup file; not read by default")
    parser.add_argument("--xml-reason", help="explicitly state the information gap in the PDF to be supplemented")
    parser.add_argument("--xml-query", action='append', default=[], help="supplement search term, repeatable; only matching XML text blocks are selected")
    parser.add_argument("--preflight", action="store_true", help="only check inputs and extraction segmentation locally; reads no keys and calls no model")
    parser.add_argument('--extract-only', action='store_true', help='only perform asset extraction and merging and save the follow-up budget plan; no search')
    parser.add_argument('--probe-output-limit', action='store_true', help='paid small-scale API probe, at most 4 calls; does not read the protocol')
    parser.add_argument('--probe-parameter', choices=('max_tokens', 'max_completion_tokens'),
                        help='probe only the given output parameter, at most 2 calls; requires --probe-output-limit')
    parser.add_argument("--dataset", type=Path, help="preflight only: a set containing case subdirectories with source.pdf/source.xml")
    parser.add_argument("--run", required=True, help="independent run name")
    parser.add_argument("--config", type=Path)
    parser.add_argument('--max-model-calls', type=int, help='explicit model-call budget for this run; can be raised on resume')
    parser.add_argument('--max-reserved-tokens', type=int, help='explicit conservative token-reservation budget for this run; not a real cost limit')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upgrade-run", action="store_true", help="only upgrade a paused checkpoint of a supported version offline; calls no model")
    parser.add_argument("--retry-uncertain", action="store_true", help="resend requests with uncertain outcomes after confirmation; may be billed twice")
    args = parser.parse_args(argv)
    if args.probe_parameter and not args.probe_output_limit:
        parser.error('--probe-parameter is only for --probe-output-limit')
    if args.probe_output_limit and any((args.pdf, args.xml, args.dataset, args.preflight, args.upgrade_run, args.extract_only)):
        parser.error('API probing cannot be combined with protocol input or other modes')
    if args.extract_only and (args.preflight or args.upgrade_run):
        parser.error('extract-only cannot be combined with offline check modes')
    if args.xml_reason or args.xml_query:
        if not (args.xml and args.xml_reason and args.xml_query) or args.dataset:
            parser.error('targeted supplementation requires --xml, --xml-reason and --xml-query together, and applies only to a single input')
    if args.upgrade_run and (args.resume or args.retry_uncertain or args.preflight or args.dataset):
        parser.error('--upgrade-run only performs an offline upgrade and cannot be mixed with execution or preflight options')
    if args.dataset and (not args.preflight or args.pdf or args.xml):
        parser.error("--dataset is only for --preflight and cannot be combined with --pdf/--xml")
    if not args.dataset and not args.pdf and not args.probe_output_limit:
        parser.error("specify --pdf, or use --preflight --dataset; XML is an optional backup file")
    if args.preflight and (args.resume or args.retry_uncertain):
        parser.error("offline preflight uses a new run name, not resume options")
    root = Path(__file__).resolve().parents[2]
    try:
        config = Config.load(args.config or root / "config/default.json")
        overrides = {name:getattr(args, name) for name in ('max_model_calls', 'max_reserved_tokens')
                     if getattr(args, name) is not None}
        if overrides:
            config = replace(config, **overrides)
        if args.probe_output_limit:
            from .probe import probe_limits
            run_root = run_directory(root, args.run)
            with run_lock(run_root):
                report = probe_limits(config, credentials(root), run_root, Transport(),
                                      resume=args.resume, retry_uncertain=args.retry_uncertain, parameter=args.probe_parameter)
            print(json.dumps({'status':report['status'], 'output':str(run_root/'output_limit_probe.json'),
                              'budget':report['budget']}, ensure_ascii=False))
            return 0 if report['status']=='finished_requires_review' else 3
        if args.preflight:
            run_root = run_directory(root, args.run)
            with run_lock(run_root):
                if any(p.name != ".lock" for p in run_root.iterdir()):
                    raise PrepError("run directory already has outputs; use a new run name for preflight")
                report = inspect_inputs(config, dataset=args.dataset, pdf=args.pdf, xml=args.xml,
                                        xml_reason=args.xml_reason, xml_queries=args.xml_query)
                output = safe_path(run_root, "preflight.json")
                atomic_json(output, report)
            print(json.dumps({"output": str(output), "total": report["total"],
                              "input_checked": report["input_checked"], "blocked": report["blocked"],
                              "model_calls": 0}, ensure_ascii=False))
            return 2 if report["blocked"] else 0
        documents = load_documents(args.pdf.resolve(), args.xml.resolve() if args.xml else None,
                                   config.max_document_chars, xml_reason=args.xml_reason, xml_queries=args.xml_query)
        if args.upgrade_run:
            from .upgrade import upgrade_run
            run_root = run_directory(root, args.run)
            with run_lock(run_root):
                print(json.dumps(upgrade_run(run_root, run_identity(documents, config), documents, config), ensure_ascii=False))
            return 0
        keys = credentials(root)
        run_root = run_directory(root, args.run)
        with run_lock(run_root):
            package = execute(documents, config, keys, run_root, Transport(), resume=args.resume,
                              retry_uncertain=args.retry_uncertain, extract_only=args.extract_only)
        print(json.dumps({"status": package["status"], "output": str(run_root / "asset_package.json"),
                          "budget": package["budget"]}, ensure_ascii=False))
        return 0
    except (BudgetExceeded, UncertainRequest, ProviderLimitError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except PrepError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; use --resume when running again.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError) as exc:
        # Do not print raw exception text or stack traces, so path/configuration errors cannot leak sensitive content.
        print(f"Local file or record error ({type(exc).__name__}); check the inputs and the run directory.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
