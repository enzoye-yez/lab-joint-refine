"""Task material preparation entry point; preflight loads no credentials and creates no network clients."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .blocks import plan
from .config import Config, credentials
from .documents import load_documents
from .errors import PrepError
from .storage import atomic_json, encoded, run_directory, run_lock
from .task_contracts import BOUNDARY, public_environment


def inspect(documents, config, environment):
    public = public_environment(environment)
    chunks = plan(documents['sources'], config.chunk_chars)
    from .blocks import excerpts
    from .providers import model_messages
    from .prepare_prompts import INDEX
    from .workflow import plan_tasks
    extraction_blocks = len(plan_tasks(documents,config))
    # Without a task the actual numbers of instances, interactions and searches cannot be estimated; only stages that will certainly be traversed are counted.
    requests = [len(encoded(model_messages(config,INDEX,{'operation':'protocol_index',
        'sources':excerpts(c['units'],documents,context=0)}))) for c in chunks]
    base_calls = 4*len(chunks)+extraction_blocks
    warnings = list(documents['warnings'])
    if base_calls > config.max_model_calls:
        warnings.append('the current call limit is below the base requirement for a full-text pass; instance assignment, identity matching and review need additional calls.')
    if max(requests,default=0)>min(config.max_input_bytes,96000):
        warnings.append('at least one index request exceeds the input limit; reduce the segmentation settings.')
    return {'status':'input_checked', 'workflow':'asset_workflow_v3', 'network_calls':0,
            'provider':config.provider, 'model':config.model, 'base_url':config.base_url,
            'scope':BOUNDARY, 'environment_sha256':public['sha256'],
            'robot':public['profile']['robot'].get('name'), 'source_blocks':len(chunks),
            'source_characters':sum(len(s['text']) for s in documents['sources']),
            'initial_index_calls':len(chunks), 'extraction_blocks':extraction_blocks,
            'baseline_document_calls':base_calls, 'largest_index_request_bytes':max(requests,default=0),
            'initial_index_token_reserve':sum(n+512+config.max_output_tokens for n in requests),
            'warnings':warnings,
            'budget':{k:getattr(config,k) for k in ('max_model_calls','max_reserved_tokens','max_search_calls')},
            'estimate_basis':'base calls = 4 x index blocks + initial extraction blocks; excludes subdivision, identity and field matching, instance assignment, interactions, final review, search and revisions. Token reservation is a byte-based upper-bound estimate, not cost.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Select the protocol execution flow and prepare mobile dual-arm task materials')
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--xml', type=Path)
    parser.add_argument('--xml-reason')
    parser.add_argument('--xml-query', action='append', default=[])
    parser.add_argument('--public-environment', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--requirements-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-uncertain', action='store_true')
    for name in ('max-model-calls','max-reserved-tokens','max-search-calls'):
        parser.add_argument('--'+name,type=int)
    args = parser.parse_args(argv)
    if args.preflight and (args.resume or args.retry_uncertain): parser.error('preflight uses a new run and does not resume network requests')
    if (args.xml_reason or args.xml_query) and not (args.xml and args.xml_reason and args.xml_query):
        parser.error('targeted XML supplementation requires the file, the reason and search terms together')
    project = Path(__file__).resolve().parents[2]
    try:
        config = Config.load(args.config or project/'config/default.json')
        config = replace(config, **{n:getattr(args,n) for n in ('max_model_calls','max_reserved_tokens','max_search_calls') if getattr(args,n) is not None})
        path = args.public_environment or project/'config/scene_environment.json'
        environment = json.loads(path.read_text(encoding='utf-8-sig')); public_environment(environment)
        documents = load_documents(args.pdf.resolve(), args.xml.resolve() if args.xml else None,
            config.max_document_chars, xml_reason=args.xml_reason, xml_queries=args.xml_query)
        root = run_directory(project, args.run)
        with run_lock(root):
            if args.preflight:
                if any(p.name!='.lock' for p in root.iterdir()): raise PrepError('preflight directory already exists; use a new name')
                result = inspect(documents, config, environment)
                atomic_json(root/'preflight.json', result)
            else:
                from .prepare_pipeline import execute_preparation
                from asset_generation.provider_diagnostics import DiagnosticTransport
                result = execute_preparation(documents, config, credentials(project), root, DiagnosticTransport(),
                    environment, resume=args.resume, retry_uncertain=args.retry_uncertain,
                    requirements_only=args.requirements_only)
        print(json.dumps(result,ensure_ascii=False)); return 0
    except PrepError as error:
        print(str(error)); return 2
    except KeyboardInterrupt:
        print('Task preparation interrupted; progress and request budget are kept.'); return 130
    except (OSError, ValueError, KeyError) as error:
        print('Failed to read or save task materials: '+type(error).__name__); return 2


if __name__ == '__main__':
    raise SystemExit(main())
