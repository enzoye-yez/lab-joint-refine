"""Version 3 execution and reference stages; used only in new runs, old experiments are not migrated."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from .budget import reference_plan, check_reference_capacity
from .contracts import fallback_selection, validate_selection, require
from .errors import PrepError, ProviderError, ReferenceUnavailable, ValidationError, BudgetExceeded, UncertainRequest, ProviderLimitError
from .images import fetch_image
from .pages import PageFetcher
from .pipeline import code_identity
from .prepare_workflow import Preparation
from .prompts import SELECTION
from .providers import combine_references
from .search import Search
from .storage import Store, atomic_json, digest, file_hash
from .task_contracts import BOUNDARY, WORKFLOW_VERSION, validate_package
from .task_delivery import publish, load_delivery


def placeholder(asset):
    return {'asset_id':asset['id'], 'target':deepcopy(asset), 'status':'pending',
            'selection':fallback_selection('References not prepared; generate from task requirements.'),
            'references':{'sources':[], 'images':[], 'warnings':[], 'search_records':[]},
            'downloaded_images':[], 'warnings':[]}


def unsupported_search(error):
    details = error.diagnostics.get('provider_error', {})
    words = ' '.join(str(details.get(k,'')) for k in ('code','type','message')).lower()
    return error.diagnostics.get('http_status') in {400, 404, 422, 501} and (
        any(t in words for t in ('web_search','tool_choice','tools','responses')) and
        any(t in words for t in ('unsupported','not support','unknown','not found','unrecognized')))


def prepare_references(prep, keys, transport):
    state, config, store = prep.state, prep.config, prep.store
    assets = state['inventory']['assets']
    remaining = [a for a in assets if digest(a) not in state['reference_cache']]
    plan = reference_plan(remaining, config, store.totals())
    state['reference_budget_plan'] = plan; prep.save()
    atomic_json(store.root/'reference_budget_plan.json', plan)
    if remaining and not state.get('search_unavailable'):
        try: check_reference_capacity(plan, config, store.totals())
        except BudgetExceeded as error:
            raise BudgetExceeded(str(error).replace('reference_budget_plan in asset_package.json','reference_budget_plan.json')) from error
    search = Search(config, keys, store, transport); pages = PageFetcher(config, keys, store, transport)
    result = []
    for asset in assets:
        key = digest(asset)
        if key in state['reference_cache']:
            result.append(deepcopy(state['reference_cache'][key])); continue
        item = placeholder(asset); batches = []
        if not state.get('search_unavailable'):
            for query in asset['queries'][:config.queries_per_asset]:
                try: batches.append(search.query(query))
                except ProviderError as error:
                    if unsupported_search(error):
                        state['search_unavailable'] = {'reason':'Configured search tool is unsupported.', 'diagnostics':error.diagnostics}
                        prep.save(); break
                    # Missing structured sources is a reference gap; explicit service refusals and authentication errors must stop.
                    if not isinstance(error,ReferenceUnavailable): raise
                    item['warnings'].append(str(error))
        if state.get('search_unavailable'): item['warnings'].append(state['search_unavailable']['reason'])
        references = combine_references(batches)
        if config.search_provider == 'model_web': pages.enrich(references)
        item['references'] = references; item['warnings'].extend(references['warnings'])
        if references['sources']:
            item['selection'] = prep.model.structured(SELECTION+'\nWrite prose in English; never modify target requirements.',
                {'target':asset, 'references':references}, lambda v:validate_selection(v, asset, references))
        if config.download_images:
            selected = set(item['selection']['image_ids'][:config.images_per_asset])
            for image in references['images']:
                if image['id'] not in selected: continue
                try: downloaded = fetch_image(image, store, transport, config)
                except (ProviderError, ValidationError) as error:
                    item['warnings'].append(str(error)); continue
                if not any(im['sha256']==downloaded['sha256'] for im in item['downloaded_images']):
                    item['downloaded_images'].append(downloaded)
        item['status'] = 'prepared_with_gaps' if (item['warnings'] or not item['downloaded_images'] or
                          item['selection']['unknowns'] or item['selection']['mode']=='requirements_only') else 'prepared'
        state['reference_cache'][key] = deepcopy(item); result.append(item)
        state['assets'] = result; prep.save()
    state['assets'] = result; prep.phase('delivery_review')


def used_excerpts(sources, *values):
    """Publish only excerpts with actual evidence citations, keeping the original source id and character offsets."""
    quoted = {}
    def visit(value):
        if isinstance(value,dict):
            if 'source_id' in value and 'quote' in value:
                quoted.setdefault(value['source_id'],set()).add(value['quote'])
            for child in value.values(): visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    for value in values: visit(value)
    result = []
    for source in sources:
        spans = []
        for quote in quoted.get(source['id'],[]):
            start = source['text'].find(quote)
            require(start>=0, 'published excerpt citation is not part of the source text')
            spans.append((start,start+len(quote)))
        merged = []
        for start,end in sorted(spans):
            if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
            else: merged.append([start,end])
        if merged:
            result.append({**deepcopy(source), 'text':'\n[…]\n'.join(source['text'][a:b] for a,b in merged),
                           'excerpt_spans':[{'start':a,'end':b} for a,b in merged]})
    return result


def package_of(prep, status):
    state = prep.state
    issues = [i for r in state['reviews'].values() for i in r['issues']]
    assets = state.get('assets') or [placeholder(a) for a in state['inventory']['assets']]
    gaps = list(state['task']['issues'])
    gaps.extend(i['description'] for i in issues)
    for item in assets:
        gaps.extend(item['warnings']); gaps.extend(item['selection']['unknowns'])
        if item['status']=='prepared_with_gaps' and not item['downloaded_images']:
            gaps.append('No downloaded reference image for asset '+item['asset_id']+'.')
    return {'schema_version':'2.0', 'status':status, 'provenance':prep.store.state['identity']['provenance'],
            'task_definition':deepcopy(state['task']), 'task_revision':state['task_revision'],
            'task_sha256':digest(state['task']), 'inventory':deepcopy(state['inventory']),
            'instance_roles':deepcopy(prep.all_roles()), 'requirements':prep.all_links(),
            'public_environment':deepcopy(prep.environment),
            'sources':used_excerpts(prep.sources,state['task'],state['inventory'],assets),
            'assets':deepcopy(assets), 'gaps':list(dict.fromkeys(gaps)),
            'traceability':{'stages':[{ 'stage_id':s['id'], 'object_ids':s['object_ids'],
                'asset_ids':sorted({r['asset_id'] for r in prep.all_roles() if s['id'] in r['stage_ids']})}
                for s in state['task']['stages']]},
            'provider':prep.config.provider, 'requested_model':prep.config.model,
            'checks':{'structural_and_quote_checks':True, 'semantic_review':'model_reviewed',
                      'visual_identity_verified':False}}


def execute_preparation(documents, config, keys, run_root, transport, environment, *,
                        resume=False, retry_uncertain=False, requirements_only=False,
                        provenance='model_generated', model=None):
    config = replace(config, scope=BOUNDARY+' All generated descriptions in English; quotes remain verbatim.')
    identity = {'workflow_contract':WORKFLOW_VERSION, 'documents':documents['files'],
                'sources_sha256':digest(documents['sources']), 'config':config.identity(),
                'public_environment':digest(environment), 'code':code_identity(), 'provenance':provenance}
    store = Store(run_root, identity, config, resume=resume, retry_uncertain=retry_uncertain)
    prep = Preparation(store, documents, config, keys, transport, environment, model=model)
    try:
        if prep.state['phase']=='complete' and prep.state.get('requirements_only') and not requirements_only:
            prep.state['reviews'] = {}; prep.phase('references')
        prep.state['requirements_only'] = requirements_only; prep.save()
        prep.advance_requirements()
        while prep.state['phase'] != 'complete':
            if prep.state['phase']=='references':
                if requirements_only:
                    prep.state['assets'] = [placeholder(a) for a in prep.state['inventory']['assets']]
                    prep.phase('delivery_review')
                else: prepare_references(prep, keys, transport)
            elif prep.state['phase']=='delivery_review': prep.review_delivery()
            elif prep.state['phase']=='publish':
                status = 'requirements_completed' if requirements_only else 'completed'
                package = package_of(prep, status)
                if (package['gaps'] or any(a['status']!='prepared' for a in package['assets'])) and not requirements_only:
                    package['status']='completed_with_gaps'
                validate_package(package)
                destination = publish(store, package)
                atomic_json(Path(run_root)/'asset_package.json', package)
                prep.state.update(phase='complete', status=package['status'], delivery=str(destination)); prep.save()
            else: raise ValidationError('unknown task preparation stage')
        load_delivery(prep.state['delivery'])
        return {'status':prep.state['status'], 'delivery':prep.state['delivery'], 'budget':store.totals()}
    except (PrepError, KeyboardInterrupt) as error:
        prep.state['status'] = 'paused' if isinstance(error,(BudgetExceeded,UncertainRequest,ProviderLimitError,KeyboardInterrupt,ProviderError)) else 'needs_review'
        prep.state['last_error'] = {'kind':type(error).__name__, 'message':str(error)}; prep.save()
        raise
