"""Resumable asset preparation flow; execution evidence and geometry verification are explicitly left to later modules."""

import copy
import json
import time
from pathlib import Path

from .contracts import fallback_selection, validate_selection
from .workflow import run_workflow, VERSION
from .errors import BudgetExceeded, ProviderError, StaleEvidence, UncertainRequest, ValidationError, ProviderLimitError
from .budget import reference_plan, check_reference_capacity
from .images import fetch_image
from .prompts import SELECTION
from .pages import PageFetcher
from .providers import Model, combine_references, redact
from .search import Search
from .storage import Store, atomic_json, digest, file_hash, safe_path


def code_identity():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def run_identity(documents, config, provenance='model_generated'):
    return {"documents": documents["files"], "config": config.identity(),
            "parsed_sources": digest(documents["sources"]), "parsing": documents.get("parsing"),
            "code": code_identity(), "workflow_contract":VERSION, "provenance": provenance}


def execute(documents, config, credentials, run_root, transport, *, resume=False,
            retry_uncertain=False, provenance="model_generated", extract_only=False):
    identity = run_identity(documents, config, provenance)
    store = Store(run_root, identity, config, resume=resume, retry_uncertain=retry_uncertain)
    workflow=store.state.get('workflow')
    if workflow is not None and digest(workflow)!=store.state.get('workflow_sha256'):
        raise StaleEvidence('workflow state was modified; cannot resume')
    if store.state.get('checkpoint') and workflow is None:
        raise StaleEvidence('old checkpoint has not been converted to task state; migrate offline first')
    intent=store.state.get('publication_intent',{})
    for relative in store.state.get('outputs',{}).keys() | intent.keys():
        expected=store.state.get('outputs',{}).get(relative)
        path = safe_path(store.root, relative)
        allowed={expected,intent.get(relative,{}).get('new')}
        if (not path.exists() and expected is not None) or (path.exists() and file_hash(path) not in allowed):
            raise StaleEvidence("existing outputs were modified or are missing; this run's evidence cannot be reused")
    previous_package = {}
    package_path = safe_path(store.root, 'asset_package.json')
    if resume and package_path.exists():
        previous_package = json.loads(package_path.read_text(encoding='utf-8'))
    model, search = Model(config, credentials, store, transport), Search(config, credentials, store, transport)
    pages = PageFetcher(config, credentials, store, transport)
    package = {"schema_version": "1.0", "status": "in_progress", "provenance": provenance,
               "provider": config.provider, "base_url": config.base_url, "requested_model": config.model,
               "search_provider": config.search_provider, "search_url": search.endpoint,
               "identity": identity, "inventory": None, "extraction": {}, "assets": [], "warnings": list(documents["warnings"]),
               "checks": {"structural_and_quote_checks": False, "semantic_completeness_verified": False,
                          "geometry_verified": False, "execution_verified": False}, "generated_at": time.time()}
    for key in ('inventory', 'assets', 'extraction'):
        if key in previous_package:
            package[key] = previous_package[key]

    def save():
        package["budget"] = store.totals()
        limit_reports = [e['status'] for c in store.state['calls'] for e in c.get('validation_events', [])
                         if e.get('kind') == 'output_limit']
        package['checks']['output_limit_reports'] = {status: limit_reports.count(status) for status in
            ('reported_within_limit', 'unverifiable', 'reported_violation')}
        returned = set()
        for call in store.state["calls"]:
            if call["kind"] in {"model", "model_search"} and call["status"] == "done":
                record = store.cached(call["key"])
                if isinstance(record.get("model"), str):
                    returned.add(record["model"])
        package["returned_models"] = sorted(returned)
        package["warnings"] = list(dict.fromkeys(package["warnings"]))
        output = redact(package, credentials.values())
        contents=(("documents.json", documents), ("asset_package.json", output))
        store.state['publication_intent']={name:{'new':digest(content)} for name,content in contents}
        store.save()
        for relative, content in contents:
            path = safe_path(store.root, relative)
            atomic_json(path, content)
            store.state.setdefault("outputs", {})[relative] = file_hash(path)
        store.state.pop('publication_intent',None)
        store.save()

    progress = {}
    def persist_workflow(value):
        store.state['workflow']=value
        store.state['workflow_sha256']=digest(value)
        store.save()

    def publish_workflow(catalog,current_progress):
        progress.clear(); progress.update(current_progress)
        publish_catalog(catalog)

    def publish_catalog(catalog):
        # Publishing only advances committed revisions; a validation or budget failure does not roll back confirmed assets.
        if progress.get('revision', 0) < package['extraction'].get('revision', 0):
            return
        existing = {item['asset_id']: item for item in package['assets']}
        package['inventory'] = copy.deepcopy(catalog)
        package['extraction'] = copy.deepcopy(progress)
        package['assets'] = []
        for asset in catalog['assets']:
            item = existing.get(asset['id'], {'asset_id': asset['id'],
                'selection': fallback_selection('not searched yet; can still be generated later from the target requirements'),
                'references': {'sources': [], 'images': []}, 'downloaded_images': [], 'status': 'pending', 'warnings': []})
            item['target'] = copy.deepcopy(asset)
            package['assets'].append(item)
        package['warnings'].extend(catalog['issues'])
        save()

    try:
        inventory = run_workflow(documents, config, model, workflow, persist_workflow, publish_workflow)
        package["checks"]["structural_and_quote_checks"] = True
        package['reference_budget_plan'] = (previous_package.get('reference_budget_plan')
            if store.state.get('reference_phase_started') else None) or reference_plan(inventory['assets'], config, store.totals())
        save()
        if extract_only:
            package['status'] = 'extraction_completed'
            save()
            return package
        # Check the full plan before first entering the reference stage; resuming a started stage is still checked call by call against the call ledger.
        if not store.state.get('reference_phase_started'):
            if config.search_provider == 'model_web' or credentials.get('TAVILY_API_KEY'):
                check_reference_capacity(package['reference_budget_plan'], config, store.totals())
            store.state['reference_phase_started'] = True
            store.save()
        for item in package["assets"]:
            asset = item["target"]
            item['warnings'] = []
            item['downloaded_images'] = []
            batches = []
            for query in asset["queries"][:config.queries_per_asset]:
                try:
                    batches.append(search.query(query))
                except ProviderError as exc:
                    item["warnings"].append(str(exc))
            references = combine_references(batches)
            item["references"] = references
            if config.search_provider == "model_web":
                pages.enrich(references)
            item["warnings"].extend(references["warnings"])
            if references["sources"]:
                try:
                    item["selection"] = model.structured(SELECTION, {"target": asset, "references": references},
                                                         lambda v: validate_selection(v, asset, references))
                except (ProviderError, ValidationError) as exc:
                    item["selection"] = fallback_selection("selection failed; keeping the target specification and continuing generation")
                    item["warnings"].append(str(exc))
            else:
                item["selection"] = fallback_selection("no usable reference material found; generating from protocol requirements and explicit assumptions")
            if config.download_images:
                selected = set(item["selection"]["image_ids"][:config.images_per_asset])
                for image in references["images"]:
                    if image["id"] in selected:
                        try:
                            item["downloaded_images"].append(fetch_image(image, store, transport, config))
                        except (ProviderError, ValidationError) as exc:
                            item["warnings"].append(str(exc))
            item["status"] = "prepared_with_gaps" if (item["selection"]["mode"] == "requirements_only"
                                  or item["selection"]["unknowns"] or item["warnings"]
                                  or not item["downloaded_images"]) else "prepared"
            save()
        package["status"] = "completed_with_gaps" if (package["warnings"] or
                            any(i["status"] != "prepared" for i in package["assets"])) else "completed"
        save()
        return package
    except (BudgetExceeded, UncertainRequest, ProviderError, ValidationError, ProviderLimitError) as exc:
        package["status"] = "paused" if isinstance(exc, (BudgetExceeded, UncertainRequest, ProviderLimitError)) else "failed"
        package["warnings"].append(str(exc))
        save()
        raise
    except KeyboardInterrupt:
        package["status"] = "interrupted"
        package["warnings"].append("interrupted by user; inspect unfinished requests, then resume")
        save()
        raise
