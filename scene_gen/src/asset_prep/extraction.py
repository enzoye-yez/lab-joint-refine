"""Segmented extraction merged incrementally into a single catalog; existing assets keep stable ids."""

import unicodedata
import copy
from .contracts import fields, require, text, validate_inventory
from .documents import chunks
from .errors import ValidationError, OutputTooLarge
from .prompts import INVENTORY
from .providers import model_messages
from .storage import encoded


EXTRACT = INVENTORY + "\nThis is a segmented extraction. Extract only this segment; empty lists are allowed when there really are no assets/steps. existing_assets is the index of the confirmed catalog: the same object should reuse its existing id; independent objects with different specifications use different ids. Repeated mentions do not mean the quantity must increase; the draft for the current segment will later be merged incrementally into the catalog."
MERGE = INVENTORY + """
This time merge several validated drafts. Unify asset identities and quantities without adding up the same object repeatedly; on conflict the PDF takes precedence.
Return the outer object {"inventory": the complete asset list above, "coverage": [{"draft_index":0,"asset_id":"input draft asset id","target_id":"merged asset id","reason":"basis for the correspondence or deduplication"}]}.
coverage must provide exactly one mapping for every asset of every draft, with nothing omitted. Different drafts may use the same id for different objects; judge from the evidence.
When existing_catalog is true, drafts[0] is the existing asset catalog: all of its asset ids must stay unchanged and must not be renumbered, deleted or merged into other ids. Map duplicate objects of the current segment onto existing ids, adding evidence and requirements; add a new id only for a genuinely different object. Do not add up quantities per segment.
Keep the original evidence of mapped assets (including quantity and requirement evidence) in the target asset's evidence for traceability; final quantities and requirements are still decided with the PDF taking precedence.
The given excerpts are source text fragments and offset is the character position in the source text; they are not new sources; cite them with the original source_id.
Details without sufficient source support stay unknown/assumed; do not shorten drafts by guessing. Steps, relations, conflicts and unresolved issues must also be merged.
"""


def request_size(config, instruction, data):
    return len(encoded(model_messages(config, instruction, data)))


def planning_limit(config):
    # Leave headroom for the limited format-repair prompts; the actual send is still re-checked against the full byte count.
    return config.max_input_bytes - min(4096, config.max_input_bytes // 8)


def extraction_data(index, sources, existing, config):
    return {'segment': index, 'sources': sources, 'existing_assets': existing,
            'asset_limit': config.extraction_max_assets,
            'output_policy': 'Return at most asset_limit candidates; if there are more, return {"needs_split":true} and do not omit other assets. Keep explanations concise; steps keep only the asset-usage links.'}


def fit_segments(parts, config, existing_assets=None):
    planned = []
    limit = planning_limit(config)

    def divide(parts):
        if request_size(config, EXTRACT, extraction_data(1000000, parts, existing_assets or [], config)) <= limit:
            planned.append(parts)
            return
        if len(parts) > 1:
            middle = len(parts) // 2
            divide(parts[:middle])
            divide(parts[middle:])
            return
        source = parts[0]
        if len(source["text"]) < 2:
            raise ValidationError("a single source id plus the prompt already exceeds the input byte limit; adjust the request limit")
        middle = len(source["text"]) // 2
        # Prefer splitting at nearby whitespace, keeping all characters; the second half is never cut off.
        cut = source["text"].rfind(" ", middle // 2, middle)
        middle = cut + 1 if cut >= 0 else middle
        divide([dict(source, text=source["text"][:middle])])
        divide([dict(source, text=source["text"][middle:], offset=source.get("offset", 0) + middle)])

    divide(parts)
    return planned


def plan_segments(sources, config):
    planned = []
    for group in chunks(sources, config.chunk_chars):
        planned.extend(fit_segments(group, config))
    if not planned:
        raise ValidationError("no extractable document text")
    return planned


def all_evidence(value):
    if isinstance(value, dict):
        if set(value) == {"source_id", "quote"}:
            yield value
        else:
            for item in value.values():
                yield from all_evidence(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_evidence(item)


def merge_data(drafts, sources, *, existing_catalog=False):
    """Send only the source windows cited by drafts; every extraction segment is still processed in full, and final validation is against the original full text."""
    by_id = {s["id"]: s for s in sources}
    intervals = {}
    for ref in all_evidence(drafts):
        source = by_id.get(ref["source_id"])
        if source is None or not ref["quote"] or ref["quote"] not in source["text"]:
            raise ValidationError("merge input contains invalid original evidence")
        start = source["text"].index(ref["quote"])
        intervals.setdefault(source["id"], set()).add((max(0, start - 160), min(len(source["text"]), start + len(ref["quote"]) + 160)))
    excerpts = []
    for source_id, spans in intervals.items():
        joined = []
        for start, end in sorted(spans):
            if joined and start <= joined[-1][1]:
                joined[-1][1] = max(joined[-1][1], end)
            else:
                joined.append([start, end])
        source = by_id[source_id]
        excerpts.append({"id": source_id, "kind": source["kind"],
                         "excerpts": [{"offset": start, "text": source["text"][start:end]} for start, end in joined]})
    return {"drafts": drafts, "sources": excerpts, "existing_catalog": existing_catalog}


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


def validate_unique_assets(inventory):
    """Block explicit name/alias duplicates; semantic synonym detection is still done by the model that sees the catalog index."""
    previous = []
    for asset in inventory['assets']:
        names = {normalized(n) for n in [asset['name'], *asset['aliases']] if normalized(n)}
        # Specifications keep decimal points and unit case; 1.0 vs 10, or m vs M, must not be treated as the same value.
        specs = {normalized(r['key']): (unicodedata.normalize('NFKC', r['value']).strip(),
                                       unicodedata.normalize('NFKC', r['unit']).strip())
                 for r in asset['requirements']}
        for other, other_names, other_specs in previous:
            if other['category'] == asset['category'] and names & other_names:
                differences = any(specs[k] != other_specs[k] for k in specs.keys() & other_specs.keys())
                require(differences, 'duplicate asset name or alias with no distinguishing specification; reuse the existing id instead of adding a duplicate asset per segment')
        previous.append((asset, names, specs))


def asset_index(inventory):
    return [{'id': a['id'], 'name': a['name'], 'category': a['category'], 'aliases': a['aliases'],
             'requirements': [{k: r[k] for k in ('key', 'value', 'unit')} for r in a['requirements']]}
            for a in inventory['assets']]


def validate_merge(value, drafts, sources, config, *, existing_catalog=False):
    fields(value, "inventory coverage")
    inventory = validate_inventory(value["inventory"], sources, config.max_assets, allow_empty=True)
    targets = {asset["id"]: asset for asset in inventory["assets"]}
    expected = {(i, a["id"]): a for i, draft in enumerate(drafts) for a in draft["assets"]}
    require(isinstance(value["coverage"], list), "merge is missing the asset mapping list")
    seen = set()
    for mapping in value["coverage"]:
        fields(mapping, "draft_index asset_id target_id reason")
        require(type(mapping["draft_index"]) is int, "draft_index must be an integer")
        text(mapping["asset_id"])
        text(mapping["target_id"])
        text(mapping["reason"])
        key = mapping["draft_index"], mapping["asset_id"]
        require(key in expected and key not in seen and mapping["target_id"] in targets, "merge mapping is duplicated or references an unknown asset")
        seen.add(key)
        if existing_catalog and mapping['draft_index'] == 0:
            require(mapping['target_id'] == mapping['asset_id'], 'existing catalog asset ids must not be renumbered, merged away or deleted')
        original = {(r["source_id"], r["quote"]) for r in all_evidence(expected[key])}
        retained = {(r["source_id"], r["quote"]) for r in all_evidence(targets[mapping["target_id"]])}
        require(original <= retained, "merge lost original evidence of an input asset")
    require(seen == set(expected), "merge is missing a mapping for an input asset")
    validate_unique_assets(inventory)
    # A mapping only proves every asset was accounted for; it does not on its own prove that deduplication or semantic correspondence is correct.
    for field in ("assumptions", "issues"):
        inventory[field] = list(dict.fromkeys([x for draft in drafts for x in draft[field]] + inventory[field]))
    return value


def extract_inventory(documents, config, model, progress=None, on_update=None, checkpoint=None, capture=None):
    """Read-only migration adapter for historical requests; the normal execution entry point is workflow.run_workflow."""
    sources = documents["sources"]
    segments = plan_segments(sources, config)
    catalog = {k: [] for k in ('assets', 'steps', 'relations', 'conflicts', 'assumptions', 'issues')}
    progress = progress if progress is not None else {}
    progress.update(total_segments=len(segments), completed_segments=0, applied_segments=0,
                    merge_calls=0, revision=0, updates=[], phase='extracting')
    if checkpoint:
        catalog = copy.deepcopy(checkpoint['inventory'])
        progress.update(copy.deepcopy(checkpoint['extraction']))
        count = progress['applied_segments']
        segments = segments[count:]
        progress['completed_segments'] = count
    while segments:
        existing = asset_index(catalog)
        original = segments.pop(0)
        fitted = fit_segments(original, config, existing)
        segment, remaining = fitted[0], fitted[1:]
        segments[0:0] = remaining
        progress['total_segments'] += len(remaining)
        index = progress['completed_segments']
        def capture_point(draft=None, batch=None):
            if capture:
                capture({'catalog':catalog,'progress':progress,'segment':segment,'queue':segments,
                         'draft':draft,'batch':batch})
        progress.update(phase='extracting', current_segment=index)
        if on_update:
            on_update(catalog)
        capture_point()
        try:
            draft = model.structured(EXTRACT, extraction_data(index, segment, existing, config),
                lambda v: validate_inventory(v, segment, config.max_assets, allow_empty=True))
        except OutputTooLarge as exc:
            # Unconfirmed overlong output is not merged; all source text and source offsets are kept and re-segmented.
            if sum(len(s['text']) for s in segment) <= 200:
                raise ValidationError('the smallest extraction fragment still exceeds the output limit; the response is kept and splitting will not continue indefinitely') from exc
            smaller = []
            for source in segment:
                for group in chunks([source], max(1, sum(len(s['text']) for s in segment) // 2)):
                    for part in group:
                        part['offset'] += source.get('offset', 0)
                    smaller.append(group)
            segments[0:0] = smaller
            progress['total_segments'] += len(smaller) - 1
            progress.setdefault('output_splits', []).append({'segment': index, 'parts': len(smaller), 'reason': str(exc)})
            if on_update:
                on_update(catalog)
            continue
        progress.update(completed_segments=index + 1, phase='merging')
        if on_update:
            on_update(catalog)  # Candidates are not yet validated and merged; the catalog stays at the last confirmed version.
        from .incremental import merge_incrementally
        capture_point(draft)
        updated, merge_calls, decisions = merge_incrementally(catalog, draft, sources, config, model, index,
            on_batch=(lambda batch:capture_point(draft,batch)) if capture else None)
        previous = {a['id']: a for a in catalog['assets']}
        current = {a['id']: a for a in updated['assets']}
        progress['revision'] += 1
        progress['merge_calls'] += merge_calls
        progress['applied_segments'] += 1
        progress['updates'].append({'revision': progress['revision'], 'segment': index,
                                    'decisions': decisions, 'added_ids': sorted(current.keys() - previous.keys()),
                                    'updated_ids': sorted(k for k in current.keys() & previous.keys() if current[k] != previous[k])})
        catalog = updated
        progress['phase'] = 'applied'
        if on_update:
            on_update(catalog)
    return validate_inventory(catalog, sources, config.max_assets)
