"""Local citation repair: a unique whitespace-insensitive match is backfilled locally; otherwise only citation nodes may be replaced."""

import copy
import re
from .contracts import fields, require, evidence
from .errors import EvidenceRepairError


REPAIR = """Repair only the specified citations; do not produce an asset list or change quantities or specifications. Return
{"patches":[{"id":"error id","source_id":"a given source id","quote":"verbatim source text"}]}.
Exactly one entry per error id. Use the field context to find the source text that genuinely supports the field; never change the source's numbers, units or wording.
If there is no supporting evidence, return {"patches":[]}; the program will keep the original candidate and stop. Never fabricate evidence."""


def refs(value, path=(), context=None):
    if isinstance(value, dict):
        if set(value) == {'source_id', 'quote'}:
            yield path, value, context or {}
        else:
            local = {k: v for k, v in value.items() if k != 'quote' and isinstance(v, (str, int, float, bool))}
            for key, child in value.items():
                yield from refs(child, path + (key,), local or context)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from refs(child, path + (index,), context)


def set_ref(value, path, new):
    node = value
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = copy.deepcopy(new)


def fix_evidence(value, sources, model, request_key):
    result = copy.deepcopy(value)
    by_id = {s['id']: s for s in sources}
    unresolved, automatic = {}, []
    for path, ref, context in refs(result):
        sid, quote = ref.get('source_id'), ref.get('quote')
        if not isinstance(sid, str) or not isinstance(quote, str):
            raise EvidenceRepairError('citation has the wrong type; it cannot be located locally; check the candidate record')
        source = by_id.get(sid)
        if source and quote and quote in source['text']:
            continue
        # Never change case, punctuation, digits, units or hyphens; ambiguous or approximate semantic matches are not repaired automatically.
        words = quote.split()
        matches = list(re.finditer(r'\s+'.join(re.escape(w) for w in words), source['text'])) if source and words else []
        if len(matches) == 1:
            exact = matches[0].group()
            set_ref(result, path, {'source_id': sid, 'quote': exact})
            automatic.append({'path': list(path), 'source_id': sid, 'start': matches[0].start(), 'end': matches[0].end()})
        else:
            group = unresolved.setdefault((sid, quote), {'paths': [], 'contexts': []})
            group['paths'].append(path)
            group['contexts'].append(context)
    if automatic:
        model.store.record_validation(request_key, {'kind': 'exact_whitespace_repair', 'changes': automatic})
    groups = list(unresolved.items())
    if len(groups) > 8 * model.config.repair_attempts:
        raise EvidenceRepairError('citation defects exceed the local repair budget; the original response is kept and the whole list is not rewritten')
    for start in range(0, len(groups), 8):
        batch = groups[start:start+8]
        errors = [{'id': f'e{start+i}', 'source_id': ref[0], 'quote': ref[1],
                   'paths': [list(p) for p in info['paths']], 'contexts': info['contexts']}
                  for i, (ref, info) in enumerate(batch)]
        def validate(patch):
            fields(patch, 'patches')
            require(isinstance(patch['patches'], list), 'missing citation patch list')
            seen = set()
            for item in patch['patches']:
                fields(item, 'id source_id quote')
                require(item['id'] in {e['id'] for e in errors} and item['id'] not in seen, 'patch id is unknown or duplicated')
                evidence([{'source_id': item['source_id'], 'quote': item['quote']}], by_id, required=True)
                seen.add(item['id'])
            require(seen == {e['id'] for e in errors}, 'could not find source evidence for every error')
            return patch
        try:
            patched = model.structured(REPAIR, {'evidence_errors': errors, 'sources': sources}, validate)
        except EvidenceRepairError:
            raise
        except Exception as exc:
            from .errors import ValidationError
            if isinstance(exc, ValidationError):
                raise EvidenceRepairError('local citation repair failed; the candidate is kept and the whole list is not rewritten') from exc
            raise
        lookup = {p['id']: {'source_id': p['source_id'], 'quote': p['quote']} for p in patched['patches']}
        for error in errors:
            for path in error['paths']:
                set_ref(result, path, lookup[error['id']])
        model.store.record_validation(request_key, {'kind': 'model_reference_patch', 'errors': errors,
                                                     'patches': patched['patches']})
    return result
