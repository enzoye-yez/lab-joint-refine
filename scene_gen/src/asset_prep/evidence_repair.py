"""引用局部修复：唯一空白匹配由本地回填，其余仅允许替换引用节点。"""

import copy
import re
from .contracts import fields, require, evidence
from .errors import EvidenceRepairError


REPAIR = """只修复指定引用，不生成资产清单、不修改数量或规格。返回
{"patches":[{"id":"错误编号","source_id":"给定来源id","quote":"逐字原文"}]}。
每个错误编号恰好一项。结合字段上下文寻找真正支持该字段的原文；不能改变原文数值、单位或措辞。
若没有支持依据，返回 {"patches":[]}，由程序保留原候选并停止，不能编造依据。"""


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
            raise EvidenceRepairError('引用类型错误；不能局部定位，请检查候选记录')
        source = by_id.get(sid)
        if source and quote and quote in source['text']:
            continue
        # 不改大小写、标点、数字、单位、连字符；歧义或近似语义不自动修复。
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
        raise EvidenceRepairError('引用缺陷超过局部修复预算；已保留原响应，不重写整份清单')
    for start in range(0, len(groups), 8):
        batch = groups[start:start+8]
        errors = [{'id': f'e{start+i}', 'source_id': ref[0], 'quote': ref[1],
                   'paths': [list(p) for p in info['paths']], 'contexts': info['contexts']}
                  for i, (ref, info) in enumerate(batch)]
        def validate(patch):
            fields(patch, 'patches')
            require(isinstance(patch['patches'], list), '缺少引用补丁列表')
            seen = set()
            for item in patch['patches']:
                fields(item, 'id source_id quote')
                require(item['id'] in {e['id'] for e in errors} and item['id'] not in seen, '补丁编号未知或重复')
                evidence([{'source_id': item['source_id'], 'quote': item['quote']}], by_id, required=True)
                seen.add(item['id'])
            require(seen == {e['id'] for e in errors}, '无法为每个错误找到原文依据')
            return patch
        try:
            patched = model.structured(REPAIR, {'evidence_errors': errors, 'sources': sources}, validate)
        except EvidenceRepairError:
            raise
        except Exception as exc:
            from .errors import ValidationError
            if isinstance(exc, ValidationError):
                raise EvidenceRepairError('局部引用修复失败；已保留候选，不重写整份清单') from exc
            raise
        lookup = {p['id']: {'source_id': p['source_id'], 'quote': p['quote']} for p in patched['patches']}
        for error in errors:
            for path in error['paths']:
                set_ref(result, path, lookup[error['id']])
        model.store.record_validation(request_key, {'kind': 'model_reference_patch', 'errors': errors,
                                                     'patches': patched['patches']})
    return result
