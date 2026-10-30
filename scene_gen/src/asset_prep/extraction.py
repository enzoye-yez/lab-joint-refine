"""分段提取、增量合并到单一总表；已有资产使用稳定标识。"""

import unicodedata
import copy
from .contracts import fields, require, text, validate_inventory
from .documents import chunks
from .errors import ValidationError, OutputTooLarge
from .prompts import INVENTORY
from .providers import model_messages
from .storage import encoded


EXTRACT = INVENTORY + "\n本次为分段提取。仅提取这一段，资产/步骤确实为空时允许空列表。existing_assets 是已确认总表的索引：同一对象优先复用已有 id；不同规格的独立对象使用不同 id。重复出现不表示需要增加数量，当前段的草稿之后会增量合并到总表。"
MERGE = INVENTORY + """
本次合并若干已校验草稿。统一资产身份和数量，不把相同对象重复累加；发现冲突时 PDF 优先。
返回外层对象 {"inventory": 上述完整资产清单, "coverage": [{"draft_index":0,"asset_id":"输入草稿资产id","target_id":"合并后资产id","reason":"对应或去重依据"}]}。
coverage 必须为每份草稿的每个资产恰好提供一条映射，不允许遗漏。不同草稿可能使用同名 id 指代不同对象，应结合证据判断。
当 existing_catalog 为 true 时，drafts[0] 是已有资产总表，其所有资产 id 必须保留不变，不能重新编号、删除或合并到其他 id。将当前段的重复对象映射到已有 id，补充证据和要求；只有确为不同对象才新增 id。不要按段累加数量。
将映射资产的原始证据（包括数量和要求证据）保留在目标资产的 evidence 中，便于追溯；最终数量及要求仍按 PDF 优先决定。
给出的 excerpts 是原文片段，offset 为原文字符位置，不是新来源；使用原 source_id 引用。
没有足够原文支持的细节保留为未知/假设，不通过猜测缩短草稿。步骤、关系、冲突与未解决事项也应合并。
"""


def request_size(config, instruction, data):
    return len(encoded(model_messages(config, instruction, data)))


def planning_limit(config):
    # 给限定次数的格式修复提示留余量；实际发送仍按完整字节数再次检查。
    return config.max_input_bytes - min(4096, config.max_input_bytes // 8)


def extraction_data(index, sources, existing, config):
    return {'segment': index, 'sources': sources, 'existing_assets': existing,
            'asset_limit': config.extraction_max_assets,
            'output_policy': '最多返回 asset_limit 项候选；超过时返回 {"needs_split":true}，不得遗漏其他资产。说明简洁，步骤只保留资产使用关联。'}


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
            raise ValidationError("单个来源标识及提示已超过输入字节上限；请调整请求上限")
        middle = len(source["text"]) // 2
        # 优先在附近空白处分割，完整保留字符；不会截掉后半段。
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
        raise ValidationError("没有可提取的文档文本")
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
    """只传草稿引用的原文窗口，所有提取段仍完整处理，最终校验对照原始全文。"""
    by_id = {s["id"]: s for s in sources}
    intervals = {}
    for ref in all_evidence(drafts):
        source = by_id.get(ref["source_id"])
        if source is None or not ref["quote"] or ref["quote"] not in source["text"]:
            raise ValidationError("合并输入包含无效原始证据")
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
    """拦截名称/别名明确重复；语义同义识别仍由看到总表索引的模型完成。"""
    previous = []
    for asset in inventory['assets']:
        names = {normalized(n) for n in [asset['name'], *asset['aliases']] if normalized(n)}
        # 规格保留小数点和单位大小写，不能把 1.0 与 10、m 与 M 当成相同值。
        specs = {normalized(r['key']): (unicodedata.normalize('NFKC', r['value']).strip(),
                                       unicodedata.normalize('NFKC', r['unit']).strip())
                 for r in asset['requirements']}
        for other, other_names, other_specs in previous:
            if other['category'] == asset['category'] and names & other_names:
                differences = any(specs[k] != other_specs[k] for k in specs.keys() & other_specs.keys())
                require(differences, '资产名称或别名重复且没有区分规格；复用已有 id，不按段新增重复资产')
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
    require(isinstance(value["coverage"], list), "合并缺少资产映射列表")
    seen = set()
    for mapping in value["coverage"]:
        fields(mapping, "draft_index asset_id target_id reason")
        require(type(mapping["draft_index"]) is int, "草稿序号必须是整数")
        text(mapping["asset_id"])
        text(mapping["target_id"])
        text(mapping["reason"])
        key = mapping["draft_index"], mapping["asset_id"]
        require(key in expected and key not in seen and mapping["target_id"] in targets, "合并映射重复或引用未知资产")
        seen.add(key)
        if existing_catalog and mapping['draft_index'] == 0:
            require(mapping['target_id'] == mapping['asset_id'], '已有总表资产 id 不得重新编号或合并删除')
        original = {(r["source_id"], r["quote"]) for r in all_evidence(expected[key])}
        retained = {(r["source_id"], r["quote"]) for r in all_evidence(targets[mapping["target_id"]])}
        require(original <= retained, "合并丢失了输入资产的原始证据")
    require(seen == set(expected), "合并遗漏输入资产映射")
    validate_unique_assets(inventory)
    # 映射只证明有交代，不能独立证明去重或语义对应正确。
    for field in ("assumptions", "issues"):
        inventory[field] = list(dict.fromkeys([x for draft in drafts for x in draft[field]] + inventory[field]))
    return value


def extract_inventory(documents, config, model, progress=None, on_update=None, checkpoint=None, capture=None):
    """历史请求的只读迁移适配；正常执行入口为 workflow.run_workflow。"""
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
            # 未确认的过长输出不合并；保留全部原文及来源偏移重新分段。
            if sum(len(s['text']) for s in segment) <= 200:
                raise ValidationError('最小提取片段仍超过输出限制；已保留响应，不无限拆分') from exc
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
            on_update(catalog)  # 候选尚未校验合并，总表保持上一个已确认版本。
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
