"""任务准备第二版资料契约；语义对象不依赖尚未生成的仿真名称。"""

from copy import deepcopy

from .contracts import evidence, fields, identifier, origin, require, text, texts, validate_inventory, validate_selection
from .storage import digest

PACKAGE_VERSION = '2.0'
WORKFLOW_VERSION = 'asset_workflow_v3'
READY = {'requirements_completed', 'completed', 'completed_with_gaps'}
BOUNDARY = (
    'Select one source-supported complete workflow. Prepare reagents, sample provenance and '
    'non-goal material fabrication beforehand. Retain required transfer, pipetting, mixing, '
    'cap/lid handling, loading, unloading and device operations. Never remove an operation '
    'because modeling, search or planning is difficult. Do not generate reagents or geometry.'
)


def public_environment(value):
    """保留公开原对象；工作区兼容只用于读取，不改变摘要。"""
    require(isinstance(value, dict) and isinstance(value.get('robot'), dict), '缺少公开机器人规格')
    workspace = value.get('workspace_design') or value.get('workspace')
    if workspace is None and 'table' in value:
        workspace = {'reference_table': value['table'], 'design':
                     'Bounded mobile dual-arm room; variable tables, real equipment scale, justified clearances.'}
    require(isinstance(workspace, dict) and workspace, '缺少工作区设计说明')
    require(value.get('robot_count', 1) == 1, '当前任务使用一台机器人')
    return {'profile': deepcopy(value), 'sha256': digest(value), 'workspace_design': deepcopy(workspace)}


def source_map(sources):
    return {s['id']: s for s in sources}


def unique(rows, name):
    require(isinstance(rows, list), name + '须为列表')
    result = {}
    for row in rows:
        require(isinstance(row, dict), name + '条目须为对象')
        identifier(row.get('id'))
        require(row['id'] not in result, name + '标识重复')
        result[row['id']] = row
    return result


def validate_task(task, sources, *, partial=False):
    fields(task, 'objective selected_flow selection_reason selection_evidence excluded_branches preparations objects stages final_conditions abstractions assumptions issues')
    for key in ('objective', 'selected_flow', 'selection_reason'):
        text(task[key])
    refs = source_map(sources)
    evidence(task['selection_evidence'],refs,required=True)
    objects = unique(task['objects'], '任务对象')
    stages = unique(task['stages'], '任务阶段')
    require(partial or (objects and stages and task['final_conditions']), '任务缺少对象、阶段或终态')
    for obj in objects.values():
        fields(obj, 'id description initial_state contents quantity evidence')
        for k in ('description', 'initial_state'):
            text(obj[k])
        texts(obj['contents'])
        fields(obj['quantity'], 'value origin basis evidence')
        q = obj['quantity']
        require(type(q['value']) is int and 1 <= q['value'] <= 100000, '对象数量须为正整数')
        origin(q, refs)
        evidence(obj['evidence'], refs, required=True)
    for stage in stages.values():
        fields(stage, 'id description depends_on object_ids outcome evidence')
        text(stage['description']); text(stage['outcome'])
        texts(stage['depends_on']); texts(stage['object_ids'])
        require(set(stage['object_ids']) <= objects.keys() and stage['object_ids'], '阶段对象不存在或为空')
        require(set(stage['depends_on']) <= stages.keys() and stage['id'] not in stage['depends_on'], '阶段依赖不存在')
        evidence(stage['evidence'], refs, required=True)
    visited = set()
    while len(visited) < len(stages):
        ready = {sid for sid, s in stages.items() if sid not in visited and set(s['depends_on']) <= visited}
        require(ready, '阶段依赖形成循环'); visited.update(ready)
    used = {oid for s in stages.values() for oid in s['object_ids']}
    require(partial or used == objects.keys(), '对象没有关联执行阶段')
    for key in ('preparations', 'final_conditions'):
        for row in unique(task[key], key).values():
            fields(row, 'id description object_ids evidence' + (' basis' if key == 'preparations' else ''))
            text(row['description']); texts(row['object_ids'])
            require(set(row['object_ids']) <= objects.keys(), '准备或终态引用未知对象')
            evidence(row['evidence'], refs, required=key == 'final_conditions')
            if key == 'preparations': text(row['basis'])
    require(isinstance(task['excluded_branches'], list), '排除分支须为列表')
    for row in task['excluded_branches']:
        fields(row, 'name reason evidence'); text(row['name']); text(row['reason'])
        evidence(row['evidence'], refs, required=True)
    for key in ('abstractions', 'assumptions', 'issues'): texts(task[key])
    return task


def validate_roles(roles, task, inventory, *, partial=False):
    objects = {o['id']: o for o in task['objects']}
    assets = {a['id']: a for a in inventory['assets']}
    stages = {s['id']: s for s in task['stages']}
    seen = set(); counts = {a: 0 for a in assets}
    for role in unique(roles, '实例角色').values():
        fields(role, 'id asset_id object_ids count stage_ids initial_state basis')
        require(role['asset_id'] in assets, '角色引用未知资产')
        require(type(role['count']) is int and role['count'] > 0, '角色数量无效')
        texts(role['object_ids']); texts(role['stage_ids'])
        require(set(role['object_ids']) <= objects.keys() and not seen.intersection(role['object_ids']), '对象角色重复或不存在')
        require(len(set(role['object_ids'])) == len(role['object_ids']), '角色内对象重复')
        require(role['stage_ids'] and set(role['stage_ids']) <= stages.keys(), '角色没有有效阶段')
        if role['object_ids']:
            require(role['count'] == sum(objects[o]['quantity']['value'] for o in role['object_ids']), '角色数量与任务对象不一致')
            required = {sid for sid, stage in stages.items() if set(role['object_ids']) & set(stage['object_ids'])}
            require(required <= set(role['stage_ids']), '角色遗漏任务阶段')
        seen.update(role['object_ids']); counts[role['asset_id']] += role['count']
        text(role['initial_state']); text(role['basis'])
    require(partial or seen == objects.keys(), '任务对象未全部分配资产角色')
    require(partial or all(counts.values()), '资产没有实例需求')
    return counts


def validate_links(links, task, inventory):
    fields(links, 'interactions mating_requirements assistance_needs')
    aids = {a['id'] for a in inventory['assets']}; sids = {s['id'] for s in task['stages']}
    for group in links.values():
        for row in unique(group, '交互要求').values():
            fields(row, 'id description asset_ids stage_ids basis')
            text(row['description']); text(row['basis'])
            texts(row['asset_ids']); texts(row['stage_ids'])
            require(row['asset_ids'] and set(row['asset_ids']) <= aids, '交互资产不存在')
            require(row['stage_ids'] and set(row['stage_ids']) <= sids, '交互阶段不存在')


def validate_package(package):
    require(package.get('schema_version') == PACKAGE_VERSION, '不是第二版任务资料包')
    fields(package, 'schema_version status provenance task_definition task_revision task_sha256 inventory instance_roles requirements public_environment sources assets gaps traceability provider requested_model checks')
    require(type(package['task_revision']) is int and package['task_revision']>0, '任务修订号无效')
    require(package.get('status') in READY, '任务资料尚未完整交付')
    sources = package['sources']; task = package['task_definition']; inv = package['inventory']
    validate_task(task, sources); validate_inventory(inv, sources, max_assets=100000)
    counts = validate_roles(package['instance_roles'], task, inv)
    validate_links(package['requirements'], task, inv)
    require(all(a['quantity']['value'] == counts[a['id']] for a in inv['assets']), '类型数量未按实例汇总')
    require(package['task_sha256'] == digest(task), '任务摘要不一致')
    env = package['public_environment']
    require(public_environment(env['profile']) == env, '公开环境摘要不一致')
    ids = [a['asset_id'] for a in package['assets']]
    require(len(set(ids)) == len(ids) and set(ids) == set(counts), '资料与资产登记表不一致')
    by_id = {a['id']: a for a in inv['assets']}
    require(all(a['target'] == by_id[a['asset_id']] for a in package['assets']), '参考资料改写了目标需求')
    for item in package['assets']:
        validate_selection(item['selection'],item['target'],item['references'])
    return package


def project_task(package, selected):
    """只给资产作者有关阶段和配合摘要，避免重复携带全文登记表。"""
    selected = set(selected); roles = [r for r in package['instance_roles'] if r['asset_id'] in selected]
    stage_ids = {s for r in roles for s in r['stage_ids']}
    object_ids = {o for r in roles for o in r['object_ids']}
    task = package['task_definition']
    links = {key: [r for r in rows if selected.intersection(r['asset_ids'])]
             for key, rows in package['requirements'].items()}
    partner_ids = {a for rows in links.values() for r in rows for a in r['asset_ids']} - selected
    return {'task_sha256': package['task_sha256'], 'task_revision': package['task_revision'],
            'objective': task['objective'], 'selected_flow': task['selected_flow'],
            'preparations': [p for p in task['preparations'] if object_ids.intersection(p['object_ids'])],
            'objects': [o for o in task['objects'] if o['id'] in object_ids],
            'stages': [s for s in task['stages'] if s['id'] in stage_ids],
            'final_conditions': [f for f in task['final_conditions'] if object_ids.intersection(f['object_ids'])],
            'abstractions': task['abstractions'], 'instance_roles': roles, 'requirements': links,
            'partners': [{k: a[k] for k in ('id', 'name', 'category', 'requirements')}
                         for a in package['inventory']['assets'] if a['id'] in partner_ids],
            'public_environment': package['public_environment']}
