"""Version 2 material contract for task preparation; semantic objects do not depend on simulation names that have not been generated yet."""

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
    """Keep the original public object; workspace compatibility is for reading only and does not change the digest."""
    require(isinstance(value, dict) and isinstance(value.get('robot'), dict), 'missing public robot specification')
    workspace = value.get('workspace_design') or value.get('workspace')
    if workspace is None and 'table' in value:
        workspace = {'reference_table': value['table'], 'design':
                     'Bounded mobile dual-arm room; variable tables, real equipment scale, justified clearances.'}
    require(isinstance(workspace, dict) and workspace, 'missing workspace design notes')
    require(value.get('robot_count', 1) == 1, 'the current task uses one robot')
    return {'profile': deepcopy(value), 'sha256': digest(value), 'workspace_design': deepcopy(workspace)}


def source_map(sources):
    return {s['id']: s for s in sources}


def unique(rows, name):
    require(isinstance(rows, list), name + ' must be a list')
    result = {}
    for row in rows:
        require(isinstance(row, dict), name + ' entries must be objects')
        identifier(row.get('id'))
        require(row['id'] not in result, name + ' has a duplicate id')
        result[row['id']] = row
    return result


def validate_task(task, sources, *, partial=False):
    fields(task, 'objective selected_flow selection_reason selection_evidence excluded_branches preparations objects stages final_conditions abstractions assumptions issues')
    for key in ('objective', 'selected_flow', 'selection_reason'):
        text(task[key])
    refs = source_map(sources)
    evidence(task['selection_evidence'],refs,required=True)
    objects = unique(task['objects'], 'task objects')
    stages = unique(task['stages'], 'task stages')
    require(partial or (objects and stages and task['final_conditions']), 'task is missing objects, stages or final conditions')
    for obj in objects.values():
        fields(obj, 'id description initial_state contents quantity evidence')
        for k in ('description', 'initial_state'):
            text(obj[k])
        texts(obj['contents'])
        fields(obj['quantity'], 'value origin basis evidence')
        q = obj['quantity']
        require(type(q['value']) is int and 1 <= q['value'] <= 100000, 'object quantity must be a positive integer')
        origin(q, refs)
        evidence(obj['evidence'], refs, required=True)
    for stage in stages.values():
        fields(stage, 'id description depends_on object_ids outcome evidence')
        text(stage['description']); text(stage['outcome'])
        texts(stage['depends_on']); texts(stage['object_ids'])
        require(set(stage['object_ids']) <= objects.keys() and stage['object_ids'], 'stage objects do not exist or are empty')
        require(set(stage['depends_on']) <= stages.keys() and stage['id'] not in stage['depends_on'], 'stage dependency does not exist')
        evidence(stage['evidence'], refs, required=True)
    visited = set()
    while len(visited) < len(stages):
        ready = {sid for sid, s in stages.items() if sid not in visited and set(s['depends_on']) <= visited}
        require(ready, 'stage dependencies form a cycle'); visited.update(ready)
    used = {oid for s in stages.values() for oid in s['object_ids']}
    require(partial or used == objects.keys(), 'an object is not linked to any execution stage')
    for key in ('preparations', 'final_conditions'):
        for row in unique(task[key], key).values():
            fields(row, 'id description object_ids evidence' + (' basis' if key == 'preparations' else ''))
            text(row['description']); texts(row['object_ids'])
            require(set(row['object_ids']) <= objects.keys(), 'a preparation or final condition references an unknown object')
            evidence(row['evidence'], refs, required=key == 'final_conditions')
            if key == 'preparations': text(row['basis'])
    require(isinstance(task['excluded_branches'], list), 'excluded_branches must be a list')
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
    for role in unique(roles, 'instance roles').values():
        fields(role, 'id asset_id object_ids count stage_ids initial_state basis')
        require(role['asset_id'] in assets, 'role references an unknown asset')
        require(type(role['count']) is int and role['count'] > 0, 'invalid role count')
        texts(role['object_ids']); texts(role['stage_ids'])
        require(set(role['object_ids']) <= objects.keys() and not seen.intersection(role['object_ids']), 'object role is duplicated or does not exist')
        require(len(set(role['object_ids'])) == len(role['object_ids']), 'duplicate object within a role')
        require(role['stage_ids'] and set(role['stage_ids']) <= stages.keys(), 'role has no valid stages')
        if role['object_ids']:
            require(role['count'] == sum(objects[o]['quantity']['value'] for o in role['object_ids']), 'role count does not match the task objects')
            required = {sid for sid, stage in stages.items() if set(role['object_ids']) & set(stage['object_ids'])}
            require(required <= set(role['stage_ids']), 'role misses a task stage')
        seen.update(role['object_ids']); counts[role['asset_id']] += role['count']
        text(role['initial_state']); text(role['basis'])
    require(partial or seen == objects.keys(), 'not all task objects are assigned an asset role')
    require(partial or all(counts.values()), 'an asset has no instance demand')
    return counts


def validate_links(links, task, inventory):
    fields(links, 'interactions mating_requirements assistance_needs')
    aids = {a['id'] for a in inventory['assets']}; sids = {s['id'] for s in task['stages']}
    for group in links.values():
        for row in unique(group, 'interaction requirements').values():
            fields(row, 'id description asset_ids stage_ids basis')
            text(row['description']); text(row['basis'])
            texts(row['asset_ids']); texts(row['stage_ids'])
            require(row['asset_ids'] and set(row['asset_ids']) <= aids, 'interaction asset does not exist')
            require(row['stage_ids'] and set(row['stage_ids']) <= sids, 'interaction stage does not exist')


def validate_package(package):
    require(package.get('schema_version') == PACKAGE_VERSION, 'not a version 2 task package')
    fields(package, 'schema_version status provenance task_definition task_revision task_sha256 inventory instance_roles requirements public_environment sources assets gaps traceability provider requested_model checks')
    require(type(package['task_revision']) is int and package['task_revision']>0, 'invalid task revision number')
    require(package.get('status') in READY, 'task materials have not been fully delivered')
    sources = package['sources']; task = package['task_definition']; inv = package['inventory']
    validate_task(task, sources); validate_inventory(inv, sources, max_assets=100000)
    counts = validate_roles(package['instance_roles'], task, inv)
    validate_links(package['requirements'], task, inv)
    require(all(a['quantity']['value'] == counts[a['id']] for a in inv['assets']), 'type quantities are not summed over instances')
    require(package['task_sha256'] == digest(task), 'task digest mismatch')
    env = package['public_environment']
    require(public_environment(env['profile']) == env, 'public environment digest mismatch')
    ids = [a['asset_id'] for a in package['assets']]
    require(len(set(ids)) == len(ids) and set(ids) == set(counts), 'materials do not match the asset registry')
    by_id = {a['id']: a for a in inv['assets']}
    require(all(a['target'] == by_id[a['asset_id']] for a in package['assets']), 'reference material rewrote the target requirements')
    for item in package['assets']:
        validate_selection(item['selection'],item['target'],item['references'])
    return package


def project_task(package, selected):
    """Give the asset author only the relevant stages and mating summaries, avoiding repeatedly carrying the full registry."""
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
