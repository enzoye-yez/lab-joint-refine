"""独立、不可变的公开交付目录；不复制请求账本或评分材料。"""

import json
import shutil
from pathlib import Path

from .contracts import require
from .storage import atomic_json, digest, file_hash, safe_path
from .task_contracts import validate_package, public_environment


def load_delivery(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    require(manifest.get('kind') == 'prepared_task_input' and manifest.get('version') == 2,
            '不是任务准备公开输入清单')
    listed = {}
    for row in manifest['files']:
        name = row['path']
        require(name not in listed and (name in {'asset_package.json', 'public_environment.json', 'README.md'}
                or name.startswith('images/')), '公开交付包含未允许或重复文件')
        path = safe_path(root, name)
        require(path.is_file() and file_hash(path) == row['sha256'], '交付文件摘要不一致：'+name)
        listed[name] = row['sha256']
    require({'asset_package.json', 'public_environment.json'} <= listed.keys(), '交付缺少资料或环境')
    package = validate_package(json.loads((root/'asset_package.json').read_text(encoding='utf-8')))
    profile = json.loads((root/'public_environment.json').read_text(encoding='utf-8'))
    require(public_environment(profile) == package['public_environment'], '公开环境与资料包不一致')
    require(manifest['task_sha256'] == package['task_sha256'], '清单与任务版本不一致')
    for item in package['assets']:
        for image in item['downloaded_images']:
            require(listed.get(image['local_path']) == image['sha256'], '图片不在公开输入清单中')
    return package, manifest


def publish(store, package):
    validate_package(package)
    relative = 'deliveries/'+digest(package)[:24]
    destination = safe_path(store.root, relative)
    if (destination/'input_manifest.json').exists():
        saved, _ = load_delivery(destination)
        require(saved == package, '已交付版本不能覆盖')
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for item in package['assets']:
            for image in item['downloaded_images']:
                src = safe_path(store.root, image['local_path'])
                require(file_hash(src) == image['sha256'], '待交付图片摘要不一致')
                target = safe_path(destination, image['local_path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists(): require(file_hash(target) == image['sha256'], '交付图片已修改')
                else: shutil.copyfile(src, target)
        atomic_json(destination/'asset_package.json', package)
        atomic_json(destination/'public_environment.json', package['public_environment']['profile'])
        task = package['task_definition']
        lines = ['# Prepared task', '', task['objective'], '', '## Selected workflow', task['selected_flow'],
                 task['selection_reason'], '', '## Preparations']
        lines += ['- '+p['description']+' — '+p['basis'] for p in task['preparations']]
        lines += ['', '## Initial objects'] + ['- '+o['id']+': '+o['initial_state'] for o in task['objects']]
        lines += ['', '## Required stages'] + ['- '+s['id']+': '+s['description']+' → '+s['outcome'] for s in task['stages']]
        lines += ['', '## Final conditions'] + ['- '+f['description'] for f in task['final_conditions']]
        lines += ['', '## Assets'] + ['- '+a['id']+': '+a['name']+'; count='+str(a['quantity']['value']) for a in package['inventory']['assets']]
        lines += ['', '## Gaps'] + ['- '+x for x in package['gaps']]
        lines += ['', 'The JSON package is authoritative. Images are downloaded references, not visually verified.']
        (destination/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        names = ['asset_package.json', 'public_environment.json', 'README.md']
        names += sorted({im['local_path'] for a in package['assets'] for im in a['downloaded_images']})
        atomic_json(destination/'input_manifest.json', {'kind':'prepared_task_input', 'version':2,
                    'task_id':store.root.name, 'task_sha256':package['task_sha256'],
                    'files':[{'path':n, 'sha256':file_hash(destination/n)} for n in names]})
        load_delivery(destination)
    atomic_json(store.root/'delivery.json', {'directory':relative, 'task_sha256':package['task_sha256'],
                                          'package_sha256':file_hash(destination/'asset_package.json')})
    return destination


def validate_role_bindings(plan, package):
    roles = {r['id']:r for r in package['instance_roles']}
    rows = plan.get('role_bindings')
    require(isinstance(rows, list) and len(rows) == len(roles), '场景计划缺少实例角色分配')
    instances = {i['id']:i for i in plan['instances']}; jobs = {j['id']:j for j in plan['jobs']}
    seen = set(); allocated = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'role_id', 'instances'}, '角色分配字段无效')
        rid = row['role_id']
        require(rid in roles and rid not in seen, '角色分配重复或未知'); seen.add(rid)
        role = roles[rid]; names = row['instances']
        require(isinstance(names, list) and len(names) == role['count'] and len(set(names)) == len(names), '角色实例数量不一致')
        for name in names:
            require(name in instances, '角色引用未知场景实例')
            require(role['asset_id'] in jobs[instances[name]['job']]['asset_ids'], '场景实例与角色资产类型不一致')
            key = (role['asset_id'], name)
            require(key not in allocated, '同一对象被重复分配角色'); allocated.add(key)
    return rows


def validate_task_coverage(coverage, definition, compiled):
    """语义要求逐项映射；不可检测项明确保留，不能用空映射删除要求。"""
    required = {'stage:'+s['id'] for s in definition['stages']}
    required |= {'initial:'+o['id'] for o in definition['objects']}
    required |= {'final:'+f['id'] for f in definition['final_conditions']}
    stage_ids = {s['id'] for s in compiled['stages']}; seen = set(); unresolved = []
    require(isinstance(coverage, list), '缺少上游任务覆盖表')
    for row in coverage:
        require(isinstance(row, dict) and set(row) == {'requirement', 'targets', 'unresolved'}, '任务覆盖字段无效')
        key = row['requirement']; require(key in required and key not in seen, '任务覆盖重复或未知'); seen.add(key)
        require(isinstance(row['targets'], list) and isinstance(row['unresolved'], str), '任务覆盖值无效')
        allowed = stage_ids if key.startswith('stage:') else {'initial'} if key.startswith('initial:') else {'final'}
        require(set(row['targets']) <= allowed and (row['targets'] or row['unresolved'].strip()), '要求未落实也未说明缺口')
        if row['unresolved'].strip(): unresolved.append(key+': '+row['unresolved'])
    require(seen == required, '任务覆盖遗漏必要要求')
    return unresolved
