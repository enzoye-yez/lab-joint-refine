"""已知新项目版本的离线检查点升级；不请求网络，不重置预算。"""

import copy
import json
import time

from .contracts import require, validate_inventory
from .extraction import plan_segments, validate_unique_assets
from .storage import atomic_json, digest, file_hash, safe_path


LEGACY_CODE = 'ca8f91e97d627d4aa6b8b52dbb9a60c9965eaf0ee34043eb4b7f788ec771e5f7'
# 此版本的成功请求格式完全保留；仅在原来本地失败的位置启用类别索引。
SCOPED_REPLAY_CODE = '245ec04872e75a90240e268abc6ede9a720cf7e8df340bea7c3dbf01c8eeef5b'


def upgrade_run(root, identity, documents, config):
    if identity.get('workflow_contract')=='asset_workflow_v2':
        from .migration import migrate_run
        return migrate_run(root,identity,documents,config)
    return legacy_upgrade_run(root,identity,documents,config)


def legacy_upgrade_run(root, identity, documents, config):
    path = safe_path(root, 'state.json')
    state = json.loads(path.read_text(encoding='utf-8'))
    if state['identity'] == identity:
        return {'status': 'already_current', 'network_calls': 0}
    old = state['identity']
    require(old.get('code') in {LEGACY_CODE, SCOPED_REPLAY_CODE}, '只支持已核对版本的检查点升级，不能绕过其他源码变化')
    require({k: v for k, v in old.items() if k != 'code'} ==
            {k: v for k, v in identity.items() if k != 'code'}, '输入、解析结果、模型或语义配置已改变，不能升级')
    for name, sha in state.get('outputs', {}).items():
        require(file_hash(safe_path(root, name)) == sha, '旧输出已修改，不能升级')
    require({'asset_package.json', 'documents.json'} <= state.get('outputs', {}).keys(), '检查点缺少输出摘要')
    for call in state['calls']:
        require(call['kind'] == 'model', '只支持尚未开始检索的提取检查点')
        if call['status'] == 'done':
            require(file_hash(safe_path(root, call['file'])) == call['sha256'], '旧响应已修改，不能升级')
    package = json.loads(safe_path(root, 'asset_package.json').read_text(encoding='utf-8'))
    require(package['identity'] == old and package['status'] in {'paused', 'failed', 'interrupted'}, '旧运行不处于可升级的暂停状态')
    progress = package['extraction']
    count = progress['applied_segments']
    if old['code'] == SCOPED_REPLAY_CODE:
        require(not state.get('checkpoint'), '缓存重放升级不接受其他迁移的检查点')
        require(count == progress['revision'] == len(progress['updates']), '旧检查点修订记录不一致')
        require([u['segment'] for u in progress['updates']] == list(range(count)), '旧检查点段落序号不连续')
        validate_inventory(package['inventory'], documents['sources'], config.max_assets, allow_empty=True)
        validate_unique_assets(package['inventory'])
        # 动态分段不能用 applied_segments 切原始段落列表；从原始输入重放缓存
        # 才能恢复相同的拆分队列，并复用当前段中已完成的合并批次。
        state.setdefault('upgrades', []).append({'at': time.time(), 'from_identity': old,
            'to_identity': identity, 'mode': 'cached_replay',
            'outputs_before': copy.deepcopy(state['outputs']), 'calls_preserved': len(state['calls'])})
        state['identity'] = identity
        atomic_json(path, state)
        return {'status': 'upgraded', 'mode': 'cached_replay', 'applied_segments': count,
                'assets_preserved': len(package['inventory']['assets']), 'network_calls': 0}
    total = len(plan_segments(documents['sources'], config))
    require(progress['total_segments'] == total, '发生过动态分段的旧运行暂不支持检查点升级')
    require(0 <= count < total and count == progress['revision'] == len(progress['updates']), '旧检查点修订记录不一致')
    require([u['segment'] for u in progress['updates']] == list(range(count)), '旧检查点段落序号不连续')
    validate_inventory(package['inventory'], documents['sources'], config.max_assets, allow_empty=True)
    validate_unique_assets(package['inventory'])
    checkpoint = copy.deepcopy({k: package[k] for k in ('inventory', 'extraction')})
    checkpoint['extraction']['completed_segments'] = count
    # 新身份仅写账本；旧资料包保留旧身份直到真正恢复，由迁移审计明确解释。
    state['checkpoint'] = checkpoint
    state['checkpoint_sha256'] = digest(checkpoint)
    state.setdefault('upgrades', []).append({'at': time.time(), 'from_identity': old,
        'to_identity': identity, 'checkpoint_sha256': digest(checkpoint),
        'outputs_before': copy.deepcopy(state['outputs']), 'calls_preserved': len(state['calls'])})
    state['identity'] = identity
    atomic_json(path, state)
    return {'status': 'upgraded', 'applied_segments': count,
            'assets_preserved': len(checkpoint['inventory']['assets']), 'network_calls': 0}
