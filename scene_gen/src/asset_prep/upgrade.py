"""Offline checkpoint upgrade for known newer project versions; makes no network requests and does not reset budgets."""

import copy
import json
import time

from .contracts import require, validate_inventory
from .extraction import plan_segments, validate_unique_assets
from .storage import atomic_json, digest, file_hash, safe_path


LEGACY_CODE = 'ca8f91e97d627d4aa6b8b52dbb9a60c9965eaf0ee34043eb4b7f788ec771e5f7'
# This version keeps the successful request format exactly; the category index is enabled only where the original run failed locally.
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
    require(old.get('code') in {LEGACY_CODE, SCOPED_REPLAY_CODE}, 'checkpoint upgrade is supported only for verified versions; other source changes cannot be bypassed')
    require({k: v for k, v in old.items() if k != 'code'} ==
            {k: v for k, v in identity.items() if k != 'code'}, 'input, parse results, model or semantic configuration changed; cannot upgrade')
    for name, sha in state.get('outputs', {}).items():
        require(file_hash(safe_path(root, name)) == sha, 'old outputs were modified; cannot upgrade')
    require({'asset_package.json', 'documents.json'} <= state.get('outputs', {}).keys(), 'checkpoint lacks output digests')
    for call in state['calls']:
        require(call['kind'] == 'model', 'only extraction checkpoints that have not started search are supported')
        if call['status'] == 'done':
            require(file_hash(safe_path(root, call['file'])) == call['sha256'], 'old responses were modified; cannot upgrade')
    package = json.loads(safe_path(root, 'asset_package.json').read_text(encoding='utf-8'))
    require(package['identity'] == old and package['status'] in {'paused', 'failed', 'interrupted'}, 'old run is not in an upgradable paused state')
    progress = package['extraction']
    count = progress['applied_segments']
    if old['code'] == SCOPED_REPLAY_CODE:
        require(not state.get('checkpoint'), 'cache-replay upgrade does not accept checkpoints from other migrations')
        require(count == progress['revision'] == len(progress['updates']), 'old checkpoint revision records are inconsistent')
        require([u['segment'] for u in progress['updates']] == list(range(count)), 'old checkpoint segment indices are not contiguous')
        validate_inventory(package['inventory'], documents['sources'], config.max_assets, allow_empty=True)
        validate_unique_assets(package['inventory'])
        # With dynamic segmentation, applied_segments cannot be used to slice the original segment list; only replaying the cache from the original input
        # restores the same split queue and reuses the merge batches already completed in the current segment.
        state.setdefault('upgrades', []).append({'at': time.time(), 'from_identity': old,
            'to_identity': identity, 'mode': 'cached_replay',
            'outputs_before': copy.deepcopy(state['outputs']), 'calls_preserved': len(state['calls'])})
        state['identity'] = identity
        atomic_json(path, state)
        return {'status': 'upgraded', 'mode': 'cached_replay', 'applied_segments': count,
                'assets_preserved': len(package['inventory']['assets']), 'network_calls': 0}
    total = len(plan_segments(documents['sources'], config))
    require(progress['total_segments'] == total, 'checkpoint upgrade is not yet supported for old runs that underwent dynamic segmentation')
    require(0 <= count < total and count == progress['revision'] == len(progress['updates']), 'old checkpoint revision records are inconsistent')
    require([u['segment'] for u in progress['updates']] == list(range(count)), 'old checkpoint segment indices are not contiguous')
    validate_inventory(package['inventory'], documents['sources'], config.max_assets, allow_empty=True)
    validate_unique_assets(package['inventory'])
    checkpoint = copy.deepcopy({k: package[k] for k in ('inventory', 'extraction')})
    checkpoint['extraction']['completed_segments'] = count
    # The new identity is written only to the ledger; the old package keeps the old identity until an actual resume, as explained explicitly by the migration audit.
    state['checkpoint'] = checkpoint
    state['checkpoint_sha256'] = digest(checkpoint)
    state.setdefault('upgrades', []).append({'at': time.time(), 'from_identity': old,
        'to_identity': identity, 'checkpoint_sha256': digest(checkpoint),
        'outputs_before': copy.deepcopy(state['outputs']), 'calls_preserved': len(state['calls'])})
    state['identity'] = identity
    atomic_json(path, state)
    return {'status': 'upgraded', 'applied_segments': count,
            'assets_preserved': len(checkpoint['inventory']['assets']), 'network_calls': 0}
