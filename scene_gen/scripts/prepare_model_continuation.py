"""Offline model-only batch continuation; never sends requests or changes source runs."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_scene_batch as batch
from asset_generation.pipeline import snapshot
from asset_generation.contracts import require
from asset_prep import implementation
from asset_prep.config import Config
from asset_prep.storage import atomic_json, file_hash, run_lock, safe_path


def inventory(root):
    files = {}
    for path in root.rglob('*'):
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Linked run paths are unsupported')
        if path.is_file() and path.name != '.lock':
            files[path.relative_to(root).as_posix()] = file_hash(path)
    return files


def validate_state(project, root, state, provider, preserve_unknown=False):
    identity = state['identity']
    require(all(identity[k] == provider[k] for k in ('provider', 'model', 'base_url', 'output_token_parameter')), 'Provider identity mismatch')
    implementation.verify(identity['implementation'], project=project, where='model continuation')
    require(identity['binding'] == implementation.from_identity(project, identity), 'Runtime binding mismatch')
    require(state['scene']['status'] == 'paused', 'Only paused runs may switch model')
    require(not any(c['status'] == 'reserved' for c in state['calls']), 'Unsent reservation: recover ledger before switching model')
    for call in state['calls']:
        if call['status'] != 'pending':
            continue
        require(preserve_unknown, 'Unresolved request: reconcile or explicitly use --preserve-unknown')
        require(call.get('kind') == 'model' and type(call.get('index')) is int
                and call['index'] >= 0 and isinstance(call.get('key'), str)
                and call.get('requested_model') == provider['model'], 'Unknown request lacks model provenance')
        require(not (root / f"responses/{call['index']:05d}.json").exists(),
                'Recover saved response before switching model; it must not be abandoned')
    for call in state['calls']:
        if call['status'] == 'done':
            require(file_hash(safe_path(root, call['file'])) == call['sha256'], 'Response archive mismatch')
    for stage in state['scene']['stages']:
        require(snapshot(safe_path(root, stage['directory'])) == stage['files'], 'Stage archive mismatch')
    if state['scene'].get('pending'):
        # Exercise the actual resume contract without committing or altering source state.
        from types import SimpleNamespace
        from scene_generation.checkpoint import recover_review_checkpoint
        probe = SimpleNamespace(root=root, state=deepcopy(state['scene']), commit=lambda *args: None)
        if probe.state['pending'].get('review_checkpoint') is not None:
            recover_review_checkpoint(probe)
        else:
            # Resume archives unfinished construction without admitting it as an asset.
            require(safe_path(root, probe.state['pending']['directory']).is_dir(), 'Pending workspace missing')

def prepare(project, source_name, target_name, model, preserve_unknown=False):
    for name in (source_name, target_name):
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', name), 'Invalid batch name')
    require(isinstance(model, str) and model.strip() == model and bool(model), 'Invalid model ID')
    source = safe_path(project / 'runs/batches', source_name)
    target = safe_path(project / 'runs/batches', target_name)
    require((source / 'batch.json').is_file() and not target.exists(), 'Source missing or target already exists')
    with ExitStack() as stack:
        stack.enter_context(run_lock(source))
        manifest = batch.read(source / 'batch.json')
        batch.verify(manifest, project)
        require(manifest['status'] in {'interrupted', 'paused_provider', 'paused_user'}, 'Source batch must be stopped')
        provider = batch.read(source / 'config/provider.json')
        require(provider['model'] != model, 'Model is unchanged')
        new_provider = {**provider, 'model': model}
        Config(**new_provider)
        prepared = []
        require(any(e['status'] not in batch.TERMINAL for e in manifest['entries']), 'No unfinished entries')
        for entry in manifest['entries']:
            if entry['status'] in batch.TERMINAL:
                continue
            old = safe_path(project / 'runs', entry['run'])
            new_name = target_name + '_' + entry['protocol'].lower().replace('-', '')
            new = safe_path(project / 'runs', new_name)
            require(not new.exists(), 'Target run already exists')
            state = None
            before = None
            if old.exists():
                stack.enter_context(run_lock(old))
                state = batch.read(old / 'state.json')
                validate_state(project, old, state, provider, preserve_unknown)
                before = inventory(old)
            else:
                require(entry['status'] == 'queued', 'Missing started run')
            prepared.append((entry, old, new, state, before))
        receipt = dict(kind='explicit_model_continuation', source_batch=source_name, target_batch=target_name,
                       created_at=batch.now(), from_provider=provider['provider'], from_model=provider['model'],
                       to_provider=new_provider['provider'], to_model=model, model_calls_added=0,
                       preserve_unknown=preserve_unknown,
                       inherited_evidence_policy='Same inputs, geometry, checker, thresholds and budgets; original call provenance retained. New requests use new model.',
                       source_manifest_sha256=file_hash(source / 'batch.json'), runs=[])
        # Publish batch.json only after all copied evidence is verified. Partial targets are never executable.
        target.mkdir(parents=True)
        shutil.copytree(source / 'config', target / 'config')
        atomic_json(target / 'config/provider.json', new_provider)
        result = deepcopy(manifest)
        result.update(batch=target_name, status='prepared', created_at=batch.now())
        result['entries'] = [deepcopy(e) for e in manifest['entries'] if e['status'] in batch.TERMINAL]
        for entry, old, new, state, before in prepared:
            child = {k: entry[k] for k in ('protocol', 'pdf', 'package')}
            child.update(run=new.name, status='queued', inherited_from=entry['run'] if state else None)
            if state:
                shutil.copytree(old, new, ignore=shutil.ignore_patterns('.lock'))
                require(inventory(new) == before and inventory(old) == before, 'Copy/source integrity mismatch')
                original_identity = deepcopy(state['identity'])
                state['identity']['model'] = model
                state['identity']['binding'] = implementation.from_identity(project, state['identity'])
                transition = dict(source_run=entry['run'], source_state_sha256=before['state.json'],
                                  previous_identity=original_identity, model=model, at=batch.now(),
                                  inherited_calls=len(state['calls']), mixed_model=True)
                unknown_indices = []
                for call in state['calls']:
                    if call['status'] == 'pending':
                        # Retire only the old model's locally unresolved request. Its remote
                        # outcome and full reservation remain unknown, never failed/refunded.
                        unknown_indices.append(call['index'])
                        call.update(status='unknown_not_replayed', outcome_unknown=True,
                                    model_transition_source_status='pending',
                                    model_transition_target=model)
                transition['unknown_calls_not_replayed'] = unknown_indices
                state.setdefault('model_history', []).append(transition)
                from asset_prep.state_archive import write_state
                write_state(new / 'state.json', state)
                atomic_json(new / 'model_transition.json', transition)
                if (new / 'result.json').exists():
                    old_result = batch.read(new / 'result.json')
                    old_result.update(status='paused', model_transition=transition,
                                      continuation_note='Inherited ' + provider['model'] + ' evidence; subsequent calls use ' + model)
                    atomic_json(new / 'result.json', old_result)
                receipt['runs'].append(dict(source=old.name, target=new.name, inherited_calls=len(state['calls']),
                                            source_files=before, unknown_calls_not_replayed=unknown_indices))
            result['entries'].append(child)
        frozen = {}
        for name, sha in manifest['frozen'].items():
            path = Path(name)
            if path.is_relative_to(source / 'config'):
                path = target / 'config' / path.relative_to(source / 'config')
                sha = file_hash(path)
            frozen[str(path)] = sha
        order={e['protocol']:i for i,e in enumerate(manifest['entries'])}
        result['entries'].sort(key=lambda e:order[e['protocol']])
        result['frozen'] = frozen
        result['model_transition'] = {k:v for k,v in receipt.items() if k != 'runs'}
        batch.verify(result, project)
        for _, old, _, state, before in prepared:
            if state:
                require(inventory(old) == before, 'Source changed during preparation')
        atomic_json(target / 'model_transition.json', receipt)
        atomic_json(target / 'batch.json', result)
        return dict(batch=target_name, model=model, inherited_runs=len(receipt['runs']), model_calls_added=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--preserve-unknown', action='store_true',
                        help='Explicitly retain unresolved old-model calls and their budget; remote charges remain unknown')
    args = parser.parse_args()
    print(json.dumps(prepare(PROJECT, args.source, args.target, args.model, args.preserve_unknown), ensure_ascii=False))

