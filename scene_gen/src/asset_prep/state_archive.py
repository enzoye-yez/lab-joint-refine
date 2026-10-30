"""Lossless immutable report archives; the live state keeps the budget and indexes."""
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid

from .errors import StaleEvidence
from .storage import atomic_json, encoded, safe_path

MARKER = '__harness_archive_v1__'


@lru_cache(maxsize=2)
def _load(path, sha, modified, size):
    with gzip.open(path, 'rb') as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise StaleEvidence('历史报告摘要不一致：' + str(path))
    return json.loads(raw)


class ArchivedRecord(Mapping):
    """Read-only record. Returned payloads are copies, so callers cannot alter history."""
    def __init__(self, root, reference):
        self.root, self.reference = Path(root), reference

    def __iter__(self):
        return iter(self.reference['keys'])

    def __len__(self):
        return len(self.reference['keys'])

    def __contains__(self, key):
        return key in self.reference['keys']

    def __getitem__(self, key):
        if key in self.reference['index']:
            return deepcopy(self.reference['index'][key])
        if key not in self.reference['keys']:
            raise KeyError(key)
        path = safe_path(self.root, self.reference['path'])
        stat = path.stat()
        return deepcopy(_load(path, self.reference['sha256'], stat.st_mtime_ns, stat.st_size)[key])

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)


def archive(root, record, heavy=('report', 'actions')):
    if isinstance(record, ArchivedRecord):
        return record
    payload = dict(record) if heavy is None else {k:v for k,v in record.items() if k in heavy}
    raw = encoded(payload)
    sha = hashlib.sha256(raw).hexdigest()
    relative = 'state_reports/' + sha + '.json.gz'
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('xb') as stream:
                stream.write(gzip.compress(raw, compresslevel=1, mtime=0))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    else:
        stat = path.stat()
        _load(path, sha, stat.st_mtime_ns, stat.st_size)
    return ArchivedRecord(root, {'path':relative, 'sha256':sha, 'keys':list(record),
        'index':{} if heavy is None else {k:deepcopy(v) for k,v in record.items() if k not in heavy}})


def compact_state(root, state):
    scene = state.get('scene', {})
    for i, stage in enumerate(scene.get('stages', [])):
        if 'report' in stage:
            scene['stages'][i] = archive(root, stage)
    for key, record in scene.get('scene_measurements', {}).items():
        scene['scene_measurements'][key] = archive(root, record)
    for key, record in scene.get('local_checks', {}).items():
        scene['local_checks'][key] = archive(root, record)
    for key, record in scene.get('check_recovery', {}).items():
        if key != scene.get('active_check_recovery'):
            scene['check_recovery'][key] = archive(root, record)
    for i, history in enumerate(scene.get('check_recovery_history', [])):
        scene['check_recovery_history'][i] = archive(root, history, heavy=None)


def write_state(path, state):
    path = Path(path)
    compact_state(path.parent, state)
    def reference(value):
        if isinstance(value, ArchivedRecord):
            return {MARKER:value.reference}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(type(value).__name__)
    atomic_json(path, state, default=reference)


def read_state(path):
    path = Path(path)
    def restore(value):
        if set(value) == {MARKER}:
            return ArchivedRecord(path.parent, value[MARKER])
        return value
    return json.loads(path.read_text(encoding='utf-8-sig'), object_hook=restore)
