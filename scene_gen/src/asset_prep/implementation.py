"""Implementation version binding: source manifest, dependency versions and input identity, each with a declared coverage.

The main process, workers, follow-up checks and offline replay use one manifest algorithm, avoiding the unprovable practice of "two ad-hoc
current digests being equal". Manifest keys are always ``package/name.py`` and do not depend on
the platform path separator, so the host and the container obtain the same digest.
"""

import importlib.metadata
import sys
from pathlib import Path

from .errors import ValidationError, StaleEvidence
from .storage import digest as value_digest, file_hash

PACKAGES = ("asset_prep", "asset_generation", "scene_generation", "robot_operation")
DEPENDENCIES = ("mujoco", "numpy", "Pillow", "trimesh", "defusedxml", "scipy")

COVERS = {
    "implementation": "digest of the recursive .py sources of the four project packages under src plus runtime_bootstrap.py; "
                      "the official entry point loads from pinned in-memory sources and does not read the project bytecode cache; excludes resources, artifacts and third-party sources.",
    "dependencies": "version numbers of the third-party packages actually imported in this execution; does not cover their source or binary digests.",
    "inputs": "digests of the protocol, asset list, robot arm files and scene configuration; does not cover model responses or run artifacts.",
    "configuration": "digests of scene, generation and service configuration; excludes credentials.",
}


class ImplementationDrift(StaleEvidence):
    """On-disk sources do not match the implementation bound to the run; continuing with mixed-in new code is not allowed."""


def package_manifest(base):
    """Produce per-file digests for the src root containing the three package directories.

    The Docker backend mounts src at /app and the local backend points directly at the project src; both pass the same
    base semantics, so manifest keys and digests are identical. An empty manifest is an environment error: otherwise an empty directory
    would also yield a "valid" identity, passing off empty sources as the bound implementation. A missing package need not be rejected separately:
    it changes the per-file set, so the post-binding check naturally reports drift.
    """
    base = Path(base)
    manifest = {}
    for package in PACKAGES:
        directory = base / package
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            manifest[path.relative_to(base).as_posix()] = file_hash(path)
    if (base/'runtime_bootstrap.py').is_file():
        manifest['runtime_bootstrap.py'] = file_hash(base/'runtime_bootstrap.py')
    if not manifest:
        raise ValidationError("implementation manifest is empty; refusing to pass off empty sources as the bound implementation")
    return manifest


def source_manifest(project):
    """Accepts the project root and points at its src."""
    return package_manifest(source_root(project))


def source_root(project):
    return Path(project) / "src"


def normalize_manifest(value):
    """Accepts the various key forms of historical --before-sources and normalizes them to package/name.py."""
    if not isinstance(value, dict):
        raise ValidationError("invalid source manifest")
    normalized = {}
    for key, item in value.items():
        name = str(key).replace("\\", "/")
        if name.startswith("./"):
            name = name[2:]
        if name.startswith("src/"):
            name = name[4:]
        if name in normalized and normalized[name] != item:
            raise ValidationError("source manifest has conflicting entries with the same name")
        normalized[name] = item
    return normalized


def implementation_digest(manifest):
    return value_digest(normalize_manifest(manifest))


def dependency_versions():
    """Read the dependency versions actually available; missing dependencies are reported by the caller's preflight."""
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_record(project, *, inputs, configuration):
    """Complete binding record: digest + per-file manifest + coverage declaration."""
    active = getattr(sys, '_harness_runtime', None)
    manifest = (dict(active['manifest']) if active and Path(active['base']) == source_root(project).resolve()
                else source_manifest(project))
    identity = implementation_digest(manifest)
    return {
        "implementation": identity,
        "manifest": manifest,
        "dependencies": dependency_versions(),
        "inputs": inputs,
        "configuration": configuration,
        "python": sys.version.split()[0],
        "covers": dict(COVERS),
    }


def verify(expected, *, base=None, project=None, where="process"):
    """Check whether the on-disk sources are still the bound implementation; reject any mismatch outright without compatibility guessing."""
    require_identity(expected, where=where)
    root = Path(base) if base is not None else source_root(project)
    active = getattr(sys, '_harness_runtime', None)
    if active and Path(active['base']) == root.resolve() and active['implementation'] != expected:
        raise ImplementationDrift(where+': the pinned sources actually loaded do not match the bound identity')
    actual = implementation_digest(package_manifest(root))
    if actual != expected:
        raise ImplementationDrift(
            f"{where}: on-disk sources do not match the bound implementation (expected {expected[:12]}, actual {actual[:12]}); "
            "new code from the working directory may not be mixed in during a run; create a new run with a supported migration")
    return actual


def require_identity(identity, *, where="process"):
    if not isinstance(identity, str) or not identity:
        raise ValidationError(f"{where}: missing implementation version binding; refusing to execute unbound sources")
    return identity


class Guard:
    """A boundary check failure is a run-consistency abort and must not become asset repair feedback."""
    def __init__(self, project, expected, *, base=None):
        self.project, self.expected = Path(project), expected
        self.base = Path(base) if base is not None else source_root(project)
        self.dependencies = dependency_versions()
        self.check()

    def check(self):
        verify(self.expected, base=self.base, where='execution boundary')
        active = getattr(sys, '_harness_runtime', None)
        if active and active.get('entrypoint'):
            entry = active['entrypoint']
            if file_hash(entry['path']) != entry['sha256']:
                raise ImplementationDrift('execution entry script changed; refusing to consume the results')
        if dependency_versions() != self.dependencies:
            raise ImplementationDrift('execution dependency versions changed; stopped consuming the results')


def from_identity(project, identity):
    """The main flow and migration share the binding-record construction, so two identity layers are not updated separately."""
    return runtime_record(project,
        inputs={k: identity[k] for k in ('catalogue', 'environment', 'contract', 'robot_files') if k in identity},
        configuration={k: identity[k] for k in ('generation', 'limits', 'provider', 'model',
                                              'base_url', 'output_token_parameter') if k in identity})


def archive_sources(project, root, expected):
    """Archive the pinned execution sources as project provenance; no editable copy is given to the generating model."""
    import base64
    from .storage import atomic_json
    verify(expected, project=project)
    active = getattr(sys, '_harness_runtime', None)
    if active and Path(active['base']) == source_root(project).resolve():
        sources = active['sources']
    else:
        sources = {name: (source_root(project)/name).read_bytes() for name in source_manifest(project)}
    import hashlib
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    if implementation_digest(manifest) != expected:
        raise ImplementationDrift('version changed while archiving the sources')
    target = Path(root)/'implementation_sources'/f'{expected}.json'
    record = {'implementation': expected, 'manifest': manifest,
              'sources_base64': {name: base64.b64encode(data).decode('ascii') for name, data in sources.items()}}
    if target.exists():
        import json
        if json.loads(target.read_text(encoding='utf-8')) != record:
            raise ImplementationDrift('archived execution sources were modified')
    else:
        atomic_json(target, record)
    if active and active.get('entrypoint'):
        entry = active['entrypoint']
        atomic_json(Path(root)/'implementation_sources'/f"entry-{entry['sha256']}.json",
                    {'entrypoint':entry,'source_base64':base64.b64encode(active['entrypoint_source']).decode('ascii')})
