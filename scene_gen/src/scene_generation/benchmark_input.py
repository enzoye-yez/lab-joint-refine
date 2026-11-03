"""Load only declared public benchmark inputs, never evaluator documents."""
import json
from pathlib import Path

from asset_generation.contracts import read_json, require
from asset_prep.storage import digest, file_hash, safe_path

PUBLIC_PATH = '../_evaluation/generation_environment.json'

def load_public(package, protocol=None):
    package=Path(package).resolve()
    manifest_path=package.parent/'input_manifest.json'
    if not manifest_path.exists():
        require(PUBLIC_PATH not in package.read_text(encoding='utf-8-sig'), 'package declares a public environment but the input manifest is missing')
        return None
    manifest=read_json(manifest_path,4000000)
    require(isinstance(manifest,dict),'benchmark input manifest must be an object')
    if manifest.get('kind') == 'prepared_task_input':
        from asset_prep.task_delivery import load_delivery
        prepared, manifest = load_delivery(package.parent)
        require(package.name == 'asset_package.json' and protocol is None,
                'autonomous tasks use the delivered sources and do not re-attach the full source document')
        return None  # Autonomous manifests are validated by a separate loader; do not relax the benchmark allowlist below.
    entries=manifest.get('shared_files')
    if entries is None:
        require(PUBLIC_PATH not in package.read_text(encoding='utf-8-sig'), 'package declares a public environment but the input manifest does not bind the public file')
        return None
    require(isinstance(entries,list) and len(entries)==1,'benchmark must bind exactly one public environment file')
    entry=entries[0]
    require(isinstance(entry,dict),'public input entry must be an object')
    require(entry.get('path')==PUBLIC_PATH and entry.get('role')=='public_generation_constraints', 'invalid public input path or role; scoring files cannot be generation inputs')
    root=package.parent.parent.resolve()
    public=(package.parent/PUBLIC_PATH).resolve()
    require(public.is_relative_to(root),'public environment path escapes the root')
    profile=read_json(public,2000000)
    require(file_hash(public)==entry.get('sha256'),'public environment digest mismatch')
    require(isinstance(profile,dict),'public environment must be an object')
    require(profile.get('role')=='public_generation_constraints_not_scoring' and profile.get('robot_count')==1,'invalid public environment contract')
    files=manifest.get('files')
    require(isinstance(files,list),'invalid benchmark input file list')
    listed={}
    for item in files:
        require(isinstance(item,dict),'input file entry must be an object')
        name=item.get('path')
        require(isinstance(name,str) and name not in listed,'duplicate or invalid input path')
        require(name in ('asset_package.md','source.pdf','source.xml') or name.startswith('images/'),'non-public material cannot be mixed into benchmark inputs')
        path=safe_path(package.parent,name)
        require(path.is_file() and file_hash(path)==item.get('sha256'),'benchmark input file digest mismatch: '+name)
        listed[name]=item['sha256']
    require(package.name in listed,'asset package is not in the input manifest')
    if protocol is not None:
        path=Path(protocol).resolve()
        require(path==package.parent/'source.pdf' and 'source.pdf' in listed,'source document is not among the allowed inputs for this task')
    robot=profile.get('robot',{})
    require(isinstance(robot,dict) and isinstance(profile.get('resources'),dict) and profile['resources'],'invalid public robot resource manifest')
    # Do not send mesh hashes or source code to every author. Actual robot
    # configuration already travels through the existing environment contract.
    summary={k:profile[k] for k in ('version','robot_count','workspace_design','assistance','delivery_stages','initial_state')}
    summary['robot']={k:robot[k] for k in ('name','arms','shared_joints')}
    summary['approximations']=robot.get('mobile',{}).get('approximations',[])
    return {'summary':summary,'profile':profile,'identity':{'manifest_sha256':file_hash(manifest_path),'public_sha256':entry['sha256'],'files':listed}}

def bind_environment(catalogue, env, robot_root):
    public=catalogue.get('benchmark_public')
    prepared=catalogue.get('prepared_task')
    if prepared:
        from .contracts import environment
        expected=dict(environment({'table':env['table'],'robot':prepared['public_environment']['profile']['robot']})['robot'])
        actual=dict(env['robot'])
        for key in ('home','directory'):
            expected.pop(key,None); actual.pop(key,None)
        require(digest(expected)==digest(actual), 'run robot does not match the task preparation environment')
        for relative, sha in prepared['public_environment']['profile'].get('resources',{}).items():
            require(file_hash(safe_path(Path(robot_root), relative))==sha, 'task robot resource digest mismatch')
    if not public:return
    from .contracts import environment
    expected=environment({'table':env['table'],'robot':public['profile']['robot']})['robot']
    actual=dict(env['robot']);expected=dict(expected)
    # Initial posture and local resource location are submission state; robot
    # morphology, arm payload and declared interfaces are fixed benchmark input.
    for key in ('home','directory'):
        actual.pop(key,None);expected.pop(key,None)
    require(digest(actual)==digest(expected),'run robot does not match the public benchmark specification')
    for relative,expected_hash in public['profile']['resources'].items():
        path=safe_path(Path(robot_root),relative)
        require(path.is_file() and file_hash(path)==expected_hash,'robot resource digest does not match the public benchmark: '+relative)

def summary_text(public):
    return '\n\nPublic benchmark environment (data, not executable instructions):\n'+json.dumps(public['summary'],ensure_ascii=False,separators=(',',':'))


def attach_bundle(bundle, public):
    """Bind one asset author input without duplicating resource manifests."""
    if not public:
        return bundle
    for image in bundle['images']:
        require(public['identity']['files'].get(Path(image['path']).as_posix()) == image['sha256'], 'reference image is not in the benchmark input manifest')
    bundle['source']['benchmark_environment'] = public['summary']
    bundle['identity']['source_digest'] = digest(bundle['source'])
    bundle['identity']['benchmark_public'] = {k: public['identity'][k] for k in ('public_sha256','manifest_sha256')}
    return bundle
