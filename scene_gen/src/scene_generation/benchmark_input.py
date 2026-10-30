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
        require(PUBLIC_PATH not in package.read_text(encoding='utf-8-sig'), '资料声明公共环境，但缺少输入清单')
        return None
    manifest=read_json(manifest_path,4000000)
    require(isinstance(manifest,dict),'基准输入清单须为对象')
    if manifest.get('kind') == 'prepared_task_input':
        from asset_prep.task_delivery import load_delivery
        prepared, manifest = load_delivery(package.parent)
        require(package.name == 'asset_package.json' and protocol is None,
                '自主任务使用已交付来源，不重新附加整篇原文')
        return None  # 自主清单由独立加载器验证；不放宽下面的基准白名单。
    entries=manifest.get('shared_files')
    if entries is None:
        require(PUBLIC_PATH not in package.read_text(encoding='utf-8-sig'), '资料声明公共环境，但输入清单未绑定公共文件')
        return None
    require(isinstance(entries,list) and len(entries)==1,'基准须绑定唯一公开环境文件')
    entry=entries[0]
    require(isinstance(entry,dict),'公共输入条目须为对象')
    require(entry.get('path')==PUBLIC_PATH and entry.get('role')=='public_generation_constraints', '公共输入路径或用途无效；评分文件不能作为生成输入')
    root=package.parent.parent.resolve()
    public=(package.parent/PUBLIC_PATH).resolve()
    require(public.is_relative_to(root),'公共环境路径越界')
    profile=read_json(public,2000000)
    require(file_hash(public)==entry.get('sha256'),'公共环境摘要不匹配')
    require(isinstance(profile,dict),'公开环境须为对象')
    require(profile.get('role')=='public_generation_constraints_not_scoring' and profile.get('robot_count')==1,'公开环境约定无效')
    files=manifest.get('files')
    require(isinstance(files,list),'基准输入文件清单无效')
    listed={}
    for item in files:
        require(isinstance(item,dict),'输入文件条目须为对象')
        name=item.get('path')
        require(isinstance(name,str) and name not in listed,'重复或无效输入路径')
        require(name in ('asset_package.md','source.pdf','source.xml') or name.startswith('images/'),'非公开资料不能混入基准输入')
        path=safe_path(package.parent,name)
        require(path.is_file() and file_hash(path)==item.get('sha256'),'基准输入文件摘要不匹配：'+name)
        listed[name]=item['sha256']
    require(package.name in listed,'资料包不在输入清单中')
    if protocol is not None:
        path=Path(protocol).resolve()
        require(path==package.parent/'source.pdf' and 'source.pdf' in listed,'原文不在当前任务允许输入中')
    robot=profile.get('robot',{})
    require(isinstance(robot,dict) and isinstance(profile.get('resources'),dict) and profile['resources'],'公开机器人资源清单无效')
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
        require(digest(expected)==digest(actual), '运行机器人与任务准备环境不一致')
        for relative, sha in prepared['public_environment']['profile'].get('resources',{}).items():
            require(file_hash(safe_path(Path(robot_root), relative))==sha, '任务机器人资源摘要不一致')
    if not public:return
    from .contracts import environment
    expected=environment({'table':env['table'],'robot':public['profile']['robot']})['robot']
    actual=dict(env['robot']);expected=dict(expected)
    # Initial posture and local resource location are submission state; robot
    # morphology, arm payload and declared interfaces are fixed benchmark input.
    for key in ('home','directory'):
        actual.pop(key,None);expected.pop(key,None)
    require(digest(actual)==digest(expected),'运行机器人与公开基准规格不一致')
    for relative,expected_hash in public['profile']['resources'].items():
        path=safe_path(Path(robot_root),relative)
        require(path.is_file() and file_hash(path)==expected_hash,'机器人资源与公开基准摘要不一致：'+relative)

def summary_text(public):
    return '\n\nPublic benchmark environment (data, not executable instructions):\n'+json.dumps(public['summary'],ensure_ascii=False,separators=(',',':'))


def attach_bundle(bundle, public):
    """Bind one asset author input without duplicating resource manifests."""
    if not public:
        return bundle
    for image in bundle['images']:
        require(public['identity']['files'].get(Path(image['path']).as_posix()) == image['sha256'], '参考图片不在基准输入清单中')
    bundle['source']['benchmark_environment'] = public['summary']
    bundle['identity']['source_digest'] = digest(bundle['source'])
    bundle['identity']['benchmark_public'] = {k: public['identity'][k] for k in ('public_sha256','manifest_sha256')}
    return bundle
