"""实现版本绑定：源码清单、依赖版本与输入身份，并声明各自覆盖范围。

主进程、工作进程、补查与离线回放使用同一份清单算法，避免"两次临时计算的
当前摘要相等"这种不可自证的做法。清单键统一为 ``package/name.py``，不依赖
平台路径分隔符，因此宿主与容器可以得到同一摘要。
"""

import importlib.metadata
import sys
from pathlib import Path

from .errors import ValidationError, StaleEvidence
from .storage import digest as value_digest, file_hash

PACKAGES = ("asset_prep", "asset_generation", "scene_generation", "robot_operation")
DEPENDENCIES = ("mujoco", "numpy", "Pillow", "trimesh", "defusedxml", "scipy")

COVERS = {
    "implementation": "src 下四个工程包的递归 .py 源码及 runtime_bootstrap.py 摘要；"
                      "正式入口从固定内存源码加载，不读取工程字节码缓存；不含资源、产物和第三方源码。",
    "dependencies": "本次执行实际导入的第三方包版本号；不覆盖其源码或二进制摘要。",
    "inputs": "协议、资产清单、机械臂文件与场景配置的摘要；不覆盖模型响应或运行产物。",
    "configuration": "场景、生成与服务配置的摘要；不含凭据。",
}


class ImplementationDrift(StaleEvidence):
    """磁盘源码与运行已绑定的实现不一致；不允许混入新代码继续执行。"""


def package_manifest(base):
    """对包含三个包目录的 src 根目录生成逐文件摘要。

    Docker 后端把 src 挂到 /app，本地后端直接指向项目 src；两者传入同一个
    base 语义，因此清单键与摘要完全一致。空清单视为环境错误：否则空目录
    也能算出"合法"身份，等于用空源码冒充已绑定实现。缺包不必单独拒绝——
    缺包会改变逐文件集合，绑定后的核对自然判为漂移。
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
        raise ValidationError("实现清单为空；拒绝用空源码冒充已绑定实现")
    return manifest


def source_manifest(project):
    """接受项目根目录，指向其 src。"""
    return package_manifest(source_root(project))


def source_root(project):
    return Path(project) / "src"


def normalize_manifest(value):
    """接受历史 --before-sources 的多种键形式，统一为 package/name.py。"""
    if not isinstance(value, dict):
        raise ValidationError("源码清单无效")
    normalized = {}
    for key, item in value.items():
        name = str(key).replace("\\", "/")
        if name.startswith("./"):
            name = name[2:]
        if name.startswith("src/"):
            name = name[4:]
        if name in normalized and normalized[name] != item:
            raise ValidationError("源码清单存在冲突的同名条目")
        normalized[name] = item
    return normalized


def implementation_digest(manifest):
    return value_digest(normalize_manifest(manifest))


def dependency_versions():
    """读取实际可用的依赖版本；缺失依赖由调用方的 preflight 报错。"""
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_record(project, *, inputs, configuration):
    """完整绑定记录：摘要 + 逐文件清单 + 覆盖范围声明。"""
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
    """核对磁盘源码是否仍是绑定的实现；不一致直接拒绝，不做兼容猜测。"""
    require_identity(expected, where=where)
    root = Path(base) if base is not None else source_root(project)
    active = getattr(sys, '_harness_runtime', None)
    if active and Path(active['base']) == root.resolve() and active['implementation'] != expected:
        raise ImplementationDrift(where+'：实际加载的固定源码与绑定身份不一致')
    actual = implementation_digest(package_manifest(root))
    if actual != expected:
        raise ImplementationDrift(
            f"{where}：磁盘源码与已绑定实现不一致（期望 {expected[:12]}，实际 {actual[:12]}）；"
            "运行期间不允许混入工作目录的新代码，请用受支持的迁移创建新运行")
    return actual


def require_identity(identity, *, where="process"):
    if not isinstance(identity, str) or not identity:
        raise ValidationError(f"{where}：缺少实现版本绑定，拒绝执行未绑定源码")
    return identity


class Guard:
    """边界核对失败是运行一致性中止，不能变成资产修复反馈。"""
    def __init__(self, project, expected, *, base=None):
        self.project, self.expected = Path(project), expected
        self.base = Path(base) if base is not None else source_root(project)
        self.dependencies = dependency_versions()
        self.check()

    def check(self):
        verify(self.expected, base=self.base, where='执行边界')
        active = getattr(sys, '_harness_runtime', None)
        if active and active.get('entrypoint'):
            entry = active['entrypoint']
            if file_hash(entry['path']) != entry['sha256']:
                raise ImplementationDrift('执行入口脚本已改变，拒绝消费结果')
        if dependency_versions() != self.dependencies:
            raise ImplementationDrift('执行依赖版本已改变，停止消费结果')


def from_identity(project, identity):
    """主流程和迁移共用绑定记录构造，避免两个身份层次各自更新。"""
    return runtime_record(project,
        inputs={k: identity[k] for k in ('catalogue', 'environment', 'contract', 'robot_files') if k in identity},
        configuration={k: identity[k] for k in ('generation', 'limits', 'provider', 'model',
                                              'base_url', 'output_token_parameter') if k in identity})


def archive_sources(project, root, expected):
    """固定执行源码作为工程来源归档；不向生成模型提供可编辑副本。"""
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
        raise ImplementationDrift('归档源码期间版本变化')
    target = Path(root)/'implementation_sources'/f'{expected}.json'
    record = {'implementation': expected, 'manifest': manifest,
              'sources_base64': {name: base64.b64encode(data).decode('ascii') for name, data in sources.items()}}
    if target.exists():
        import json
        if json.loads(target.read_text(encoding='utf-8')) != record:
            raise ImplementationDrift('已归档的执行源码被修改')
    else:
        atomic_json(target, record)
    if active and active.get('entrypoint'):
        entry = active['entrypoint']
        atomic_json(Path(root)/'implementation_sources'/f"entry-{entry['sha256']}.json",
                    {'entrypoint':entry,'source_base64':base64.b64encode(active['entrypoint_source']).decode('ascii')})
