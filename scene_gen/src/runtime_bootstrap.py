"""在导入工程包前固定源码；只使用标准库，不读取工程字节码缓存。"""

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import runpy
import sys

PACKAGES = ('asset_prep', 'asset_generation', 'scene_generation', 'robot_operation')
DRIFT_EXIT = 78


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def capture(base):
    base = Path(base).resolve()
    sources = {p.relative_to(base).as_posix(): p.read_bytes()
               for package in PACKAGES for p in sorted((base/package).rglob('*.py'))}
    sources['runtime_bootstrap.py'] = (base/'runtime_bootstrap.py').read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    return {'base': str(base), 'sources': sources, 'manifest': manifest,
            'implementation': digest(manifest)}


class SourceLoader(importlib.abc.Loader):
    def __init__(self, record, name, relative):
        self.record, self.name, self.relative = record, name, relative

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        return str(Path(self.record['base'])/self.relative)

    def is_package(self, fullname):
        return self.relative.endswith('/__init__.py')

    def get_code(self, fullname):
        return compile(self.record['sources'][self.relative], self.get_filename(fullname), 'exec')

    def exec_module(self, module):
        exec(self.get_code(module.__name__), module.__dict__)


class SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, record):
        self.record = record

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in PACKAGES:
            return None
        relative = fullname.replace('.', '/')
        for candidate in (relative+'/__init__.py', relative+'.py'):
            if candidate in self.record['sources']:
                loader = SourceLoader(self.record, fullname, candidate)
                return importlib.util.spec_from_file_location(fullname, loader.get_filename(fullname),
                    loader=loader, submodule_search_locations=[] if loader.is_package(fullname) else None)
        # 不允许从磁盘补入快照中不存在的新模块。
        raise ModuleNotFoundError('固定源码中没有模块：'+fullname)


def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--module')
    group.add_argument('--script', type=Path)
    parser.add_argument('--expected-code')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    record = capture(Path(__file__).resolve().parent)
    if args.expected_code and record['implementation'] != args.expected_code:
        print('执行源码与指定验证版本不一致，拒绝启动', file=sys.stderr)
        return DRIFT_EXIT
    if any(name.split('.')[0] in PACKAGES for name in sys.modules):
        raise RuntimeError('启动层必须在工程包导入之前执行')
    sys._harness_runtime = record
    sys.meta_path.insert(0, SourceFinder(record))
    sys.dont_write_bytecode = True
    if args.script:
        script = args.script.resolve()
        record['entrypoint_source'] = script.read_bytes()
        record['entrypoint'] = {'path':str(script), 'sha256':hashlib.sha256(record['entrypoint_source']).hexdigest()}
    from asset_prep.implementation import Guard, ImplementationDrift
    from asset_prep.storage import atomic_json
    guard = Guard(Path(record['base']).parent, record['implementation'], base=record['base'])
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    code = 0
    receipt = {'status':'running','implementation':record['implementation'], 'manifest':record['manifest'],
               'dependencies':guard.dependencies, 'python':sys.version.split()[0]}
    if args.receipt:
        atomic_json(args.receipt, receipt)
    try:
        guard.check()
        sys.argv = [args.module or str(args.script), *arguments]
        try:
            if args.module:
                runpy.run_module(args.module, run_name='__main__', alter_sys=True)
            else:
                exec(compile(record['entrypoint_source'], str(script), 'exec'),
                     {'__name__':'__main__', '__file__':str(script), '__package__':None})
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
        guard.check()
        receipt.update(status='passed' if code == 0 else 'failed', exit_code=code, command=sys.argv)
    except ImplementationDrift as exc:
        print(str(exc), file=sys.stderr)
        code = DRIFT_EXIT
        receipt.update(status='failed', exit_code=code, reason='implementation_drift')
    finally:
        if args.receipt:
            if receipt['status'] == 'running':
                receipt.update(status='failed', reason='interrupted_or_exception')
            atomic_json(args.receipt, receipt)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
