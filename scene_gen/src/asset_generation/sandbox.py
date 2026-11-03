"""容器执行边界。未安装隔离环境时拒绝执行生成代码，无宿主机回退。"""

import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from asset_prep.errors import PrepError, ValidationError
from asset_prep.storage import atomic_json, safe_path
from .contracts import read_json, require
from .check_report import checked_report, failed_report


class SandboxUnavailable(PrepError):
    pass


class BackendExecutionError(ValidationError):
    """进程超时/传输/本地环境失败，不等同于模型生成了错误几何。"""


class WorkerTimeout(BackendExecutionError):
    """工作进程超出时间预算；对已通过准入的同一资产，复查超时由调用方按该资产处理。"""


def extract_artifacts(stream, destination, config):
    """只拷贝有界普通文件；拒绝链接、设备、路径穿越，不使用extractall。"""
    destination = Path(destination)
    total = count = 0
    names = set()
    allowed = {".xml", ".json", ".obj", ".stl", ".msh", ".png", ".jpg", ".jpeg", ".webp", ".mtl", ".txt"}
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            count += 1
            require(count <= config.max_artifact_files * 2 + 1, "容器归档成员过多")
            name = member.name
            if name.startswith("./"):
                name = name[2:]
            if member.isdir():
                require(not member.issym() and ".." not in PurePosixPath(name).parts, "容器目录路径无效")
                continue
            require(member.isfile() and not member.issym() and not member.islnk(), "拒绝容器输出中的链接或特殊文件")
            require(name and name not in names and "\\" not in name and ":" not in name
                    and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                    "容器输出路径无效或重复")
            require(PurePosixPath(name).suffix.lower() in allowed, "容器输出包含不允许的文件类型")
            names.add(name); total += member.size
            require(len(names) <= config.max_artifact_files and 0 <= member.size and total <= config.max_artifact_bytes,
                    "容器输出超过文件数或大小上限")
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=65536)
    return sorted(names)


class ExecutionBackend:
    """两种执行后端共享的构建和检查编排，不提供隐式切换。"""

    # 由主进程在启动时写入；工作进程用它核对磁盘源码未被替换。
    # 未显式绑定时，后端按自身启动时看到的磁盘源码绑定一次（惰性、只算一次）。
    implementation = None

    def bind_implementation(self, identity):
        from asset_prep import implementation as binding
        binding.require_identity(identity, where=f'{type(self).__name__} 绑定')
        self.implementation = identity

    def implementation_identity(self):
        from asset_prep import implementation as binding
        if self.implementation is None:
            self.implementation = binding.implementation_digest(binding.source_manifest(self.project))
        return self.implementation

    def recovery_environment(self):
        """只观察执行前提是否变化，不使用时钟、空闲内存等易变值绕过重试预算。"""
        return {'backend':type(self).__name__,
                'executable_available':bool(self.executable and Path(self.executable).is_file()),
                'worker_available':(self.project/'src/asset_generation/worker.py').is_file(),
                'implementation':self.implementation_identity()}

    def job_identity(self):
        """工作进程作业固定携带实现身份；缺失时拒绝下发。"""
        from asset_prep import implementation as binding
        return {'implementation':binding.require_identity(self.implementation_identity(),
                                                          where=f'{type(self).__name__} 下发作业')}

    def recover_scene(self, artifact, destination, limits, contract, capability, request, input_key):
        """可信注册能力在场景副本执行；复用原进程隔离、限时及输出限制。"""
        from dataclasses import asdict, replace
        from copy import copy
        destination = Path(destination)
        destination.mkdir(parents=True)
        job = destination/'job.json'
        atomic_json(job,{'scope':'scene','limits':asdict(limits),'contract':contract,
                         'capability':capability,'request':request,'input_key':input_key,
                         **self.job_identity()})
        output = destination/'checked'
        output.mkdir()
        backend = copy(self)
        backend.config = replace(self.config,max_artifact_bytes=limits.max_artifact_bytes,
                                 max_artifact_files=limits.max_artifact_files,
                                 build_timeout_seconds=limits.build_timeout_seconds)
        backend._run('validate',[(artifact,'/input/artifact'),(job,'/input/job.json')],output)
        return read_json(output/'report.json',8000000)

    def recheck_scene(self, artifact, destination, limits, contract):
        """在原场景副本上补查，不重建资产、不修改原报告、不重新渲染。"""
        from dataclasses import asdict, replace
        from copy import copy
        destination = Path(destination)
        destination.mkdir(parents=True)
        job = destination/'job.json'
        atomic_json(job,{'scope':'scene','limits':asdict(limits),'contract':contract,'recovery':True,
                         **self.job_identity()})
        output = destination/'checked'
        output.mkdir()
        backend = copy(self)
        backend.config = replace(self.config,max_artifact_bytes=limits.max_artifact_bytes,
                                 max_artifact_files=limits.max_artifact_files,
                                 build_timeout_seconds=limits.build_timeout_seconds)
        try:
            backend._run('validate',[(artifact,'/input/artifact'),(job,'/input/job.json')],output)
            return checked_report(read_json(output/'report.json',8000000),domain='scene',mode=limits.validation_mode)
        except ValidationError as exc:
            return failed_report(exc)

    def inspect_artifact(self, artifact, destination, scope, requests):
        from .evidence import requests_valid
        requests_valid(requests,scope)
        destination = Path(destination)
        destination.mkdir(parents=True)
        job = destination/"inspection.json"
        atomic_json(job,{"scope":scope,"requests":requests,**self.job_identity()})
        output = destination/"checked"
        output.mkdir()
        try:
            self._run("inspect",[(artifact,"/input/artifact"),(job,"/input/job.json")],output)
            return read_json(output/"report.json",8000000),output/"renders"
        except ValidationError as exc:
            return {"status":"incomplete","error":str(exc),"renders":[],"simulation_steps":0},None

    def build_and_check(self, script, destination, asset_ids, profile, *, admission=False):
        destination = Path(destination)
        artifact = destination/"artifact"
        artifact.mkdir()
        phase = 'build'
        try:
            self._run("build", [(script, "/input/model.py")], artifact)
            phase = 'validate'
            job = destination/"job.json"
            atomic_json(job, {"asset_ids": asset_ids, "profile": profile, "admission":admission, **self.job_identity()})
            checked = destination/"checked"
            checked.mkdir()
            self._run("validate", [(artifact, "/input/artifact"), (job, "/input/job.json")], checked)
            return checked_report(read_json(checked/"report.json"),domain='asset',mode=profile.get('validation_mode','full')), checked/"renders"
        except ValidationError as exc:
            if phase == 'validate':
                return failed_report(exc),None
            if isinstance(exc,BackendExecutionError):
                return failed_report(exc),None
            return {"status": "fail", "checks": [{"id": "asset_build", "status": "fail", "detail": str(exc)}],
                    "renders": [], "manifest": None, "coverage": ["构建未完成，未执行后续检查"]}, None

    def build_candidate(self, script, destination, asset_ids, profile):
        report,_=self.build_and_check(script,destination,asset_ids,profile,admission=True)
        if not report.get("assembly_readiness",{}).get("loadable"):return report,None
        return self.admit_asset(Path(destination)/"artifact",Path(destination)/"candidate_views",asset_ids,profile,render=True,admitted=report)

    def admit_asset(self, artifact, destination, asset_ids, profile, *, render=False, admitted=None):
        destination=Path(destination);destination.mkdir(parents=True)
        job=destination/'job.json'
        atomic_json(job,{"asset_ids":asset_ids,"profile":profile,"admission":True,
            "candidate_render":render,**self.job_identity()})
        output=destination/'checked';output.mkdir()
        try:
            self._run('validate',[(artifact,'/input/artifact'),(job,'/input/job.json')],output)
            report=checked_report(read_json(output/'report.json',8000000),domain='asset',mode='static')
            if admitted and not report.get('assembly_readiness',{}).get('loadable'):
                return {**admitted,"render_failure":report},None
            return report,output/'renders'
        except ValidationError as exc:
            if admitted:return {**admitted,"render_failure":failed_report(exc)},None
            return failed_report(exc),None

    def check_scene(self, artifact, destination, limits, contract):
        from dataclasses import asdict, replace
        from copy import copy
        destination = Path(destination)
        destination.mkdir(parents=True)
        job = destination/'job.json'
        atomic_json(job,{'scope':'scene','limits':asdict(limits),'contract':contract,'render':True,**self.job_identity()})
        output=destination/'checked'; output.mkdir()
        backend=copy(self)
        backend.config=replace(self.config,max_artifact_bytes=limits.max_artifact_bytes,
            max_artifact_files=limits.max_artifact_files,build_timeout_seconds=limits.build_timeout_seconds)
        try:
            backend._run('validate',[(artifact,'/input/artifact'),(job,'/input/job.json')],output)
            return checked_report(read_json(output/'report.json',8000000),domain='scene',mode=limits.validation_mode),output/'renders'
        except ValidationError as exc:
            return failed_report(exc),None

    def assemble_candidate(self, plan, environment, artifacts, robot_root, destination, limits, contract):
        return self.assemble_and_check(plan,environment,artifacts,robot_root,destination,limits,contract,admission=True)

    def recheck_asset(self, artifact, destination, asset_ids, profile):
        """实现升级后重新量测已归档几何；不执行建模脚本。"""
        destination = Path(destination)
        job = destination/'job.json'
        atomic_json(job, {'asset_ids':asset_ids, 'profile':profile, **self.job_identity()})
        output = destination/'checked'
        output.mkdir()
        self._run('validate', [(artifact, '/input/artifact'), (job, '/input/job.json')], output)
        return checked_report(read_json(output/'report.json'), domain='asset',
                              mode=profile.get('validation_mode','full')), output/'renders'

    def assemble_and_check(self, plan, environment, artifacts, robot_root, destination, limits, contract, *, admission=False):
        from dataclasses import asdict, replace
        from time import monotonic
        started=monotonic()
        from copy import copy
        destination = Path(destination)
        job = destination/"scene_job.json"
        atomic_json(job, {"plan":plan, "environment":environment, "limits":asdict(limits), "contract":contract, "admission":admission,
                          **self.job_identity()})
        output = destination/"scene"
        output.mkdir()
        mounts = [(job,"/input/job.json"), (robot_root,"/input/robot")]
        mounts.extend((path,"/input/components/"+name) for name,path in artifacts.items())
        scene_backend = copy(self)
        scene_backend.config = replace(self.config,max_artifact_bytes=limits.max_artifact_bytes,
            max_artifact_files=limits.max_artifact_files,build_timeout_seconds=limits.build_timeout_seconds)
        phase = 'execute'
        try:
            scene_backend._run("scene",mounts,output)
            elapsed=monotonic()-started
            phase = 'report'
            report=checked_report(read_json(output/"report.json",8000000),domain='scene',mode=limits.validation_mode)
            report["execution_timing"]={"worker_seconds":round(elapsed,3),
                "budget_seconds":getattr(scene_backend,'last_work_budget',{}).get('seconds',limits.build_timeout_seconds),
                "base_budget_seconds":limits.build_timeout_seconds,
                "note":"总时长含输入复制与产物回收；本地日志另记实际工作时限及装配、编译、准入阶段耗时"}
            return report, output/"renders"
        except ValidationError as exc:
            if phase == 'report' or isinstance(exc,BackendExecutionError):
                report=failed_report(exc)
                report["execution_timing"]={"stage":phase,"worker_seconds":round(monotonic()-started,3),
                    "budget_seconds":getattr(scene_backend,'last_work_budget',{}).get('seconds',limits.build_timeout_seconds),
                    "base_budget_seconds":limits.build_timeout_seconds,
                    "components":len(artifacts),"instances":len(plan.get("instances",[]))}
                return report,None
            return {"status":"fail", "checks":[{"id":"scene_build", "status":"fail", "detail":str(exc),
                     "instances":[i["id"] for i in plan["instances"]]}], "renders":[],
                    "coverage":["场景构建未完成"]}, None


class DockerSandbox(ExecutionBackend):
    def __init__(self, config, project):
        self.config, self.project = config, Path(project)
        self.executable = shutil.which("docker")
        self.image_id = None

    def preflight(self):
        if not self.executable:
            raise SandboxUnavailable("未找到 Docker；可离线预检，但不能执行模型生成的 Python 程序")
        try:
            result = subprocess.run([self.executable, "info", "--format", "{{.OSType}}"], capture_output=True,
                                    timeout=15, check=True, text=True)
            require(result.stdout.strip() == "linux", "需要 Linux 容器后端")
            image = subprocess.run([self.executable, "image", "inspect", self.config.image, "--format", "{{.Id}}"],
                                   capture_output=True, timeout=15, check=True, text=True)
            self.image_id = image.stdout.strip()
            require(self.image_id.startswith("sha256:"), "无法确认构建镜像身份")
            return {"backend": "docker", "image_id": self.image_id}
        except (OSError, subprocess.SubprocessError, ValidationError) as exc:
            raise SandboxUnavailable("Docker 服务、Linux 后端或构建镜像尚未就绪；不会自动下载或退回宿主执行") from exc

    def command(self, name, mounts, mode):
        require(self.image_id is not None, "隔离后端须先预检")
        command = [self.executable, "create", "--name", name, "--pull=never", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=1g",
                   "--memory-swap=1g", "--cpus=2", "--user=65534:65534", "--log-driver=none",
                   "--tmpfs=/tmp:rw,size=128m,mode=1777", f"--tmpfs=/output:rw,size={self.config.max_artifact_bytes+64000000},mode=1777",
                   "--workdir=/", "--env=PYTHONPATH=/app", "--env=PYTHONDONTWRITEBYTECODE=1",
                   "--env=MUJOCO_GL=osmesa", "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=2",
                   "--env=MKL_NUM_THREADS=1", "--env=NUMEXPR_NUM_THREADS=1", "--env=LP_NUM_THREADS=2"]
        for source, target in mounts:
            source = str(Path(source).resolve())
            require("," not in source, "容器挂载路径暂不支持逗号")
            command.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
        require(mode in {"build", "validate", "scene", "inspect"}, "工作进程模式无效")
        identity = self.job_identity()
        # 保持容器存活至产物取回；容器退出后tmpfs会卸载。
        launcher = (f"python -B -P /app/runtime_bootstrap.py --expected-code {identity['implementation']} "
                    f"--module asset_generation.worker -- {mode} "
                    f"--implementation {identity['implementation']}; "
                    "echo $? > /tmp/worker_exit; sleep 86400")
        return command + [self.image_id, "/bin/sh", "-c", launcher]

    def _run(self, mode, mounts, output):
        from asset_prep.implementation import Guard, ImplementationDrift
        guard = Guard(self.project, self.implementation_identity())
        name = "harness-asset-" + uuid.uuid4().hex
        mounts = [(self.project/"src", "/app"), *mounts]
        try:
            subprocess.run(self.command(name, mounts, mode), check=True, capture_output=True, timeout=20)
            subprocess.run([self.executable, "start", name], check=True, capture_output=True, timeout=20)
            deadline = time.monotonic() + self.config.build_timeout_seconds
            while True:
                done = subprocess.run([self.executable, "exec", name, "cat", "/tmp/worker_exit"],
                                      capture_output=True, timeout=5, text=True)
                if done.returncode == 0:
                    break
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("isolated worker", self.config.build_timeout_seconds)
                time.sleep(0.2)
            guard.check()
            if done.stdout.strip() == '78':
                raise ImplementationDrift('容器工作进程拒绝执行不一致的实现版本')
            # 输出只存在于容器tmpfs。不把容器可写目录直接挂载到仓库。
            with tempfile.TemporaryFile(dir=output.parent) as errors:
                # Docker cp不支持tmpfs内容，按官方说明在存活容器中流式归档。
                proc = subprocess.Popen([self.executable, "exec", name, "tar", "-C", "/output", "-cf", "-", "."],
                                        stdout=subprocess.PIPE, stderr=errors)
                timer = threading.Timer(20, proc.kill)
                timer.daemon = True
                timer.start()
                try:
                    files = extract_artifacts(proc.stdout, output, self.config)
                    require(proc.wait(timeout=20) == 0, "读取容器产物失败")
                    if done.stdout.strip() != "0":
                        error_file = output/"build_error.json"
                        detail = read_json(error_file).get("detail", "构建失败") if error_file.is_file() else "工作进程失败"
                        raise ValidationError(str(detail)[:5000])
                    return files
                finally:
                    timer.cancel()
                    proc.stdout.close()
                    if proc.poll() is None:
                        proc.kill(); proc.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise BackendExecutionError("隔离工作进程超时，已请求清理本轮容器") from exc
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
            raise BackendExecutionError("隔离工作进程或产物传输失败") from exc
        finally:
            try:
                removed = subprocess.run([self.executable, "rm", "--force", name], capture_output=True, timeout=20)
                if removed.returncode != 0:
                    raise SandboxUnavailable(f"无法确认临时容器 {name} 已清理，请检查容器服务")
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxUnavailable(f"清理临时容器 {name} 失败，请检查容器服务") from exc

