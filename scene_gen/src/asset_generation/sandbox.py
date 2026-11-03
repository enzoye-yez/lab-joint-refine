"""Container execution boundary. Without an installed isolation environment, generated code is refused; there is no host fallback."""

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
    """Process timeout/transport/local environment failure; not the same as the model generating wrong geometry."""


class WorkerTimeout(BackendExecutionError):
    """The worker exceeded its time budget; for the same asset that already passed admission, the caller handles a recheck timeout as that asset's problem."""


def extract_artifacts(stream, destination, config):
    """Copy only bounded regular files; reject links, devices and path traversal; never use extractall."""
    destination = Path(destination)
    total = count = 0
    names = set()
    allowed = {".xml", ".json", ".obj", ".stl", ".msh", ".png", ".jpg", ".jpeg", ".webp", ".mtl", ".txt"}
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            count += 1
            require(count <= config.max_artifact_files * 2 + 1, "Too many members in the container archive")
            name = member.name
            if name.startswith("./"):
                name = name[2:]
            if member.isdir():
                require(not member.issym() and ".." not in PurePosixPath(name).parts, "Invalid container directory path")
                continue
            require(member.isfile() and not member.issym() and not member.islnk(), "Links or special files in container output rejected")
            require(name and name not in names and "\\" not in name and ":" not in name
                    and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                    "Container output path invalid or duplicated")
            require(PurePosixPath(name).suffix.lower() in allowed, "Container output contains a disallowed file type")
            names.add(name); total += member.size
            require(len(names) <= config.max_artifact_files and 0 <= member.size and total <= config.max_artifact_bytes,
                    "Container output exceeds the file count or size limit")
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=65536)
    return sorted(names)


class ExecutionBackend:
    """Build and check orchestration shared by both execution backends; no implicit switching."""

    # Written by the main process at startup; the worker uses it to verify that the source on disk was not replaced.
    # Without explicit binding, the backend binds once to the on-disk source it sees at its own startup (lazily, computed only once).
    implementation = None

    def bind_implementation(self, identity):
        from asset_prep import implementation as binding
        binding.require_identity(identity, where=f'{type(self).__name__} binding')
        self.implementation = identity

    def implementation_identity(self):
        from asset_prep import implementation as binding
        if self.implementation is None:
            self.implementation = binding.implementation_digest(binding.source_manifest(self.project))
        return self.implementation

    def recovery_environment(self):
        """Only observe whether the execution preconditions changed; never use volatile values such as the clock or free memory to bypass the retry budget."""
        return {'backend':type(self).__name__,
                'executable_available':bool(self.executable and Path(self.executable).is_file()),
                'worker_available':(self.project/'src/asset_generation/worker.py').is_file(),
                'implementation':self.implementation_identity()}

    def job_identity(self):
        """Worker jobs always carry the implementation identity; refuse to dispatch when it is missing."""
        from asset_prep import implementation as binding
        return {'implementation':binding.require_identity(self.implementation_identity(),
                                                          where=f'{type(self).__name__} job dispatch')}

    def recover_scene(self, artifact, destination, limits, contract, capability, request, input_key):
        """Trusted registered capabilities run on the scene copy, reusing the original process isolation, time limits and output limits."""
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
        """Recheck on the original scene copy without rebuilding assets, modifying the original report or re-rendering."""
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
                    "renders": [], "manifest": None, "coverage": ["Build not finished; subsequent checks not run"]}, None

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
        """After an implementation upgrade, re-measure the archived geometry; never execute the modelling script."""
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
                "note":"Total duration includes input copying and artifact retrieval; the local log separately records the actual worker time limit and the durations of the assembly, compile and admission stages"}
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
                    "coverage":["Scene build not finished"]}, None


class DockerSandbox(ExecutionBackend):
    def __init__(self, config, project):
        self.config, self.project = config, Path(project)
        self.executable = shutil.which("docker")
        self.image_id = None

    def preflight(self):
        if not self.executable:
            raise SandboxUnavailable("Docker not found; offline preflight is possible, but Python programs generated by the model cannot be executed")
        try:
            result = subprocess.run([self.executable, "info", "--format", "{{.OSType}}"], capture_output=True,
                                    timeout=15, check=True, text=True)
            require(result.stdout.strip() == "linux", "Requires a Linux container backend")
            image = subprocess.run([self.executable, "image", "inspect", self.config.image, "--format", "{{.Id}}"],
                                   capture_output=True, timeout=15, check=True, text=True)
            self.image_id = image.stdout.strip()
            require(self.image_id.startswith("sha256:"), "Cannot confirm the build image identity")
            return {"backend": "docker", "image_id": self.image_id}
        except (OSError, subprocess.SubprocessError, ValidationError) as exc:
            raise SandboxUnavailable("Docker service, Linux backend or build image not ready yet; will not download automatically or fall back to host execution") from exc

    def command(self, name, mounts, mode):
        require(self.image_id is not None, "The isolated backend must be preflighted first")
        command = [self.executable, "create", "--name", name, "--pull=never", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=1g",
                   "--memory-swap=1g", "--cpus=2", "--user=65534:65534", "--log-driver=none",
                   "--tmpfs=/tmp:rw,size=128m,mode=1777", f"--tmpfs=/output:rw,size={self.config.max_artifact_bytes+64000000},mode=1777",
                   "--workdir=/", "--env=PYTHONPATH=/app", "--env=PYTHONDONTWRITEBYTECODE=1",
                   "--env=MUJOCO_GL=osmesa", "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=2",
                   "--env=MKL_NUM_THREADS=1", "--env=NUMEXPR_NUM_THREADS=1", "--env=LP_NUM_THREADS=2"]
        for source, target in mounts:
            source = str(Path(source).resolve())
            require("," not in source, "Container mount paths do not support commas yet")
            command.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
        require(mode in {"build", "validate", "scene", "inspect"}, "Invalid worker mode")
        identity = self.job_identity()
        # Keep the container alive until artifacts are retrieved; the tmpfs is unmounted once the container exits.
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
                raise ImplementationDrift('Container worker refused to run a mismatched implementation version')
            # Output exists only in the container tmpfs. Never mount a container-writable directory directly onto the repository.
            with tempfile.TemporaryFile(dir=output.parent) as errors:
                # Docker cp does not support tmpfs contents, so following the official guidance we stream an archive from the live container.
                proc = subprocess.Popen([self.executable, "exec", name, "tar", "-C", "/output", "-cf", "-", "."],
                                        stdout=subprocess.PIPE, stderr=errors)
                timer = threading.Timer(20, proc.kill)
                timer.daemon = True
                timer.start()
                try:
                    files = extract_artifacts(proc.stdout, output, self.config)
                    require(proc.wait(timeout=20) == 0, "Failed to read container artifacts")
                    if done.stdout.strip() != "0":
                        error_file = output/"build_error.json"
                        detail = read_json(error_file).get("detail", "build failed") if error_file.is_file() else "worker failed"
                        raise ValidationError(str(detail)[:5000])
                    return files
                finally:
                    timer.cancel()
                    proc.stdout.close()
                    if proc.poll() is None:
                        proc.kill(); proc.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise BackendExecutionError("Isolated worker timed out; cleanup of this round's container requested") from exc
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
            raise BackendExecutionError("Isolated worker or artifact transfer failed") from exc
        finally:
            try:
                removed = subprocess.run([self.executable, "rm", "--force", name], capture_output=True, timeout=20)
                if removed.returncode != 0:
                    raise SandboxUnavailable(f"Cannot confirm temporary container {name} was cleaned up; check the container service")
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxUnavailable(f"Failed to clean up temporary container {name}; check the container service") from exc

