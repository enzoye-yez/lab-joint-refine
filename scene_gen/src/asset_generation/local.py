"""Explicit local execution: a separate process with bounded artifact checks; not an operating-system security sandbox."""

import importlib.metadata
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from xml.etree.ElementTree import ParseError
from defusedxml.common import DefusedXmlException

from asset_prep.errors import ValidationError
from .contracts import require
from .sandbox import ExecutionBackend, SandboxUnavailable, BackendExecutionError, WorkerTimeout
from .process_tree import ProcessTree

RENDER_ENVIRONMENT = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_PLATFORM", "__EGL_VENDOR_LIBRARY_FILENAMES",
                      "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "DISPLAY")


def work_budget(mode, source, base_seconds):
    """Bound local scene work by staged geometry, independently of paid-call budgets."""
    from defusedxml import ElementTree as ET
    from asset_prep.storage import safe_path
    from .contracts import read_json
    source = Path(source)
    if mode not in {"scene", "validate", "inspect"}:
        return {"seconds":base_seconds, "estimated_geoms":None}
    job = read_json(source/"job.json", 32000000)
    if mode != "scene" and job.get("scope") != "scene":
        # Each declared configuration of an asset is loaded and checked on its own.
        if mode == "validate" and (source/"artifact"/"interfaces.json").is_file():
            from .configurations import catalogue
            configurations = len(catalogue(source/"artifact"))
            return {"seconds":base_seconds*min(8, configurations), "estimated_geoms":None, "configurations":configurations}
        return {"seconds":base_seconds, "estimated_geoms":None}
    paths = []
    if mode == "scene":
        from .configurations import selection
        from scene_generation.robot_contract import model_file
        paths.append(safe_path(source/"robot",model_file(job["plan"],job["environment"])))
        for item in job["plan"]["instances"]:
            root = safe_path(source/"components", item["job"])
            _, filename, _ = selection(root,item.get("configuration"))
            paths.append(safe_path(root,filename))
    else:
        paths.append(source/"artifact"/"model.xml")
    counts = {}
    for path in paths:
        if path not in counts:
            require(path.stat().st_size <= 32000000, "Scene budget estimation input file too large")
            counts[path] = sum(1 for _ in ET.parse(path).iter("geom"))
    count = sum(counts[p] for p in paths)
    # Estimate includes repeated instances; the cap keeps process cancellation bounded.
    # Assembly compiles once to initialize/export and once to admit the saved XML.
    passes = 2 if mode == "scene" else 1
    return {"seconds":base_seconds*max(1.0,min(8.0,passes*count/10000)), "estimated_geoms":count}


def artifact_files(root, config):
    """Check the complete output tree before copying; never follow symlinks or Windows reparse points."""
    root = Path(root)
    allowed = {".xml", ".json", ".obj", ".stl", ".msh", ".png", ".jpg", ".jpeg", ".webp", ".mtl", ".txt"}
    files, pending = [], [root]
    total = entries = 0
    while pending:
        path = pending.pop()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0), "Local artifacts must not contain links or reparse points")
        if stat.S_ISDIR(info.st_mode):
            for child in path.iterdir():
                entries += 1
                require(entries <= config.max_artifact_files * 2 + 1, "Too many entries in the local artifact directory")
                pending.append(child)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Local artifacts must not contain special files or hard links")
            require(path.suffix.lower() in allowed, "Local artifacts contain a disallowed file type")
            files.append(path.relative_to(root))
            total += info.st_size
            require(len(files) <= config.max_artifact_files and total <= config.max_artifact_bytes,
                    "Local artifacts exceed the file count or size limit")
    return sorted(files)


class LocalBackend(ExecutionBackend):
    def __init__(self, config, project):
        self.config, self.project = config, Path(project).resolve()
        self.executable = sys.executable

    def preflight(self):
        try:
            versions = {name: importlib.metadata.version(name)
                        for name in ("mujoco", "numpy", "Pillow", "trimesh", "scipy", "defusedxml")}
        except importlib.metadata.PackageNotFoundError as exc:
            raise SandboxUnavailable("Local generation dependencies missing; install the dependencies in requirements-generation.txt") from exc
        require((self.project/"src/asset_generation/worker.py").is_file(), "Local worker entry point missing")
        return {"backend": "local", "python": self.executable, "python_version": sys.version.split()[0],
                "packages": versions, "os_isolation": False,
                "note": "Separate host process; does not isolate host files, network or system permissions"}

    def worker_environment(self, temp):
        # Do not inherit model keys; this does not stop the program from actively reading host files or accessing the network.
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
        # Linux hosts without a display need an explicit rendering backend (e.g. Mesa software EGL); pass through only rendering-related variables.
        env.update({key: os.environ[key] for key in RENDER_ENVIRONMENT if key in os.environ})
        system = Path(os.environ.get("SystemRoot", "C:/Windows"))/"System32"
        env.update(PATH=os.pathsep.join((str(Path(self.executable).parent), str(system)))
                   if os.name == "nt" else os.defpath,
                   PYTHONPATH=str(self.project/"src"), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
                   TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", LP_NUM_THREADS="2")
        return env

    def _run(self, mode, mounts, output):
        from asset_prep.implementation import Guard, ImplementationDrift
        guard = Guard(self.project, self.implementation_identity())
        require(mode in {"build", "validate", "scene", "inspect"}, "Invalid worker mode")
        output = Path(output)
        log_path = output.parent/("local-" + mode + ".log")
        from asset_prep.storage import atomic_json
        receipt_path = output.parent/("local-"+mode+"-exit.json")
        receipt = {"mode":mode,"status":"preparing","started_at":time.time(),"pid":None,"exit_code":None}
        budget = {"seconds":self.config.build_timeout_seconds, "estimated_geoms":None}
        try:
            # Copy only this round's inputs; the temporary directory, logs and final artifacts all live on the same drive as the run directory.
            with tempfile.TemporaryDirectory(prefix="local-worker-", dir=output.parent) as directory:
                workspace = Path(directory).resolve()
                source, built, temp = workspace/"input", workspace/"output", workspace/"tmp"
                for folder in (source, built, temp):
                    folder.mkdir()
                for path, target in mounts:
                    require(target.startswith("/input/") and ".." not in Path(target).parts,
                            "Invalid local worker input path")
                    staged = source/target[len("/input/"):]
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    if Path(path).is_dir():
                        shutil.copytree(path, staged)
                    else:
                        shutil.copyfile(path, staged)
                try:
                    budget = work_budget(mode, source, self.config.build_timeout_seconds)
                except (OSError, ValueError, KeyError, TypeError, ParseError, DefusedXmlException, ValidationError) as exc:
                    # Estimation is not admission: malformed inputs still reach the
                    # worker's authoritative validator with the original time bound.
                    budget["estimation_error"] = type(exc).__name__
                self.last_work_budget = budget
                # Register the process tree before allowing the worker to start, closing the window in which children could spawn before registration.
                gate = "import sys,runpy; token=sys.stdin.buffer.read(1); sys.stdin.close(); " \
                       "sys.argv=sys.argv[1:]; " \
                       "runpy.run_path(sys.argv[0],run_name='__main__') if token==b'1' else sys.exit(1)"
                identity = self.job_identity()
                command = [self.executable, "-B", "-P", "-c", gate, str(self.project/'src/runtime_bootstrap.py'),
                           '--expected-code', identity['implementation'], '--module', 'asset_generation.worker', '--', mode,
                           "--input", str(source), "--output", str(built),
                           "--implementation", identity['implementation']]
                if mode in {"scene","validate"}:
                    checkpoints = self.project/".cache/scene-checkpoints"
                    command.extend(["--checkpoints",str(checkpoints)])
                    receipt["checkpoints"] = str(checkpoints)
                tree = ProcessTree()
                try:
                    process = subprocess.Popen(command, cwd=workspace, env=self.worker_environment(temp),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        start_new_session=os.name != "nt")
                except BaseException:
                    tree.close()
                    raise
                # Keep draining the pipe and keep at most 1 MiB in total from the head and tail, so verbose output cannot block or grow the log without bound.
                log_errors = []
                def collect():
                    cap = 1024 * 1024
                    try:
                        with log_path.open("wb") as log:
                            import json
                            head = bytearray((json.dumps({"event":"work_budget", **budget})+"\n").encode())
                            tail = bytearray()
                            log.write(head);log.flush()
                            while chunk := process.stdout.read(65536):
                                take = min(max(0,65536-len(head)),len(chunk))
                                head.extend(chunk[:take]);tail.extend(chunk[take:])
                                del tail[:max(0,len(tail)-(cap-len(head)))]
                                log.seek(0);log.write(head);log.write(tail);log.truncate();log.flush()
                    except (OSError, ValueError) as exc:
                        log_errors.append(type(exc).__name__)
                reader = threading.Thread(target=collect, daemon=True)
                started = False
                try:
                    receipt.update(status="running",pid=process.pid,budget=budget)
                    atomic_json(receipt_path,receipt)
                    tree.attach(process)
                    reader.start()
                    started = True
                    process.stdin.write(b"1")
                    process.stdin.close()
                    deadline = time.monotonic() + budget["seconds"]
                    while True:
                        if getattr(self, 'cancel_guard', None):
                            self.cancel_guard()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, budget['seconds'])
                        try:
                            code = process.wait(timeout=min(1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    receipt.update(exit_code=code,status="exited" if code==0 else "process_failed")
                finally:
                    original = sys.exception()
                    try:
                        tree.stop(process)
                    except Exception as exc:
                        if original is None:
                            raise
                        original.add_note("Worker cleanup also failed: " + type(exc).__name__)
                    finally:
                        process.stdin.close()
                        if started:
                            reader.join(timeout=5)
                        # An unbuffered pipe does not hold the BufferedReader read lock, which keeps the close path bounded.
                        process.stdout.close()
                        if started and reader.is_alive():
                            reader.join(timeout=1)
                require(not reader.is_alive(), "Log pipe not closed after the local worker exited; check for spawned processes")
                require(not log_errors, "Local log reading failed: " + ",".join(log_errors))
                guard.check()
                if code == 78:
                    raise ImplementationDrift('Worker refused to run a mismatched implementation version; log: '+str(log_path))
                if code != 0:
                    detail = log_path.read_bytes()[-5000:].decode("utf-8", errors="replace")
                    message=f"Local {mode} process exit code {code}: {detail}"
                    if mode != "build" or not (built/"build_error.json").is_file():
                        raise BackendExecutionError(message)
                    raise ValidationError(message)
                files = artifact_files(built, self.config)
                for relative in files:
                    destination = output/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with (built/relative).open("rb") as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                receipt["status"]="completed"
                return [path.as_posix() for path in files]
        except subprocess.TimeoutExpired as exc:
            receipt["status"]="timeout"
            raise WorkerTimeout(
                f"Local worker {mode} timed out: not finished after {budget['seconds']} s; termination of the process tree requested; "
                f"log kept at {log_path} (the assembly log has stage start/end and durations; a stage with a start but no end may have been interrupted by the timeout)") from exc
        except OSError as exc:
            receipt["status"]="io_error"
            raise BackendExecutionError("Local worker or artifact copy failed: " + str(exc)) from exc
        finally:
            if sys.exception() is not None and receipt["status"] in {"preparing","running","exited"}:
                receipt["status"]="failed"
            receipt["finished_at"]=time.time()
            failure=sys.exception()
            try:atomic_json(receipt_path,receipt)
            except OSError as exc:
                if failure is not None:failure.add_note("Failed to save the exit receipt: "+type(exc).__name__)
                else:raise
