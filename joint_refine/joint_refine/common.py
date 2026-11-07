"""Paths, the experiment ledger and resource-guarded subprocess launches shared by every agent."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
TRAJ = BASE / "np002_traj"
SCENE_GEN = BASE / "np002_agent_joint"      # worktree of np002_agent_v3, branch joint_refine
GUARDED = TRAJ / "scripts/guarded.sh"
CORES = os.cpu_count() or 1


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_identity(directory):
    def run(*args):
        return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True).stdout.strip()
    return {"path": str(directory), "commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


class Ledger:
    """Append-only event log of one experiment (one JSON line per step start/end)."""

    def __init__(self, directory):
        self.path = Path(directory) / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, **fields):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": now(), **fields}, ensure_ascii=False, default=float) + "\n")

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_per_core():
    return os.getloadavg()[0] / CORES


def wait_for_load(limit=1.0, log=print, poll=30):
    """Do not start a heavy step while the machine is already busy with other jobs."""
    waited = 0
    while load_per_core() >= limit:
        if waited % 600 == 0:
            log(f"load {os.getloadavg()[0]:.0f} on {CORES} cores; waiting before the next heavy step")
        time.sleep(poll)
        waited += poll
    return waited


def guarded(command, *, cpus, mem, cwd, log_path, env=None):
    """Run a command inside the guarded cgroup scope (CPU quota, hard memory cap, no swap)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full = [str(GUARDED), "--cpus", str(cpus), "--mem", mem, "--", *map(str, command)]
    started = time.time()
    with log_path.open("ab") as sink:
        code = subprocess.run(full, cwd=str(cwd), stdout=sink, stderr=subprocess.STDOUT,
                              env={**os.environ, **(env or {})}).returncode
    return code, time.time() - started
