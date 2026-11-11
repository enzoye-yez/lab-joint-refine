"""Resource guard for long CPU runs on the shared devbox: never let a run take the machine down.

Three layers, loose enough not to disturb normal runs:
  1. scripts/guarded.sh puts the run in its own cgroup scope (kernel-enforced memory cap, no swap,
     CPU quota, low priority): if everything else fails, the kernel kills only this scope.
  2. Guard (a watchdog thread in the parent) samples the run's process-tree memory, the machine's
     available memory and the load every few seconds; on a trip it kills its own workers first,
     writes guard.json with the reason and exits.
  3. worker_check() runs in each worker before every task: a worker above its memory cap stops
     taking tasks (the pool also recycles workers after a fixed number of tasks against leaks).
"""
import json
import os
import signal
import threading
import time
from pathlib import Path

GB = 1024 ** 3


def meminfo():
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        rows[key] = int(value.split()[0]) * 1024
    return rows


def cgroup_limit():
    """Memory limit of this process's cgroup (bytes), or None if unlimited."""
    try:
        path = Path("/proc/self/cgroup").read_text().split("::", 1)[1].strip()
        value = (Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.max").read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, IndexError, ValueError):
        return None


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def descendants(root):
    """All live descendant pids of `root` (from /proc, no psutil needed)."""
    children = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    out, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def default_limits(memory_fraction=0.25, available_floor_fraction=0.10, load_factor=1.5):
    total = meminfo()["MemTotal"]
    limit = cgroup_limit()
    budget = memory_fraction * total
    if limit:
        budget = min(budget, 0.8 * limit)
    return {"tree_bytes": int(budget), "available_floor_bytes": int(available_floor_fraction * total),
            "load_limit": load_factor * (os.cpu_count() or 1), "worker_bytes": 8 * GB}


def safe_workers(requested, per_worker_bytes=1.5 * GB, limits=None, cpu_fraction=0.5):
    """Workers that fit in half the CPUs and in the memory budget."""
    limits = limits or default_limits()
    by_cpu = max(1, int((os.cpu_count() or 1) * cpu_fraction))
    by_memory = max(1, int(limits["tree_bytes"] / per_worker_bytes))
    return max(1, min(requested, by_cpu, by_memory))


class Guard:
    def __init__(self, out_dir, limits=None, interval=2.0, load_window=60.0):
        self.out_dir, self.limits = Path(out_dir), limits or default_limits()
        self.interval, self.load_window = interval, load_window
        self.on_trip = None
        self.tripped = None
        self._high_load_since = None
        self._stop = threading.Event()
        self.peak = 0

    def sample(self):
        pids = [os.getpid()] + descendants(os.getpid())
        tree = sum(rss(p) for p in pids)
        self.peak = max(self.peak, tree)
        return {"tree_bytes": tree, "available_bytes": meminfo()["MemAvailable"], "load1": os.getloadavg()[0],
                "processes": len(pids)}

    def check(self, now=None):
        """The reason to stop, or None."""
        s, lim = self.sample(), self.limits
        if s["tree_bytes"] > lim["tree_bytes"]:
            return f"run memory {s['tree_bytes'] / GB:.1f} GB above its budget {lim['tree_bytes'] / GB:.1f} GB", s
        if s["available_bytes"] < lim["available_floor_bytes"]:
            return f"machine available memory {s['available_bytes'] / GB:.1f} GB below {lim['available_floor_bytes'] / GB:.1f} GB", s
        now = time.time() if now is None else now
        if s["load1"] > lim["load_limit"]:
            self._high_load_since = self._high_load_since or now
            if now - self._high_load_since > self.load_window:
                return f"load {s['load1']:.0f} above {lim['load_limit']:.0f} for {self.load_window:.0f} s", s
        else:
            self._high_load_since = None
        return None, s

    def _loop(self):
        while not self._stop.wait(self.interval):
            reason, s = self.check()
            if reason:
                self.trip(reason, s)
                return

    def trip(self, reason, sample):
        """Kill this run's own workers first, record why, then exit the run."""
        self.tripped = reason
        for pid in descendants(os.getpid()):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        record = {"reason": reason, "sample": sample, "limits": self.limits, "peak_tree_bytes": self.peak,
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "guard.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError:
            pass
        print(f"GUARD TRIPPED: {reason}; workers killed", flush=True)
        if self.on_trip:
            try:
                self.on_trip()
            except Exception:
                pass
        os._exit(3)

    def start(self, on_trip=None):
        self.on_trip = on_trip
        threading.Thread(target=self._loop, name="resource-guard", daemon=True).start()
        return self

    def stop(self):
        self._stop.set()


def worker_check(limit_bytes):
    """In a worker, before a task: refuse to go on above the per-worker memory cap."""
    used = rss(os.getpid())
    if used > limit_bytes:
        raise MemoryError(f"worker {os.getpid()} uses {used / GB:.1f} GB (cap {limit_bytes / GB:.1f} GB)")
