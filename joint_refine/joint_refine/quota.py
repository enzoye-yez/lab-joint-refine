"""Codex (ChatGPT login) quota: remaining percent of the individual limit, via the app-server API.

The experiment stops before any model step when less than STOP_BELOW percent is left, and
`python -m joint_refine.quota --watch` stops running repair processes when it drops below that.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, now, write

STOP_BELOW = 50
STOP_FILE = ROOT / "experiments/QUOTA_STOP.json"


def read_limits(timeout=30):
    requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "joint_refine", "version": "0.1"}}},
                {"jsonrpc": "2.0", "method": "initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "account/rateLimits/read"}]
    process = subprocess.Popen(["codex", "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, cwd=str(Path.home()), text=True)
    try:
        for r in requests:
            process.stdin.write(json.dumps(r) + "\n")
        process.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            message = json.loads(line)
            if message.get("id") == 2:
                return message["result"]["rateLimits"]
    finally:
        process.kill()
    raise RuntimeError("codex app-server did not report rate limits")


def remaining_percent():
    limits = read_limits()
    individual = limits.get("individualLimit") or {}
    if "remainingPercent" in individual:
        return float(individual["remainingPercent"]), individual
    used = [w.get("usedPercent") for w in (limits.get("primary"), limits.get("secondary")) if w]
    return (100.0 - max(used) if used else 100.0), limits


class QuotaLow(RuntimeError):
    pass


def require(threshold=STOP_BELOW):
    """Raise QuotaLow (and leave the stop file) when the remaining quota is under the threshold."""
    percent, detail = remaining_percent()
    if percent < threshold:
        write(STOP_FILE, {"at": now(), "remaining_percent": percent, "detail": detail})
        raise QuotaLow(f"Codex quota {percent:.0f}% < {threshold}%: experiment stopped")
    return percent


def _stop_repairs():
    """Terminate running scene-repair processes (by PID; this process's own command line never matches)."""
    found = subprocess.run(["pgrep", "-f", "apply_robot_feedback.py"], capture_output=True, text=True).stdout.split()
    killed = []
    for pid in map(int, found):
        if pid in (os.getpid(), os.getppid()):
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
    return killed


def watch(period=300, threshold=STOP_BELOW):
    while True:
        try:
            percent, detail = remaining_percent()
        except Exception as exc:          # transient app-server failure: try again next period
            print(f"[{now()}] quota read failed: {exc}", flush=True)
            time.sleep(period)
            continue
        print(f"[{now()}] codex quota remaining {percent:.0f}%", flush=True)
        if percent < threshold:
            killed = _stop_repairs()
            write(STOP_FILE, {"at": now(), "remaining_percent": percent, "detail": detail, "stopped_pids": killed})
            print(f"[{now()}] below {threshold}%: stopped {killed}; wrote {STOP_FILE}", flush=True)
            return
        time.sleep(period)


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        print(json.dumps(remaining_percent()[1]))
