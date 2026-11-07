"""One Codex (ChatGPT login) request for a text-only agent, isolated with bwrap, fully recorded.

The home directory is hidden except the Codex login and installed runtimes; the request runs in
its own folder, which also receives the input files the agent may read.
"""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .common import write

MODEL = "gpt-6.1-sol"


def _isolation(folder):
    home = Path.home()
    return ["bwrap", "--dev-bind", "/", "/", "--tmpfs", str(home),
            "--bind", str(home / ".codex"), str(home / ".codex"),
            "--ro-bind", str(home / ".local"), str(home / ".local"),
            "--bind", str(folder), str(folder)]


def _parse(stdout):
    message, usage, error = None, {}, None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            message = item.get("text")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        elif event.get("type") in {"error", "turn.failed"}:
            error = event.get("message") or (event.get("error") or {}).get("message") or "codex reported failure"
    return message, usage, error


def ask(prompt, record_dir, *, files=None, effort="high", timeout=1800, retries=2):
    """Send `prompt` (plus readable `files` {name: text}) and return (reply text, usage). Every
    attempt's prompt, raw events and usage are kept under `record_dir`."""
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    for attempt in range(retries + 1):
        with tempfile.TemporaryDirectory(prefix="joint-llm-", dir=str(Path.home() / ".cache")) as folder:
            for name, text in (files or {}).items():
                (Path(folder) / name).write_text(text, encoding="utf-8")
                if attempt == 0:
                    (record_dir / "inputs").mkdir(exist_ok=True)
                    (record_dir / "inputs" / name).write_text(text, encoding="utf-8")
            command = _isolation(folder) + [shutil.which("codex") or "codex", "exec", "--model", MODEL,
                                            "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                                            "--ignore-user-config", "--json", "--cd", folder,
                                            "-c", f'model_reasoning_effort="{effort}"', "-"]
            started = time.time()
            try:
                done = subprocess.run(command, input=prompt.encode(), capture_output=True, timeout=timeout)
                stdout, code = done.stdout.decode("utf-8", "replace"), done.returncode
            except subprocess.TimeoutExpired:
                stdout, code = "", -1
            elapsed = time.time() - started
        (record_dir / f"events_{attempt}.jsonl").write_text(stdout, encoding="utf-8")
        message, usage, error = _parse(stdout)
        write(record_dir / f"usage_{attempt}.json", {"model": MODEL, "usage": usage, "exit": code,
                                                     "error": error, "seconds": round(elapsed, 1)})
        if code == 0 and message and not error:
            (record_dir / "reply.md").write_text(message, encoding="utf-8")
            return message, {"model": MODEL, **usage, "seconds": round(elapsed, 1), "attempts": attempt + 1}
    raise RuntimeError(f"codex request failed after {retries + 1} attempts: {error or code}")


def json_reply(text):
    """The JSON object in a reply (tolerates a fenced block)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])
