"""Call the model through the locally logged-in Codex CLI; returns the same response shape as chat completions.

Uses the ChatGPT login session and reads no API key. Each request runs one
`codex exec` in its own temporary directory (read-only sandbox, no persisted session); images are passed as attachments and text via stdin.

Coding agent mode (agent=True) is only for author requests: codex edits model.py in a writable work directory,
may run the local ./check self-check repeatedly, and at the end returns the final script under the original author contract (model_py on the first round, whole-file replacement
on repair rounds). The framework still independently rebuilds, checks and reviews; self-check results are not grounds for passing.
"""

import base64
import binascii
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from asset_prep.errors import ProviderError

REPLY_RULES = ("You are called as a plain model endpoint, not as a coding agent. "
               "Do not run commands, read files or use tools. Answer only with the single JSON "
               "object the instructions below ask for, without Markdown fences or extra text.")
FILE_RULES = ("You are called as a model endpoint. Long inputs are saved as files in the current directory "
              "and named in the prompt; read them with read-only shell commands (cat, grep, jq, python). Do not modify "
              "files or use other tools. Answer only with the single JSON object the instructions below ask for, "
              "without Markdown fences or extra text.")
INLINE_TEXT_BYTES = 60000  # longer text parts go to files a whole-context request reads on demand
IMAGE_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}

AGENT_TASK = """# Asset modelling task (coding agent mode)

You work in a work directory that belongs to this asset only. Goal: write a `model.py` that satisfies instructions.md and context.json.

- `instructions.md`: the modelling contract (mandatory; identical to normal mode).
- `context.json`: materials, task, dependencies, previous-round feedback (repair_hint in feedback / repair_tasks gives local-frame directions and distances).
- `references/`: reference images (also given to you as image attachments).
- `model.py`: the current script{current_note}.
- `./check`: builds the current model.py, runs the single-asset checks and prints a JSON summary (failing, incomplete, renders).

How to work:
1. Read the contract and materials, then edit `model.py` (edit only this file; do not create other deliverable files).
2. Run `./check`. Entries in `failing` must be fixed; an `incomplete` entry may remain if it is only due to unreliable measurement or a missing probe.
3. When needed, look at the images in `renders` to verify the shape.
4. Repeat 2-3, at most {max_checks} checks; once nothing is failing, do one final overall review and stop.
5. Do not modify check, selfcheck.json, instructions.md, context.json or references; do not access the network.

When finished reply with only one line of JSON: {{"done": true, "checks_run": <number of checks>, "remaining_failing": [<ids of checks still failing>], "summary": "<one sentence on what you did>"}}.
The framework will independently rebuild, check and review your model.py; passing the self-check is not acceptance.
"""

SKILLS = Path(__file__).resolve().parent / "skills"
SKILLS_NOTE = """
Skills: `skills/*/SKILL.md` hold modelling recipes for this framework's checks (stability, material integrity,
buttons, connections, lids, slides, hoses). Read them before the first edit and follow them where they apply;
instructions.md stays authoritative.
"""

CHECK_SCRIPT = """#!/usr/bin/env bash
# Local self-check: build model.py and run the single-asset checks (CPU rendering, no GPU).
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export PYTHONPATH={src} TMPDIR="$(pwd)/tmp" TEMP="$(pwd)/tmp" TMP="$(pwd)/tmp"
mkdir -p "$TMPDIR"
exec {python} -B -m asset_generation.selfcheck "$(pwd)" 2> >(grep -v '^libEGL' >&2)
"""


def author_context(payload):
    """The author's structured context, or None when the request is paged or not an author request."""
    try:
        context = json.loads(payload["messages"][1]["content"][0]["text"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(context, dict) or any(k in context for k in ("_context", "context_page", "context_pages", "context_delivery")):
        return None
    if "profile" not in context or "current_model_py" not in context:
        return None
    return context


def prompt_and_images(messages, folder, inline_limit=None):
    """Join system/user messages into one prompt and write data URL images as sequentially numbered attachments.

    With inline_limit, a longer text part is written verbatim to input_<n>.txt and the prompt names the file.
    """
    sections, images, files = [REPLY_RULES], [], 0
    for message in messages:
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        texts = []
        for part in parts:
            if part["type"] == "text":
                if inline_limit is not None and len(part["text"].encode("utf-8")) > inline_limit:
                    files += 1
                    (Path(folder) / f"input_{files}.txt").write_text(part["text"], encoding="utf-8")
                    texts.append(f"[full text is in file input_{files}.txt in the current directory ({len(part['text'])} characters); read it as needed]")
                else:
                    texts.append(part["text"])
            elif part["type"] == "image_url":
                header, _, data = part["image_url"]["url"].partition(",")
                mime = header.removeprefix("data:").split(";")[0]
                if not header.startswith("data:") or ";base64" not in header or mime not in IMAGE_SUFFIX:
                    raise ProviderError("Codex transport only supports base64 data URL images", uncertain=False,
                                        diagnostics={"failure": "unsupported_image"})
                path = Path(folder) / f"image_{len(images) + 1:03d}{IMAGE_SUFFIX[mime]}"
                try:
                    path.write_bytes(base64.b64decode(data, validate=True))
                except binascii.Error:
                    raise ProviderError("Image data URL cannot be decoded", uncertain=False,
                                        diagnostics={"failure": "unsupported_image"}) from None
                images.append(path)
                texts.append(f"[attached image #{len(images)}]")
        sections.append(f"## {message['role']}\n" + "\n".join(texts))
    if files:
        sections[0] = FILE_RULES
    return "\n\n".join(sections), images


def strip_fence(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_events(stdout):
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


PROJECT = Path(__file__).resolve().parents[2]
# Evaluation material and everything written about it; no model request may read these.
HIDDEN = ("benchmark/NP-002/rubric.json", "benchmark/_evaluation", "benchmark/_dev", "docs", "reports", "scripts",
          "AGENTS.md")


def isolation(writable):
    """bwrap prefix for a codex process: the home directory holds only the codex login, the
    installed runtimes and this project (read-only apart from the request folder), and the
    project's evaluation material is hidden. Codex's own sandbox restricts writes, not reads,
    so this is what keeps the rubric and other project copies out of reach."""
    if os.name == "nt":
        return []
    if shutil.which("bwrap") is None:
        raise ProviderError("bwrap is required to keep evaluation files out of model requests", uncertain=False,
                            diagnostics={"failure": "isolation_missing"})
    home = Path.home()
    prefix = ["bwrap", "--dev-bind", "/", "/", "--tmpfs", str(home),
              "--bind", str(home/".codex"), str(home/".codex"), "--ro-bind", str(home/".local"), str(home/".local"),
              "--ro-bind", str(PROJECT), str(PROJECT)]
    # The project's virtual environment may be a link into another project copy; only it is shown.
    venv = Path(sys.prefix).resolve()
    if venv.is_relative_to(home) and not venv.is_relative_to(PROJECT):
        prefix += ["--ro-bind", str(venv), str(venv)]
    for name in HIDDEN:
        path = PROJECT/name
        if path.is_dir():
            prefix += ["--tmpfs", str(path)]
        elif path.exists():
            prefix += ["--ro-bind", "/dev/null", str(path)]
    # The process temp directory stays writable where it lives under the home directory.
    temp = Path(tempfile.gettempdir()).resolve()
    if temp.is_relative_to(home):
        prefix += ["--bind", str(temp), str(temp)]
    folder = Path(writable).resolve()
    return prefix + ["--bind", str(folder), str(folder), "--chdir", str(folder), "--die-with-parent", "--"]


def is_author_phase(phase):
    return phase == "author" or (phase or "").startswith("asset:")


class CodexTransport:
    """Same post interface as GenerationTransport; url and key are unused."""

    def __init__(self, executable="codex", reasoning_effort="high", *, agent=False, workspace_root=None, max_checks=12,
                 skills=False):
        self.executable, self.reasoning_effort = executable, reasoning_effort
        self.agent, self.workspace_root, self.max_checks, self.skills = agent, workspace_root, max_checks, skills
        self.check_cancel = lambda: None
        self.phase = None  # GenerationModel sets the request phase before each post.

    def whole_context(self, phase):
        """In agent mode no request is paged: authors get context.json in their workspace and other
        requests read over-long text from files in their request folder."""
        return bool(self.agent)

    def command(self, model, folder, images, sandbox="read-only"):
        command = isolation(folder) + [self.executable, "exec", "--model", model, "--sandbox", sandbox, "--ephemeral",
                   "--skip-git-repo-check", "--ignore-user-config", "--json",
                   "--cd", str(folder), "-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        for image in images:
            command += ["--image", str(image)]
        return command + ["-"]

    def post(self, url, payload, key, timeout):
        if shutil.which(self.executable) is None:
            raise ProviderError("codex command not found; install and log in to the Codex CLI first", uncertain=False,
                                diagnostics={"failure": "codex_missing"})
        context = author_context(payload) if self.agent and is_author_phase(self.phase) else None
        if context is not None:
            return self.author_session(payload, context, timeout)
        with tempfile.TemporaryDirectory(prefix="codex-request-") as folder:
            prompt, images = prompt_and_images(payload["messages"], folder,
                                               INLINE_TEXT_BYTES if self.agent else None)
            stdout, stderr, code, elapsed = self.run(self.command(payload["model"], folder, images), folder, prompt, timeout)
        message, usage = self.reply(stdout, stderr, code)
        return self.response(payload, strip_fence(message), usage, elapsed)

    def author_session(self, payload, context, timeout):
        """One coding-agent session in a kept workspace; returns the final script under the author contract."""
        from datetime import datetime, timezone
        previous = context.get("current_model_py")
        asset_ids = (context.get("job") or {}).get("asset_ids") or (context.get("input") or {}).get("asset_ids")
        if not asset_ids:
            raise ProviderError("Author context is missing the asset id; cannot set up the self-check", uncertain=False,
                                diagnostics={"failure": "agent_context"})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        workspace = (Path(self.workspace_root or tempfile.gettempdir()) / "codex_agent_sessions"
                     / f"{stamp}_{(self.phase or 'author').replace(':', '_')}")
        (workspace / "references").mkdir(parents=True)
        _, images = prompt_and_images(payload["messages"], workspace / "references")
        (workspace / "instructions.md").write_text(payload["messages"][0]["content"], encoding="utf-8")
        (workspace / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=1), encoding="utf-8")
        (workspace / "model.py").write_text(previous or "", encoding="utf-8")
        (workspace / "selfcheck.json").write_text(json.dumps({"asset_ids": asset_ids, "profile": context["profile"],
                                                             "role": (context.get("job") or {}).get("role", "asset")},
                                                             ensure_ascii=False), encoding="utf-8")
        check = workspace / "check"
        check.write_text(CHECK_SCRIPT.format(src=Path(__file__).resolve().parents[1], python=sys.executable),
                         encoding="utf-8")
        check.chmod(0o755)
        note = " (previous round's version; repair it from here)" if previous else " (currently empty; write it from scratch)"
        task_text = AGENT_TASK.format(current_note=note, max_checks=self.max_checks)
        if self.skills:
            # Modelling recipes distilled from repeated check failures, read before the first edit.
            shutil.copytree(SKILLS, workspace / "skills")
            task_text += SKILLS_NOTE
        (workspace / "AGENT_TASK.md").write_text(task_text, encoding="utf-8")
        task = "Read AGENT_TASK.md in the current directory and follow it exactly."
        stdout, stderr, code, elapsed = self.run(
            self.command(payload["model"], workspace, images, sandbox="workspace-write"), workspace, task, timeout)
        (workspace / "events.jsonl").write_bytes(stdout)
        message, usage = self.reply(stdout, stderr, code)
        source = (workspace / "model.py").read_text(encoding="utf-8")
        # Whole-file replacement keeps the unique-match patch contract; an unchanged
        # script is rejected by author_result like any other no-op repair.
        value = {"model_py": source} if previous is None else {"edits": [{"old": previous, "new": source}]}
        checks_run = 0
        for line in stdout.decode("utf-8", "replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item") or {}
            if (event.get("type") == "item.completed" and item.get("type") == "command_execution"
                    and "./check" in str(item.get("command", ""))):
                checks_run += 1
        response = self.response(payload, json.dumps(value, ensure_ascii=False), usage, elapsed)
        response["agent_session"] = {"workspace": str(workspace), "checks_run": checks_run,
                                     "final_message": strip_fence(message)[:2000]}
        return response

    def run(self, command, folder, prompt, timeout):
        from .process_tree import ProcessTree
        started = time.monotonic()
        env = {k: v for k, v in os.environ.items()
               if k not in {"OPENAI_API_KEY", "HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "TAVILY_API_KEY"}}
        tree, process = ProcessTree(), None
        try:
            self.check_cancel()
            process = subprocess.Popen(command, cwd=folder, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=os.name != "nt")
            tree.attach(process)
            pending = prompt.encode("utf-8")
            while True:
                self.check_cancel()
                left = timeout - (time.monotonic() - started)
                if left <= 0:
                    raise ProviderError("Codex request timed out; process killed", uncertain=False,
                                        diagnostics={"failure": "timeout", "timeout_seconds": timeout})
                try:
                    stdout, stderr = process.communicate(input=pending, timeout=min(1, left))
                    break
                except subprocess.TimeoutExpired:
                    pending = None
        finally:
            if process is not None:
                tree.stop(process)
            else:
                tree.close()
        return stdout, stderr, process.returncode, time.monotonic() - started

    @staticmethod
    def reply(stdout, stderr, code):
        message, usage, error = parse_events(stdout.decode("utf-8", "replace"))
        if code != 0 or error or message is None:
            detail = error or (stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"])[0]
            # Login-session calls are not billed per call; a failure is handled as a connection error and may be retried with bounds by the caller.
            raise ProviderError("Codex request failed: " + detail[:500], uncertain=False,
                                diagnostics={"failure": "connection_error", "exit_code": code,
                                             "provider_error": {"message": detail[:1024]}})
        return message, usage

    @staticmethod
    def response(payload, content, usage, elapsed):
        prompt_tokens, completion = usage.get("input_tokens"), usage.get("output_tokens")
        return {"model": payload["model"], "provider": "codex",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                          "total_tokens": (prompt_tokens or 0) + (completion or 0),
                          "cached_tokens": usage.get("cached_input_tokens"),
                          "reasoning_tokens": usage.get("reasoning_output_tokens")},
                "elapsed_seconds": round(elapsed, 3)}
