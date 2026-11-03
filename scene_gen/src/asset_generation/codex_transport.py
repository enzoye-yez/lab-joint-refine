"""通过本机已登录的 Codex CLI 调用模型；返回与 chat completions 相同的响应形状。

使用 ChatGPT 登录态，不读取任何 API 密钥。每次请求在独立临时目录中运行一次
`codex exec`（只读沙箱、不持久化会话），图片以附件传入，文本经标准输入传入。

编码 agent 模式（agent=True）只用于作者请求：codex 在可写工作目录里编辑 model.py，
可反复运行 ./check 本地自检，结束后把最终脚本按原作者契约返回（首轮 model_py，修复轮
整文件替换）。框架照旧独立重建、检查和审查，自检结论不作为通过依据。
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

AGENT_TASK = """# 资产建模任务（编码 agent 模式）

你在一个只属于本资产的工作目录中工作。目标：写出满足 instructions.md 与 context.json 的 `model.py`。

- `instructions.md`：建模契约（必须遵守，与普通模式相同）。
- `context.json`：资料、任务、依赖、上一轮反馈（feedback / repair_tasks 中的 repair_hint 给出局部坐标方向和距离）。
- `references/`：参考图片（也已作为图片附件给你）。
- `model.py`：当前脚本{current_note}。
- `./check`：构建当前 model.py 并运行单资产检查，输出 JSON 摘要（failing、incomplete、renders）。

工作方式：
1. 阅读契约与资料，编辑 `model.py`（只改这个文件；不要创建其他交付文件）。
2. 运行 `./check`。`failing` 中的条目必须修复；`incomplete` 若只因量测不可靠或缺少探测器，可以保留。
3. 需要时查看 `renders` 中的图片核对外形。
4. 重复 2–3，最多 {max_checks} 次检查；没有 failing 后再做一次整体核对即可结束。
5. 不要修改 check、selfcheck.json、instructions.md、context.json 或 references；不要联网。

结束时只回复一行 JSON：{{"done": true, "checks_run": 次数, "remaining_failing": [仍失败的检查编号], "summary": "一句话说明做了什么"}}。
框架会独立重建、检查和审查你的 model.py；自检通过不等于验收。
"""

CHECK_SCRIPT = """#!/usr/bin/env bash
# 本地自检：构建 model.py 并运行单资产检查（CPU 渲染，不使用 GPU）。
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
    """把 system/user 消息拼成一段提示，并把 data URL 图片写成按顺序编号的附件。

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
                    texts.append(f"[完整原文见当前目录文件 input_{files}.txt（{len(part['text'])} 字符），按需读取]")
                else:
                    texts.append(part["text"])
            elif part["type"] == "image_url":
                header, _, data = part["image_url"]["url"].partition(",")
                mime = header.removeprefix("data:").split(";")[0]
                if not header.startswith("data:") or ";base64" not in header or mime not in IMAGE_SUFFIX:
                    raise ProviderError("Codex 传输只支持 base64 data URL 图片", uncertain=False,
                                        diagnostics={"failure": "unsupported_image"})
                path = Path(folder) / f"image_{len(images) + 1:03d}{IMAGE_SUFFIX[mime]}"
                try:
                    path.write_bytes(base64.b64decode(data, validate=True))
                except binascii.Error:
                    raise ProviderError("图片 data URL 无法解码", uncertain=False,
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
            error = event.get("message") or (event.get("error") or {}).get("message") or "codex 报告失败"
    return message, usage, error


def is_author_phase(phase):
    return phase == "author" or (phase or "").startswith("asset:")


class CodexTransport:
    """与 GenerationTransport 相同的 post 接口；url 与 key 不使用。"""

    def __init__(self, executable="codex", reasoning_effort="high", *, agent=False, workspace_root=None, max_checks=12):
        self.executable, self.reasoning_effort = executable, reasoning_effort
        self.agent, self.workspace_root, self.max_checks = agent, workspace_root, max_checks
        self.check_cancel = lambda: None
        self.phase = None  # GenerationModel sets the request phase before each post.

    def whole_context(self, phase):
        """In agent mode no request is paged: authors get context.json in their workspace and other
        requests read over-long text from files in their request folder."""
        return bool(self.agent)

    def command(self, model, folder, images, sandbox="read-only"):
        command = [self.executable, "exec", "--model", model, "--sandbox", sandbox, "--ephemeral",
                   "--skip-git-repo-check", "--ignore-user-config", "--json",
                   "--cd", str(folder), "-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        for image in images:
            command += ["--image", str(image)]
        return command + ["-"]

    def post(self, url, payload, key, timeout):
        if shutil.which(self.executable) is None:
            raise ProviderError("未找到 codex 命令；请先安装并登录 Codex CLI", uncertain=False,
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
            raise ProviderError("作者上下文缺少资产编号，无法建立自检", uncertain=False,
                                diagnostics={"failure": "agent_context"})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        workspace = (Path(self.workspace_root or tempfile.gettempdir()) / "codex_agent_sessions"
                     / f"{stamp}_{(self.phase or 'author').replace(':', '_')}")
        (workspace / "references").mkdir(parents=True)
        _, images = prompt_and_images(payload["messages"], workspace / "references")
        (workspace / "instructions.md").write_text(payload["messages"][0]["content"], encoding="utf-8")
        (workspace / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=1), encoding="utf-8")
        (workspace / "model.py").write_text(previous or "", encoding="utf-8")
        (workspace / "selfcheck.json").write_text(json.dumps({"asset_ids": asset_ids, "profile": context["profile"]},
                                                             ensure_ascii=False), encoding="utf-8")
        check = workspace / "check"
        check.write_text(CHECK_SCRIPT.format(src=Path(__file__).resolve().parents[1], python=sys.executable),
                         encoding="utf-8")
        check.chmod(0o755)
        note = "（上一轮版本，需要在此基础上修复）" if previous else "（当前为空，需要从头编写）"
        (workspace / "AGENT_TASK.md").write_text(AGENT_TASK.format(current_note=note, max_checks=self.max_checks),
                                                 encoding="utf-8")
        task = "Read AGENT_TASK.md in the current directory and follow it exactly. 先读 AGENT_TASK.md。"
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
                    raise ProviderError("Codex 请求超时；进程已终止", uncertain=False,
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
            detail = error or (stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["无输出"])[0]
            # 登录态调用不按次计费，失败按连接错误处理，可由上层有界重试。
            raise ProviderError("Codex 请求失败：" + detail[:500], uncertain=False,
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
