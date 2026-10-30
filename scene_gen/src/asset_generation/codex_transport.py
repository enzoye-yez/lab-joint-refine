"""通过本机已登录的 Codex CLI 调用模型；返回与 chat completions 相同的响应形状。

使用 ChatGPT 登录态，不读取任何 API 密钥。每次请求在独立临时目录中运行一次
`codex exec`（只读沙箱、不持久化会话），图片以附件传入，文本经标准输入传入。
"""

import base64
import binascii
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from asset_prep.errors import ProviderError

REPLY_RULES = ("You are called as a plain model endpoint, not as a coding agent. "
               "Do not run commands, read files or use tools. Answer only with the single JSON "
               "object the instructions below ask for, without Markdown fences or extra text.")
IMAGE_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


def prompt_and_images(messages, folder):
    """把 system/user 消息拼成一段提示，并把 data URL 图片写成按顺序编号的附件。"""
    sections, images = [REPLY_RULES], []
    for message in messages:
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        texts = []
        for part in parts:
            if part["type"] == "text":
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


class CodexTransport:
    """与 GenerationTransport 相同的 post 接口；url 与 key 不使用。"""

    def __init__(self, executable="codex", reasoning_effort="high"):
        self.executable, self.reasoning_effort = executable, reasoning_effort
        self.check_cancel = lambda: None

    def command(self, model, folder, images):
        command = [self.executable, "exec", "--model", model, "--sandbox", "read-only", "--ephemeral",
                   "--skip-git-repo-check", "--ignore-user-config", "--json",
                   "--cd", str(folder), "-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        for image in images:
            command += ["--image", str(image)]
        return command + ["-"]

    def post(self, url, payload, key, timeout):
        if shutil.which(self.executable) is None:
            raise ProviderError("未找到 codex 命令；请先安装并登录 Codex CLI", uncertain=False,
                                diagnostics={"failure": "codex_missing"})
        from .process_tree import ProcessTree
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="codex-request-") as folder:
            prompt, images = prompt_and_images(payload["messages"], folder)
            env = {k: v for k, v in os.environ.items()
                   if k not in {"OPENAI_API_KEY", "HARNESS_MODEL_API_KEY", "HARNESS_MODEL_API_KEY2", "TAVILY_API_KEY"}}
            tree, process = ProcessTree(), None
            try:
                self.check_cancel()
                process = subprocess.Popen(self.command(payload["model"], folder, images), cwd=folder, env=env,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=os.name != "nt")
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
        message, usage, error = parse_events(stdout.decode("utf-8", "replace"))
        if process.returncode != 0 or error or message is None:
            detail = error or stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["无输出"]
            detail = detail if isinstance(detail, str) else detail[0]
            # 登录态调用不按次计费，失败按连接错误处理，可由上层有界重试。
            raise ProviderError("Codex 请求失败：" + detail[:500], uncertain=False,
                                diagnostics={"failure": "connection_error", "exit_code": process.returncode,
                                             "provider_error": {"message": detail[:1024]}})
        prompt_tokens = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        return {"model": payload["model"], "provider": "codex",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": strip_fence(message)}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                          "total_tokens": (prompt_tokens or 0) + (completion or 0),
                          "cached_tokens": usage.get("cached_input_tokens"),
                          "reasoning_tokens": usage.get("reasoning_output_tokens")},
                "elapsed_seconds": round(time.monotonic() - started, 3)}
