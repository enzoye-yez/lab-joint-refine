"""原子写入、证据摘要、调用账本及跨进程锁。"""

import hashlib
import json
import os
import re
import time
import errno
import uuid
from contextlib import contextmanager
from pathlib import Path

from .errors import BudgetExceeded, StaleEvidence, UncertainRequest, ValidationError


def encoded(value, *, default=None):
    from collections.abc import Mapping
    def mapping(item):
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      default=default or mapping).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = root / relative
    if Path(relative).is_absolute() or ":" in relative or ".." in Path(relative).parts:
        raise ValidationError("输出路径越界")
    if not path.resolve().is_relative_to(root):
        raise ValidationError("输出路径通过链接越界")
    return path


def atomic_json(path: Path, value, *, default=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    if path.is_symlink():
        raise ValidationError("拒绝写入符号链接")
    content = encoded(value, default=default)
    try:
        with temp.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(10):
            try:
                os.replace(temp, path)
                return
            except OSError as exc:
                transient = exc.errno in {errno.EACCES, errno.EBUSY, errno.EPERM} or getattr(exc, "winerror", None) in {5, 32, 33}
                if not transient or attempt == 9:
                    exc.add_note("本地文件保存失败，原文件保持完整；目标：" + str(path))
                    raise
                delay = min(.1 * 2 ** attempt, 2)
                print(f"本地保存暂时受阻：{path.name}；{delay:g} 秒后重试（{attempt+1}/9）", flush=True)
                time.sleep(delay)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass  # 不让临时文件清理掩盖原始错误。


@contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(root, ".lock")
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValidationError("同一运行正在被另一个进程使用") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Store:
    def __init__(self, root, identity, config, *, resume=False, retry_uncertain=False):
        self.root = Path(root)
        self.config = config
        self.retry_uncertain = retry_uncertain
        self.path = safe_path(self.root, "state.json")
        if self.path.exists():
            if not resume:
                raise ValidationError("运行已存在；使用 --resume 或新的运行名称")
            from .state_archive import read_state
            self.state = read_state(self.path)
            if self.state["identity"] != identity:
                old=self.state['identity']
                compatible=(old.get('workflow_contract')=='asset_workflow_v2' and
                            identity.get('workflow_contract')==old['workflow_contract'] and
                            {k:v for k,v in old.items() if k!='code'}=={k:v for k,v in identity.items() if k!='code'})
                if not compatible:
                    raise StaleEvidence("输入、配置或工作流契约已变化；请先执行受支持的离线迁移")
                self.state.setdefault('implementation_history',[]).append({'from':old['code'],'to':identity['code'],'at':time.time()})
                self.state['identity']=identity
                self.save()
        elif resume:
            raise ValidationError("没有可恢复的运行")
        else:
            self.state = {"identity": identity, "calls": [], "created_at": time.time()}
            self.save()
        self.recover_responses()

    def recover_responses(self):
        """响应已落盘但账本替换失败时恢复；未发送预留不冒充未知请求。"""
        changed = False
        for call in self.state["calls"]:
            receipt = safe_path(self.root, f"responses/{call['index']:05d}.json")
            if receipt.exists() and call["status"] != "done":
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if saved.get("key") != call["key"] or digest(saved["payload"]) != saved.get("sha256"):
                    raise StaleEvidence("待恢复响应与调用账本不匹配")
                self.finish(call, saved["payload"])
            elif call["status"] == "reserved":
                call.update(status="not_sent", finished_at=time.time())
                changed = True
        if changed:
            self.save()

    def save(self):
        from .state_archive import write_state
        write_state(self.path, self.state)

    def cached(self, key):
        for call in reversed(self.state["calls"]):
            if call["key"] != key:
                continue
            if call["status"] == "pending" and not self.retry_uncertain:
                raise UncertainRequest("上次请求结果不明；核对服务商记录后用 --retry-uncertain 重试，可能重复计费")
            if call["status"] == "done":
                path = safe_path(self.root, call["file"])
                if not path.is_file() or file_hash(path) != call["sha256"]:
                    raise StaleEvidence("请求缓存被修改或缺失，旧证据失效")
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    def reserve(self, key, kind, tokens=0, *, defer_send=False):
        calls = self.state["calls"]
        if kind not in {"model", "search", "model_search", "image", "page"} or type(tokens) is not int or tokens < 0:
            raise ValidationError("无效的调用类型或预算预留")
        totals = self.totals()
        checks = []
        if kind in {"model", "model_search"}:
            checks.append(("model_calls", self.config.max_model_calls))
        if kind in {"search", "model_search"}:
            checks.append(("search_calls", self.config.max_search_calls))
        if kind == "image":
            checks.append(("image_downloads", self.config.max_image_downloads))
        if kind == "page":
            checks.append(("page_fetches", self.config.max_page_fetches))
        for counter, limit in checks:
            if totals[counter] >= limit:
                raise BudgetExceeded(f"{counter} 调用预算耗尽")
        if totals["accounted_tokens"] + tokens > self.config.max_reserved_tokens:
            raise BudgetExceeded("保守估算的令牌预算耗尽")
        from .phase_budget import check
        check(self,tokens,kind in {'model','model_search'})
        call = {"key": key, "kind": kind, "status": "reserved" if defer_send else "pending", "reserved_tokens": tokens,
                "started_at": time.time(), "index": len(calls)}
        calls.append(call)
        self.save()  # 先扣预算再发送；崩溃、失败、未知结果均不退回预算。
        return call

    def finish(self, call, payload):
        receipt = safe_path(self.root, f"responses/{call['index']:05d}.json")
        atomic_json(receipt, {"key":call["key"], "sha256":digest(payload), "payload":payload})
        relative = f"records/{call['index']:05d}.json"
        path = safe_path(self.root, relative)
        atomic_json(path, payload)
        call.update(status="done", file=relative, sha256=file_hash(path), finished_at=time.time())
        if call["kind"] in {"model", "model_search"}:
            usage = payload.get("usage")
            actual = usage.get("total_tokens") if isinstance(usage, dict) else None
            if type(actual) is int and actual >= 0:
                call["reported_total_tokens"] = actual
                # Codex CLI reports the provider's own count, images included; such a call is
                # settled at that count and its pre-send reservation is released.
                if payload.get("provider") == "codex" and actual > 0:
                    call["settled_by_usage"] = True
        self.save()
        try:
            receipt.unlink(missing_ok=True)
        except OSError:
            pass

    def mark_sent(self, call):
        call.update(status="pending", send_started_at=time.time())
        try:
            self.save()
        except OSError:
            call.update(status="not_sent")
            call.pop("send_started_at", None)
            raise

    def check_token_budget(self):
        if self.totals()["accounted_tokens"] > self.config.max_reserved_tokens:
            raise BudgetExceeded("服务报告的实际令牌超过预留后已达到预算上限；响应已保存，请提高预算后恢复")

    def fail(self, call, error_type):
        call.update(status="failed", error_type=error_type, finished_at=time.time())
        self.save()

    def record_error(self, call, error):
        # 仅允许传输层生成的结构化诊断；不保存密钥、异常原文或服务正文。
        allowed = {'phase', 'elapsed_seconds', 'timeout_seconds', 'http_status', 'received_bytes', 'failure',
                   'exception_type', 'errno', 'winerror'}
        call['diagnostics'] = {k: v for k, v in error.diagnostics.items() if k in allowed}
        call['error_at'] = time.time()
        self.save()

    def record_validation(self, key, event):
        for call in reversed(self.state['calls']):
            if call['key'] == key and call['status'] == 'done':
                events = call.setdefault('validation_events', [])
                if event not in events:
                    events.append(event)
                    self.save()
                break

    def totals(self):
        calls = self.state["calls"]
        return {"model_calls": sum(c["kind"] in {"model", "model_search"} for c in calls),
                "search_calls": sum(c["kind"] in {"search", "model_search"} for c in calls),
                "image_downloads": sum(c["kind"] == "image" for c in calls),
                "page_fetches": sum(c["kind"] == "page" for c in calls),
                "reserved_tokens": sum(c["reserved_tokens"] for c in calls),
                "reported_total_tokens": sum(c.get("reported_total_tokens", 0) for c in calls),
                "accounted_tokens": sum(c["reported_total_tokens"] if c.get("settled_by_usage") else
                                        max(c["reserved_tokens"], c.get("reported_total_tokens", 0)) for c in calls)}


def run_directory(project: Path, name: str):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValidationError("运行名称只允许英文字母、数字、短横线和下划线")
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValidationError("运行名称不能是 Windows 保留设备名")
    return safe_path(project, f"runs/{name}")
