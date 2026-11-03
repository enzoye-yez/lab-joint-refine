"""复用原项目传输与账本；实际图片像素进入多模态请求。"""

import json
from contextlib import contextmanager
import time

from asset_prep.errors import ProviderError, UncertainRequest, ValidationError, ContextLimitError
from asset_prep.providers import redact
from asset_prep.storage import digest, encoded
from .provider_diagnostics import safe_details


# 传输重试独立于模型几何修复轮次，但每次仍扣除调用与令牌预算。
TRANSPORT_RETRY_DELAYS = (15, 30, 60, 60, 60)


def request_summary(calls):
    model = [c for c in calls if c.get("kind") in {"model", "model_search"}]
    return {"attempts": len(model), "successful_responses": sum(c["status"] == "done" for c in model),
            "failed_attempts": sum(c["status"] == "failed" for c in model),
            "unknown_attempts": sum(c["status"] == "pending" or c.get("outcome_unknown", False) for c in model),
            "retry_attempts": sum("retry_of" in c for c in model),
            "not_sent": sum(c["status"] in {"reserved", "not_sent"} for c in model),
            "logical_requests": len({c["key"] for c in model})}


def transient_error(error):
    status = error.diagnostics.get("http_status")
    if status is not None and status != 200:
        return status in {408, 429} or (500 <= status <= 599 and status not in {501, 505})
    return error.diagnostics.get("failure") in {
        "timeout", "process_deadline", "connection_error", "dns_error",
        "protocol_error", "network_worker_exit", "invalid_json",
    }


def request_failure_kind(error):
    """服务明确拒绝不盲目重发；无说明的 400 只作有限诊断重试。"""
    details = error.diagnostics.get("provider_error", {})
    if not isinstance(details, dict):
        details = {}
    code = " ".join(str(details.get(k, "")) for k in ("code", "type", "message")).lower()
    if any(token in code for token in ("context_length_exceeded", "input tokens exceed",
            "maximum context length", "too many input tokens", "context window", "messages resulted in")):
        return "context_limit"
    permanent = ("insufficient_quota", "insufficient_balance", "invalid_api_key", "authentication_error",
                 "invalid_request_error", "context_length_exceeded", "model_not_found", "permission_denied",
                 "invalid parameter", "invalid api key", "insufficient balance", "quota exceeded",
                 "余额不足", "额度不足", "无效密钥", "参数错误")
    if any(token in code for token in permanent):
        return "permanent"
    if transient_error(error):
        return "transient"
    if error.diagnostics.get("http_status") == 400:
        temporary = ("overloaded", "temporarily unavailable", "upstream timeout", "server busy", "服务器繁忙")
        return "transient" if any(token in code for token in temporary) else "unclassified_http_400"
    return "permanent"


class GenerationModel:
    def __init__(self, config, generation, credentials, store, transport):
        self.config, self.generation = config, generation
        self.credentials, self.store, self.transport = credentials, store, transport
        self.execution_guard = lambda: None
        self.cancel_guard = lambda: self.execution_guard()
        self.progress = lambda phase, event: None
        if hasattr(transport, "check_cancel"):
            transport.check_cancel = lambda: self.cancel_guard()

    manages_context = True

    def whole_context(self, phase):
        check = getattr(type(self.transport), "whole_context", None)
        return check is not None and check(self.transport, phase) is True

    def ask(self, instruction, context, images, phase):
        from .request_context import RequestContext, measure
        self.execution_guard()
        # Do not let a different compacted payload bypass an unresolved paid request.
        inflight = getattr(self.store, "inflight", set())
        if not self.store.retry_uncertain and any(c["status"] == "pending" and c.get("index") not in inflight
                                                  for c in self.store.state["calls"]):
            raise UncertainRequest("账本存在未知请求，不能通过压缩或读取详情绕过显式恢复")
        if self.whole_context(phase):
            # The codex agent transport reads over-long context from files,
            # so the request is neither projected nor paged.
            return self._ask_once(instruction, context, images, phase)
        view = RequestContext(self.store.root, self.store.state.get("identity", {}),
                              instruction, context, images, phase, self.generation)
        self._context_binding = view.binding
        current, pages, notes = view.context, {}, view.notes([])
        retained_pages = {}
        stalled = 0
        batch_findings = []
        while True:
            self.execution_guard()
            try:
                value, call = self._ask_once(view.instruction, current, view.images, phase)
            except BaseException:
                view.save("interrupted")
                raise
            if view.paged and isinstance(value,dict) and (
                    "context_images" in value or set(value)=={"context_notes"}):
                try:
                    if not set(value) <= {"context_images","context_notes"}:
                        raise ValidationError("图片读取或分页笔记不能混入最终结果")
                    if "context_notes" in value:
                        notes = view.notes(value["context_notes"])
                    if "context_images" in value:
                        signature = digest({"context_images":value["context_images"]})
                        view.select_images(value["context_images"])
                        stalled = stalled+1 if signature in pages else 0
                        pages[signature] = True
                        if stalled >= self.generation.max_context_reads:
                            raise ContextLimitError("图片对照持续无新内容；保留当前任务和全部原图")
                        view.receipt["reads"].append({"request":value,"call":call,"kind":"image_comparison"})
                        current = {**view.context,"context_working_notes":notes}
                        view.save("image_read")
                        continue
                except ValidationError as exc:
                    stalled += 1
                    if stalled > self.generation.max_context_reads:
                        raise ContextLimitError("分页协议持续无效；保留资料，不认证未完成结果") from exc
                    current = {**view.context,"context_working_notes":notes,
                               "context_delivery_error":{"attempt":stalled,"error":str(exc)}}
                    continue
            if not isinstance(value, dict) or "context_read" not in value:
                if view.delivery_index < view.delivery_count-1:
                    # Early task answers are archived proposals, never a result for
                    # the unseen remainder. Known visual defects survive later pages.
                    if isinstance(value,dict) and value.get("verdict")=="repair":
                        batch_findings.append(value)
                    view.receipt.setdefault("batch_responses",[]).append(
                        {"batch":view.delivery_index,"call":call,"response":value})
                    view.delivery_index += 1
                    view.set_delivery()
                    current = {**view.context,"context_working_notes":notes,
                               "prior_batch_findings":view.fold(batch_findings,8192),
                               "prior_batch_rule":"prior_batch_findings是此前图片批次提出的缺陷。最终结论须结合全部批次逐条判断：仍成立的写入findings并给repair；被后续图片证据否定的不再列出。"}
                    stalled = 0
                    view.save("next_batch")
                    continue
                if view.paged and isinstance(value,dict) and set(value)=={"context_notes"}:
                    stalled += 1
                    if stalled > self.generation.max_context_reads:
                        raise ContextLimitError("资料已送达，整理后仍未提交任务结果；保留任务")
                    current = {**view.context,"context_working_notes":notes,
                               "context_delivery_error":{"attempt":stalled,"instruction":"所有批次已送达，请按原任务契约给出结论"}}
                    continue
                if batch_findings and isinstance(value,dict) and "verdict" in value:
                    from .review import retain_batch_defects
                    value, superseded = retain_batch_defects(batch_findings, value)
                    view.receipt["superseded_batch_findings"] = superseded
                view.receipt["reported_usage"] = call.get("usage", {})
                view.save("answered")
                return value, {**call, "context_binding": view.binding}
            # A complete window without new usable evidence asks for a decision first.
            if stalled >= self.generation.max_context_reads:
                view.save("detail_task_deferred")
                raise ContextLimitError("详情读取持续无新增内容，整理提示后仍未提交结论；暂挂当前任务，其他任务可继续")
            request = value.get("context_read")
            signature = digest(request)
            number = len(view.receipt["reads"])
            progress = {"completed": number + 1, "window": self.generation.max_context_reads,
                        "previous_requests": [entry["request"] for entry in view.receipt["reads"][-8:]],
                        "instruction": "可重读缓存核对；窗口结束时整理工作笔记并尽量提交当前任务结论。"}
            reused = signature in pages
            try:
                if not {"context_read"} <= set(value) <= {"context_read", "context_notes"}:
                    raise ValidationError("读取详情时不得同时提交任务结果")
                if "context_notes" in value:
                    notes = view.notes(value["context_notes"])
                skeleton = {**view.context, "context_working_notes": notes,
                            "context_read_progress": progress}
                remaining = (self.generation.max_context_tokens -
                             measure(view.instruction, skeleton, view.images, self.generation)["required"] - 1024)
                # Re-read through the hash-checked reader; cache identity never bypasses integrity.
                page = view.read(request, budget=min(32768, remaining))
                # Different requested limits can return the exact same page.
                page_key = digest({k:page[k] for k in ("reference","path","offset","next_offset")})
                reused = page_key in pages
                fresh = not reused and bool(page.get("data"))
                pages[page_key] = digest(page)
                retained_pages.pop(page_key,None)
                retained_pages[page_key] = page
            except (ValidationError, ContextLimitError) as exc:
                page = {"error": str(exc), "request": request,
                        "instruction": "缩小字段路径或分页；若已有证据足够则提交结果，否则明确未解决项。"}
                fresh = False
            stalled = 0 if fresh else stalled + 1
            progress.update(reused=reused, stalled=stalled)
            if stalled >= self.generation.max_context_reads:
                progress["instruction"] = "本窗口没有新增可用详情。请按原任务契约提交结论；证据不足明确保留未知，不再重读同一内容。"
            view.receipt["reads"].append({"request": request, "call": call, "page_sha256": digest(page),
                                          "working_notes": notes, "reused": reused, "new_information": fresh})
            view.save("detail_read")
            current = {**view.context, "context_page": page, "context_working_notes": notes,
                       "context_read_progress": progress}
            # Keep exact returned pages, not model-written summaries. Evict only
            # older pages when their shared context allowance is full.
            while retained_pages:
                older = [p for p in retained_pages.values() if p != page]
                candidate = {**current,"context_pages":older}
                if (len(encoded(older))+len(encoded(page)) <= self.generation.review_context_bytes
                        and measure(view.instruction,candidate,view.images,self.generation)["required"] <= self.generation.max_context_tokens):
                    current = candidate
                    break
                retained_pages.pop(next(iter(retained_pages)))

    def _ask_once(self, instruction, context, images, phase):
        # The ledger lock is held for every ledger step and released only while waiting
        # on the provider, so parallel authors (scene prefetch) share one consistent ledger.
        lock = getattr(self.store, "lock", None)
        if lock is None:
            return self._ask_locked(instruction, context, images, phase)
        with lock:
            return self._ask_locked(instruction, context, images, phase)

    @contextmanager
    def _unlocked(self):
        lock = getattr(self.store, "lock", None)
        if lock is None:
            yield
            return
        lock.release()
        try:
            yield
        finally:
            lock.acquire()

    def _ask_locked(self, instruction, context, images, phase):
        from .request_context import messages_for, measure
        self.execution_guard()
        key = self.config.model_key(self.credentials)
        messages = messages_for(instruction, context, images)
        payload = {"model": self.config.model, "messages": messages, "stream": False,
                   self.config.output_token_parameter: self.generation.max_output_tokens}
        metrics = measure(instruction, context, images, self.generation)
        estimate, required = metrics["input_upper_estimate"], metrics["required"]
        whole = self.whole_context(phase)
        if required > self.generation.max_context_tokens and not whole:
            raise ContextLimitError(f"请求上下文预算不足：保守估计含输出和余量 {required}，上限 {self.generation.max_context_tokens}；未发送，不自动重试相同请求")
        if len(encoded(payload)) > self.generation.max_request_bytes and not whole:
            raise ContextLimitError("图文请求字节超限；未发送，需调整上下文投影或任务范围")
        request_key = digest({"phase": phase, "payload": payload,
                              "implementation":self.store.state.get('identity',{}).get('implementation')})
        if any(c.get("key")==request_key and c.get("failure_kind")=="context_limit" for c in self.store.state["calls"]):
            raise ContextLimitError("相同请求已被服务明确拒绝为上下文超限；保留账本，需改变上下文后才能再发")
        response = self.store.cached(request_key)
        previous_calls = ([call for call in self.store.state["calls"]
                           if call["status"] == "pending" and call.get("phase") == phase]
                          if self.store.retry_uncertain else [])
        if response is not None:
            for call in previous_calls:
                if call["key"] == request_key:
                    call.update(status="unknown_superseded", outcome_unknown=True,
                                recovered_from_cache=request_key)
            if previous_calls:
                self.store.save()
        retry_limit = self.generation.max_transport_retries
        unknown_retries = 0
        for retry in range(retry_limit + 1):
            self.execution_guard()
            if response is not None:
                break
            # 图像由服务分词，不能用base64字节当作精确令牌；保留显式工程估计。
            reserve = estimate + self.generation.max_output_tokens
            call = self.store.reserve(request_key, "model", reserve, defer_send=True)
            call.update(phase=phase, provider=self.config.provider, requested_model=self.config.model,
                        request_sha256=digest(payload), images=[{"path": im["path"], "sha256": im["sha256"]} for im in images],
                        image_token_estimate=True, transport_attempt=retry + 1,
                        context_binding=getattr(self, "_context_binding", None),
                        context_budget=metrics)
            if previous_calls:
                call["retry_of"] = previous_calls[-1]["index"]
            for previous_call in previous_calls:
                previous_call["retried_by"] = call["index"]
                if previous_call["status"] == "pending":
                    # 未知结果不伪装成服务端失败；只有下一次成功预留后才解除阻塞。
                    previous_call.update(status="unknown_retried", outcome_unknown=True)
            self.execution_guard()
            self.store.mark_sent(call)
            self.progress(phase, "request_sent")
            print(f"模型请求 #{call['index']+1}：{phase}，传输尝试 {retry+1}，图片 {len(images)} 张", flush=True)
            try:
                # Transports that act differently per phase (codex agent author) read this.
                self.transport.phase = phase
                inflight = getattr(self.store, "inflight", None)
                if inflight is not None:
                    inflight.add(call["index"])
                try:
                    with self._unlocked():
                        response = self.transport.post(self.config.base_url.rstrip("/")+"/chat/completions",
                                                       payload, key, self.config.timeout_seconds)
                finally:
                    if inflight is not None:
                        inflight.discard(call["index"])
                response = redact(response, self.credentials.values())
                self.store.finish(call, response)
                self.progress(phase, "response_saved")
                # 先保存已付费响应，再拒绝消费漂移版本下的结果；恢复不丢账本。
                self.execution_guard()
            except ProviderError as exc:
                details = safe_details(exc.diagnostics.get("provider_error"), self.credentials.values())
                if details:
                    call["provider_error"] = details
                self.store.record_error(call, exc)
                if not exc.uncertain:
                    self.store.fail(call, "ProviderError")
                failure_kind = request_failure_kind(exc)
                call["failure_kind"] = failure_kind
                can_retry = not exc.uncertain and (failure_kind == "transient" or
                    (failure_kind == "unclassified_http_400" and unknown_retries < self.generation.max_unknown_http_retries))
                if can_retry and retry < retry_limit:
                    if failure_kind == "unclassified_http_400":
                        unknown_retries += 1
                    delay = (TRANSPORT_RETRY_DELAYS[retry] if retry < len(TRANSPORT_RETRY_DELAYS)
                             else self.generation.transport_cooldown_seconds)
                    retry_after = details.get("retry_after", "")
                    if retry_after.isascii() and retry_after.isdigit():
                        delay = max(delay, min(int(retry_after), self.generation.transport_cooldown_seconds))
                    call["automatic_retry"] = {"delay_seconds": delay, "number": retry + 1,
                                               "limit": retry_limit, "failure_kind":failure_kind}
                    self.store.save()
                    status = exc.diagnostics.get("http_status")
                    reason = f"服务状态码{status}" if status else "连接中断、超时或响应异常"
                    uncertainty = "上次请求结果未知，可能重复计费；" if exc.uncertain else ""
                    print(f"{reason}；{delay}秒后进行第{retry+1}/{retry_limit}次自动重试。"
                          f"{uncertainty}每次尝试保留预算记录。", flush=True)
                    previous_calls = [call]
                    self.progress(phase, "retry_wait")
                    remaining = delay
                    while remaining > 0:
                        # 长冷却保持可中断；每分钟有进度消息。
                        self.cancel_guard()
                        interval = min(remaining, 1)
                        with self._unlocked():
                            time.sleep(interval)
                        self.cancel_guard()
                        remaining -= interval
                        if remaining and remaining % 60 == 0:
                            print(f"等待服务恢复；距重试约 {remaining:.0f} 秒", flush=True)
                    continue
                self.store.save()
                if exc.uncertain:
                    raise UncertainRequest("生成请求结果未知，预留预算和现场已保存；不自动重发，核对后可显式使用 --retry-uncertain") from exc
                raise
        # Paid, archived responses are consumable even if reported usage exceeds
        # the reservation. Store.reserve enforces the total budget BEFORE any new
        # request. Consumption must not discard work or force a duplicate request.
        try:
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValidationError("生成响应被截断；不执行不完整代码")
            usage = response.get("usage", {})
            completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
            usage_observation = None
            if type(completion) is int and completion > self.generation.max_output_tokens:
                usage_observation = {"kind":"reported_output_exceeds_request",
                    "parameter":self.config.output_token_parameter,
                    "requested_limit":self.generation.max_output_tokens,
                    "reported_completion_tokens":completion,
                    "completion_tokens_details":usage.get("completion_tokens_details"),
                    "interpretation":"provider_accounting_or_enforcement_unknown"}
                self.store.record_validation(request_key, usage_observation)
            value = json.loads(choice["message"]["content"])
            encoded(value)
            return value, {"request_key": request_key, "response_model": response.get("model"),
                           "usage": usage, "usage_verified": type(completion) is int,
                           "usage_observation":usage_observation}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValidationError("模型返回的结构化内容无效，原响应已归档") from exc
