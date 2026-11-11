"""Reuses the original project's transport and ledger; actual image pixels go into the multimodal requests."""

import json
from contextlib import contextmanager
import time

from asset_prep.errors import ProviderError, UncertainRequest, ValidationError, ContextLimitError
from asset_prep.providers import redact
from asset_prep.storage import digest, encoded
from .provider_diagnostics import safe_details


# Transport retries are independent of model geometry-repair rounds, but every attempt still consumes call and token budget.
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
    """Explicit service rejections are not blindly resent; an unexplained 400 gets only limited diagnostic retries."""
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
            raise UncertainRequest("The ledger has a request with unknown outcome; compaction or detail reads cannot bypass explicit recovery")
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
                        raise ValidationError("Image reads or paging notes must not be mixed into the final result")
                    if "context_notes" in value:
                        notes = view.notes(value["context_notes"])
                    if "context_images" in value:
                        signature = digest({"context_images":value["context_images"]})
                        view.select_images(value["context_images"])
                        stalled = stalled+1 if signature in pages else 0
                        pages[signature] = True
                        if stalled >= self.generation.max_context_reads:
                            raise ContextLimitError("Image comparison keeps producing nothing new; keeping the current task and all original images")
                        view.receipt["reads"].append({"request":value,"call":call,"kind":"image_comparison"})
                        current = {**view.context,"context_working_notes":notes}
                        view.save("image_read")
                        continue
                except ValidationError as exc:
                    stalled += 1
                    if stalled > self.generation.max_context_reads:
                        raise ContextLimitError("Paging protocol keeps being invalid; keeping the materials and not certifying the unfinished result") from exc
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
                               "prior_batch_rule":"prior_batch_findings are defects raised by earlier image batches. The final verdict must weigh all batches item by item: those still valid go in findings with repair; those refuted by later image evidence are no longer listed."}
                    stalled = 0
                    view.save("next_batch")
                    continue
                if view.paged and isinstance(value,dict) and set(value)=={"context_notes"}:
                    stalled += 1
                    if stalled > self.generation.max_context_reads:
                        raise ContextLimitError("Materials delivered, but no task result submitted even after consolidation; keeping the task")
                    current = {**view.context,"context_working_notes":notes,
                               "context_delivery_error":{"attempt":stalled,"instruction":"All batches have been delivered; give the verdict under the original task contract"}}
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
                raise ContextLimitError("Detail reads keep adding nothing new and no verdict was submitted even after the consolidation prompt; suspending the current task, other tasks can continue")
            request = value.get("context_read")
            signature = digest(request)
            number = len(view.receipt["reads"])
            progress = {"completed": number + 1, "window": self.generation.max_context_reads,
                        "previous_requests": [entry["request"] for entry in view.receipt["reads"][-8:]],
                        "instruction": "You may re-read the cache to verify; when the window ends, consolidate your working notes and submit the current task's verdict if at all possible."}
            reused = signature in pages
            try:
                if not {"context_read"} <= set(value) <= {"context_read", "context_notes"}:
                    raise ValidationError("Do not submit a task result while reading details")
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
                        "instruction": "Narrow the field path or page; if the evidence is already sufficient submit the result, otherwise state the unresolved items explicitly."}
                fresh = False
            stalled = 0 if fresh else stalled + 1
            progress.update(reused=reused, stalled=stalled)
            if stalled >= self.generation.max_context_reads:
                progress["instruction"] = "This window added no new usable details. Submit the verdict under the original task contract; where evidence is insufficient keep it explicitly unknown, and do not re-read the same content."
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
            raise ContextLimitError(f"Request context budget insufficient: conservative estimate including output and margin {required}, limit {self.generation.max_context_tokens}; not sent, the same request is not retried automatically")
        if len(encoded(payload)) > self.generation.max_request_bytes and not whole:
            raise ContextLimitError("Image-and-text request exceeds the byte limit; not sent, adjust the context projection or task scope")
        request_key = digest({"phase": phase, "payload": payload,
                              "implementation":self.store.state.get('identity',{}).get('implementation')})
        if any(c.get("key")==request_key and c.get("failure_kind")=="context_limit" for c in self.store.state["calls"]):
            raise ContextLimitError("The same request was already explicitly rejected by the service as exceeding the context; ledger kept, the context must change before resending")
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
            # Images are tokenized by the service, so base64 bytes are not exact tokens; keep the explicit engineering estimate.
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
                    # An unknown outcome is not disguised as a server-side failure; only the next successful reservation lifts the block.
                    previous_call.update(status="unknown_retried", outcome_unknown=True)
            self.execution_guard()
            self.store.mark_sent(call)
            self.progress(phase, "request_sent")
            print(f"Model request #{call['index']+1}: {phase}, transport attempt {retry+1}, {len(images)} images", flush=True)
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
                # Save the paid response first, then refuse to consume a result produced under a drifted version; recovery loses no ledger entries.
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
                    reason = f"Service status code {status}" if status else "Connection dropped, timed out or abnormal response"
                    uncertainty = "Previous request outcome unknown, possibly billed twice; " if exc.uncertain else ""
                    print(f"{reason}; automatic retry {retry+1}/{retry_limit} in {delay} s. "
                          f"{uncertainty}Every attempt keeps its budget record.", flush=True)
                    previous_calls = [call]
                    self.progress(phase, "retry_wait")
                    remaining = delay
                    while remaining > 0:
                        # Long cooldowns stay interruptible; a progress message every minute.
                        self.cancel_guard()
                        interval = min(remaining, 1)
                        with self._unlocked():
                            time.sleep(interval)
                        self.cancel_guard()
                        remaining -= interval
                        if remaining and remaining % 60 == 0:
                            print(f"Waiting for the service to recover; about {remaining:.0f} s until retry", flush=True)
                    continue
                self.store.save()
                if exc.uncertain:
                    raise UncertainRequest("Generation request outcome unknown; reserved budget and state saved; not resent automatically; after verifying, you may explicitly use --retry-uncertain") from exc
                raise
        # Paid, archived responses are consumable even if reported usage exceeds
        # the reservation. Store.reserve enforces the total budget BEFORE any new
        # request. Consumption must not discard work or force a duplicate request.
        try:
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValidationError("Generation response truncated; incomplete code is not executed")
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
            raise ValidationError("Structured content returned by the model is invalid; the original response is archived") from exc
