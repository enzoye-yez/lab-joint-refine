"""Chat model and optional Tavily search; the native web adapter is in search.py."""

import copy
import json
import sys
from urllib.parse import urlsplit

from .errors import ProviderError, UncertainRequest, ValidationError, OutputTooLarge, EvidenceRepairError
from .limits import check_output_limit
from .prompts import SYSTEM
from .storage import digest, encoded


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    return value


class Model:
    def __init__(self, config, credentials, store, transport):
        self.config, self.credentials, self.store, self.transport = config, credentials, store, transport

    def messages(self, instruction, data):
        return model_messages(self.config, instruction, data)

    def structured(self, instruction, data, validate):
        key = self.config.model_key(self.credentials)
        messages = self.messages(instruction, data)
        is_extraction = 'segment' in data or data.get('operation')=='local_extract'
        output_limit = min(self.config.max_output_tokens, self.config.extraction_output_tokens) if is_extraction else self.config.max_output_tokens
        if 'incoming_assets' in data or 'evidence_errors' in data or data.get('operation') in {'identity','field_identity','field_value'}:
            output_limit = min(output_limit, 4000)
        for attempt in range(self.config.repair_attempts + 1):
            size = len(encoded(messages))
            if size > self.config.max_input_bytes:
                raise ValidationError("model input exceeds the configured byte limit; evidence is never silently truncated")
            payload = {"model": self.config.model, "messages": messages,
                       self.config.output_token_parameter: output_limit, "stream": False}
            request_key = digest({"kind": "model", "payload": payload})
            response = self.store.cached(request_key)
            if response is None:
                # UTF-8 bytes + per-message overhead + output limit: a conservative reservation, not the real bill.
                call = self.store.reserve(request_key, "model", size + 256 * len(messages) + output_limit)
                stage = ('local extraction' if data.get('operation')=='local_extract' else 'extraction' if is_extraction else
                         'identity page' if data.get('operation')=='identity' else 'field page' if data.get('operation')=='field_identity' else
                         'field value choice' if data.get('operation')=='field_value' else 'local citation repair' if 'evidence_errors' in data else 'incremental asset merge' if 'incoming_assets' in data else 'reference selection')
                stage = {'protocol_index':'full-text index','flow_selection':'flow selection','task_definition':'task definition',
                         'task_review':'task review','asset_roles':'instance assignment','missing_roles':'supplementary assignment',
                         'task_interactions':'interaction requirements','delivery_review':'delivery review',
                         'targeted_repair':'targeted semantic revision'}.get(data.get('operation'), stage)
                print(f"model request #{call['index'] + 1}: {stage}, input {size} bytes, wait limit {self.config.timeout_seconds} s", file=sys.stderr, flush=True)
                try:
                    response = self.transport.post(self.config.base_url.rstrip("/") + "/chat/completions",
                                                   payload, key, self.config.timeout_seconds)
                    response = redact(response, self.credentials.values())
                    self.store.finish(call, response)
                    print(f"model request #{call['index'] + 1} received a response, validating", file=sys.stderr, flush=True)
                except ProviderError as exc:
                    self.store.record_error(call, exc)
                    if exc.uncertain:
                        detail = encoded(exc.diagnostics).decode() if exc.diagnostics else 'transport provided no detailed diagnostics'
                        raise UncertainRequest("model request outcome is uncertain; diagnostics=" + detail + "; the budget reservation is kept; verify before resuming") from exc
                    self.store.fail(call, "ProviderError")
                    raise
            self.store.check_token_budget()
            check_output_limit(response, output_limit, self.config.output_token_parameter, self.store, request_key)
            content = ""
            try:
                choice = response["choices"][0]
                if (is_extraction or 'incoming_assets' in data or data.get('operation')=='protocol_index') and choice.get('finish_reason') == 'length':
                    raise OutputTooLarge('response truncated; shrink the source text or candidate batch')
                if choice.get("finish_reason") != "stop":
                    raise ValidationError("model output did not finish normally; possibly truncated or refused")
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValidationError("model response is not text")
                if is_extraction and len(content) > self.config.extraction_max_chars:
                    raise OutputTooLarge('extraction output exceeds the character limit; the source text must be split')
                if 'incoming_assets' in data and len(content) > 16000:
                    raise OutputTooLarge('merge decisions exceed the character limit; shrink the candidate batch')
                value = json.loads(content)
                # Reject non-standard JSON numbers such as NaN/Infinity.
                encoded(value)
                if is_extraction:
                    inventory = value.get('inventory',value) if isinstance(value,dict) else value
                    if value in ({'needs_split': True},{'split_unit':True}) or (isinstance(inventory, dict) and isinstance(inventory.get('assets'), list)
                            and len(inventory['assets']) > self.config.extraction_max_assets):
                        raise OutputTooLarge('candidate assets exceed this extraction limit; the source text must be split')
                    if data.get('operation')=='local_extract':
                        value, paths = normalize_requirement_numbers(value)
                        if paths:
                            self.store.record_validation(request_key, {'kind':'numeric_requirement_format','paths':paths})
                    from .evidence_repair import fix_evidence
                    value = fix_evidence(value, data['sources'], self, request_key)
                validate(value)
                return value
            except EvidenceRepairError:
                raise
            except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
                if attempt == self.config.repair_attempts:
                    raise ValidationError("model output still violates the fixed format or evidence contract after the allowed number of repairs") from exc
                # Do not re-attach the whole invalid body each time, to avoid growing the context; the raw response is already archived.
                reason = str(exc) if isinstance(exc, ValidationError) else "invalid output format"
                messages.append({"role": "user", "content": "The previous output failed the fixed validation: " + reason + ". Output the complete object again."})
        raise AssertionError("unreachable")


def normalize_requirement_numbers(value):
    """Only normalizes how specification numbers are represented in the new workflow; never infers values or units or rewrites the raw response."""
    result = copy.deepcopy(value)
    inventory = result.get('inventory') if isinstance(result, dict) else None
    assets = inventory.get('assets') if isinstance(inventory, dict) else None
    paths = []
    if not isinstance(assets, list):
        return result, paths
    for asset_index, asset in enumerate(assets):
        requirements = asset.get('requirements') if isinstance(asset, dict) else None
        if not isinstance(requirements, list):
            continue
        for field_index, field in enumerate(requirements):
            if isinstance(field, dict) and type(field.get('value')) in (int, float):
                field['value'] = json.dumps(field['value'], allow_nan=False)
                paths.append(['inventory', 'assets', asset_index, 'requirements', field_index, 'value'])
    return result, paths


def model_messages(config, instruction, data):
    return [{"role": "system", "content": SYSTEM + "\nTask scope: " + config.scope},
            {"role": "user", "content": instruction + "\nTask data follows:\n" + encoded(data).decode()}]


def valid_public_url_shape(value):
    if not isinstance(value, str):
        return False
    try:
        url = urlsplit(value)
        return url.scheme == "https" and bool(url.hostname) and not url.username and not url.password and not url.fragment
    except ValueError:
        return False


def normalize_search(response, query):
    sources, images = [], []
    for entry in response.get("results", []):
        if not isinstance(entry, dict) or not valid_public_url_shape(entry.get("url")):
            continue
        raw = entry.get("raw_content") or entry.get("content") or ""
        if not isinstance(raw, str):
            continue
        source_id = f"r{len(sources)}"
        sources.append({"id": source_id, "url": entry["url"], "title": str(entry.get("title", "")),
                        "content": raw[:12000], "content_truncated": len(raw) > 12000, "query": query})
        for image in entry.get("images", []):
            if isinstance(image, str):
                image = {"url": image}
            if isinstance(image, dict) and valid_public_url_shape(image.get("url")):
                images.append({"id": f"i{len(images)}", "url": image["url"], "source_id": source_id,
                               "description": str(image.get("description", "")), "association": "search_source_linked"})
    for image in response.get("images", []):
        if isinstance(image, str):
            image = {"url": image}
        if not isinstance(image, dict) or not valid_public_url_shape(image.get("url")):
            continue
        if any(i["url"] == image["url"] for i in images):
            continue
        linked = next((s["id"] for s in sources if s["url"] == image.get("source_url")), None)
        images.append({"id": f"i{len(images)}", "url": image["url"], "source_id": linked,
                       "description": str(image.get("description", "")),
                       "association": "search_source_linked" if linked else "unverified"})
    return {"sources": sources, "images": images[:30]}


class TavilySearch:
    def __init__(self, config, credentials, store, transport):
        self.config, self.credentials, self.store, self.transport = config, credentials, store, transport
        self.endpoint = config.search_url

    def query(self, query):
        key = self.credentials.get("TAVILY_API_KEY")
        if not key:
            raise ProviderError("TAVILY_API_KEY is not configured; generating from requirements and marking reference material as missing")
        payload = {"query": query, "search_depth": "basic", "max_results": self.config.results_per_query,
                   "include_images": True, "include_image_descriptions": True, "include_raw_content": True,
                   "include_answer": False, "include_usage": True, "auto_parameters": False}
        request_key = digest({"kind": "search", "payload": payload})
        response = self.store.cached(request_key)
        if response is None:
            call = self.store.reserve(request_key, "search")
            try:
                response = self.transport.post(self.config.search_url, payload, key, self.config.timeout_seconds)
                response = redact(response, self.credentials.values())
                self.store.finish(call, response)
            except ProviderError as exc:
                self.store.record_error(call, exc)
                if exc.uncertain:
                    raise UncertainRequest("search request outcome is uncertain; the budget reservation is kept; check the provider records, then resume with --retry-uncertain") from exc
                self.store.fail(call, "ProviderError")
                raise
        try:
            return normalize_search(response, query)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("search service returned an invalid structure") from exc


def combine_references(batches):
    sources, images, warnings, search_records = [], [], [], []
    for batch in batches:
        warnings.extend(batch.get("warnings", []))
        search_records.extend(batch.get("search_records", []))
        remap = {}
        for source in batch["sources"]:
            existing = next((s for s in sources if s["url"] == source["url"]), None)
            new_id = existing["id"] if existing else f"r{len(sources)}"
            remap[source["id"]] = new_id
            if not existing:
                sources.append(dict(source, id=new_id))
            elif not existing["content"] and source["content"]:
                existing.update(dict(source, id=new_id))
        for image in batch["images"]:
            existing = next((i for i in images if i["url"] == image["url"]), None)
            linked = remap.get(image["source_id"])
            if not existing:
                images.append(dict(image, id=f"i{len(images)}", source_id=linked,
                                   source_ids=[linked] if linked else []))
            elif linked and linked not in existing["source_ids"]:
                existing["source_ids"].append(linked)
                if existing["source_id"] is None:
                    existing.update(source_id=linked, association=image["association"])
    return {"sources": sources, "images": images, "warnings": list(dict.fromkeys(warnings)),
            "search_records": search_records}
