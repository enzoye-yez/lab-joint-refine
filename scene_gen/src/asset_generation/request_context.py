"""Bounded request views. Original requirements, source and evidence stay immutable.

Developer-authored infrastructure; summaries are not acceptance evidence.
"""
import json
import re
import zipfile
from pathlib import Path

from asset_prep.storage import encoded, digest, atomic_json, safe_path
from asset_prep.errors import ContextLimitError, ValidationError, StaleEvidence

POLICY = """
_context_encoding="rows_v1" is a lossless list representation: columns are the field names, and rows hold every value of each record in the same column order;
no interface or value has been removed. items of literal_v1 hold the original objects with the same field names. Coordinates, units and constraints keep their original definitions.
Context management protocol: context_ref points to the complete read-only materials bound to this input. omitted=true means not expanded;
it never means a check passed, there is no defect, or a requirement may be ignored. When you need details, do not output the final task result; return only:
{"context_read":{"reference":"original context_ref value","path":[],"offset":0,"limit":8}}
path is a sequence of field names or list indices, one per level; it can locate a specific interface, configuration or check row. Lists/objects are paged by entry,
strings by character (limit at most 32768); next_offset is the next page. context_page is the current page; context_pages keeps recently read raw pages within budget, so you can compare across pages directly without re-requesting; older pages that were evicted can still be located again.
When reading the next page you may also return context_notes:[{"reference":"known read-only reference","text":"summary needed for later decisions"}],
which replaces the working notes (at most 32 notes, 8192 bytes in total) rather than appending to a chat log. Omitting it keeps the existing notes.
Notes are clues summarized by the model itself and not certified by the checker; they cannot override numeric facts, unresolved items or fixed constraints.
Re-reading old pages from the cache for comparison is allowed. Reading materials shares the call and token budget with the real task; the read window is for consolidation, not a cap on the total count. context_read_progress will prompt you to consolidate or to stop reads that make no progress. Once the materials suffice, return the result under the original task contract;
if necessary evidence is still missing, keep the uncertainty explicit; never treat a reference as already read or verified.
"""
DELIVERY_POLICY = """
context_delivery describes which part of the original materials and images each batch delivers; every page still belongs to the same task.
While remaining_batches>0 do not submit a final design or acceptance; return only {"context_notes":[{"reference":"currently valid read-only reference","text":"requirements, defects and corresponding image numbers to keep across pages"}]}.
Submit the result under the original task contract only on the last page; images not seen and verbatim references cannot be treated as verified. Notes are only working memory; the original evidence is unchanged.
When you need to compare images across pages you may return {"context_images":[image index,...],"context_notes":[...]},
with indices from image_catalog; at most per_request_limit per request, and images are never deleted. Other materials can still be paged verbatim with context_read.
"""

# These are decision/validation anchors, never byte-truncated or replaced by a summary.
PINNED = {
    "job", "plan",
    "profile", "contract", "fixed_contract", "environment", "current_model_py",
    "instance_contract", "scene_layout", "repair_directive", "reference_policy",
    "table_edit_policy", "table_modifications", "edge_mount_policy", "edge_mounts",
    "allowed_jobs", "allowed_instances", "review_scope", "scope",
    "deferred_checks", "unresolved_checks", "questions", "capabilities",
    "review_format_correction", "authoring_capabilities", "task_recovery",
    "layout_review_targets", "evidence_not_resent",
}
MATERIAL = {"input", "source", "protocol", "source_requirements"}
FACTS = {
    "id", "status", "complete", "instances", "kind", "configuration", "unit",
    "component_scope", "tolerance_m", "measurement_error_count", "unexamined_pairs",
    "collision_count", "disconnected_count", "version", "name", "body", "type",
}
HISTORIES = {"previous_evidence", "supplemental_evidence", "recovery_history"}



def pack_tables(value):
    """Factor repeated field names, never values. Decoding must equal the original."""
    if isinstance(value, dict):
        if "_context_encoding" in value:
            return {"_context_encoding": "literal_v1",
                    "items": [[k, pack_tables(v)] for k, v in value.items()]}
        return {k: pack_tables(v) for k, v in value.items()}
    if isinstance(value, list):
        ordinary = [pack_tables(v) for v in value]
        if len(value) >= 6 and all(isinstance(v, dict) for v in value):
            columns = sorted(value[0])
            if columns and all(set(v) == set(columns) and "_context_encoding" not in v for v in value):
                packed = {"_context_encoding": "rows_v1", "columns": columns,
                          "rows": [[pack_tables(v[k]) for k in columns] for v in value]}
                if len(encoded(packed)) < len(encoded(ordinary)):
                    return packed
        return ordinary
    return value


def unpack_tables(value):
    if isinstance(value, dict):
        if value.get("_context_encoding") == "rows_v1":
            return [dict(zip(value["columns"], [unpack_tables(v) for v in row])) for row in value["rows"]]
        if value.get("_context_encoding") == "literal_v1":
            return {k: unpack_tables(v) for k, v in value["items"]}
        return {k: unpack_tables(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack_tables(v) for v in value]
    return value

def messages_for(instruction, context, images):
    content = [{"type": "text", "text": encoded(context).decode("utf-8")}]
    for im in images:
        content.append({"type": "text", "text": "Image: " + im["path"]})
        content.append({"type": "image_url", "image_url": {"url": im["data_url"], "detail": "high"}})
    return [{"role": "system", "content": instruction}, {"role": "user", "content": content}]


def measure(instruction, context, images, config):
    messages = messages_for(instruction, context, images)
    text_bytes = len(encoded(messages)) - sum(len(im["data_url"].encode("utf-8")) for im in images)
    image_estimate = len(images) * 16384
    return {
        "estimator": "utf8_bytes_plus_fixed_image_allowance_v1",
        "exact_token_count": False,
        "text_and_structure_bytes": text_bytes,
        "image_count": len(images), "image_allowance": image_estimate,
        "input_upper_estimate": text_bytes + image_estimate,
        "output": config.max_output_tokens, "margin": config.context_margin_tokens,
        "required": text_bytes + image_estimate + config.max_output_tokens + config.context_margin_tokens,
        "limit": config.max_context_tokens,
        "context_fields_bytes": {k: len(encoded(v)) for k, v in context.items()},
    }


def reference_bytes(root, key):
    """Historical packs are immutable; current receipts use a file overlay."""
    path = safe_path(root, key + '.json')
    if path.is_file():
        return path.read_bytes()
    archive = safe_path(Path(root).parent, Path(root).name + '.zip')
    if archive.is_file():
        with zipfile.ZipFile(archive) as packed:
            try:return packed.read(key + '.json')
            except KeyError:pass
    raise FileNotFoundError(path)


class RequestContext:
    def __init__(self, root, identity, instruction, context, images, phase, config):
        self.config, self.phase = config, phase
        self.original = context
        self.binding = digest({"identity": identity, "phase": phase, "instruction": instruction,
                               "context": context,
                               "images": [(i["path"], digest(i["data_url"])) for i in images]})
        self.root = safe_path(root, "request_context/" + self.binding)
        self.refs = set()
        # Remove identical pixels only; retain every path label in the receipt and prompt.
        self.images, seen, self.aliases = [], {}, {}
        for im in images:
            key = digest(im["data_url"])
            if key not in seen:
                seen[key] = im["path"]
                self.images.append(im)
            self.aliases.setdefault(seen[key], []).append(im["path"])
        self.before = measure(instruction, context, images, config)
        self.raw_instruction = instruction
        self.instruction = instruction
        self.context = context
        self.projected = False
        self.source_ref = self.reference(context)
        self.receipt = {"binding": self.binding, "phase": phase, "source_ref": self.source_ref,
                        "before": self.before, "image_aliases": self.aliases, "reads": []}
        self.materials = []
        self.all_images, self.images = self.images, []
        self._prepare()
        self._delivery()
        self.save("prepared")

    def reference(self, value):
        key = digest(value)
        path = safe_path(self.root, key + ".json")
        try:
            previous = reference_bytes(self.root, key)
        except FileNotFoundError:
            atomic_json(path, value)
        else:
            if digest(json.loads(previous)) != key:
                raise StaleEvidence("Original context archive was modified")
        self.refs.add(key)
        return key

    def ref(self, value, pending=None):
        if pending is None:
            key = self.reference(value)
        else:
            key = digest(value)
            pending[key] = value
        result = {"context_ref": key, "omitted": True,
                  "type": type(value).__name__, "count": len(value) if hasattr(value, "__len__") else 1}
        if isinstance(value, dict):
            result["fields"] = list(value)[:32]
            result["field_count"] = len(value)
            result.update({k: v for k, v in value.items() if k in FACTS and len(encoded(v)) <= 2048})
        return result

    def fold(self, value, budget):
        # Recursive projections are speculative. Only retained references are
        # published; the complete original input is already archived separately.
        pending = {}
        result = self._fold(value, budget, pending)
        def publish(item):
            if isinstance(item, dict):
                key = item.get('context_ref')
                if isinstance(key,str) and key in pending:
                    self.reference(pending.pop(key))
                for child in item.values():publish(child)
            elif isinstance(item,list):
                for child in item:publish(child)
        publish(result)
        return result

    def _fold(self, value, budget, pending):
        if len(encoded(value)) <= budget:
            return value
        if isinstance(value, dict):
            result = {}
            share = max(180, budget // max(1, len(value)))
            for key, child in value.items():
                if key in FACTS:
                    result[key] = child
                else:
                    result[key] = self._fold(child, share, pending)
            if len(encoded(result)) <= budget:
                return result
        elif isinstance(value, list) and len(value) <= 512:
            rows = [self._fold(child, max(180, budget // max(1, len(value))), pending) for child in value]
            if len(encoded(rows)) <= budget:
                return rows
        return self.ref(value, pending)

    def check_index(self, value, prefix=""):
        """All check identities/statuses stay inline, independent of detail volume."""
        rows = []
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "checks" and isinstance(child, list):
                    for r in child:
                        if isinstance(r, dict) and "id" in r:
                            rows.append({**{k: v for k, v in r.items() if k in FACTS},
                                         "location": prefix + "/checks",
                                         "context_ref": self.reference(r)})
                elif key not in HISTORIES and key != "validation_feedback":
                    rows.extend(self.check_index(child, prefix + "/" + str(key)))
        return rows

    def current_history(self, entries):
        """Request/outcome index replaces repeated reports; original history remains addressable."""
        rows = []
        for entry in entries:
            if not isinstance(entry, dict):
                rows.append(self.ref(entry))
                continue
            row = {k: v for k, v in entry.items() if k in {
                "status", "input_key", "output_key", "problem_key", "remaining_problem",
                "action_key", "cycle", "outcome", "scene", "request", "error", "complete"}}
            if "action" in entry:
                row["action"] = self.fold(entry["action"], 4096)
            results = entry.get("results", [])
            if isinstance(results, list):
                row["observations"] = [
                    {**{k: r[k] for k in ("request", "status", "complete", "error", "quality") if k in r},
                     "context_ref": self.reference(r)}
                    for r in results if isinstance(r, dict)]
            row["context_ref"] = self.reference(entry)
            rows.append(row)
        return {"current_action_index": rows, "history_ref": self.reference(entries),
                "policy": "Current checks and unresolved_checks are the current state; older observations are kept for provenance, and full reports are not cumulatively resent."}

    def dependency_snapshot(self, dependencies):
        """Keep actual interfaces/constraints/poses inline; bulk body detail is queryable."""
        result = {}
        for name, dep in dependencies.items():
            if not isinstance(dep, dict):
                result[name] = dep
                continue
            geometry = dep.get("geometry", {})
            result[name] = {k: v for k, v in dep.items() if k != "geometry"}
            if isinstance(geometry, dict):
                result[name]["geometry"] = {
                    k: self.ref(v) if k == "bodies" and len(encoded(v)) > 2048 else v
                    for k, v in geometry.items()}
            else:
                result[name]["geometry"] = geometry
        return pack_tables(result)

    def material_view(self, value, budget, location):
        """Page large original material, not a model-authored replacement summary."""
        if len(encoded(value)) <= budget:
            return value
        if isinstance(value, dict):
            return {k:self.material_view(v, max(2048, budget//max(1,len(value))), location+"/"+k)
                    for k,v in value.items()}
        # JSON text is an exact reversible representation, including ordered lists.
        text = value if isinstance(value, str) else encoded(value).decode("utf-8")
        reference = self.reference(text)
        if not any(d["reference"]==reference for d in self.materials):
            self.materials.append({"reference":reference, "location":location, "characters":len(text),
                                   "encoding":"text" if isinstance(value,str) else "json"})
        return {**self.ref(value), "required_material":reference, "location":location,
                "policy":"Original text delivered in batches; no requirement removed; do not submit the final result before all materials are read"}

    def anchor_view(self, value, location):
        """Keep object ids, poses and numeric contracts; page long narrative requirements."""
        if isinstance(value, dict):
            return {k:(self.material_view(v, 8192, location+"/"+k)
                       if k in {"requirements","source_requirements","description"} else
                       self.anchor_view(v, location+"/"+k)) for k,v in value.items()}
        if isinstance(value, list):
            return [self.anchor_view(v, location+"/"+str(i)) for i,v in enumerate(value)]
        return value

    def _prepare(self):
        reduced = dict(self.original)
        if any(len(paths) > 1 for paths in self.aliases.values()):
            reduced["image_aliases"] = self.aliases
        for field in HISTORIES:
            if isinstance(reduced.get(field), list) and reduced[field]:
                reduced[field] = self.current_history(reduced[field])
        if isinstance(reduced.get("validation_feedback"), list) and reduced["validation_feedback"]:
            records = reduced["validation_feedback"]
            reduced["validation_feedback"] = {
                "latest": records[-1],
                "attempts": [{k: v for k, v in r.items() if k in {"error", "attempt", "reason", "request"}}
                             for r in records if isinstance(r, dict)],
                "history_ref": self.reference(records)}
        changed = reduced != self.original
        # The shared projection applies to planning, authoring, review and repair.
        if (not changed and len(encoded(reduced)) <= self.config.review_context_bytes
                and measure(self.instruction, reduced, self.images, self.config)["required"] <= self.config.max_context_tokens):
            self.context = reduced
            self._check()
            return
        self.projected = True
        self.instruction = self.raw_instruction + POLICY
        fixed = {k: v for k, v in reduced.items() if k in PINNED}
        other = {k: v for k, v in reduced.items() if k not in PINNED}
        for field in ("plan","job"):
            if field in fixed:
                fixed[field] = self.anchor_view(fixed[field], field)
        for field in sorted(MATERIAL & set(other)):
            fixed[field] = self.material_view(other.pop(field), 24000, field)
        if isinstance(other.get("dependencies"), dict):
            snapshot = self.dependency_snapshot(other.pop("dependencies"))
            fixed["dependencies"] = (snapshot if len(encoded(snapshot)) <= 60000
                                     else pack_tables(self.material_view(unpack_tables(snapshot), 60000, "dependencies")))
        index = self.check_index({k: v for k, v in reduced.items() if k not in PINNED})
        fixed["_context"] = {"binding": self.binding, "source_ref": self.source_ref,
                             "read_protocol": "context_read", "check_index": index,
                             "required_materials":self.materials}
        # Optional detail allocation adapts locally; mandatory constraints never shrink.
        for budget in (self.config.review_context_bytes, self.config.review_context_bytes // 2,
                       self.config.review_context_bytes // 4, 4096):
            projected = {k: self.fold(v, max(512, budget // max(1, len(other)))) for k, v in other.items()}
            self.context = {**fixed, **projected}
            if measure(self.instruction, self.context, self.images, self.config)["required"] <= self.config.max_context_tokens:
                self._check()
                return
        self._check()

    def _check(self):
        self.after = measure(self.instruction, self.context, self.images, self.config)
        if self.after["required"] > self.config.max_context_tokens:
            self.save("blocked_mandatory_context_or_images")
            raise ContextLimitError(
                "Mandatory context or complete visual evidence still exceeds the budget; itemized diagnostics saved; no images removed, no constraints or source truncated; not sent")
        # Match the exact provider envelope, not only the context estimate.
        return self.after

    def _delivery(self):
        """Bound transport pages independently of the immutable evidence archive."""
        self.base_context = self.context
        self.delivery_index = 0
        self.material_pages = []
        for document in self.materials:
            offset = 0
            while offset < document["characters"]:
                page = self.read({"reference":document["reference"], "offset":offset, "limit":32768},
                                 budget=40000)
                self.material_pages.append({**page, "location":document["location"],
                                            "encoding":document["encoding"]})
                offset = page["next_offset"] or document["characters"]
        # Several small dependency documents should share one request, not each
        # consume a model call. Page identities and offsets remain independent.
        self.material_batches = []
        batch = []
        for page in self.material_pages:
            if batch and len(encoded(batch+[page]))>40000:
                self.material_batches.append(batch)
                batch = []
            batch.append(page)
        if batch:
            self.material_batches.append(batch)
        self.image_pages = []
        batch = []
        whole_fits = (not self.material_pages and len(self.all_images)<=self.config.max_images
                      and measure(self.instruction,self.context,self.all_images,self.config)["required"]<=self.config.max_context_tokens
                      and len(encoded(messages_for(self.instruction,self.context,self.all_images)))+512<=self.config.max_request_bytes)
        # Reserve room for notes, a detail page and delivery metadata. Raw envelope
        # size is checked too; token estimates alone cannot bound base64 transport.
        for im in ([] if whole_fits else self.all_images):
            proposed = batch + [im]
            required = measure(self.instruction+DELIVERY_POLICY, self.context, proposed, self.config)["required"]
            size = len(encoded(messages_for(self.instruction+DELIVERY_POLICY, self.context, proposed)))
            if batch and (len(proposed)>self.config.max_images or required+65536>self.config.max_context_tokens
                          or size+65536>self.config.max_request_bytes):
                self.image_pages.append(batch)
                batch = []
            batch.append(im)
            if (measure(self.instruction+DELIVERY_POLICY,self.context,batch,self.config)["required"]+65536>self.config.max_context_tokens
                    or len(encoded(messages_for(self.instruction+DELIVERY_POLICY,self.context,batch)))+65536>self.config.max_request_bytes):
                raise ContextLimitError("A single original image plus mandatory context cannot fit; evidence kept; this cannot be bypassed by removing images or lowering acceptance")
        if batch:
            self.image_pages.append(batch)
        if whole_fits and self.all_images:
            self.image_pages = [self.all_images]
        self.delivery_count = max(1,len(self.image_pages),len(self.material_batches))
        self.paged = self.delivery_count>1 or bool(self.material_pages)
        self.receipt["delivery"] = {"batches":self.delivery_count, "material_pages":len(self.material_pages),
                                  "material_batches":len(self.material_batches),
                                  "image_pages":[[i["path"] for i in p] for p in self.image_pages]}
        if self.paged:
            self.instruction += POLICY if POLICY not in self.instruction else ""
            self.instruction += DELIVERY_POLICY
        self.set_delivery()

    def set_delivery(self):
        n = self.delivery_index
        self.images = self.image_pages[n] if n<len(self.image_pages) else []
        self.context = dict(self.base_context)
        if self.paged:
            self.context["context_delivery"] = {
                "batch":n, "remaining_batches":self.delivery_count-n-1,
                "per_request_limit":self.config.max_images, "source_ref":self.source_ref,
                "image_catalog":[{"index":i,"path":im["path"],"sha256":im.get("sha256")}
                                 for i,im in enumerate(self.all_images)],
                "current_images":[im["path"] for im in self.images],
                "material_pages":self.material_batches[n] if n<len(self.material_batches) else []}
        self._check()

    def select_images(self, indices):
        if (not isinstance(indices,list) or not 0<len(indices)<=self.config.max_images
                or any(type(i) is not int or not 0<=i<len(self.all_images) for i in indices)
                or len(set(indices))!=len(indices)):
            raise ValidationError("context_images must be non-repeating, bounded indices within the current image list")
        selected = [self.all_images[i] for i in indices]
        if (measure(self.instruction,self.context,selected,self.config)["required"]+49152>self.config.max_context_tokens
                or len(encoded(messages_for(self.instruction,self.context,selected)))+49152>self.config.max_request_bytes):
            raise ValidationError("Selected comparison images are too large; shrink this comparison set; all original images are still kept")
        self.images = selected
        self.context = {**self.context, "context_image_selection":indices}
        if "context_delivery" in self.context:
            self.context["context_delivery"] = {**self.context["context_delivery"],
                                                "current_images":[im["path"] for im in selected]}

    def save(self, status):
        self.receipt.update(status=status, projected=self.projected,
                            after=measure(self.instruction, self.context, self.images, self.config))
        atomic_json(self.root / "receipt.json", self.receipt)

    def notes(self, value):
        if not isinstance(value, list) or len(value) > 32 or len(encoded(value)) > 8192:
            raise ValidationError("context_notes must be working state of at most 32 notes and 8192 bytes")
        for note in value:
            if (not isinstance(note, dict) or set(note) != {"reference", "text"}
                    or not isinstance(note["reference"], str) or note["reference"] not in self.refs
                    or not isinstance(note["text"], str) or not 0 < len(note["text"]) <= 2048):
                raise ValidationError("context_notes must reference this request's read-only materials and carry bounded text")
        return {"provenance": "model_working_notes_not_verified_evidence", "items": value}

    def read(self, request, budget=32768):
        if not isinstance(request, dict) or not {"reference"} <= set(request) <= {"reference", "path", "offset", "limit"}:
            raise ValidationError("Invalid context_read fields")
        key = request["reference"]
        if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key) or key not in self.refs:
            raise ValidationError("context_read reference does not belong to the current request")
        value = json.loads(reference_bytes(self.root, key))
        if digest(value) != key:
            raise StaleEvidence("Original text of the context detail has changed")
        parts = request.get("path", [])
        if not isinstance(parts, list) or len(parts) > 32:
            raise ValidationError("Invalid context_read path")
        try:
            for part in parts:
                if isinstance(value, dict) and isinstance(part, str):
                    value = value[part]
                elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                    value = value[part]
                else:
                    raise KeyError()
        except (KeyError, IndexError):
            raise ValidationError("context_read path does not exist") from None
        start, limit = request.get("offset", 0), request.get("limit", 8)
        if type(start) is not int or start < 0 or type(limit) is not int or not 1 <= limit <= 32768:
            raise ValidationError("Invalid context_read paging")
        if isinstance(value, str):
            items, unit = value, "characters"
        elif isinstance(value, dict):
            items, unit = [{"field": k, "value": v} for k, v in value.items()], "fields"
        elif isinstance(value, list):
            items, unit = value, "items"
        else:
            items, unit = [value], "items"
        if budget < 1024:
            raise ContextLimitError("Current request cannot fit the detail page; original text and constraints kept; not sent")
        limit = min(limit, 16) if unit != "characters" else limit
        while True:
            end = min(len(items), start + limit)
            data = (items[start:end] if unit == "characters"
                    else [self.fold(v, max(256, (budget - 512) // max(1, limit))) for v in items[start:end]])
            page = {"reference": key, "path": parts, "offset": start, "unit": unit, "total": len(items),
                    "next_offset": end if end < len(items) else None, "data": data}
            if len(encoded(page)) <= budget:
                return page
            if limit <= 1:
                raise ContextLimitError("Even the index of a single detail entry exceeds the remaining budget; not sent")
            limit = max(1, limit // 2)
