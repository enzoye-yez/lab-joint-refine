"""Persistent evidence ledger: measured facts are kept separate from model interpretation; never relies on the model's self-assessed information value."""
from copy import deepcopy
from pathlib import Path
from asset_prep.storage import atomic_json, file_hash
from .common import OperationError, need, digest


class EvidenceLedger:
    def __init__(self, state, settings, save=lambda: None, archive_root=None):
        self.state = state
        self.settings, self.save = settings, save
        self.archive_root=Path(archive_root) if archive_root else None
        for key, default in (("records", {}), ("attempts", []), ("sim_seconds_reserved", 0.0),
                             ("stalls", {})):
            state.setdefault(key, default)

    def acquire(self, query, binding, measure, *, sim_seconds=0.0):
        key = digest({"query": query, "binding": binding})
        records = self.state["records"]
        if key in records:
            row = records[key]
            self.validate_archive(row)
            return {**deepcopy(row), "delivery": "reused", "information_gain": False}
        for row in records.values():
            if row["status"] == "valid" and row["binding"] == binding and covers(row["query"], query):
                self.validate_archive(row)
                return {**deepcopy(row), "delivery": "covered_by_existing", "information_gain": False,
                        "requested_query": deepcopy(query)}
        if binding.get("local_geometry_sha256"):
            for previous in list(records.values()):
                same={k:v for k,v in previous["binding"].items() if k!="scene_sha256"}=={k:v for k,v in binding.items() if k!="scene_sha256"}
                if previous["status"]=="valid" and previous["query"]==query and same:
                    self.validate_archive(previous)
                    rebound={**deepcopy(previous),"id":key,"binding":deepcopy(binding),"rebound_from":previous["id"],
                             "information_gain":False,"sim_seconds_reserved":0}
                    records[key]=rebound;self.save()
                    return {**deepcopy(rebound),"delivery":"local_dependency_reused"}
        spent=sum("rebound_from" not in r for r in records.values())
        need(spent < self.settings.diagnostic_queries, "Diagnostic measurement budget exhausted", "diagnostic_budget")
        need(self.state["sim_seconds_reserved"] + sim_seconds <= self.settings.diagnostic_total_sim_seconds,
             "Local probe simulation budget exhausted", "diagnostic_budget")
        self.state["sim_seconds_reserved"] += sim_seconds
        row = {"id": key, "query": deepcopy(query), "binding": deepcopy(binding),
               "status": "interrupted", "result": None, "information_gain": False,
               "sim_seconds_reserved": sim_seconds}
        records[key] = row
        self.save()  # Interrupted requests are not retried automatically; reserved spend must not be reset on resume.
        try:
            result = measure()
            need(isinstance(result, dict) and result.get("status") in {"valid", "unknown", "invalid"},
                 "Measurement tool returned a malformed result", "measurement_error")
            row["status"], row["result"] = result["status"], result
            # New question wording, numbering and return paths are not information; the coverage itself is valid measurement content.
            facts = {k: result.get(k) for k in ("observations", "coverage", "limitations", "rows")}
            def semantic(value):
                if isinstance(value, dict): return {k: semantic(v) for k, v in value.items() if k not in {"directory", "source"}}
                if isinstance(value, list): return [semantic(v) for v in value]
                return value
            fingerprint = digest(semantic(facts))
            same_context = [r for k, r in records.items() if k != key and r["binding"] == binding]
            row["information_gain"] = (row["status"] == "valid" and
                                        not any(r.get("fingerprint") == fingerprint for r in same_context))
            row["fingerprint"] = fingerprint
            if self.archive_root:
                path=self.archive_root/(key+".json")
                atomic_json(path,result)
                row["result_reference"]={"path":str(path.resolve()),"sha256":file_hash(path)}
                row["result"]=deepcopy(result)
                window=row["result"].get("observations",{}).get("window")
                if isinstance(window,list) and len(window)>6:
                    row["result"]["observations"].update(window=window[:3]+window[-3:],window_total=len(window),window_preview_only=True)
        except OperationError as exc:
            if exc.code in {"stale_scene", "stale_evidence"}:
                self.save()
                raise
            row.update(status="invalid", result={"status": "invalid", "code": exc.code,
                       "detail": exc.detail, "repair_authorized": False})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row.update(status="invalid", result={"status": "invalid", "code": "measurement_error",
                       "detail": str(exc), "repair_authorized": False})
        self.save()
        return {**deepcopy(row), "delivery": "measured"}

    def validate_archive(self, row):
        reference=row.get("result_reference")
        if not reference:return
        path=Path(reference["path"])
        need(self.archive_root is not None and path.resolve().parent==self.archive_root.resolve()
             and path.is_file() and file_hash(path)==reference["sha256"],
             "Existing evidence archive is missing or modified; cannot be reused silently", "stale_evidence")

    def finish_batch(self, binding, records):
        context = digest(binding)
        gain = any(r.get("information_gain") for r in records)
        count = 0 if gain else self.state["stalls"].get(context, 0) + 1
        self.state["stalls"][context] = count
        receipt = {"context": context, "evidence_ids": [r["id"] for r in records],
                   "information_gain": gain, "consecutive_stalls": count,
                   "exhausted": count >= self.settings.diagnostic_stalls}
        # Request receipts are bounded too; cache hits never build an unbounded history.
        self.state["attempts"].append(receipt)
        self.state["attempts"] = self.state["attempts"][-self.settings.diagnostic_queries:]
        self.save()
        return receipt

    def current(self, scene_hash):
        return [deepcopy(r) for r in self.state["records"].values()
                if r["binding"].get("scene_sha256") == scene_hash]


def covers(existing, requested):
    """Reuse only provable coverage relations; never infer measurement equivalence from text similarity."""
    if not existing.get("kind") or existing.get("kind") != requested.get("kind"): return False
    kind = existing["kind"]
    field = {"mechanism": "samples", "fit": "samples", "contacts": "bodies", "joint_state": "joints", "trace": "window"}.get(kind)
    if field is None: return False
    if {k: v for k, v in existing.items() if k != field} != {k: v for k, v in requested.items() if k != field}: return False
    a, b = existing[field], requested[field]
    if field == "samples": return a >= b and (a-1) % (b-1) == 0
    if field == "window": return a >= b
    if kind == "contacts" and not a: return True
    return bool(b) and set(b) <= set(a)
