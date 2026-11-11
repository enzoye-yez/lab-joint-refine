"""Visual-review format retries are separate from geometry repair, sharing the original call ledger and budget."""

import json
from pathlib import Path

from asset_prep.errors import ValidationError, ContextLimitError
from asset_prep.storage import atomic_json
from .contracts import visual_result


LIST_LIMIT = 8
# A format correction resends only context fields up to this size; the evidence the reviewer already judged is not resent.
FORMAT_FIELD_BYTES = 16384


def compact_report(report, *, passed_detail=True):
    """A check report as a model reads it: list evidence capped at LIST_LIMIT entries (with the
    total kept), without the manifest and render list the model already has as files/images.
    passed_detail=False reduces passing rows to their id."""
    if not isinstance(report, dict):
        return report
    def row(r):
        if not passed_detail and r.get("status") == "pass":
            return {"id": r.get("id"), "status": "pass"}
        out = {}
        for k, v in r.items():
            if isinstance(v, list) and len(v) > LIST_LIMIT:
                out[k], out[k+"_total"] = v[:LIST_LIMIT], len(v)
            else:
                out[k] = v
        return out
    out = {k: v for k, v in report.items() if k not in {"checks", "manifest", "renders"}}
    out["checks"] = [row(r) for r in report.get("checks", []) if isinstance(r, dict)]
    return out


def author_feedback(feedback):
    """What an author acts on: confirmed defects, failing checks and motion requirements that are
    missing or wrong. A reviewer's unverified notes are the model's own unconstrained design
    choices, and deferrals belong to later stages; neither is a defect, so neither is sent."""
    if not isinstance(feedback, dict):
        return feedback
    out = dict(feedback)
    if isinstance(out.get("numerical"), dict):
        out["numerical"] = compact_report(out["numerical"], passed_detail=False)
    visual = out.get("visual")
    if isinstance(visual, dict):
        out["visual"] = {k: visual[k] for k in ("verdict", "findings", "source") if k in visual}
        gaps = [r for r in visual.get("motion_coverage") or [] if isinstance(r, dict) and r.get("verdict") != "ok"]
        if gaps:
            out["visual"]["motion_coverage"] = gaps
    return out


def retain_batch_defects(previous, current):
    """The final page decides, having been shown every earlier page's findings.

    A final repair keeps the union of all pages' findings; a final accept stands, and the
    earlier findings it overruled are returned for the call receipt. Final stage validation
    still checks ids, units and the full response; this grants no new edit rights.
    """
    if current.get("verdict") == "accept":
        return current, [f for v in previous for f in v.get("findings", [])]
    result = dict(current)
    reviews = previous + [current]
    result["verdict"] = "repair"
    for field in ("findings", "unverified", "affected_instances", "affected_layout"):
        values = [v[field] for v in reviews if field in v]
        if any(not isinstance(rows,list) or any(not isinstance(r,str) for r in rows) for rows in values):
            raise ValidationError("Invalid cross-image-batch review field: "+field)
        if values:
            result[field] = list(dict.fromkeys(r for rows in values for r in rows))
    blockers = result.pop("blocking", [])
    if not isinstance(blockers,list) or any(not isinstance(r,str) for r in blockers):
        raise ValidationError("Invalid cross-image-batch review field: blocking")
    result.setdefault("unverified", []).extend(blockers)
    return result, []


def format_request(context, correction):
    """The small contract fields (ids, scope, deferred checks) plus the rejected response and its error."""
    kept = {k: v for k, v in context.items()
            if len(json.dumps(v, ensure_ascii=False, default=str).encode()) <= FORMAT_FIELD_BYTES}
    return {**kept, "review_format_correction": correction, "evidence_not_resent": sorted(set(context) - set(kept))}


def ask_review(model, instruction, context, images, phase, destination, validator=visual_result, *, session=None, save=lambda:None):
    session = session if session is not None else {}
    correction = session.get("correction")
    last_call = session.get("call")
    if "review" in session:
        return validator(session["review"]), last_call
    for attempt in range(session.get("attempts",0),3):
        request = context if correction is None else format_request(context, correction)
        value = candidate = None
        last_call = None
        try:
            value, last_call = model.ask(instruction, request, images if correction is None else [], phase)
            candidate=value
            if isinstance(value,dict) and session.get("accepted_results"):
                # A small patch changes only answers supplied in this response.
                patch_only=set(value)=={"deferred_results"}
                candidate={**session.get("draft",{}),**value} if patch_only else dict(value)
                rows=value.get("deferred_results",[])
                if isinstance(rows,list):
                    replaced={r.get("id") for r in rows if isinstance(r,dict) and isinstance(r.get("id"),str)}
                    candidate["deferred_results"]=[r for r in session["accepted_results"] if r["id"] not in replaced]+rows
            review = validator(candidate)
        except ContextLimitError as exc:
            session["deferred_reason"] = str(exc); save()
            result = {"verdict":"uncertain","findings":[],"unverified":[str(exc)]}
            if validator is not visual_result:
                result["blocking"] = [str(exc)]
            return result, {**(last_call or {}),"task_deferred":str(exc)}
        except ValidationError as exc:
            session["attempts"]=attempt+1
            if hasattr(exc,"accepted_results"):
                session["accepted_results"]=exc.accepted_results
                session["draft"]=candidate
            correction = {"retained_results":session.get("accepted_results",[]),
                          "item_feedback":getattr(exc,"review_feedback",None),
                          "patch_rule":"Valid items for the same input version have been kept; you may return only a deferred_results patch to fix missing or wrong ids. Errors in any other field still require the complete corrected review.",
                          "attempt": attempt + 1, "error": str(exc), "previous_response": value,
                          "instruction": "This is a format correction only. Images and the evidence fields named in evidence_not_resent are not resent: keep the judgements of previous_response and change only what the error requires. Do not modify the asset. findings lists only defects that need repair;"
                          "do not put positive observations in findings. When accept, findings must be empty;"
                          "if there really is a defect, give repair; when evidence is insufficient follow the current stage's verdict and deferred-verification rules, and put unverified items in unverified."}
            atomic_json(Path(destination)/f"review_format_{attempt + 1}.json",
                        {"call": last_call, "response": value, "error": str(exc)})
            session.update(correction=correction,call=last_call); save()
            continue
        session.update(review=review,call=last_call,attempts=attempt+1); save()
        return review, last_call
    # Do not treat a contradictory response as a pass, and do not send format problems to the modeller.
    reason = "Visual review format invalid three times in a row; suspending this review and continuing other reliable tasks: " + (correction or {}).get("error","review contract not yet restored")
    result = {"verdict": "uncertain", "findings": [], "unverified": [reason]}
    if validator is not visual_result:
        result["blocking"] = [reason]
    if context.get("deferred_checks"):
        accepted={r["id"]:r for r in session.get("accepted_results",[])}
        result["deferred_results"]=[accepted.get(d["id"],{"id":d["id"],"status":"unresolved",
            "evidence":"framework record: review response not yet validly completed; resolution not certified"}) for d in context["deferred_checks"] if d.get("stage")=="scene"]
    return result, last_call
