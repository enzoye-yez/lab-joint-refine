"""Task types: tasks that are the same operation on repeated elements (the 96 tips of a rack, the 24 clips
of a holder, twelve tubes of one kind) are one type. The benchmark is scored per type, each type weighted
equally, so a rack of identical wells counts once, not 96 times."""
import re

GRID = re.compile(r"(?<![a-z])[A-H]\d{1,2}(?![a-z])")     # well / slot labels such as A01, H12
NUMBER = re.compile(r"\d+")


def task_type(interface_id, instance, skill):
    """(instance kind, skill, interface template): indices and well labels replaced by '#'."""
    local = interface_id.split("__", 1)[1] if "__" in interface_id else interface_id
    local = NUMBER.sub("#", GRID.sub("#", local))
    return f"{NUMBER.sub('#', instance)}|{skill}|{local}"


def representatives(ids, per_type):
    """Up to `per_type` members of each type: the first and the last (the ends of a row or grid),
    then evenly between them; deterministic."""
    ids = sorted(ids)
    if per_type <= 1 or len(ids) <= per_type:
        return ids[:max(per_type, 1)] if per_type <= 1 else ids
    step = (len(ids) - 1) / (per_type - 1)
    return sorted({ids[round(k * step)] for k in range(per_type)})


def type_scores(results):
    """{type: (successes, tasks)} from benchmark result rows (each with task, instance, skill, success)."""
    scores = {}
    for r in results:
        key = r.get("type") or task_type(r["task"], r.get("instance", ""), r.get("skill", ""))
        done, total = scores.get(key, (0, 0))
        scores[key] = (done + bool(r.get("success")), total + 1)
    return scores


def summary(results):
    """Type-level numbers: types, types with a success, mean per-type success rate (each type weight 1)."""
    scores = type_scores(results)
    if not scores:
        return {"types": 0, "types_solved": 0, "type_mean": 0.0}
    return {"types": len(scores), "types_solved": sum(1 for d, _ in scores.values() if d),
            "type_mean": sum(d / t for d, t in scores.values()) / len(scores)}
