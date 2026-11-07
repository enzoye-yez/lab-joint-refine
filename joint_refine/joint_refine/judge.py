"""Judge: compare round k with round k+1 per task and per finding.

Task transitions: success->success, fail->success (fixed), success->fail (regressed), fail->fail,
and tasks that appear or disappear because the generator changed interfaces.
Finding outcomes, for every finding that round k sent (non-method):
  resolved    every blocked task that still exists now succeeds,
  improved    some of them succeed or fail later in the pipeline (a different, non-scene category),
  persisting  the same finding id is diagnosed again,
  changed     the tasks still fail, but under another scene finding,
  vanished    none of its tasks exist any more (interfaces removed or renamed).
"""
import collections
import json
from pathlib import Path

from .common import read, write


def _results(round_dir):
    path = Path(round_dir) / "trajectory/results.jsonl"
    with path.open() as f:
        return {r["task"]: r for r in map(json.loads, f)}


def _finding_of(findings):
    owner = {}
    for f in findings:
        for t in f["blocked_tasks"]:
            owner.setdefault(t, f)
    return owner


def judge(previous_dir, current_dir):
    before, after = _results(previous_dir), _results(current_dir)
    transitions = collections.Counter()
    rows = []
    for task in sorted(set(before) | set(after)):
        a, b = before.get(task), after.get(task)
        if a is None:
            state = "new_success" if b["success"] else "new_fail"
        elif b is None:
            state = "removed"
        else:
            state = {(True, True): "kept_success", (False, True): "fixed", (True, False): "regressed",
                     (False, False): "still_fail"}[(a["success"], b["success"])]
        transitions[state] += 1
        rows.append({"task": task, "state": state, "before": a and a["reason"], "after": b and b["reason"]})

    sent = [f for f in read(Path(previous_dir) / "diagnosis/findings.json")["findings"] if f["attribution"] != "method"]
    now = read(Path(current_dir) / "diagnosis/findings.json")["findings"]
    now_ids, owner = {f["id"] for f in now}, _finding_of(now)
    outcomes = []
    for f in sent:
        existing = [t for t in f["blocked_tasks"] if t in after]
        succeeded = [t for t in existing if after[t]["success"]]
        if not existing:
            outcome = "vanished"
        elif len(succeeded) == len(existing):
            outcome = "resolved"
        elif f["id"] in now_ids:
            outcome = "persisting"
        elif succeeded or all(owner.get(t, {}).get("attribution", "method") == "method" for t in existing if t not in succeeded):
            outcome = "improved"
        else:
            outcome = "changed"
        outcomes.append({"finding": f["id"], "attribution": f["attribution"], "kind": f["kind"], "instance": f["instance"],
                         "outcome": outcome, "blocked": len(f["blocked_tasks"]), "existing": len(existing),
                         "succeeded": len(succeeded),
                         "now_under": sorted({owner[t]["id"] for t in existing if t in owner and t not in succeeded})[:5]})
    summary = {"transitions": dict(transitions),
               "success_before": sum(r["success"] for r in before.values()), "success_after": sum(r["success"] for r in after.values()),
               "tasks_before": len(before), "tasks_after": len(after),
               "findings": dict(collections.Counter(o["outcome"] for o in outcomes)),
               "new_findings": sorted(now_ids - {f["id"] for f in sent} - {f["id"] for f in now if f["attribution"] == "method"})}
    out = Path(current_dir) / "judge"
    write(out / "task_transitions.json", rows)
    write(out / "finding_outcomes.json", outcomes)
    write(out / "summary.json", summary)
    return summary
