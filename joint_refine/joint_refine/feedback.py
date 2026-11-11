"""Feedback Writer: measured findings -> the scene generator's planner-feedback message.

The LLM (Codex, gpt-6.1-sol) only merges and words the findings; every number it writes must come
from the findings file (checked below), and the instances must be scene plan instances. Findings
attributed to the trajectory method are never sent.
"""
import json
import re
from pathlib import Path

from . import llm
from .common import read, write

KINDS = {"collision", "unreachable", "grasp", "mechanism", "fit", "support"}
MAX_FAILURES = 100
MAX_JOBS = None         # every asset job with a finding may be repaired. Round 0 (61 items, 17 of 22 jobs
                        # rewritten) ran out of its 80M-token budget; the repair budget now covers all jobs
                        # (repair.MAX_TOKENS) and the generator's precheck rejects assets that regress.
MAX_FINDINGS = 200      # the Feedback Writer merges them into at most MAX_FAILURES rows

PROMPT = """You are the Feedback Writer in a joint-refinement loop between a robot-trajectory generator and a
3D asset / lab scene generator. A robot (holonomic base + 0.6 m lift + two Franka FR3 arms, reach 0.855 m,
each with a 64 mm parallel gripper: 0-50 mm stroke, 5 mm thick fingertips with 20 mm pads, plus a 5 mm
press pin protruding 20 mm and a 1.2 mm x 12 mm blade protruding 25 mm; base footprint 0.6 x 0.5 m)
attempted one single-skill task on every operable interface of the scene in physics simulation.

`findings.json` lists the failures the trajectory side attributes to the scene (asset geometry and
mechanisms, layout, or interface metadata), with measurements taken from the simulated model and the
numeric requirement derived from the robot. `instances.json` lists the scene plan instances (id, job,
configuration) the scene generator knows.

Write the feedback message for the scene generator as JSON:
{"failures": [{"kind": one of collision|unreachable|grasp|mechanism|fit|support,
               "instances": [plan instance ids involved, at least one],
               "detail": "what is wrong and exactly what to change, with the numeric requirement",
               "evidence": "measured values, the interfaces affected and how many robot tasks it blocks"}],
 "omitted": [{"finding": id, "why": "..."}]}

Rules:
- At most 100 failures; order by blocked tasks, most first. Merge findings with the same root cause on the
  same asset; keep different root causes apart.
- Every number you write must appear in findings.json (copy it; do not compute or round new values).
- Name parts by the geom/body/joint names in the findings. Say what to change in the asset or layout,
  not how the robot should behave.
- Use only instance ids from instances.json. If a finding has no matching instance, put it in "omitted".
- English only. Reply with the JSON object only.
"""


def plan_instances(scene_artifact):
    plan = json.loads((Path(scene_artifact) / "scene.json").read_text())["plan"]
    return [{"id": i["id"], "job": i["job"], "configuration": i.get("configuration")} for i in plan["instances"]]


def numbers(text):
    return {float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)}


def validate(message, findings, instances):
    """Problems with a feedback message; empty when it can be sent."""
    errors = []
    known = {i["id"] for i in instances}
    allowed = numbers(json.dumps(findings)) | {float(n) for n in range(0, 11)}
    failures = message.get("failures")
    if not isinstance(failures, list) or not 1 <= len(failures) <= MAX_FAILURES:
        return [f"failures must be a list of 1..{MAX_FAILURES} items"]
    for n, f in enumerate(failures):
        path = f"failures[{n}]"
        if set(f) != {"kind", "instances", "detail", "evidence"}:
            errors.append(f"{path}: fields must be exactly kind, instances, detail, evidence")
            continue
        if f["kind"] not in KINDS:
            errors.append(f"{path}.kind: {f['kind']!r} not in {sorted(KINDS)}")
        bad = [i for i in f["instances"] if i not in known] if isinstance(f["instances"], list) else ["<not a list>"]
        if bad or not f["instances"]:
            errors.append(f"{path}.instances: unknown {bad}")
        for field in ("detail", "evidence"):
            if not isinstance(f[field], str) or not 0 < len(f[field]) <= 12000:
                errors.append(f"{path}.{field}: must be a non-empty string of at most 12000 characters")
                continue
            invented = sorted(numbers(f[field]) - allowed)
            if invented:
                errors.append(f"{path}.{field}: numbers not in findings.json: {invented[:8]}")
    return errors


def select(findings, instances, attempted=(), max_jobs=MAX_JOBS):
    """Pick the findings to send this round: most blocked tasks first, within `max_jobs` asset jobs.

    Findings already sent in an earlier round that persisted (`attempted`) go after all fresh ones,
    so the generator is not asked to repeat a change that did not help while other problems wait.
    Returns (selected, skipped) with the reason for each skip."""
    job = {i["id"]: i["job"] for i in instances}
    ranked = sorted(findings, key=lambda f: (f["id"] in attempted, -len(f["blocked_tasks"]), f["id"]))
    jobs, selected, skipped = set(), [], []
    for f in ranked:
        need = {job[f["instance"]]} if f["instance"] in job else set()
        if not need:
            skipped.append({"finding": f["id"], "why": "instance not in the scene plan"})
        elif max_jobs is not None and len(jobs | need) > max_jobs:
            skipped.append({"finding": f["id"], "why": f"outside the {max_jobs} jobs focused this round"})
        elif len(selected) >= MAX_FINDINGS:
            skipped.append({"finding": f["id"], "why": f"beyond {MAX_FINDINGS} findings this round"})
        else:
            jobs |= need
            selected.append(f)
    return selected, skipped, sorted(jobs)


def scene_findings(findings, round_dir):
    """Keep only what the scene must change: the task types triaged to the scene side (see triage.py);
    a finding keeps just its tasks of those types. Trajectory and deferred types are not sent."""
    from .triage import task_type, write_triage
    table = (read(round_dir / "triage.json") if (round_dir / "triage.json").exists() else {"types": write_triage(round_dir)})["types"]
    results = {r["task"]: r for r in map(json.loads, (round_dir / "trajectory/results.jsonl").open())}
    kept, dropped = [], []
    for f in findings:
        classes = {}
        for t in f["blocked_tasks"]:
            r = results.get(t)
            key = task_type(t, r.get("instance", "") if r else f.get("instance", ""), r.get("skill", "") if r else "")
            classes[t] = table.get(key, {}).get("class", "scene")
        scene = [t for t in f["blocked_tasks"] if classes[t] == "scene"]
        if scene:
            kept.append({**f, "blocked_tasks": scene})
        else:
            dropped.append({"finding": f["id"], "why": "triaged to " + "/".join(sorted(set(classes.values()))) + ", not the scene"})
    return kept, dropped


def write_feedback(round_dir, scene_artifact, scene_sha256, max_rounds=3, attempted=()):
    """Ask the LLM for the message, validate, let it correct up to `max_rounds` times; record all."""
    round_dir = Path(round_dir)
    out = round_dir / "feedback"
    diagnosis = read(round_dir / "diagnosis/findings.json")
    instances = plan_instances(scene_artifact)
    scene_side, triaged_out = scene_findings([f for f in diagnosis["findings"] if f["attribution"] != "method"], round_dir)
    sendable, skipped, jobs = select(scene_side, instances, set(attempted))
    skipped += triaged_out
    write(out / "selection.json", {"max_jobs": MAX_JOBS, "max_findings": MAX_FINDINGS, "jobs": jobs, "sent": [f["id"] for f in sendable],
                                   "attempted_before": sorted(attempted), "skipped": skipped})
    findings = {"findings": sendable, "tasks": diagnosis["tasks"], "success": diagnosis["success"]}
    files = {"findings.json": json.dumps(findings, indent=1, ensure_ascii=False, default=float),
             "instances.json": json.dumps(instances, indent=1)}
    prompt, usage_total, message, errors = PROMPT, [], None, []
    for attempt in range(max_rounds):
        text, usage = llm.ask(prompt, out / f"llm_{attempt}", files=files)
        usage_total.append(usage)
        try:
            message = llm.json_reply(text)
            errors = validate(message, sendable, instances)
        except (ValueError, json.JSONDecodeError) as exc:
            message, errors = None, [f"reply is not a JSON object: {exc}"]
        if not errors:
            break
        prompt = PROMPT + "\n\nYour previous reply had these problems; fix all of them and reply again:\n- " + \
            "\n- ".join(errors[:40]) + "\n\nPrevious reply:\n" + text
    if errors:
        raise RuntimeError(f"feedback still invalid after {max_rounds} attempts: {errors[:5]}")
    payload = {"scene_sha256": scene_sha256, "failures": message["failures"]}
    write(out / "feedback.json", payload)
    write(out / "omitted.json", message.get("omitted", []))
    write(out / "usage.json", usage_total)
    (out / "feedback.md").write_text(render(payload, message.get("omitted", [])), encoding="utf-8")
    return payload


def render(payload, omitted):
    lines = [f"# Feedback to the scene generator ({len(payload['failures'])} items)", "",
             f"scene_sha256 `{payload['scene_sha256']}`", ""]
    for n, f in enumerate(payload["failures"], 1):
        lines += [f"## {n}. [{f['kind']}] {', '.join(f['instances'])}", "", f"**Detail.** {f['detail']}", "",
                  f"**Evidence.** {f['evidence']}", ""]
    if omitted:
        lines += ["## Omitted", ""] + [f"- {o.get('finding')}: {o.get('why')}" for o in omitted]
    return "\n".join(lines)
