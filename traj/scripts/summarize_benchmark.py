"""Summarise a benchmark run: success by skill and mechanism, failure reasons, blocking geometry, timing.

  .venv/bin/python scripts/summarize_benchmark.py runs/bench_01
"""
import collections
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in (Path(sys.argv[1]) / "results.jsonl").open()]
by = collections.defaultdict(collections.Counter)
reasons = collections.Counter()
blockers = collections.Counter()
for r in rows:
    key = f"{r['skill']}/{r['mechanism']}"
    by[key]["total"] += 1
    by[key]["success"] += r["success"]
    reason = r["reason"] or ("disturbed" if r["disturbed"] else None) or ("goal_not_met" if not r["success"] else None)
    if reason:
        reasons[(r["skill"], reason)] += 1
    for pair, value in ((r.get("detail") or {}).get("blocking") or {}).items():
        n = value[0] if isinstance(value, list) else value
        other = [g for g in pair.split("/") if not g.startswith("robot__")]
        blockers["robot/" + ("own part" if not other else "other") if not other else "fingers vs " + other[0].split("__")[0]] += n
total = sum(r["success"] for r in rows)
print(f"{total}/{len(rows)} succeeded")
for key, c in sorted(by.items()):
    print(f"  {key:18s} {c['success']:3d}/{c['total']:3d}")
print("failure reasons:")
for (skill, reason), n in reasons.most_common():
    print(f"  {skill:8s} {reason:24s} {n}")
print("most frequent blocking contacts (dock search):", dict(blockers.most_common(8)))
print(f"wall per task: median {sorted(r['wall_s'] for r in rows)[len(rows)//2]:.0f} s; sim per task median "
      f"{sorted(r['sim_s'] for r in rows)[len(rows)//2]:.1f} s")
