# lab-joint-refine

Robot-in-the-loop generation of articulated laboratory scenes: an LLM agent writes MuJoCo assets and
lays out a lab for a written protocol, a mobile dual-arm robot tries every operable interface, and the
failures are turned into verified, numeric repair requirements for the assets and the layout.

```
protocol ──> scene_gen/      asset generation (one coding agent per asset) + scene planning, layout, assembly
                 │                                         ▲
                 ▼ scene artifact (model.xml, scene.json,   │ repair requirements (verified in simulation)
                 │  interfaces.json)                        │
             traj/           single-skill benchmark: dual Franka FR3 on a mobile base, every interface
                 │
                 ▼ per-task results (reason, detail, log)
             joint_refine/   cycles, judges and the repair-skill harness:
                             failure -> counterfactual scene edit -> rerun -> verdict -> requirement
```

| Directory | What it is |
|---|---|
| [`scene_gen/`](scene_gen/README.md) | Protocol → asset jobs → per-asset MuJoCo authoring → arrangement → assembled scene. `scripts/apply_robot_feedback.py` (LLM repair from feedback) and `scripts/apply_directed_repair.py` (deterministic, already-verified edits) feed robot findings back. |
| [`traj/`](traj/README.md) | The robot side. `tasks/run.py` runs one task in-process; `scripts/run_benchmark.py` runs all of them; `scene/perturb.py` + `scripts/run_counterfactual.py` rerun tasks on edited copies of the scene. The robot's limits are frozen in `data/robot/capability.json`. |
| `joint_refine/` | Orchestration (`cycle.py`), per-cycle judges, the older diagnose/LLM feedback path, and the repair-skill harness (`joint_refine/joint_refine/harness/`): skills for dock space, grasp geometry, mechanism effort, blocker chains, part retention and robot limits. |

The three directories were separate repositories; their histories are kept under these prefixes.

## The protocol input

The benchmark protocol (NP-002) is the Nature Protocols article *Optimized DNA extraction and metagenomic
sequencing of airborne microbial communities* (2015, PMID 25906115, PMCID PMC7086576). Its PDF, JATS XML
and figures are **not** redistributed here; download them from PubMed Central into
`scene_gen/benchmark/NP-002/` as `source.pdf`, `source.xml` and `images/` (file names and SHA-256 sums are
listed in `scene_gen/benchmark/NP-002/input_manifest.json`).

## Third-party models

Robot models under `scene_gen/data/robots/` and `traj/data/vendor/` (Franka FR3, ROBOTIS FFW-SG2, PAL TIAGo
omni base) keep their own Apache-2.0 licenses next to the files.

## Running

Each directory has its own README, requirements and tests. CPU only; the trajectory benchmark is meant to
run under `traj/scripts/guarded.sh` (cgroup CPU/memory limits). The generator's LLM calls go through the
Codex CLI login (no API keys are stored in this repository).
