# lab-scene-gen

An agent pipeline that turns a written lab protocol (NP-002, a nucleic-acid workflow) into a
MuJoCo lab scene for a mobile dual-arm robot. It produces articulated instruments, consumables
and fixtures, places them on benches, and hands the result to robot-trajectory planning.

```
protocol PDF + asset package
  -> asset_prep        extract instruments, containers, consumables and their requirements
  -> scene plan        LLM: jobs, instances, benches, floor/bench, support chains, bench order
  -> asset authoring   one codex coding-agent session per asset writes model.py (MuJoCo via a
                       small builder); it self-checks with ./check, the framework rebuilds,
                       checks (geometry, material, mechanisms, buttons, connections, stability)
                       and reviews (vision) independently; failed assets go back to their author
  -> arrangement       deterministic: benches against the walls in two rows facing one robot
                       aisle, one row of instruments per bench, operating face to the aisle,
                       a dock per instrument
  -> scene checks      collisions, support, support fit, table bounds, reach from docks
  -> scene review      vision review; repair tasks per colliding pair / defect, scheduled
  -> operation handoff planning_handoff.json: scene, plan and operation_obligations (what the
                       static stages left for trajectories to verify)
```

Every prompt, contract and feedback text an agent reads is English. The HTML report is Chinese.

## Requirements

- Linux, Python 3.11, MuJoCo 3.3.7 (pinned in `requirements.txt`), Mesa EGL for CPU rendering
  (`scripts/env.sh` hides every GPU and selects Mesa).
- `bwrap` (bubblewrap): every model request runs inside it, so the agent cannot read the rubric,
  the evaluation files, docs, reports or other copies of the project (codex's own sandbox
  restricts writes, not reads).
- Codex CLI, logged in with a ChatGPT account (`codex login`). Model calls use that login; no API
  key is read.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-generation.txt
codex login
```

## Run NP-002 with gpt-6.1-sol

Prepare a batch (no model call):

```bash
scripts/run-scene-batch.sh --batch np002_sol_01 --protocols NP-002 --input-root benchmark \
  --provider-config config/codex_sol_agent.json \
  --generation-config config/experiments/np002_sol_generation.json \
  --scene-config config/experiments/np002_sol_scene.json \
  --environment config/scene_environment.json
```

Start it (calls the model; budget 1600 calls / 480M reserved tokens, 8 parallel authors):

```bash
nohup scripts/run-scene-batch.sh --batch np002_sol_01 --resume --execute > np002_sol_01.log 2>&1 &
```

- Run directory: `runs/np002_sol_01_np002/` (`state.json`, `stages/`, `codex_agent_sessions/`).
- Batch summary: `runs/batches/np002_sol_01/index.html`.
- A batch freezes the source hashes. After editing `src/`, prepare a continuation instead of
  resuming: `scripts/prepare_batch_continuation.py --source <batch> --target <new> --kind <kind>`.

## Report

```bash
source scripts/env.sh
$PYTHON scripts/render_figures.py --run runs/np002_sol_01_np002 --workers 16
$PYTHON scripts/build_report.py --run runs/np002_sol_01_np002
```

`reports/<run>/index.html`: per asset and configuration the stills, mechanism poses, a video
that plays every mechanism, the asset assembled in the scene, the checks and reviews; the whole
scene from five views; `mujoco_assets.zip` with every asset and a `play_mechanisms.py` viewer.

## Tests

```bash
./test.sh
```

Offline only (no model calls). A few tests fail on this machine for known environment reasons
(see `docs/`); compare against a baseline checkout rather than expecting zero failures.

## Layout of the code

| Path | What |
|---|---|
| `src/asset_prep` | protocol and asset-package extraction |
| `src/asset_generation` | author/review prompts, codex transport, sandboxed build, asset checks, presentation renders, optional author skills (off by default) |
| `src/scene_generation` | plan, arrangement, assembly, scene checks, review, repair scheduling, operation handoff |
| `src/robot_operation` | task contracts, trajectory planning and diagnosis on the scene |
| `benchmark/NP-002` | inputs; `rubric.json` is for evaluation only and is hidden from every model request |
| `scripts` | batch runner, continuations, report |
| `data/robots` | vendor robot models (ROBOTIS FFW-SG2, Franka FR3, TIAGo) |
| `cases` | generated assets and scenes as MuJoCo packages, one folder per run |
