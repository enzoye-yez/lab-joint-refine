"""Create a batch continuation copy offline. Keeps the original runs; does not call the model; unknown requests are blocked individually."""

import argparse
import copy
import json
from pathlib import Path
import re
import shutil

from asset_prep.storage import run_lock
from asset_prep.storage import atomic_json
from asset_generation.contracts import GenerationConfig
from scene_generation.contracts import SceneConfig
from prepare_scene_continuation import prepare
from run_scene_batch import BUDGET, read, hashes, code_files, save, now, verify


def prepare_batch(project, source_name, target_name, before_sources, kind="evidence_recovery", *, resume_preparation=False):
    for name in (source_name,target_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}",name):
            raise ValueError("invalid batch name")
    source = project/"runs/batches"/source_name
    target = project/"runs/batches"/target_name
    if target.exists() and not resume_preparation:
        raise ValueError("target batch already exists; overwriting is forbidden")
    if resume_preparation and ((target/'batch.json').exists() or kind!='evidence_pipeline'):
        raise ValueError('only an interrupted evidence-pipeline copy of a batch not yet published is supported')
    with run_lock(source), run_lock(target):
        original = read(source/"batch.json")
        for name,expected in original["frozen"].items():
            path = Path(name)
            if path.is_relative_to(project/"src"):
                continue
            if not path.is_file() or hashes([path])[str(path.resolve())] != expected:
                raise ValueError("original batch input or config has changed: "+name)
        manifest = copy.deepcopy(original)
        manifest.update(batch=target_name,created_at=now(),status="prepared",source_batch=source_name)
        if resume_preparation:
            expected={p.name:hashes([p])[str(p.resolve())] for p in (source/'config').glob('*.json')}
            actual={p.name:hashes([p])[str(p.resolve())] for p in (target/'config').glob('*.json')}
            if actual!=expected:raise ValueError('config of the interrupted copy is inconsistent')
        else:
            shutil.copytree(source/"config",target/"config")
        if kind == "transport_recovery":
            generation = read(target/"config/generation.json")
            generation.update(max_transport_retries=30, max_unknown_http_retries=2, transport_cooldown_seconds=300)
            GenerationConfig(**generation)
            atomic_json(target/"config/generation.json",generation)
            manifest["transport_recovery"] = {k:generation[k] for k in
                ("max_transport_retries", "max_unknown_http_retries", "transport_cooldown_seconds")}
        if kind == "repair_schedule":
            scene = read(target/"config/scene.json")
            scene.pop("max_repairs_per_round", None)  # Repairs are no longer capped per round.
            SceneConfig(**scene)
            atomic_json(target/"config/scene.json", scene)
        if kind == "budget_limits":
            for name, config_class in (("scene.json", SceneConfig), ("generation.json", GenerationConfig)):
                value = read(target/"config"/name)
                value.update(BUDGET)
                config_class(**value)
                atomic_json(target/"config"/name, value)
            manifest["budgets"].update(model_calls=BUDGET["max_model_calls"], reserved_tokens=BUDGET["max_reserved_tokens"])
        entries=[]
        try:
            for old in original["entries"]:
                run = target_name+"_"+old["protocol"].lower().replace("-","")
                source_run=project/"runs"/old["run"]
                if source_run.exists():
                    receipt = prepare(project,old["run"],run,before_sources,kind,
                                      resume_copy=resume_preparation and (project/'runs'/run).exists())
                else:
                    if old["status"]!="queued":
                        raise ValueError("started source run is missing: "+old["run"])
                    if (project/"runs"/run).exists():
                        raise ValueError("target run already exists: "+run)
                    receipt = {"unknown_request_preserved":False}
                entry = {k:old[k] for k in ("protocol","pdf","package")}
                entry.update(run=run,source_run=old["run"],status="blocked_unknown_request" if receipt["unknown_request_preserved"] else "queued")
                entries.append(entry)
        except Exception:
            (target/"preparation_incomplete.json").write_text(json.dumps({"prepared_entries":entries},ensure_ascii=False),encoding="utf-8")
            raise
        manifest["entries"]=entries
        files=code_files(project)+list((target/"config").glob("*.json"))
        files += [Path(name) for name in original["frozen"] if not Path(name).is_relative_to(project/"src")
                  and not Path(name).is_relative_to(source)]
        manifest["frozen"]=hashes(files)
        verify(manifest,project)
        save(target,manifest,project)
        return {"batch":target_name,"model_calls_added":0,"entries":[{"run":e["run"],"status":e["status"]} for e in entries]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--target",required=True)
    parser.add_argument("--before-sources",type=Path,required=True)
    parser.add_argument('--resume-preparation',action='store_true')
    parser.add_argument("--kind",choices=("evidence_recovery","transport_recovery","evidence_pipeline","agent_context","review_contract","asset_checks","phase_budget","budget_limits","repair_loop","tool_draft","collision_depth","repair_schedule","partial_revert","repair_feedback","discovered_defects","checker_failure","environment_retry"),default="evidence_recovery")
    args=parser.parse_args()
    print(json.dumps(prepare_batch(Path(__file__).resolve().parents[1],args.source,args.target,args.before_sources,args.kind,
                                  resume_preparation=args.resume_preparation),ensure_ascii=False))
