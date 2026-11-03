"""One existing SceneRun repair cycle, using the parent ledger and immutable sources."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import uuid
from asset_generation.pipeline import snapshot
from asset_prep.storage import safe_path, digest, run_lock
from .common import need,read


def seed(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    need(not destination.is_relative_to(source) and not source.is_relative_to(destination),"A joint run must not overwrite or nest with the source run")
    with run_lock(source):
        from asset_prep.state_archive import read_state
        state=read_state(source/"state.json")
        if (source/'state_reports').is_dir():
            shutil.copytree(source/'state_reports', destination/'state_reports', dirs_exist_ok=True)
        need("scene" in state and state["scene"].get("plan"),"Source is not a planned scene run")
        for stage in state["scene"]["stages"]:
            origin=safe_path(source,stage["directory"])
            need(snapshot(origin)==stage["files"],"An archived source stage was modified")
            target=safe_path(destination,stage["directory"])
            present=snapshot(target) if target.exists() else {}
            need(all(stage["files"].get(n)==sha for n,sha in present.items()),"External modification during an unfinished source copy")
            for name in set(stage["files"])-set(present):
                dest=safe_path(target,name);dest.parent.mkdir(parents=True,exist_ok=True)
                temp=destination/(".seed-"+uuid.uuid4().hex+".tmp")
                try:
                    shutil.copyfile(safe_path(origin,name),temp);temp.replace(dest)
                finally:temp.unlink(missing_ok=True)
        for folder in ("records","responses"):
            if (source/folder).exists():
                for path in (source/folder).glob("*.json"):
                    target=destination/folder/path.name;target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():need(target.read_bytes()==path.read_bytes(),"Inherited call receipt mismatch")
                    else:shutil.copyfile(path,target)
        scene=deepcopy(state["scene"])
        # Interrupted sources remain preserved in the source tree; no claim of a completed candidate.
        scene.pop("pending",None)
        return {"calls":deepcopy(state["calls"]),"scene":scene,"source_state_sha256":digest(state),
                "source_identity":state["identity"],
                **({'phase_budget':deepcopy(state['phase_budget'])} if 'phase_budget' in state else {})}


class SceneBridge:
    def __init__(self,store,exchange,budget,project,package,backend,generation,provider,guard,protocol=None):
        self.store,self.exchange,self.budget=store,exchange,budget
        self.project,self.package,self.backend=Path(project),package,backend
        self.generation,self.provider,self.guard,self.protocol=generation,provider,guard,protocol

    def runner(self):
        from scene_generation.pipeline import SceneRun
        from scene_generation.inputs import catalog
        from scene_generation.contracts import SceneConfig
        from asset_generation.local import LocalBackend
        from asset_generation.sandbox import DockerSandbox
        source=self.store.state["source_identity"]
        catalogue=catalog(Path(self.package),self.generation,Path(self.protocol) if self.protocol else None)
        need(catalogue["identity"]==source["catalogue"],"Repair package differs from the original run")
        generation=replace(self.generation,max_iterations=1)
        backend=(LocalBackend if self.backend=="local" else DockerSandbox)(generation,self.project)
        # Keep source checker mode/tolerances. Operation evidence is separately recorded and verified.
        limits=SceneConfig(**source["limits"])
        limits=replace(limits,max_iterations=80,max_model_calls=800,max_reserved_tokens=240000000)
        env=source["environment"];robot=safe_path(self.project,env["robot"]["directory"])
        run=SceneRun(catalogue,env,source["contract"],self.provider,generation,limits,{},self.store.root,None,
                     backend,robot,self.project,provenance=source["provenance"],
                     expected_implementation=self.store.state["identity"]["implementation"],
                     shared_store=self.store,shared_model=self.exchange,before_candidate=self.budget.asset,
                     before_repair=lambda action:self.budget.round(digest(["scene",run.repair_input_key(),action]),"scene"))
        original=run.check_pause
        run.check_pause=lambda:(self.guard(),original())
        run.model.execution_guard=run.check_pause;run.model.cancel_guard=run.check_pause
        return run

    def repair(self,diagnosis):
        run=self.runner();state=run.state
        operation=self.store.state["joint"].setdefault("scene_work",{})
        key=digest(diagnosis)
        if not operation:
            known={i["id"] for i in state["plan"]["instances"]}
            ticket=self.store.state["joint"].get("repair_ticket")
            if ticket:
                need(ticket["scene_sha256"]==diagnosis["scene_sha256"],"Repair ticket does not match the diagnosis version")
                known &= set(ticket["instances"])
            rows=[]
            for row in diagnosis["rows"]:
                involved=sorted(set(row.get("instances",[]))&known)
                selected = not ticket or row.get("evidence_id") in ticket["evidence_ids"]
                design_trial = bool(ticket and ticket["target"]=="fixture" and selected)
                if selected and (row.get("repair_authorized") or design_trial) and involved:
                    rows.append({"id":"operation:"+digest(row)[:16],"status":"fail","instances":involved,
                        "kind":row["kind"],"detail":"Fixture design hypothesis backed by contact evidence; must be verified by comparison after the change" if design_trial else "Local issue re-confirmed by the fixed diagnostic",
                        "evidence":row, "repair_ticket":ticket})
            need(rows,"No independently re-checked grounds for a scene or asset repair")
            operation.update(key=key,phase="plan",baseline=deepcopy(state))
            state.update(feedback={"status":"fail","checks":rows,"coverage":diagnosis.get("coverage_gaps",[])},repair_pending=True)
            state.pop("repair_planning",None)
            self.store.save()
        need(operation["key"]==key,"Diagnosis version changed during scene repair")
        if operation["phase"]=="plan":
            if not run.route_repair():return {"status":"deferred","reason":state.get("repair_error","No executable local plan")}
            operation["phase"]="assets";self.store.save()
        if operation["phase"]=="assets":
            jobs={j["id"]:j for j in state["plan"]["jobs"]}
            done=operation.setdefault("jobs_done",[])
            for job in state["order"]:
                if job in done:continue
                current=state["jobs"].get(job)
                if current is None or current.get("dirty"):
                    run.generate_job(jobs[job],state.get("feedback"),draft=run.limits.validation_mode=="static")
                done.append(job);self.store.save()
            operation["phase"]="assemble";self.store.save()
        if operation["phase"]=="assemble":
            # Reuse existing partial publication; failed new candidates must not erase usable parts.
            from scene_generation.candidate import usable,inspect_candidate
            # Full-check legacy records predate assembly_readiness; remeasure the preserved artifact.
            # Never label missing/incompatible assets ready merely because an XML file exists.
            from asset_prep.errors import PrepError
            for key,record in state["jobs"].items():
                if record.get("artifact") and "assembly_readiness" not in record:
                    job=next(j for j in state["plan"]["jobs"] if j["id"]==key)
                    try:
                        receipt=inspect_candidate(safe_path(run.root,record["artifact"]),job["asset_ids"],
                            {**run.contract["jobs"].get(key,{}),"assembly_role":job["role"]})
                        record["assembly_readiness"]=receipt["assembly_readiness"]
                    except (PrepError,ValueError,OSError) as exc:
                        record["assembly_readiness"]={"loadable":False,"reason":str(exc)}
                    self.store.save()
            available={k:v for k,v in state["jobs"].items() if usable(run,v)}
            missing=sorted({j["id"] for j in state["plan"]["jobs"]}-set(available))
            if missing:
                from scene_generation.scene_first import publish
                published=publish(run,missing)
                entry=next((h for h in reversed(state.get('repair_history',[])) if h.get('status')=='applied'),None)
                if entry:
                    comparison={'outcome':'coverage_lost','adopt':False,'coverage_lost':entry['target_checks'],
                        'new_defects':[],'resolved_checks':[],'remaining_checks':entry['target_checks']}
                    entry.update(status='evaluated',outcome='coverage_lost',comparison=comparison,
                        selection='baseline_preserved',failure='candidate_missing_jobs',
                        calls_consumed=len(self.store.state['calls'])-entry['call_start'])
                    self.store.save()
                return {"status":"deferred","reason":"Candidate has missing parts; a partial scene was published and task success is not claimed", "missing":missing,
                        "partial_scene":str(safe_path(run.root,published["artifact"])) if published else None}
            from scene_generation.local_repair import precheck
            if precheck(run):
                return {'status':'deferred','reason':'Local re-check rejected regressing or non-improving candidates; the original scene is kept'}
            report,review=run.build_scene()
            latest=state["latest"]
            operation.update(phase="done",artifact=latest["artifact"],report=report)
            self.store.save()
        return {"status":"candidate","scene":str(safe_path(run.root,operation["artifact"])),
                "static_quality":operation["report"]["status"],"report":operation["report"],"missing":[]}

    def restore(self,baseline):
        # Restore selection, never stages, ledger or candidate counts.
        current=self.store.state["scene"]
        restored=deepcopy(baseline)
        for key in ("stages","rounds","repair_history","candidate_attempts","local_checks"):
            if key in current:restored[key]=deepcopy(current[key])
        restored.pop("pending",None)
        self.store.state["scene"]=restored
        self.store.save()
