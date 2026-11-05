"""Static workflow: durable diagnostic assembly before quality and repair.

Generation, availability and acceptance are independent. No quality tolerance is
changed and no unknown result is used as a geometry repair instruction.
"""
from copy import deepcopy
from pathlib import Path
from asset_generation.contracts import check_profile, require
from asset_generation.check_report import checked_report, checked_previews, report_status, incomplete, failed_report
from asset_generation.inputs import image_record
from asset_generation.pipeline import snapshot
from asset_prep.storage import atomic_json, digest, safe_path
from asset_prep.errors import StaleEvidence, BudgetExceeded, ProviderError, UncertainRequest, ContextLimitError, ValidationError
from .candidate import usable
from .layout import review_targets
from .inputs import task_bundle
from .checkpoint import save_review_checkpoint
from .recovery import decide


def load_images(run, report, directory):
    return checked_previews(report,directory,
        lambda root,name:image_record(safe_path(root,name),run.root,run.generation))


def dependencies_blocking(run, job):
    """Dependencies that already stopped, so waiting cannot produce new evidence.

    A tool or fixture declares every asset it serves as a dependency by contract. That
    contract is about interfaces, not about blocking on a job that will never load.
    """
    blocked=run.state.get("blocked_jobs",{})
    return sorted(k for k in job["depends_on"] if blocked.get(k))


def record_awaiting(run, key, dependencies):
    for dependency in dependencies:
        waiting=run.state.setdefault("waiting_on",{}).setdefault(dependency,{})
        waiting[key]={"reason":"Dependency job has not yet produced a loadable candidate"}
    run.store.save()


def record_job_exhausted(run, key, history):
    """Persist why a job stopped instead of silently retrying it forever."""
    blocked=run.state.setdefault("blocked_jobs",{})
    if blocked.get(key,{}).get("kind")=="allocation_exhausted":return
    blocked[key]={"kind":"allocation_exhausted","attempts":len(history),
        "reason":"This job has used up its cumulative candidate attempts; archived candidates are kept, and resume or dependency changes do not reset the count",
        "last":{k:v for k,v in (history[-1] if history else {}).items() if k in {"cycle","status","loadable"}}}
    run.store.save()


def record_job_blocked(run, key, reason):
    blocked=run.state.setdefault("blocked_jobs",{})
    if blocked.get(key,{}).get("kind")==reason:return
    blocked[key]={"kind":reason,
        "reason":"Checker fault or paused job; regenerating will not change the result. State is kept while waiting for an environment or implementation fix"}
    run.store.save()


def prefetch_candidates(run, jobs, attempts):
    """Asset jobs the sweep below will author, with every dependency already usable.

    A conservative mirror of the sweep's eligibility: a job included wrongly costs one
    cached request; a job left out is simply authored serially. Tools and fixtures are
    excluded because layout refinement may change their request first.
    """
    blocked=run.state.get("blocked_jobs",{})
    result=[]
    for key in run.state["order"]:
        job,old=jobs[key],run.state["jobs"].get(key)
        if job["role"]!="asset" or key in blocked or (usable(run,old) and not old.get("regenerate")):
            continue
        if old and old.get("artifact") and not old.get("assembly_readiness"):
            continue
        if len(attempts.get(key,[]))>=run.generation.max_iterations:
            continue
        # A dependency this sweep re-authors changes the request; author after it instead.
        if any(not usable(run,run.state["jobs"].get(k))
               or (run.state["jobs"].get(k) or {}).get("regenerate") and k not in blocked for k in job["depends_on"]):
            continue
        result.append(key)
    return result


def thread_model(run):
    """A model client for one worker thread; requests and responses go through the shared ledger."""
    from copy import copy
    from asset_generation.model import GenerationModel
    model=GenerationModel(run.provider,run.generation,run.keys,run.store,copy(run.transport))
    model.execution_guard,model.cancel_guard=run.check_pause,run.check_cancel
    if hasattr(model.transport,"check_cancel"):
        model.transport.check_cancel=run.check_cancel
    return model


def workers(run):
    return getattr(getattr(run,"limits",None),"parallel_authors",1)


def prepared_directory(run):
    from uuid import uuid4
    return run.root/"prepared"/uuid4().hex


def prebuild(run, job, inputs, value):
    """Build an authored draft ahead in this worker thread; the serial sweep adopts the result
    (sandbox.adopt) instead of building the drafts one after another."""
    if not hasattr(run.sandbox,"prepare"):return
    from asset_generation.contracts import author_result
    try:
        source=author_result(value,inputs["previous"],run.generation.max_script_bytes)
    except ValidationError:
        return  # The serial sweep records the same authoring error.
    scratch=prepared_directory(run)
    script=scratch.with_suffix(".py")
    script.parent.mkdir(parents=True,exist_ok=True)
    script.write_text(source,encoding="utf-8")
    profile={**inputs["profile"],"assembly_role":job["role"],"assembly_configurations":job.get("configurations",["default"])}
    try:
        # A later prefetch of the same sweep asks again for jobs not yet committed; build once.
        if run.sandbox.prepared_key("build_candidate",script,job["asset_ids"],profile) not in run.sandbox.__dict__.get("prepared",{}):
            run.sandbox.prepare("build_candidate",scratch,script,job["asset_ids"],profile)
    except Exception:
        pass  # The serial sweep builds it again and meets the same failure itself.
    finally:
        script.unlink()


def prepare_rechecks(run, items):
    """Re-measure several archived assets at once; recheck_asset then adopts each result.

    items: (artifact path, asset_ids, profile). CPU-only; no state is touched."""
    items=[i for i in items if i[0] is not None]
    if not hasattr(run.sandbox,"prepare") or workers(run)<2 or len(items)<2:return
    from concurrent.futures import ThreadPoolExecutor
    def recheck(item):
        try:
            run.sandbox.prepare("recheck_asset",prepared_directory(run),*item)
        except Exception:
            pass  # The serial recheck meets and records the same failure itself.
    with ThreadPoolExecutor(max_workers=workers(run)) as pool:
        list(pool.map(recheck,items))


def prefetch(run, jobs, attempts):
    """Send the sweep's independent author requests concurrently; responses land in the ledger cache.

    The serial sweep then builds the identical request and consumes the cached response,
    so state changes, builds, checks and commits stay sequential and unchanged.
    """
    count=workers(run)
    keys=prefetch_candidates(run,jobs,attempts)
    if count<2 or len(keys)<2:
        return
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    inputs={key:run.author_inputs(jobs[key],None,draft=True) for key in keys}
    requests={key:run.author_request(jobs[key],inputs[key],0) for key in keys}
    def author(key):
        try:
            value,_=thread_model(run).ask(*requests[key])
        except Exception as exc:  # The serial sweep meets and handles the same failure itself.
            return {"job":key,"outcome":"failed","error":type(exc).__name__,"detail":str(exc)[:300]}
        prebuild(run,jobs[key],inputs[key],value)
        return {"job":key,"outcome":"cached"}
    started=datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=count) as pool:
        results=list(pool.map(author,keys))
    run.state.setdefault("prefetch",[]).append({"started_at":started,"finished_at":datetime.now(timezone.utc).isoformat(),
        "workers":count,"jobs":results})
    run.store.save()


def collect(run):
    """One attempt per eligible job per sweep; quality never consumes first-pass budget.

    Publish between sweeps, including when every draft fails. Confirmed authoring
    failures retain feedback and retry within the cumulative allocation; checker
    faults and unknown requests do not become geometric repair instructions.
    """
    jobs={j["id"]:j for j in run.state["plan"]["jobs"]}
    attempts=run.state.setdefault("candidate_attempts",{})
    while True:
        # Clear historical waiting edges once the dependency has a usable candidate.
        waiting=run.state.setdefault("waiting_on",{})
        for dependency in list(waiting):
            if usable(run,run.state["jobs"].get(dependency)):
                waiting.pop(dependency)
        prefetch(run,jobs,attempts)
        progressed=False
        for key in run.state["order"]:
            run.check_pause()
            old=run.state["jobs"].get(key)
            if old and old.get("artifact") and not old.get("assembly_readiness") and hasattr(run.sandbox,"admit_asset"):
                destination=run.begin("asset_admission")
                report,_=run.sandbox.admit_asset(safe_path(run.root,old["artifact"]),destination/"admission",jobs[key]["asset_ids"],
                    {"assembly_role":jobs[key]["role"],"assembly_configurations":jobs[key].get("configurations",["default"])})
                run.commit("asset_admission",{"job":key,"report":report,"artifact":old["artifact"]})
                old["assembly_readiness"]=report.get("assembly_readiness",{"loadable":False})
                run.store.save()
            if usable(run,old) and not old.get("regenerate"):
                run.state.setdefault("blocked_jobs",{}).pop(key,None)
                continue
            dependencies=jobs[key]["depends_on"]
            # A tool or fixture declares every served asset as a dependency by design
            # contract. That contract concerns interfaces: a dependency that already
            # stopped cannot produce new interfaces, so it must not stall the sweep.
            stopped=dependencies_blocking(run,jobs[key])
            required=[k for k in dependencies if k not in set(stopped)]
            unmet=[k for k in required if not usable(run,run.state["jobs"].get(k))]
            if unmet:
                record_awaiting(run,key,unmet)
                continue
            # Counts are input/version-bound. Resuming a failed input cannot refill its budget.
            input_key=digest({"job":jobs[key],"source":(old or {}).get("source"),
                "repair":run.state.get("repair_directive"),
                "dependencies":{k:run.state["jobs"].get(k,{}).get("assembly_readiness") for k in required}})
            history=attempts.setdefault(key,[])
            # A quality repair scheduled after the audit has its own allowance (schedule_asset_repairs).
            quality=(old or {}).get("regenerate")=="quality"
            # Initial admission failures share one bounded allocation despite generated source changes.
            cycle=digest({"job":jobs[key],"repair":run.state.get("repair_directive"),
                "dependencies":{k:run.state["jobs"].get(k,{}).get("assembly_readiness") for k in required}})
            pending=next((a for a in reversed(history) if a["cycle"]==cycle and a["status"]=="started"),None)
            # Asset allocations are cumulative; changing dependencies must not silently reset them.
            if pending is None and not quality and sum(not a.get("quality") for a in history)>=run.generation.max_iterations:
                record_job_exhausted(run,key,history)
                continue
            binding=digest({"cycle":cycle,"implementation":run.implementation_guard.expected})
            blocked=run.state.setdefault("blocked_jobs",{}).get(key)
            if blocked and blocked.get("kind")!="allocation_exhausted" and blocked.get("binding") not in (None,binding):
                run.state.setdefault("job_reactivations",[]).append({"job":key,"previous":blocked,"binding":binding,"reason":"input_changed"})
                run.state["blocked_jobs"].pop(key)
                blocked=None
            current=[a for a in history if a["cycle"]==cycle and a.get("binding",binding)==binding]
            if pending is None and len(current)>=3 and not any(a.get('loadable') for a in current):
                record_job_blocked(run,key,'authoring_stalled')
                run.state['blocked_jobs'][key].update(binding=binding,
                    reason='Initial draft build failed three times in a row; specific errors kept, independent jobs continue',
                    recovery_condition='Related dependency, repair directive or check implementation changes')
                continue
            last=current[-1] if current else {}
            if last.get("checker_fault") or last.get("task_deferred"):
                kind="checker_fault" if last.get("checker_fault") else "task_deferred"
                # One fresh decision context per input, persisted before the next paid call.
                recoveries=run.state.setdefault("task_context_recoveries",{})
                recovery_key=key+":"+binding
                if kind=="task_deferred" and recovery_key not in recoveries:
                    recoveries[recovery_key]={"job":key,"binding":binding,"status":"scheduled"}
                    run.state["blocked_jobs"].pop(key,None)
                    run.state.setdefault("task_recovery",{})[key]={"reason":kind,"binding":binding,
                        "instruction":"Keep the current script, the current job and the necessary interfaces; no need to reread the full history. Submit the current best candidate or state exactly what information is missing; do not fabricate missing images or checks."}
                else:
                    record_job_blocked(run,key,kind)
                    run.state["blocked_jobs"][key].update(binding=binding,
                        recovery_condition="Dependency, repair directive or corresponding check implementation changes",
                        reason="Checker fault" if kind=="checker_fault" else "Author or detail interaction paused; the compact-recovery chance for this input has been used")
                    continue
            elif blocked:
                continue
            entry=pending or {"cycle":cycle,"input_key":input_key,"binding":binding,"status":"started","quality":quality}
            if pending is None:history.append(entry)
            run.store.save()
            recovery=run.state.get("task_context_recoveries",{}).get(key+":"+binding)
            if recovery: recovery.update(status="started")
            # Dependencies committed earlier in this sweep unlock new independent authors;
            # send them together instead of one by one.
            if jobs[key]["role"]=="asset":
                prefetch(run,jobs,attempts)
            try:
                if jobs[key]["role"]!="asset":
                    from .layout_planning import refine
                    refine(run)
                ready=run.generate_job(jobs[key],None,draft=True)
            except (BudgetExceeded,ProviderError,UncertainRequest) as exc:
                # No further paid calls: publish whatever already exists, then preserve the original interruption.
                if run.state.get("pending"):
                    run.commit("generation_interrupted",{"job":key,"reason":type(exc).__name__})
                run.candidate_interruption=exc
                return [k for k in jobs if not usable(run,run.state["jobs"].get(k))]
            candidate=run.state["jobs"][key]
            entry.update(status="completed",stage=candidate["stage"],loadable=ready,
                checker_fault=candidate.get("feedback",{}).get("numerical",{}).get("status")=="error",
                task_deferred=candidate.get("task_deferred"))
            if recovery: recovery.update(status="completed",loadable=ready)
            run.state.get("task_recovery",{}).pop(key,None)
            progressed=True
            if ready:
                candidate.pop("regenerate",None)
            elif usable(run,old):
                # Preserve the previous loadable version; the failed replacement stays archived.
                run.state["jobs"][key]=old
                old.pop("regenerate",None)
                old["replacement_failure"]={"stage":candidate["stage"],"feedback":candidate.get("feedback")}
            run.store.save()
            # Publish the first usable asset promptly, then batch the rest of the
            # sweep. Every asset is already committed before this expensive step.
            if (ready and "layout" in run.state["plan"]
                    and not run.state.get("published_scene",{}).get("job_versions")):
                publish(run,[k for k in jobs if not usable(run,run.state["jobs"].get(k))])
        if all(usable(run,run.state["jobs"].get(k)) for k in jobs):
            if "arms" in run.env["robot"] and all(j["role"]=="asset" for j in jobs.values()):
                from .layout_planning import refine
                try:
                    refine(run)
                except (BudgetExceeded,ProviderError,UncertainRequest) as exc:
                    run.candidate_interruption=exc
            break
        if progressed:
            # A failing sweep still has useful repair feedback. Save available work
            # before retrying; the per-job allocation above also bounds no-progress.
            publish(run,[k for k in jobs if not usable(run,run.state["jobs"].get(k))])
            continue
        if not progressed:
            if "layout" in run.state["plan"]:
                from .layout_planning import refine
                try: refine(run)
                except (BudgetExceeded,ProviderError,UncertainRequest) as exc: run.candidate_interruption=exc
            run.store.save()
            break
    return [k for k in jobs if not usable(run,run.state["jobs"].get(k))]


def assembly_plan(run, missing):
    plan=deepcopy(run.state["plan"])
    plan["jobs"]=[j for j in plan["jobs"] if j["id"] not in missing]
    plan["instances"]=[i for i in plan["instances"] if i["job"] not in missing]
    ids={i["id"] for i in plan["instances"]}
    # Missing parents stay explicit missing evidence; do not turn a supported object into a
    # counterfeit table-supported success. The support checker records its empty parent set.
    if "layout" in plan:
        plan["layout"]["docks"]=[d for d in plan["layout"]["docks"] if d["instance"] in ids]
    for field in ("table_modifications","edge_mounts"):
        if field in plan:plan[field]=[r for r in plan[field] if r["instance"] in ids]
    return plan


def publish(run, missing):
    plan=assembly_plan(run,missing)
    selected={j["id"]:run.state["jobs"][j["id"]] for j in plan["jobs"]}
    key=digest({"plan":plan,"requested_plan":run.state["plan"],"assets":{k:v["assembly_readiness"] for k,v in selected.items()},
                "implementation":run.implementation_guard.expected,"missing":missing})
    previous=run.state.get("published_scene")
    if not previous or previous.get("assembly_key")!=key:
        previous=next((stage for stage in reversed(run.state["stages"])
                       if stage["kind"]=="scene_candidate" and stage.get("assembly_key")==key),previous)
    if previous and previous.get("assembly_key")==key:
        if digest(snapshot(safe_path(run.root,previous["artifact"])))!=previous["artifact_sha256"]:
            raise StaleEvidence("Published scene was modified")
        previous=deepcopy({k:v for k,v in previous.items() if k not in {"kind","files"}})
        run.state["published_scene"]=previous
        return previous
    destination=run.begin("scene_candidate")
    artifacts={k:safe_path(run.root,v["artifact"]) for k,v in selected.items()}
    report,renders=run.sandbox.assemble_candidate(plan,run.env,artifacts,run.robot_root,destination,run.limits,run.contract)
    artifact=destination/"scene"/"artifact"
    if not report.get("assembly_readiness",{}).get("loadable"):
        timing=report.get("execution_timing") or {}
        failure=report.get("execution_failure") or {}
        # Include component contents/configurations, not just job names and poses.
        # A structural repair must not inherit unrelated assembly-failure evidence.
        size=key
        history=run.state.setdefault("assembly_failures",[])
        similar=[h for h in history if h.get("size")==size]
        history.append({"size":size,"kind":failure.get("kind"),"error_kind":failure.get("error_kind"),
                        "worker_seconds":timing.get("worker_seconds"),
                        "budget_seconds":timing.get("budget_seconds") or run.limits.build_timeout_seconds})
        run.state["assembly_failure_summary"]={"attempts_at_this_size":len(similar)+1,
            "timed_out_at_this_size":sum(1 for h in similar if h.get("error_kind")=="backend_execution"),
            "last_timing":timing,
            "detail":"Records failure count and actual time limit for the same assembly input; use stage logs to locate the cause. A timeout alone does not prove the asset geometry is wrong; counts restart after components or configuration change"}
        run.store.save()
        failed=run.commit("assembly_failure",{"report":report,"plan":plan})
        from .progress import repair_comparison
        for entry in run.state.get("repair_history",[]):
            if entry.get("status")=="applied" and entry.get("baseline"):
                baseline=next(s for s in run.state["stages"] if s["directory"]==entry["baseline"]["scene"])
                entry.update(status="evaluated",outcome="coverage_lost",scene=failed["directory"],
                    output_key=run.repair_input_key(),comparison=repair_comparison(baseline["report"],{},baseline.get("review")),
                    failure="candidate_assembly_failed",calls_consumed=len(run.store.state["calls"])-entry["call_start"])
        restored = run.select_repair_candidate()
        retryable=restored or decide(report,None)=="repair"
        if retryable and not restored:
            run.state.update(feedback=deepcopy(report),latest={"directory":failed["directory"],
                "report":deepcopy(report),"review":None,"plan":deepcopy(run.state["plan"]),
                "preview_paths":[],"job_versions":{k:v["stage"] for k,v in run.state["jobs"].items()}})
        run.state["repair_pending"]=retryable
        run.state.update(status="retrying_assembly" if retryable else "assembly_failed",assembly_failure=report)
        run.store.save()
        return None
    record={"report":report,"review":None,"plan":plan,"requested_plan":deepcopy(run.state["plan"]),"missing_jobs":missing,
        "assembly_key":key,"artifact":str(artifact.relative_to(run.root)),
        "artifact_sha256":digest(snapshot(artifact)),"preview_paths":[],
        "job_versions":{k:v["stage"] for k,v in selected.items()}}
    stage=run.commit("scene_candidate",record)
    run.state["published_scene"]={**record,"directory":stage["directory"]}
    run.state.pop("assembly_failure",None)
    run.state.pop("assembly_failure_summary",None)
    run.store.save()
    run.result()  # Available before any numeric check or paid review can fail.
    return run.state["published_scene"]


# Sources that shape what an asset review sees and how it is judged; any change re-reviews.
REVIEW_SOURCES = ("scene_generation/review.py", "scene_generation/prompts.py", "asset_generation/prompts.py")


def review_key(job, readiness, layout, dependencies, report, image_hashes):
    """What an asset review judged: the asset, its context, the check rows and the rendered
    images. A checker or framework change that leaves all of these unchanged keeps the verdict."""
    src = Path(__file__).resolve().parents[1]
    sources = {name: digest((src/name).read_bytes().replace(b"\r\n", b"\n").decode()) for name in REVIEW_SOURCES}
    return digest({"job": job, "asset": readiness, "layout": layout, "dependencies": dependencies,
                   "checks": report.get("checks"), "images": sorted(image_hashes), "review_sources": sources})


def dependency_geometry(run, job):
    """How each dependency is built. Audits and reviews depend on this only, so a dependency that
    is re-reviewed or re-versioned without a geometry change re-reviews nothing downstream."""
    return {k:digest({"manifest":run.state["jobs"].get(k,{}).get("manifest"),
                      "geometry":run.state["jobs"].get(k,{}).get("geometry_summary")}) for k in job["depends_on"]}


def audit_profile(run, job):
    # Motion, settling and probes run in every mode (checks.py); static only selects the
    # kinematic geometry state used for the material checks.
    return check_profile({**run.contract["jobs"].get(job["id"],{}),"validation_mode":"static"})


def audit_assets(run):
    """Re-measure and re-review every asset whose version, dependency geometry or checker changed.

    The numerical rechecks of all due assets run at once (prepare_rechecks) and each result is
    committed in order; then their visual reviews are sent at once (review_prefetch) and each
    verdict is committed in order from the cached response."""
    from .local_repair import check_key,cached_check
    due=[]
    for job in run.state["plan"]["jobs"]:
        current=run.state["jobs"].get(job["id"])
        if not usable(run,current):continue
        layout=run.layout_context() if job['role']!='asset' else None
        audit_key=digest({"asset":current["assembly_readiness"],"layout":layout,"dependencies":dependency_geometry(run,job),
            "implementation":run.implementation_guard.expected})
        if current.get("quality_key")!=audit_key:due.append((job,layout,audit_key))
    jobs=run.state["jobs"]
    prepare_rechecks(run,[(safe_path(run.root,jobs[job["id"]]["artifact"]),job["asset_ids"],audit_profile(run,job))
        for job,_,audit_key in due if jobs[job["id"]].get("numeric_audit_key")!=audit_key
        and not cached_check(run,check_key(run,jobs[job["id"]],job,audit_profile(run,job)))])
    reviews=[(job,layout,audit_key) for job,layout,audit_key in due if measure_asset(run,job,layout,audit_key)]
    review_prefetch(run,[job for job,_,_ in reviews])
    for job,layout,audit_key in reviews:
        review_measured(run,job,layout,audit_key)


def measure_asset(run, job, layout, audit_key):
    """Commit the numerical audit of one asset; True when its visual review is still due."""
    key=job["id"]; current=run.state["jobs"][key]
    destination=run.begin("asset_review")
    record={k:deepcopy(v) for k,v in current.items() if k not in {"stage","dirty","regenerate","quality_key"}}
    artifact=safe_path(run.root,current["artifact"])
    profile=audit_profile(run,job)
    from .local_repair import check_key,cached_check
    local=cached_check(run,check_key(run,current,job,profile))
    if current.get("numeric_audit_key")==audit_key:
        report=current["feedback"]["numerical"]
        images=[image_record(safe_path(run.root,x),run.root,run.generation) for x in current.get("preview_paths",[])]
    else:
        try:
            if local:
                report=deepcopy(local['report'])
                renders=safe_path(run.root,local['render_directory']) if local.get('render_directory') else None
            else:report,renders=run.sandbox.recheck_asset(artifact,destination,job["asset_ids"],profile)
        except (ValidationError,OSError) as exc:
            report,renders=failed_report(exc),None
        report=checked_report(report,domain="asset",mode="static",require_renders=True)
        if any(incomplete(r) for r in report.get("checks",[])) and report["status"]!="error":
            from asset_generation.check_recovery import recover_checks
            report=recover_checks(run.sandbox,artifact,report,run.root/"asset_check_recovery",
                implementation=run.implementation_guard.expected,contract=profile,
                check_pause=run.check_pause,max_actions=run.limits.max_check_actions,max_retries=run.limits.max_check_retries)
        if job["role"]=="tool" and report["status"]!="error":
            from .tool_contract import printability_check
            report={**report,"checks":report["checks"]+[printability_check(artifact)]}
            report["status"]=report_status(report["checks"])
        report,images=load_images(run,report,renders)
        if not images:images=[image_record(safe_path(run.root,x),run.root,run.generation) for x in current.get("preview_paths",[])]
    record.update(feedback={"numerical":report},preview_paths=[i["path"] for i in images],status="needs_review",numeric_audit_key=audit_key)
    atomic_json(destination/"checks.json",report)
    due=False
    if report["status"]=="fail":
        # Confirmed numerical defects go straight back to the author; a visual review
        # cannot overrule them and would only spend a call. Findings of an earlier real
        # review of this same artifact still apply and travel with them.
        earlier=current.get("feedback",{}).get("visual") or {}
        seen=earlier.get("verdict")=="repair" and earlier.get("source")!="numeric"
        review={"verdict":"repair","source":"numeric","unverified":[],"motion_coverage":[],
                "findings":failing_findings(report)+(["Visual review (earlier, same version): "+f for f in earlier.get("findings",[])] if seen else [])}
        atomic_json(destination/"review.json",review)
        record.update(feedback={"numerical":report,"visual":review},status="failed",deferred_checks=[])
    elif images:
        key_now=review_key(job,current["assembly_readiness"],layout,dependency_geometry(run,job),report,[i["sha256"] for i in images])
        visual=current.get("feedback",{}).get("visual") or {}
        if current.get("review_key")==key_now and visual.get("verdict") in {"accept","repair"}:
            # Nothing the last review judged has changed: its verdict and deferrals stand.
            record.update(feedback={"numerical":report,"visual":visual},
                          status="candidate_ready" if visual["verdict"]=="accept" else "failed")
        else:
            due=True
    if not due:record["quality_key"]=audit_key
    run.commit("asset_review",record)
    if not due:run.state["jobs"][key]["dirty"]=False
    usable(run,run.state["jobs"][key])
    run.store.save()
    return due


def review_prefetch(run, jobs):
    """Send the visual reviews of one audit at once; each review in review_measured then builds
    the identical request (asset_review_request) and consumes the cached response."""
    jobs=[j for j in jobs if j["role"]=="asset"]
    if workers(run)<2 or len(jobs)<2:return
    from concurrent.futures import ThreadPoolExecutor
    from .evidence_state import session_path
    requests=[]
    for job in jobs:
        current=run.state["jobs"][job["id"]]
        bundle=task_bundle(run.catalogue,job)
        images=[image_record(safe_path(run.root,x),run.root,run.generation) for x in current["preview_paths"]]
        instruction,context,images,phase,_,artifact=run.asset_review_request(
            job,bundle,bundle["images"],images,current["feedback"]["numerical"],current,None)
        if hasattr(run.sandbox,"inspect_artifact") and session_path(run,artifact,phase,"asset",context,None)[1].exists():
            continue  # A saved review session answers from its own record.
        requests.append((instruction,context,images,phase))
    def review(request):
        try:
            thread_model(run).ask(*request)
        except Exception:
            pass  # The serial review meets and handles the same failure itself.
    with ThreadPoolExecutor(max_workers=workers(run)) as pool:
        list(pool.map(review,requests))


def review_measured(run, job, layout, audit_key):
    key=job["id"]; current=run.state["jobs"][key]
    destination=run.begin("asset_review")
    record={k:deepcopy(v) for k,v in current.items() if k not in {"stage","dirty","regenerate","quality_key"}}
    report=current["feedback"]["numerical"]
    images=[image_record(safe_path(run.root,x),run.root,run.generation) for x in current["preview_paths"]]
    bundle=task_bundle(run.catalogue,job)
    save_review_checkpoint(run,"asset",record)
    run.review_asset(job,bundle,bundle["images"],images,report,destination,record)
    record["review_key"]=review_key(job,current["assembly_readiness"],layout,dependency_geometry(run,job),report,[i["sha256"] for i in images])
    record["quality_key"]=audit_key
    run.commit("asset_review",record)
    run.state["jobs"][key]["dirty"]=False
    usable(run,run.state["jobs"][key])
    run.store.save()


QUALITY_REPAIRS = 3


def defect_score(record):
    """Failed numeric rows plus one for a visual repair verdict; lower is better."""
    feedback=record.get("feedback") or {}
    rows=(feedback.get("numerical") or {}).get("checks",[])
    return sum(r.get("status")=="fail" for r in rows)+((feedback.get("visual") or {}).get("verdict")=="repair")


def schedule_asset_repairs(run):
    """Send every audited, failed asset straight back to its author with the full evidence.

    Each asset has QUALITY_REPAIRS repairs of its own, apart from the draft allocation and the
    scene repair rounds. An asset whose last two repairs did not lower its defect score stops
    early. Stopped assets are recorded with their reason and stay in the scene as they are."""
    repairs=run.state.setdefault("asset_repairs",{})
    unresolved=run.state.setdefault("asset_repairs_unresolved",{})
    scheduled=[]
    for job in run.state["plan"]["jobs"]:
        key=job["id"]; current=run.state["jobs"].get(key)
        if not usable(run,current) or current.get("status")!="failed" or current.get("regenerate") or key in unresolved:continue
        history=repairs.setdefault(key,[])
        score=defect_score(current)
        if history:history[-1]["result_score"]=score
        scores=[h["score"] for h in history]+[score]
        if len(history)>=QUALITY_REPAIRS:
            unresolved[key]={"reason":f"Used all {QUALITY_REPAIRS} asset quality repairs","scores":scores,"artifact":current["artifact"]}
            continue
        if len(scores)>=3 and scores[-1]>=scores[-2]>=scores[-3]:
            unresolved[key]={"reason":"Two consecutive repairs did not reduce defects","scores":scores,"artifact":current["artifact"]}
            continue
        history.append({"artifact":current["artifact"],"score":score})
        current["regenerate"]="quality"
        scheduled.append(key)
    if scheduled:run.store.save()
    return scheduled


def failing_findings(report, limit=24):
    """Actionable author findings from failed check rows, with the checker's own hints."""
    out=[]
    for row in report.get("checks",[]):
        if row.get("status")!="fail":continue
        evidence=[]
        for key in ("collisions","disconnected_components","floating_articulations","surface_mismatches","problems","drifting_bodies"):
            for item in (row.get(key) or [])[:3]:
                evidence.append(item if isinstance(item,str) else {k:item[k] for k in item if k in
                    ("geom1","geom2","body","joint","depth_mm","repair_hint","free_directions","displacement_mm","fraction")})
        for side in ("high","low"):
            if row.get(f"{side}_blocking_contacts"):evidence.append({f"{side}_blocking":row[f"{side}_blocking_contacts"]})
        # Probe measurements: what the author has to change is in these numbers.
        measured={k:row[k] for k in ("rest_position","peak_position","final_position","stroke","engagement","unlock",
            "trigger","moved_when_active_mm","moved_when_released_mm","moved_when_all_released_mm","moved_after_opening_mm",
            "requires_opening","trapped_by","high_error_fraction","low_error_fraction","locked_motion_fraction") if row.get(k) not in (None,[],{})}
        if measured:evidence.append(measured)
        out.append(f"{row['id']}: check failed; basis: {str(row.get('detail',''))[:240]}; evidence: {evidence}"[:1600])
        if len(out)>=limit:break
    return out


def inherited_checks(run):
    rows=[]
    for key,job in sorted(run.state["jobs"].items()):
        if not usable(run,job):continue
        instances=[i["id"] for i in run.state["plan"]["instances"] if i["job"]==key]
        numerical=job.get("feedback",{}).get("numerical",{})
        for r in numerical.get("checks",[]):
            rows.append({**deepcopy(r),"id":"asset:"+key+":"+r["id"],"instances":instances,"asset_job":key})
        if numerical.get("status")=="error":
            rows.append({"id":"asset:"+key+":execution","status":"error","error_kind":numerical.get("error_kind","check_execution"),
                "detail":numerical.get("detail","Asset checker failed"),"instances":instances,
                "asset_job":key,"failure_scope":"asset_job",
                "execution_failure":deepcopy(numerical.get("execution_failure",{}))})
        visual=job.get("feedback",{}).get("visual",{})
        verdict=visual.get("verdict")
        rows.append({"id":"asset:"+key+":visual","status":"fail" if verdict=="repair" else "pass" if verdict=="accept" else "partial",
            "detail":"Asset visual review; see findings and unverified evidence", "instances":instances,
            "findings":visual.get("findings",[]),"unverified":visual.get("unverified",[])})
        if job.get("replacement_failure"):
            rows.append({"id":"asset:"+key+":replacement","status":"partial","instances":instances,
                "detail":"Latest replacement failed; scene preserves previous loadable asset", "failure":job["replacement_failure"]})
    return rows


def assess_scene(run,published):
    # Assets are finished before the scene is measured and judged: failed ones go straight back
    # to their authors, so a repair round measures the scene once, with the repaired versions.
    run.state['status']='reviewing_assets'
    run.progress("asset_quality", "audit_started")
    audit_assets(run)
    if schedule_asset_repairs(run):
        return None,None
    artifact=safe_path(run.root,published["artifact"])
    measurements=run.state.setdefault("scene_measurements",{})
    # A continuation that reruns a check the machine killed asks for one fresh measurement of that assembly.
    remeasure=published["assembly_key"] in run.state.get("remeasure",[])
    if remeasure:
        run.state["remeasure"].remove(published["assembly_key"])
        measurements.pop(published["assembly_key"],None)
    saved=None if remeasure else measurements.get(published["assembly_key"]) or next((s for s in reversed(run.state["stages"])
        if s["kind"] in {"scene_measurement", "scene_evidence"} and s.get("assembly_key")==published["assembly_key"]),None)
    if saved:
        report=deepcopy(saved["report"])
        images=[image_record(safe_path(run.root,x),run.root,run.generation) for x in saved["preview_paths"]]
    else:
        run.state['status']='measuring_scene'
        destination=run.begin("scene_measurement")
        report,renders=run.sandbox.check_scene(artifact,destination/"measurement",run.limits,run.contract)
        report=checked_report(report,domain="scene",mode="static",require_renders=True)
        report,images=load_images(run,report,renders)
        receipt={"report":deepcopy(report),"artifact":published["artifact"],"assembly_key":published["assembly_key"],"preview_paths":[i["path"] for i in images]}
        run.commit("scene_measurement",receipt)
        measurements[published["assembly_key"]]=receipt
    run.state["current_assessment"]={"assembly_key":published["assembly_key"],
        "artifact_sha256":published["artifact_sha256"],"report":deepcopy(report)}
    run.progress('scene_measurement', 'measurement_completed')
    if report['status'] != 'pass' and not (saved or {}).get('recovery_complete'):
        run.state['status']='recovering_scene_evidence'
        destination=run.begin('scene_evidence')
        report=run.recover_checks(report,artifact,destination)
        receipt={"report":deepcopy(report),"artifact":published["artifact"],"assembly_key":published["assembly_key"],
                 "preview_paths":[i['path'] for i in images],"recovery_complete":True}
        run.commit('scene_evidence',receipt)
        measurements[published['assembly_key']]=receipt
    published["preview_paths"]=[i["path"] for i in images]
    run.store.save()
    run.state["current_assessment"]={"assembly_key":published["assembly_key"],
        "artifact_sha256":published["artifact_sha256"],"report":deepcopy(report)}
    report["checks"]=report.get("checks",[])+inherited_checks(run)
    from .recovery import isolate_asset_failures
    report = isolate_asset_failures(report, run.state["plan"])
    missing = published.get("missing_jobs", [])
    report["checks"].append({"id":"assembly_missing_jobs","status":"partial" if missing else "pass",
        "complete":not missing, "missing_jobs":missing,
        "detail":"Assembly completeness; only states whether planned objects are all present, not that geometric quality passes"})
    report["status"]="error" if report.get("status")=="error" else report_status(report["checks"])
    run.state["current_assessment"]["report"]=deepcopy(report)
    run.state['status']='reviewing_scene'
    destination=run.begin("scene")
    review=None
    if images:
        from .prompts import SCENE_REVIEW
        from .review import SCENE_SCOPE,REQUIREMENTS_POLICY,scoped_review
        from .evidence import review_with_evidence
        originals=list({im["sha256"]:im for b in run.catalogue["bundles"].values() for im in b["images"]}.values())
        record={**{k:v for k,v in published.items() if k not in {"directory","kind","files"}},"report":report,"job_versions":{k:v["stage"] for k,v in run.state["jobs"].items()}}
        save_review_checkpoint(run,"scene",record)
        review,call=review_with_evidence(run,SCENE_REVIEW+REQUIREMENTS_POLICY+SCENE_SCOPE,
            {"plan":run.state["plan"],"checks":report,"fixed_contract":run.contract,"environment":run.env,
             "layout_scope":"The mobile layout only gives rough aisle and distance suggestions; do not require proof of complete operating trajectories at this stage. Necessary moving mechanisms, obvious obstructions and real instrument specifications must still be reviewed.",
             "protocol":run.catalogue.get("protocol"),"deferred_checks":run.deferred_checks(),
             "review_scope":{"stage":"scene","validation_mode":"static"},
             "layout_review_targets":review_targets(run.state["plan"])},
            originals+images,"scene_review",destination,
            validator=lambda v:scoped_review(v,"scene",run.deferred_checks(),preview=True,
                instance_ids={i["id"] for i in run.state["plan"]["instances"]},
                layout_targets=review_targets(run.state["plan"])),artifact=artifact,scope="scene")
        atomic_json(destination/"review.json",{"review":review,"call":call})
    record={**{k:v for k,v in published.items() if k not in {"directory","kind","files"}},"report":report,"review":review,
        "job_versions":{k:v["stage"] for k,v in run.state["jobs"].items()},"assessed":True,
        "assessment_implementation":run.implementation_guard.expected,
        "assessment_scope":"full_quality"}
    if digest(snapshot(artifact))!=published["artifact_sha256"]:raise StaleEvidence("Quality assessment mutated published scene")
    stage=run.commit("scene",record)
    run.state["published_scene"]={**record,"directory":stage["directory"]}
    run.store.save()
    return report,review


def void_stale_repairs(run):
    """A repair candidate is judged against the scene it started from. A candidate whose baseline
    was assessed by another checker implementation cannot be compared with it: it is voided, the
    assets go back to the baseline versions, and the baseline is assessed again first."""
    stages={s["directory"]:s for s in run.state["stages"]}
    for entry in run.state.get("repair_history",[]):
        scene=stages.get((entry.get("baseline") or {}).get("scene"))
        undecided=entry.get("status")=="applied" or entry.get("status")=="evaluated" and "selection" not in entry
        if not undecided or not scene or scene.get("assessment_implementation")==run.implementation_guard.expected:continue
        run.restore_baseline(entry)
        entry.update(status="voided",selection="voided",void_reason="The checker changed after the baseline scene was assessed; "
            "the candidate was not compared, its asset versions were set aside and the baseline is assessed again with the current checker")
    run.store.save()


def run_static(run):
    run.check_pause()
    import shutil
    shutil.rmtree(run.root/"prepared",ignore_errors=True)  # Results prepared by a stopped process.
    if "layout" in run.state["plan"] and not run.state.get("published_scene"):
        publish(run,[j["id"] for j in run.state["plan"]["jobs"]
                     if not usable(run,run.state["jobs"].get(j["id"]))])
    run.select_repair_candidate()  # Also finish a committed selection when the iteration budget is exhausted.
    latest=run.state.get('latest',{})
    if latest.get('assessed') and latest.get('assessment_implementation')!=run.implementation_guard.expected:
        void_stale_repairs(run)
        run.state['repair_pending']=False  # Reassess the baseline before planning against a new checker.
    while len(run.state["rounds"])<run.limits.max_iterations:
        if run.state.get("repair_pending") and not run.route_repair():break
        run.state["status"]="generating_candidates"
        missing=collect(run)
        from .local_repair import precheck
        if precheck(run):continue
        published=publish(run,missing)
        if getattr(run,"candidate_interruption",None):
            error=run.candidate_interruption
            del run.candidate_interruption
            raise error
        if published is None:
            if run.state.get("repair_pending"):continue
            break
        if (published.get("assessed") and published.get('assessment_implementation')==run.implementation_guard.expected
                and published.get('assessment_scope')=='full_quality'
                and not any(j.get("replacement_failure") for j in run.state["jobs"].values())):
            report,review=published["report"],published.get("review")
            if any(a.get("status")=="applied" for a in run.state.get("repair_history",[])):
                # A model may change source without changing geometry: evaluate that executed repair.
                run.begin("scene")
                record={k:deepcopy(v) for k,v in published.items() if k not in {"directory","kind","files"}}
                record["job_versions"]={k:v["stage"] for k,v in run.state["jobs"].items()}
                stage=run.commit("scene",record)
                run.state["published_scene"]={**record,"directory":stage["directory"]}
        else:
            report,review=assess_scene(run,published)
            if report is None:continue  # Failed assets went back to their authors first.
        if run.select_repair_candidate():continue
        action=decide(report,review)
        run.state["last_decision"]={"action":action,"scene":run.state["published_scene"]["directory"]}
        if action!="repair":
            run.state["status"]="incomplete_scene" if missing else action
            run.state["stop_reason"]={"reason":"no_runnable_tasks",
                "decision":action,"missing_jobs":missing,
                "detail":"Available candidates, checks and reliable repairs processed; gaps and local failures of remaining jobs are kept in the record"}
            break
        run.state["repair_pending"]=True
        run.store.save()
    else:run.state["status"]="scene_repair_budget_exhausted"
    run.store.save()
    return run.result()


def write_status_page(run,result):
    """Unaccepted scenes remain visible; the page never changes artifact files."""
    import html,json
    assembly=result["assembly"]
    title="Complete diagnostic scene" if assembly["complete"] else "Partial diagnostic scene (parts missing)"
    if not assembly["available"]:title="No loadable scene yet"
    verdict="Passed current static acceptance" if assembly["quality_accepted"] else "Did not pass final acceptance"
    body=f'<h1>{title}</h1><p>{verdict}; status: {html.escape(result["status"])}</p>'
    if result.get("scene_artifact"):
        artifact=safe_path(run.root,result["scene_artifact"])
        body+=f'<p><a href="{(artifact/"model.xml").as_uri()}">MuJoCo model</a> (resource files must stay with the directory)</p>'
        body+='<pre>'+html.escape(f".\\run.ps1 preview --run {run.root.name}")+'</pre>'
    body+='<h2>Missing jobs</h2><pre>'+html.escape(json.dumps(assembly["missing_jobs"],ensure_ascii=False))+'</pre>'
    if assembly.get('ready_unassembled_jobs'):
        body+='<h2>Saved assets awaiting assembly</h2><pre>'+html.escape(json.dumps(assembly['ready_unassembled_jobs'],ensure_ascii=False,indent=2))+'</pre>'
    for path in run.state.get("published_scene",{}).get("preview_paths",[])[:3]:
        body+=f'<img style="max-width:480px" src="{safe_path(run.root,path).as_uri()}">'
    body+='<h2>Current evidence and repair progress</h2><pre>'+html.escape(json.dumps(result.get('quality_progress',{}),ensure_ascii=False,indent=2))+'</pre>'
    if result.get('repair_progress'):
        last=result['repair_progress'][-1]
        body+='<p>Latest repair: '+html.escape(str(last.get('outcome','pending recheck'))+' · '+str(last.get('selection','pending selection')))+'</p>'
    body+='<p><a href="scene_document.md">Scene delivery notes</a> · <a href="result.json">Full diagnostics and repair record</a>; only a check summary follows.</p><h2>Check facts and evidence gaps</h2>'
    for row in assembly["issues"]:
        if row.get("status")=="pass":continue
        body+='<details open><summary>'+html.escape(row["id"]+' · '+row.get("status","unknown"))+'</summary><pre>'
        body+=html.escape(json.dumps({k:row[k] for k in ("id","status","instances","complete","collision_count","measurement_error_count") if k in row},ensure_ascii=False,indent=2))+'</pre></details>'
    page='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px system-ui;margin:32px;max-width:1200px}pre{white-space:pre-wrap}details{margin:12px 0}</style>'+body
    target=run.root/'scene_status.html'
    temporary=run.root/'scene_status.tmp'
    temporary.write_text(page,encoding='utf-8');temporary.replace(target)
