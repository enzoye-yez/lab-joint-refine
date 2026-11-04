"""Scene orchestration over the shared ledger: assets, devices, assembly, checks, targeted repair."""

from dataclasses import replace
from pathlib import Path

from asset_generation.contracts import author_result, check_profile, require
from asset_generation.inputs import public_bundle, image_record
from asset_generation.model import GenerationModel, request_summary
from asset_generation.pipeline import snapshot
from asset_generation.prompts import AUTHOR, REVIEW
from asset_generation.review import author_feedback, compact_report
from asset_prep.storage import Store, atomic_json, digest, file_hash, safe_path
from asset_prep import implementation
from asset_prep.errors import PrepError, StaleEvidence, UncertainRequest, ValidationError, ProviderError, ContextLimitError
from .contracts import validate_plan, repair_action
from .feedback import planning_feedback
from .inputs import task_bundle
from .prompts import PLAN, REPAIR, SCENE_REVIEW, for_environment
from .layout import revision as layout_revision, validate_layout, table_policy
from .layout import review_targets
from .review import ASSEMBLABLE, ASSET_SCOPE, SCENE_SCOPE, REQUIREMENTS_POLICY, scoped_review, deferred_records
from .evidence import review_with_evidence
from .checkpoint import save_review_checkpoint, recover_review_checkpoint
from .table import validate_cuts
from .recovery import decide, gaps as check_gaps
from asset_generation.check_report import checked_report, checked_previews, incomplete
from .edge_mounts import POLICY as EDGE_MOUNT_POLICY
from .dispatch import recover
from .capabilities import semantic, problem_identity, repair_signature, repair_mechanism, repeated_repair
from .robot_contract import mount_for, joint_catalogue, validate_model_posture


def code_identity(project):
    return implementation.implementation_digest(implementation.source_manifest(project))


def evidence_stop_status(outcome):
    reason=(outcome or {}).get('reason')
    if reason in {'capability_blocked','inspection_unavailable'}:
        return 'needs_check_support'
    if reason in {'evidence_planning_exhausted','evidence_rounds_exhausted','evidence_no_progress'}:
        return reason
    return 'needs_review'


class SceneRun:
    def __init__(self, catalogue, env, contract, provider, generation, limits, keys, root, transport,
                 sandbox, robot_root, project, resume=False, retry_uncertain=False, provenance="model_generated",
                 expected_implementation=None, shared_store=None, shared_model=None, before_candidate=None, before_repair=None):
        self.catalogue, self.env, self.contract = catalogue, env, contract
        self.generation, self.limits = generation, limits
        self.root, self.robot_root = Path(root), Path(robot_root)
        from .benchmark_input import bind_environment
        bind_environment(catalogue, env, self.robot_root)
        self.robot_joints = joint_catalogue(self.robot_root,env)
        self.sandbox = sandbox
        binding = implementation.runtime_record(
            project,
            inputs={"catalogue": catalogue["identity"], "environment": env, "contract": contract,
                    "robot_files": snapshot(robot_root)},
            configuration={"generation": generation.identity(), "limits": limits.identity(),
                           "provider": provider.provider, "model": provider.model,
                           "base_url": provider.base_url,
                           "output_token_parameter": provider.output_token_parameter})
        if provenance == 'model_generated':
            import sys
            active = getattr(sys, '_harness_runtime', None)
            if (not active or Path(active['base']) != (Path(project)/'src').resolve()
                    or expected_implementation != binding['implementation']):
                raise implementation.ImplementationDrift('A production scene must start from frozen source bound to the specified verified version')
        identity = {"catalogue":catalogue["identity"],"environment":env,"contract":contract,
                    "provider":provider.provider,"model":provider.model,"base_url":provider.base_url,
                    "output_token_parameter":provider.output_token_parameter,"generation":generation.identity(),
                    "limits":limits.identity(),"robot_files":snapshot(robot_root),"backend":sandbox.preflight(),
                    "implementation":binding["implementation"],"binding":binding,"provenance":provenance}
        # Workers and supplementary checks must run the same source; verify at startup so no new code slips in mid-run.
        if hasattr(sandbox, 'bind_implementation'):
            sandbox.bind_implementation(binding["implementation"])
        implementation.verify(identity["implementation"], project=project, where="main process startup")
        self.implementation_guard = implementation.Guard(project, identity['implementation'])
        self.store = shared_store or Store(root,identity,replace(provider,max_model_calls=limits.max_model_calls,
                           max_reserved_tokens=limits.max_reserved_tokens),resume=resume,retry_uncertain=retry_uncertain)
        if shared_store is not None:
            require(Path(shared_store.root).resolve()==self.root.resolve(), "The shared ledger and the scene root directory must match")
            previous_binding=self.store.state.setdefault("joint_scene_binding",identity)
            require(previous_binding==identity,"The joint scene inputs or the frozen configuration changed")
        self.before_candidate=before_candidate or (lambda job,stage: None)
        if shared_store is None:
            from asset_prep.phase_budget import enter
            enter(self.store,'scene')
        self.before_repair=before_repair or (lambda action: None)
        if not retry_uncertain and any(c["status"] == "pending" for c in self.store.state["calls"]):
            raise UncertainRequest("Scene request outcome unknown; retry explicitly only after reconciling the billing record")
        self.state = self.store.state.setdefault("scene",{"status":"running","stages":[],"jobs":{},"rounds":[],"feedback_receipts":[]})
        if "plan" in self.state:
            validate_plan(self.state["plan"], self.catalogue["asset_ids"], self.limits, self.env)
            self.validate_prepared_plan(self.state['plan'])
            validate_model_posture(self.state["plan"],self.robot_joints)
        for stage in self.state["stages"]:
            if snapshot(safe_path(self.root,stage["directory"])) != stage["files"]:
                raise StaleEvidence("The archived scene version changed; reuse is forbidden")
        if self.state.get("pending") and (not self.state["pending"].get("awaiting_exchange") or self.state["pending"].get("review_checkpoint")):
            recover_review_checkpoint(self)
        self.provider, self.keys, self.transport = provider, keys, transport
        self.model = shared_model or GenerationModel(provider,generation,keys,self.store,transport)
        self.model.execution_guard = self.check_pause
        self.model.cancel_guard = self.check_cancel
        self.sandbox.cancel_guard = self.check_cancel
        self.model.progress = self.progress
        implementation.archive_sources(project, root, identity['implementation'])
        self.store.save()

    def progress(self, phase, event):
        from datetime import datetime, timezone
        self.state["activity"] = {"phase":phase,"event":event,"updated_at":datetime.now(timezone.utc).isoformat()}
        self.store.save()
        if self.state.get("plan"):
            self.result()

    def begin(self, kind):
        self.check_pause()
        pending=self.state.get("pending")
        if pending and pending.get("awaiting_exchange"):
            require(pending["kind"]==kind,"A stage awaiting a response cannot resume as a different kind")
            return safe_path(self.root,pending["directory"])
        relative = f"stages/{len(self.state['stages'])+1:05d}_{kind}"
        destination = safe_path(self.root,relative)
        require(not destination.exists(), "Workspace directory conflict; preserving the current state and stopping")
        self.state["pending"] = {"directory":relative,"kind":kind}
        self.progress(kind, "stage_started")
        self.store.save()
        destination.mkdir(parents=True)
        return destination

    def check_pause(self):
        self.implementation_guard.check()
        self.check_cancel()

    def check_cancel(self):
        if (self.root / "pause.request").exists():
            raise PrepError("Pause requested; preserving the current state and the unknown-request ledger. Local network processes will be terminated; remote request outcomes may remain unknown")

    def commit(self, kind, record):
        self.implementation_guard.check()
        pending = self.state["pending"]
        destination = safe_path(self.root,pending["directory"])
        atomic_json(destination/"record.json", {"kind":kind,**record})
        stage = {"directory":pending["directory"],"kind":kind,"files":snapshot(destination),**record}
        self.state["stages"].append(stage)
        if kind in {"asset","asset_review"}:
            self.state["jobs"][record["job"]] = {**record,"stage":stage["directory"],
                "artifact_stage":record.get("artifact_stage",stage["directory"]),
                "artifact":record.get("artifact",stage["directory"]+"/artifact"),
                "dirty":record["status"] not in ASSEMBLABLE}
        elif kind == "scene":
            self.state["rounds"].append(stage["directory"])
            self.state["latest"] = {**record,"directory":stage["directory"]}
            self.state["feedback"] = record["report"]
            self.state.pop("scene_review_resume",None)
            self.finish_repair(record['report'],record.get('review'))
        elif kind == "repair":
            self.state["repair_pending"] = False
        self.state.pop("pending")
        self.store.save()
        return stage

    def images(self, directory, report):
        return [image_record(safe_path(directory,name),self.root,self.generation) for name in report.get("renders",[])]

    def validate_prepared_plan(self, plan):
        if self.catalogue.get('prepared_task'):
            from asset_prep.task_delivery import validate_role_bindings
            validate_role_bindings(plan, self.catalogue['prepared_task'])

    def initialize(self):
        self.check_pause()
        if "plan" in self.state:
            return
        feedback = None
        for attempt in range(self.limits.max_plan_attempts):
            try:
                instruction=for_environment(PLAN,self.env)
                if self.catalogue.get('prepared_task'):
                    instruction+='\nThis is a defined task dossier. Keep the selected procedure, initial state, final state and instance counts. Also return role_bindings, each item {role_id:upstream role id,instances:[actual scene instance ids]}, covering every role with matching counts. Each instance\'s generation job must include the asset corresponding to its role. Different uses may share a geometry type, but objects that should be independent must not be shared.'
                value, call = self.model.ask(instruction,{"source":self.catalogue["source"],"asset_ids":self.catalogue["asset_ids"],
                    "protocol":self.catalogue.get("protocol"),
                    "environment":self.env,"robot_joint_contract":self.robot_joints,"limits":{"max_jobs":self.limits.max_jobs,"max_instances":self.limits.max_instances},
                    "table_edit_policy":table_policy(self.env),
                    "edge_mount_policy":EDGE_MOUNT_POLICY,
                    "contract":self.contract,"validation_feedback":feedback,"attempt":attempt},[],"scene_plan")
                plan, order = validate_plan(value,self.catalogue["asset_ids"],self.limits,self.env)
                self.validate_prepared_plan(plan)
                validate_model_posture(plan,self.robot_joints)
                validate_layout(plan,self.env)
                validate_cuts(plan.get("table_modifications", []), plan, self.env)
                require(set(self.contract["jobs"]) <= {j["id"] for j in plan["jobs"]}, "Numeric acceptance references a job not in the plan")
                break
            except ValidationError as exc:
                feedback = str(exc)
        else:
            raise ValidationError("Scene plan format repair attempts exhausted: "+feedback)
        destination = self.begin("plan")
        atomic_json(destination/"plan.json",plan)
        atomic_json(destination/"input.json",{"source":self.catalogue["source"],"protocol":self.catalogue.get("protocol"),
                    "identity":self.catalogue["identity"],"contract":self.contract,"environment":self.env})
        self.state.update(plan=plan,order=order)
        self.commit("plan",{"call":call})

    def dependency_context(self, job):
        return {key:{"manifest":self.state["jobs"][key]["manifest"],
                     "assembly_readiness":self.state["jobs"][key].get("assembly_readiness",{}),
                     "interface_checks":self.state["jobs"][key].get("feedback",{}).get("numerical",{}).get("checks",[]),
                     "geometry":self.state["jobs"][key].get("geometry_summary",{}),
                     "status":self.state["jobs"][key]["status"],
                     "deferred_checks":self.state["jobs"][key].get("deferred_checks",[]),
                     "instances":[i for i in self.state['plan']['instances'] if i['job']==key],
                     "version":self.state["jobs"][key]["stage"],"source_requirements":next(j["requirements"] for j in self.state["plan"]["jobs"] if j["id"] == key)}
                for key in job["depends_on"] if key in self.state["jobs"] and self.state["jobs"][key].get("source")}

    def layout_context(self):
        return {"revision":layout_revision(self.state['plan']),
                "layout":self.state['plan'].get("layout"),
                "robot":self.env["robot"],
                "end_effectors":self.state["plan"].get("end_effectors"),
                "robot_posture":self.state["plan"].get("robot_posture",{}),
                "instances":self.state['plan']['instances'],
                "rule":"Non-robot instances use world coordinates in meters and yaw in radians about the world Z axis. Assembly formula: world=position+Rz(yaw)@local. "
                       "Dependency interfaces and geometry are in each asset's local frame; transform them to world by their actual instance transform, "
                       "then into this device's local frame. Robot tool local coordinates are relative to the fixed flange. "
                       "This layout is the only authoritative source; do not assume or substitute dependency positions."}

    def instance_contract(self, job):
        instances = [i for i in self.state["plan"]["instances"] if i["job"] == job["id"]]
        return {"instances":instances,
                "delivery_role":job["role"], "manifest_asset_ids":job["asset_ids"],
                "reference_only_jobs":job["depends_on"],
                "robot_mounts":[mount_for(self.state["plan"],self.env,i["id"]) for i in instances if i["placement"]=="robot"],
                "scope_rule":"An asset role delivers the specified asset; a tool/fixture role delivers only that device, and asset_ids is its service scope and must be written verbatim into the inventory. dependencies are only for reading interfaces and mating reference; they never add deliverables.",
                # One unit per model only when every instance is the same configuration; a
                # configuration may legitimately compose several units (a rack with its tubes).
                "expected_model_units":1 if job["role"] == "asset" and len(job["asset_ids"]) == 1 and len(instances) > 1
                                       and len({i.get("configuration") for i in instances}) == 1 else None,
                "rule":"Each scene instance copies this model once; do not repeat the full asset inside one model to match the scene total"}

    def dependency_images(self, job, required=True):
        """The first render of each dependency: what a tool is designed and judged against."""
        images = []
        for dependency_id in job["depends_on"]:
            previews = self.state["jobs"][dependency_id].get("preview_paths",[])
            require(not required or previews, "Operation device design dependencies lack verified renders")
            if previews:
                images.append(image_record(safe_path(self.root,previews[0]),self.root,self.generation))
        return images

    def review_asset(self, job, bundle, references, previews, report, destination, record):
        self.check_pause()
        if job["role"] == "tool":
            # The end-effector is a first working draft: its static checks gate it, and its use
            # (grasps, actuation, release) is judged and repaired in the robot-operation stage,
            # where trajectories show what it has to do. A visual review here cannot judge that.
            from .tool_contract import OPERATION_REVIEW, OPERATION_CHECK
            atomic_json(destination/"review.json",OPERATION_REVIEW)
            record["deferred_checks"] = [{**OPERATION_CHECK,"id":job["id"]+":tool_operation","job":job["id"]}]
            record["feedback"] = {"numerical":report,"visual":dict(OPERATION_REVIEW)}
            record['layout_revision'] = self.layout_context()['revision']
            record["status"] = "candidate_ready"
            return record["feedback"]
        instruction, context, images, phase, validator, artifact = self.asset_review_request(
            job, bundle, references, previews, report, record, destination)
        review, call = review_with_evidence(self,instruction,context,images,phase,destination,validator=validator,
                                  artifact=artifact,scope="asset")
        atomic_json(destination/"review.json",review)
        record["review_call"] = call
        record['evidence_outcome']=self.state.get('evidence_outcomes',{}).get('asset_review:'+job['id'])
        record["deferred_checks"] = deferred_records(review,job["id"],destination.name)
        record["feedback"] = {"numerical":report,"visual":review}
        record['layout_revision'] = self.layout_context()['revision']
        record["status"] = {"accept":"candidate_ready","uncertain":"needs_review","repair":"failed"}[review["verdict"]]
        return record["feedback"]

    def review_dependencies(self, job):
        """What an asset review reads of its dependencies: how they are built and placed. Their
        review status and stage versions are left out, so a dependency's re-review alone does not
        change this review's request (and so its cached response stays usable)."""
        return {key:{"manifest":dependency["manifest"],"geometry":dependency.get("geometry_summary",{}),
                     "artifact":dependency.get("artifact"),
                     "instances":[i for i in self.state['plan']['instances'] if i['job']==key],
                     "source_requirements":next(j["requirements"] for j in self.state["plan"]["jobs"] if j["id"] == key)}
                for key in job["depends_on"] for dependency in [self.state["jobs"].get(key,{})] if dependency.get("source")}

    def asset_review_request(self, job, bundle, references, previews, report, record, destination):
        """The exact first review request (and how its answer is validated); shared by the
        parallel review prefetch and the review itself."""
        if job["role"] != "asset":
            # The served instruments were judged against their photos in their own reviews; a
            # tool is judged against them as built, one render each, not every original photo.
            references, bundle = [], {**bundle, "images": self.dependency_images(job, required=False)}
        scene=self.state.get("published_scene",{})
        scene_previews=scene.get("preview_paths") or self.state.get("latest",{}).get("preview_paths",[])
        scene_images=[image_record(safe_path(self.root,path),self.root,self.generation) for path in scene_previews[:3]]
        images = list({im["sha256"]:im for im in references + bundle["images"] + previews + scene_images}.values())
        context = {"input":public_bundle(bundle),"job":job,"checks":compact_report(report),
                   "dependencies":self.review_dependencies(job),"reference_policy":record["reference_policy"],
                   "fixed_contract":self.contract["jobs"].get(job["id"],{}),
                   "instance_contract":self.instance_contract(job),
                   "scene_layout":self.layout_context(),
                   "layout_revalidation":record.get('layout_revalidation'),
                   "repair_directive":self.state.get("repair_directive"),
                   "review_scope":{"stage":"asset","validation_mode":self.limits.validation_mode},
                   "evidence":{"original_reference_images":[im["path"] for im in references],
                               "design_reference_images":[im["path"] for im in bundle["images"]],
                               "current_asset_images":[im["path"] for im in previews],
                               "assembled_scene_available":bool(self.state.get("published_scene")),
                               "assembled_scene_artifact":self.state.get("published_scene",{}).get("artifact"),
                               "assembled_scene_images":[im["path"] for im in scene_images]}}
        if record.get("reviewed_from"):
            context["previous_review"] = record.get("feedback",{}).get("visual")
            context["review_retry_from"] = record["reviewed_from"]
        if record.get('layout_revalidation'):
            context['current_model_py']=safe_path(self.root,record['source']).read_text(encoding='utf-8')
            context['current_geometry']=record.get('geometry_summary',{})
        artifact = safe_path(self.root,record["artifact"]) if record.get("artifact") else destination/"artifact"
        validator = lambda value: scoped_review(value,"asset",expected_units=context["instance_contract"]["expected_model_units"])
        return REVIEW+REQUIREMENTS_POLICY+ASSET_SCOPE, context, images, "asset_review:"+job["id"], validator, artifact

    def failure_history(self, key, limit=6):
        """Rejected designs with the reason each was rejected.

        Repair feedback used to keep only the most recent error. An author then fixed
        error A, reintroduced the design that had been rejected for error B, and the
        loop alternated indefinitely. Every attempt keeps its actionable reason here.
        """
        rows=[]
        # Audits of a version are rejections too; only the last audit of each version counts.
        audits={}
        for stage in self.state.get("stages",[]):
            if stage.get("kind")=="asset_review" and stage.get("job")==key:
                audits[stage.get("artifact")]=stage
        for stage in self.state.get("stages",[]):
            if stage.get("job")!=key or not (stage.get("kind")=="asset" or
                    stage.get("kind")=="asset_review" and audits.get(stage.get("artifact")) is stage):continue
            feedback=stage.get("feedback") or {}
            numerical=feedback.get("numerical") or {}
            reasons=[{"check":row.get("id"),"status":row.get("status"),
                      "detail":str(row.get("detail") or "")[:600]}
                     for row in numerical.get("checks",[]) if row.get("status") in {"fail","error"}]
            if feedback.get("generation_error"):
                reasons.append({"check":"generation_error","status":"error",
                                "detail":str(feedback["generation_error"])[:600]})
            visual=feedback.get("visual") or {}
            if visual.get("verdict")=="repair" and visual.get("source")!="numeric":
                reasons.extend({"check":"visual_review","status":"fail","detail":str(f)[:600]} for f in visual.get("findings",[])[:8])
            if feedback.get("task_deferred"):
                reasons.append({"check":"task_deferred","status":"error",
                                "detail":str(feedback["task_deferred"])[:600]})
            rows.append({"stage":stage.get("directory"),"reasons":reasons})
        # Scene repairs that reverted this asset, or that it answered with the same geometry.
        for entry in self.state.get("repair_history",[]):
            reasons=[{"check":d["check"],"status":"regressed","detail":str(d.get("added") or d.get("detail") or "")[:600]}
                     for d in entry.get("revert_reasons",{}).get(key,[])]
            if key in entry.get("unchanged_jobs",[]):
                reasons.append({"check":"unchanged_geometry","status":"fail",
                    "detail":"The repair returned geometry identical to the version it was asked to fix; the repair task was not addressed"})
            if reasons:rows.append({"stage":entry.get("salvage",{}).get("scene") or entry.get("scene") or "scene_repair","reasons":reasons})
        out=[row for row in rows if row["reasons"]][-limit:]
        if not out:return []
        # Collapse repeated identical rejections into a count; iteration order is kept.
        grouped={}
        for row in out:
            signature=tuple(sorted((r["check"],r["status"],r["detail"]) for r in row["reasons"]))
            grouped.setdefault(signature,{"stage":row["stage"],"reasons":row["reasons"],"occurrences":0})
            grouped[signature]["occurrences"]+=1
        history=list(grouped.values())
        history[-1]["latest"]=True
        return history

    def author_inputs(self, job, feedback=None, *, draft=False):
        """Inputs shared by every author request of this job version; no side effects."""
        current = self.state["jobs"].get(job["id"])
        previous_source = (current or {}).get("source")
        previous = safe_path(self.root,previous_source).read_text(encoding="utf-8") if previous_source else None
        if feedback is None and current:
            feedback = current.get("feedback")
        bundle = task_bundle(self.catalogue,job)
        references = list(bundle["images"])
        reference_policy = "Instrument generation uses all original reference images of the selected asset"
        if job["role"] != "asset":
            bundle["original_image_inventory"] = [{k:v for k,v in im.items() if k != "data_url"} for im in bundle["images"]]
            bundle["images"] = self.dependency_images(job, required=not draft)
            reference_policy = ("Operation device design uses actual renders and interfaces of generated dependencies; original images served the earlier asset generation, and quality is reviewed after the scene is saved; a missing image does not mean quality passed"
                                if draft else "Operation device design uses the first actual render view and measured interfaces of each generated dependency; original images were fully reviewed in the earlier asset stage, and full multi-view evidence is kept in each version")
        profile = check_profile({**self.contract["jobs"].get(job["id"],{}), "validation_mode":self.limits.validation_mode})
        previews = [image_record(safe_path(self.root,path),self.root,self.generation)
                    for path in (current or {}).get("preview_paths",[])]
        return dict(current=current, previous_source=previous_source, previous=previous, feedback=feedback,
                    bundle=bundle, references=references, reference_policy=reference_policy, profile=profile,
                    previews=previews)

    def author_request(self, job, inputs, attempt):
        """The exact author request (instruction, context, images, phase); shared by prefetch and generation."""
        key = job["id"]
        bundle, previous, feedback = inputs["bundle"], inputs["previous"], inputs["feedback"]
        profile, reference_policy, previews = inputs["profile"], inputs["reference_policy"], inputs["previews"]
        from asset_generation.authoring import capabilities
        context = {"authoring_capabilities":capabilities(self.generation,self.store.state.get("joint_backend",self.store.state["identity"].get("backend"))),"input":public_bundle(bundle),"job":job,"dependencies":self.dependency_context(job),
                   "environment":self.env,"profile":profile,"current_model_py":previous,"feedback":author_feedback(feedback),
                   "table_edit_policy":table_policy(self.env),"table_modifications":self.state["plan"].get("table_modifications", []),
                   "edge_mount_policy":EDGE_MOUNT_POLICY,"edge_mounts":self.state['plan'].get('edge_mounts',[]),
                   "fixed_contract":self.contract["jobs"].get(key,{}),
                   "instance_contract":self.instance_contract(job),
                   "scene_layout":self.layout_context(),
                   "repair_directive":self.state.get("repair_directive"),
                   "repair_tasks":self.state.get("repair_tasks",{}).get(key,[]),
                   "rejected_attempts":self.failure_history(key),
                   "task_recovery":self.state.get("task_recovery",{}).get(key),
                   "reference_policy":reference_policy,"scene_round":len(self.state["rounds"]),"attempt":attempt}
        if context["task_recovery"]:
            context["rejected_attempts"]=context["rejected_attempts"][-1:]
        instruction = AUTHOR+REQUIREMENTS_POLICY
        instruction += "\nFollow instance_contract: generate exactly one copyable unit; the framework handles the scene instance count. repair_directive is this round's repair decision and must be carried out against the specific failure evidence; when dependencies change, verify the actual new interfaces instead of only editing declarations. scene_layout is the actual assembly layout: design fixture-target mating through its coordinate transforms, and never use world coordinates directly in the local model."
        instruction += ("\nrejected_attempts lists this job's previously rejected designs and their reasons; occurrences is how many times the same rejection repeated. "
                        "When fixing the current error, do not reintroduce another representation that was already rejected; if two representations each trigger a different constraint, "
                        "the representation this part needs has not been declared correctly yet: redeclare it using the valid category and constraint type the checker gives, "
                        "instead of switching between two known rejected options. A reason with status regressed is a scene repair "
                        "of this asset that was reverted because the listed checks got worse; unchanged_geometry means a repair did "
                        "not change the model at all.")
        instruction += ("\nThe model's default pose (every joint at 0) is the instance's initial state in the scene. Scene assembly "
                        "only places each instance by its position and yaw; it never folds, loads, opens or moves parts, and asset "
                        "keyframes are dropped. Build each part in the state the protocol starts in."
                        "\nWhere a hose, tube or cable ends on a port of another instance or configuration, compute the port in "
                        "this model's local frame from scene_layout, run the last segment coaxially with the port axis, and mate "
                        "without interpenetration: a hose bore fits over a barb or nozzle whose outer radius is smaller than the bore, "
                        "or the hose stops 0.5 mm short of a solid face. Never end a hose inside solid port geometry."
                        "\nFace the operator along local -Y: the side a person works from (controls, door, lid hinge opposite, "
                        "loading slot) is the -Y face, and operation interfaces declare their approach_axis. The framework turns "
                        "that face to the robot aisle; keep controls the robot must use on that face and within the front "
                        "0.30 m of the footprint where the instrument allows.")
        if self.limits.validation_mode == "static":
            instruction += "\nOnly static scene appearance and geometry are accepted now. Still build moving joints, buttons, connections and interfaces per the dossier; skipping operation verification this round does not permit simplifying a movable instrument into an immovable shell."
        if job["role"] != "asset":
            instruction += "\nGenerate only the operation device specified by job.role this time; do not regenerate the served instrument. asset_ids is the service scope. Complete the mating design from the actual dependencies interfaces."
            from .layout import author_layout_policy
            instruction += "\n" + author_layout_policy(self.env)
        if job["role"] == "tool":
            from .tool_contract import TOOL_POLICY
            instruction += "\n" + TOOL_POLICY
        return instruction, context, bundle["images"]+previews, "asset:"+key

    def generate_job(self, job, feedback=None, *, draft=False):
        key = job["id"]
        inputs = self.author_inputs(job, feedback, draft=draft)
        current, previous_source, previous = inputs["current"], inputs["previous_source"], inputs["previous"]
        recheck_required = bool(current and current.get('needs_implementation_recheck'))
        # Decided on the caller's feedback, before falling back to the stored one.
        review_again = current and (recheck_required or current["status"] == "review_pending" or
                                    feedback is None and current["status"] == "needs_review")
        feedback, bundle, references = inputs["feedback"], inputs["bundle"], inputs["references"]
        reference_policy, profile, previews = inputs["reference_policy"], inputs["profile"], inputs["previews"]
        if not draft and current and current["status"] in {"needs_check_support","needs_environment"}:
            from asset_generation.check_recovery import recover_checks
            destination=self.begin("asset_review")
            report=current["feedback"]["numerical"]
            if report["status"]=="error" and hasattr(self.sandbox,"recheck_asset"):
                report,renders=self.sandbox.recheck_asset(safe_path(self.root,current["artifact"]),
                    destination/"recheck",job["asset_ids"],profile)
                report=checked_report(report,domain="asset",mode=profile.get("validation_mode","full"),require_renders=True)
                report,previews=checked_previews(report,renders,
                    lambda directory,name:image_record(safe_path(directory,name),self.root,self.generation))
            if report["status"]!="error":
                report=recover_checks(self.sandbox,safe_path(self.root,current["artifact"]),report,
                    self.root/"asset_check_recovery",implementation=self.implementation_guard.expected,contract=profile,
                    check_pause=self.check_pause,max_actions=self.limits.max_check_actions,max_retries=self.limits.max_check_retries)
            record={k:v for k,v in current.items() if k not in {"stage","dirty"}}
            record.update(preview_paths=[im["path"] for im in previews],manifest=report.get("manifest"),
                geometry_summary=report.get("geometry_summary",{}))
            record["feedback"]={"numerical":report}
            record["status"]={"partial":"needs_check_support","error":"needs_environment","fail":"failed","pass":"review_pending"}[report["status"]]
            if report["status"]=="pass" and previews:
                save_review_checkpoint(self,"asset",record)
                self.review_asset(job,bundle,references,previews,report,destination,record)
            self.commit("asset_review",record)
            if record["status"] in ASSEMBLABLE:return True
            if report["status"]=="fail":return self.generate_job(job,record["feedback"])
            self.state["status"]=record["status"]
            self.store.save()
            return False
        # Review evidence is insufficient or re-review is needed after an upgrade: keep the compiled assets and do not modify geometry blindly.
        if not draft and review_again and (recheck_required or current.get("feedback",{}).get("numerical",{}).get("status") == "pass") and previews:
            destination = self.begin("asset_review")
            report = current["feedback"]["numerical"]
            record = {k:v for k,v in current.items() if k not in {"stage","dirty"}}
            if record.pop('needs_implementation_recheck', False):
                report, renders = self.sandbox.recheck_asset(safe_path(self.root,current['artifact']),
                    destination, job['asset_ids'], profile)
                self.implementation_guard.check()
                record['feedback'] = {'numerical':report}
                if report['status'] != 'pass':
                    # Keep geometry and return genuinely new evidence; program/measurement problems do not enter geometry repair.
                    record['status'] = 'failed' if report['status']=='fail' else 'needs_environment'
                    if report['status'] != 'fail':
                        record['needs_implementation_recheck'] = True
                    self.commit('asset_review', record)
                    self.state['status'] = 'needs_review' if report['status']=='fail' else 'needs_environment'
                    self.store.save()
                    return self.generate_job(job, record['feedback']) if report['status']=='fail' else False
                previews = self.images(renders, report)
                record['preview_paths'] = [image['path'] for image in previews]
            if not record.pop("interrupted_review",False):
                record["reviewed_from"] = current["stage"]
            record.update(reference_policy=reference_policy,
                          artifact_stage=current.get("artifact_stage",current["stage"]))
            atomic_json(destination/"checks.json",report)
            save_review_checkpoint(self, "asset", record)
            feedback = self.review_asset(job,bundle,references,previews,report,destination,record)
            stage = self.commit("asset_review",record)
            self.state["jobs"][key] = {**record,"stage":stage["directory"],"dirty":record["status"] not in ASSEMBLABLE}
            self.store.save()
            if record["status"] in ASSEMBLABLE:
                return True
            if record["status"] == "needs_review":
                self.state["status"] = evidence_stop_status(record.get('evidence_outcome'))
                return False
        # Full-mode callers and resumes share committed author attempts too.
        # Static collection reserves its own cumulative attempts before entering.
        consumed=sum(s["kind"]=="asset" and s.get("job")==key for s in self.state["stages"])
        attempts=range(1) if draft else range(consumed,self.generation.max_iterations)
        for attempt in attempts:
            instruction, context, images, phase = self.author_request(job, {**inputs, "previous":previous,
                                                                            "feedback":feedback, "previews":previews}, attempt)
            destination = self.begin("asset")
            report = {}
            record = {"job":key,"source":previous_source,"status":"failed","reference_policy":reference_policy,"preview_paths":[]}
            try:
                value, call = self.model.ask(instruction,context,images,phase)
                source = author_result(value,previous,self.generation.max_script_bytes)
                self.before_candidate(key,str(destination.relative_to(self.root)))
                script = destination/"model.py"
                script.write_text(source,encoding="utf-8")
                previous = source
                previous_source = str(script.relative_to(self.root))
                record.update(source=previous_source,call=call)
                if draft and hasattr(self.sandbox,"build_candidate"):
                    admission_profile={**profile,"assembly_role":job["role"],"assembly_configurations":job.get("configurations",["default"])}
                    report,renders=self.sandbox.build_candidate(script,destination,job["asset_ids"],admission_profile)
                else:
                    report, renders = self.sandbox.build_and_check(script,destination,job["asset_ids"],profile)
                from asset_generation.configurations import catalogue
                from asset_generation.check_report import report_status
                if report.get("manifest") is not None:
                    actual = set(catalogue(destination/"artifact"))
                    expected = set(job.get("configurations",["default"]))
                    if actual != expected:
                        report["checks"].append({"id":"planned_configurations","status":"fail",
                            "detail":"Delivered configurations must exactly match plan", "expected":sorted(expected),"actual":sorted(actual)})
                        report["status"]=report_status(report["checks"])
                report = checked_report(report,domain='asset',mode=profile.get('validation_mode','full'),require_renders=True)
                if not draft and any(incomplete(r) for r in report.get("checks",[])) and report["status"]!="error":
                    from asset_generation.check_recovery import recover_checks
                    report=recover_checks(self.sandbox,destination/"artifact",report,self.root/"asset_check_recovery",
                        implementation=self.implementation_guard.expected,contract=profile,check_pause=self.check_pause,
                        max_actions=self.limits.max_check_actions,max_retries=self.limits.max_check_retries)
                report,previews = checked_previews(report,renders,
                    lambda directory,name:image_record(safe_path(directory,name),self.root,self.generation))
                atomic_json(destination/"checks.json",report)
                feedback = {"numerical":report}
                record["preview_paths"] = [im["path"] for im in previews]
                record.update(manifest=report.get("manifest"), geometry_summary=report.get("geometry_summary",{}),
                              coverage=report.get("coverage",[]), feedback={"numerical":report},
                              artifact=str((destination/"artifact").relative_to(self.root)),
                              artifact_stage=str(destination.relative_to(self.root)))
                if not draft and report["status"] == "pass" and previews:
                    save_review_checkpoint(self, "asset", record)
                    feedback = self.review_asset(job,bundle,references,previews,report,destination,record)
                elif report["status"] == "error":
                    record["status"] = "needs_environment"
                elif report["status"] == "partial":
                    record["status"] = "needs_check_support"
                record["manifest"] = report.get("manifest")
                record["geometry_summary"] = report.get("geometry_summary",{})
                record["coverage"] = report.get("coverage",[])
            except ContextLimitError as exc:
                feedback = {**(feedback or {}),"task_deferred":str(exc)}
                record["task_deferred"] = str(exc)
            except ValidationError as exc:
                feedback = {"generation_error":str(exc)}
            if draft:
                record["assembly_readiness"] = report.get("assembly_readiness",{"loadable":False})
                record["status"] = "unassessed" if record["assembly_readiness"].get("loadable") else record["status"]
            record["feedback"] = feedback
            stage = self.commit("asset",record)
            self.state["jobs"][key] = {**record,"stage":stage["directory"],
                "artifact_stage":stage["directory"],"artifact":stage["directory"]+"/artifact",
                "dirty":record["status"] not in ASSEMBLABLE}
            self.store.save()
            if draft:
                from .candidate import usable
                ready = usable(self,self.state["jobs"][key])
                self.state["jobs"][key]["dirty"] = not ready
                self.store.save()
                return ready
            if record.get("task_deferred"):
                self.state["status"] = "needs_review"
                return False
            if record["status"] in ASSEMBLABLE:
                return True
            if record["status"] in {"needs_review","needs_environment","needs_check_support"}:
                self.state["status"] = evidence_stop_status(record.get('evidence_outcome')) if record['status']=='needs_review' else record['status']
                return False
        self.state["status"] = "asset_repair_budget_exhausted"
        return False

    def recover_checks(self, report, artifact, destination):
        return recover(self,report,artifact,destination)

    def check_progress(self, report, artifact, event):
        from copy import deepcopy
        published = self.state.get('published_scene')
        if published and safe_path(self.root,published['artifact']) == Path(artifact):
            self.state['current_assessment'] = {'assembly_key':published['assembly_key'],
                'artifact_sha256':published['artifact_sha256'],'report':deepcopy(report)}
        self.progress('scene_evidence', event)

    def build_scene(self):
        destination = self.begin("scene")
        self.state.pop('active_check_recovery',None)
        self.state.setdefault('evidence_outcomes',{}).pop('scene_review',None)
        resumed = self.state.get("scene_review_resume")
        versions = {key:job["stage"] for key,job in self.state["jobs"].items()}
        latest = self.state.get('latest',{})
        if (not resumed and latest.get('plan')==self.state['plan'] and latest.get('job_versions')==versions
                and 'preview_paths' in latest):
            resumed = latest
        if resumed:
            require(resumed["plan"] == self.state["plan"] and resumed["job_versions"] == versions,
                    "The plan or asset versions of the built scene changed; review alone cannot be resumed")
            report = checked_report(resumed["report"],domain='scene',mode=self.limits.validation_mode)
            images = [image_record(safe_path(self.root,p),self.root,self.generation) for p in resumed["preview_paths"]]
            artifact = safe_path(self.root,resumed["artifact"])
        else:
            artifacts = {k:safe_path(self.root,v["artifact"]) for k,v in self.state["jobs"].items()}
            print(f'Starting scene assembly, checks and rendering, round {len(self.state["rounds"])+1}.',flush=True)
            report, renders = self.sandbox.assemble_and_check(self.state["plan"],self.env,artifacts,self.robot_root,
                                                           destination,self.limits,self.contract)
            report = checked_report(report,domain='scene',mode=self.limits.validation_mode,require_renders=True)
            report,images = checked_previews(report,renders,
                lambda directory,name:image_record(safe_path(directory,name),self.root,self.generation))
            artifact = destination/"scene"/"artifact"
        if artifact.is_dir() and (check_gaps(report) or report.get('status') in {'fail','error','partial'}):
            save_review_checkpoint(self,'scene_check',{'report':report,'plan':self.state['plan'],
                'job_versions':versions,'artifact':str(artifact.relative_to(self.root)),
                'preview_paths':[im['path'] for im in images]})
            report = self.recover_checks(report,artifact,destination)
        if report['status']=='pass' and not images:
            report,images = checked_previews(report,None,lambda *_:None)
        atomic_json(destination/"checks.json",report)
        review = None
        if report["status"] != "error" and images:
            save_review_checkpoint(self, "scene", {"report":report,"plan":self.state["plan"],
                "job_versions":versions,"artifact":str(artifact.relative_to(self.root)),
                "preview_paths":[im["path"] for im in images]})
            deferred = self.deferred_checks()
            originals = list({im["sha256"]:im for bundle in self.catalogue["bundles"].values() for im in bundle["images"]}.values())
            self.check_pause()
            review, call = review_with_evidence(self,SCENE_REVIEW+REQUIREMENTS_POLICY+SCENE_SCOPE,{"source":self.catalogue["source"],"plan":self.state["plan"],
                                        "protocol":self.catalogue.get("protocol"),"checks":report,
                                        "fixed_contract":self.contract,
                                        "review_scope":{"stage":"scene","validation_mode":self.limits.validation_mode},
                                        "layout_review_targets":review_targets(self.state["plan"]),
                                        "deferred_checks":deferred,
                                        "evidence":{"original_reference_images":[im["path"] for im in originals],
                                                    "assembled_scene_images":[im["path"] for im in images]}},
                                        originals+images,"scene_review",destination,
                                        validator=lambda value: scoped_review(value,"scene",deferred,preview=self.limits.validation_mode == "static",
                                            instance_ids={i["id"] for i in self.state["plan"]["instances"]},
                                            layout_targets=review_targets(self.state["plan"])),
                                        artifact=artifact,scope="scene")
            atomic_json(destination/"review.json",{"review":review,"call":call})
        record = {"report":report,"review":review,"plan":self.state["plan"],"artifact":str(artifact.relative_to(self.root)),
                  "assessed":True,"assessment_scope":"full_quality",
                  "assessment_implementation":self.implementation_guard.expected,
                  "preview_paths":[im['path'] for im in images],
                  "evidence_outcome":self.state.get('evidence_outcomes',{}).get('scene_review'),
                  "job_versions":{key:job["stage"] for key,job in self.state["jobs"].items()}}
        stage = self.commit("scene",record)
        return report, review

    def repair_input_key(self):
        return digest({'plan':semantic(self.state['plan']),
                       'sources':{k:file_hash(safe_path(self.root,v['source'])) if v.get('source') else None for k,v in self.state['jobs'].items()},
                       'artifacts':{k:snapshot(safe_path(self.root,v['artifact'])) for k,v in self.state['jobs'].items()}})

    def finish_repair(self, report, review):
        if report.get('status')=='error' and report.get('error_kind')!='measurement_error':
            return  # The checker itself failed: the candidate stays unjudged until it is checked again.
        for entry in self.state.get('repair_history',[]):
            if entry.get('status') == 'applied':
                output_key = self.repair_input_key()
                remaining = problem_identity(report,review)
                from .progress import repair_comparison
                baseline=entry.get("baseline")
                old=next((r for r in self.state["stages"] if r["directory"]==baseline.get("scene")),None) if baseline else None
                comparison=repair_comparison(old["report"],report,old.get("review"),review,
                    old_plan=old.get("requested_plan",old.get("plan")),new_plan=self.state["plan"],
                    unchanged_assets=self.unchanged_assets(baseline)) if old else None
                entry.update(status='evaluated',output_key=output_key,remaining_problem=remaining,
                             outcome=comparison["outcome"] if comparison else 'ineffective' if remaining==entry['problem_key'] else 'changed',
                             comparison=comparison,scene=self.state['latest']['directory'],
                             calls_consumed=len(self.store.state['calls'])-entry.get('call_start',len(self.store.state['calls'])))

    def unchanged_assets(self, baseline):
        """Jobs whose current artifact is byte-identical to the one in the repair baseline."""
        stages={r["directory"]:r for r in self.state["stages"]}
        sha=lambda job:(job or {}).get("assembly_readiness",{}).get("artifact_sha256")
        return {k for k,d in baseline.get("jobs",{}).items()
                if sha(stages.get(d)) and sha(stages.get(d))==sha(self.state["jobs"].get(k))}

    def select_repair_candidate(self):
        """Restore archived references only; never rewind calls, stages or attempt budgets."""
        entry=next((r for r in reversed(self.state.get("repair_history",[]))
                    if r.get("status")=="evaluated" and r.get("baseline") and "selection" not in r),None)
        if entry is None:return False
        if not entry.get("comparison") or entry["comparison"]["adopt"]:
            entry["selection"]="adopted";self.store.save();return False
        if self.salvage_repair(entry):
            self.store.save();return True
        self.restore_baseline(entry)
        entry["selection"]="baseline_preserved"
        self.store.save()
        return True

    def defect_jobs(self, check, added, reports):
        """Asset jobs a compared check row belongs to; None when it cannot be attributed."""
        items={i["id"]:i["job"] for i in self.state["plan"]["instances"]}
        head,_,tail=check.partition(":")
        if head in items:instances={head}
        elif head=="scene_collisions" and tail:instances=set(tail.split("|"))
        elif head=="scene_visual":instances=set(added or [])
        else:
            row=next((r for report in reports for r in report.get("checks",[]) if r.get("id")==check),{})
            instances=set(row.get("instances",[]))
            if head=="scene_collisions":
                # A whole-scene collision row: only the pairs that were added count.
                instances={n.split("__",1)[0] for unit in added or [] for n in unit.removeprefix("pair:").split("|")}
        jobs={items[i] for i in instances if i in items}
        return jobs or None

    def salvage_repair(self, entry):
        """A round that fixed something but also made some assets worse keeps its other changes:
        the assets behind every new defect (with the edited assets built on them) go back to their
        baseline versions and the scene is checked once more. Done at most once per round."""
        from .local_repair import with_dependents, revert, geometry_only
        comparison=entry["comparison"]
        gain=comparison.get("resolved_checks") or comparison.get("improved_checks") or comparison.get("partially_improved_checks")
        if entry.get("salvage") or not gain or comparison["outcome"] not in {"mixed","coverage_lost"}:return False
        baseline=next(r for r in self.state["stages"] if r["directory"]==entry["baseline"]["scene"])
        reports=[self.state["latest"]["report"],baseline["report"]]
        edited=set(entry["action"]["jobs"])-set(entry.get("reverted_jobs",[]))
        culprits,defects=set(),{}
        for row in comparison["new_defects"]+[{"check":k} for k in comparison["coverage_lost"]]:
            jobs=self.defect_jobs(row["check"],row.get("added"),reports)
            if not jobs or not jobs&edited:return False  # Not caused by a reversible asset edit.
            for job in jobs&edited:
                culprits.add(job);defects.setdefault(job,[]).append(row)
        reverted=with_dependents(self,entry,culprits)
        if not edited-reverted and geometry_only(entry["action"]):return False
        entry["salvage"]={"scene":entry["scene"],"comparison":comparison,"reverted_jobs":sorted(reverted)}
        revert(self,entry,reverted,defects)
        for key in ("comparison","outcome","output_key","scene","remaining_problem"):
            entry.pop(key,None)
        entry["status"]="applied"
        return True

    def archived_versions(self, versions):
        """Job records of archived stage versions {job: stage directory}; the archives must be intact."""
        from copy import deepcopy
        stages={r["directory"]:r for r in self.state["stages"]}
        jobs={}
        for key,directory in versions.items():
            stage=stages[directory]
            if snapshot(safe_path(self.root,directory))!=stage["files"]:
                raise StaleEvidence("The repair baseline archive changed; restoring old evidence is forbidden")
            jobs[key]={k:deepcopy(v) for k,v in stage.items() if k not in {"directory","kind","files"}}
            jobs[key].update(stage=directory,dirty=False)
            jobs[key].setdefault("artifact",directory+"/artifact")
            jobs[key].setdefault("artifact_stage",directory)
        return jobs

    def restore_baseline(self, entry):
        """Return the plan, assets and scene evidence to the baseline a repair started from."""
        from copy import deepcopy
        baseline=entry["baseline"]
        old=next(r for r in self.state["stages"] if r["directory"]==baseline["scene"])
        if snapshot(safe_path(self.root,baseline["scene"]))!=old["files"]:
            raise StaleEvidence("The repair baseline archive changed; restoring old evidence is forbidden")
        restored={k:deepcopy(v) for k,v in old.items() if k not in {"kind","files"}}
        self.state.update(plan=deepcopy(old.get("requested_plan",old["plan"])),jobs=self.archived_versions(baseline["jobs"]),
                          latest=restored,feedback=deepcopy(old["report"]),published_scene=deepcopy(restored),repair_pending=True)
        _,self.state["order"]=validate_plan(self.state["plan"],self.catalogue["asset_ids"],self.limits,self.env)
        for key in ("current_assessment","repair_directive","repair_tasks","scene_review_resume"):
            self.state.pop(key,None)

    def deferred_checks(self):
        return [item for job in self.state["jobs"].values() for item in job.get("deferred_checks",[])]

    def route_repair(self):
        self.check_pause()
        # Failed assembly never reaches an assessed scene round. Its applied
        # repair still consumes the cumulative allocation, including on resume.
        if len(self.state.get("repair_history", [])) >= self.limits.max_iterations:
            self.state.update(status="scene_repair_budget_exhausted", stop_reason={
                "reason":"Cumulative scene repair action budget exhausted; keeping the published scene and the failed candidate",
                "attempts":len(self.state["repair_history"]), "limit":self.limits.max_iterations})
            self.store.save()
            return False
        self.state['status']='repairing_scene'
        self.progress('scene_repair', 'repair_planning_started')
        report = self.state["feedback"]
        from .recovery import repairable_rows
        from .progress import repair_tasks
        editable=repairable_rows(report)
        affected = {i for row in editable for i in row.get("instances",[])}
        review = self.state.get("latest",{}).get("review")
        if review and review["verdict"] == "repair" and not any(f["status"]=="open" for f in report.get("recovery_failures",[])):
            affected.update(review.get("affected_instances",[]))
        items = {i["id"]:i for i in self.state["plan"]["instances"]}
        affected &= {i for i in items if self.state["jobs"].get(items[i]["job"],{}).get("source")
            and (self.limits.validation_mode!="static" or self.state["jobs"][items[i]["job"]].get("assembly_readiness",{}).get("loadable"))}
        visual_layout = (review or {}).get("affected_layout") if not any(
            r.get("status")=="error" for r in report.get("checks",[])) else []
        layout_only = "layout" in self.state["plan"] and not affected and (bool(visual_layout) or any(
            set(row.get("instances",[])) & {"robot","environment"} or row["id"].startswith("table_")
            for row in editable))
        if not affected and not layout_only:
            self.state["status"] = "needs_review"
            self.state["repair_error"] = "The failure report did not locate an editable instance; keeping the evidence and not widening the workspace blindly"
            return False
        blocked_instances={i for row in report.get("checks",[]) if row.get("status")=="error" for i in row.get("instances",[])}
        blocked_jobs={items[i]["job"] for i in blocked_instances if i in items}
        affected={i for i in affected if items[i]["job"] not in blocked_jobs}
        jobs = sorted({items[i]["job"] for i in affected})
        if not affected and not layout_only:
            self.state.update(status="needs_review",repair_error="The confirmed problem shares an asset with the faulty object; it cannot be modified without reliable evidence")
            return False
        tasks=repair_tasks(report,review,self.state["plan"],self.repair_input_key())
        from .repair_schedule import schedule, action_tasks
        tasks=schedule(self,tasks)
        if not tasks:
            self.state.update(status="repair_tasks_deferred",repair_pending=False,
                stop_reason={"reason":"no_runnable_repairs","detail":"Keeping failures and unknowns; all current problems are on hold"})
            self.store.save()
            return False
        eligible={i for t in tasks for i in t['instances']}
        affected &= eligible
        from .repair_schedule import job_order
        jobs=job_order(self,tasks,items,affected)
        input_key=self.repair_input_key()
        from .capabilities import problem_identity
        problem_key=problem_identity(report,review)
        from .progress import repair_history_context
        history_view=repair_history_context(self.state.get("repair_history",[]),problem_key,input_key,
            target_checks=[t.get('check_id',t['id']) for t in tasks])
        latest = self.state.get("latest", {})
        for task in tasks:
            task["evidence_record"]=latest.get("directory","")+"/record.json"
        cycle = latest.get("directory", str(len(self.state["rounds"])))
        planning = self.state.setdefault("repair_planning", {"cycle":cycle,"attempts":[]})
        if planning["cycle"] != cycle:
            planning = self.state["repair_planning"] = {"cycle":cycle,"attempts":[]}
        planning.setdefault("call_start",len(self.store.state["calls"]))
        # Every open task is handled in this one round: the joint plan covers all their assets,
        # and only when it cannot be planned (too large, blocked) does it narrow to one asset at
        # a time. Same state on resume consumes the saved attempts instead of restarting.
        scopes=[sorted(affected)]+[sorted(i for i in affected if items[i]["job"]==j) for j in jobs]
        solved=False
        for scope_index,allowed in enumerate(scopes):
            scoped_jobs=sorted({items[i]["job"] for i in allowed})
            scoped_tasks=[t for t in tasks if set(t["instances"]) & set(allowed) or layout_only]
            scope_key=digest({'jobs':scoped_jobs,'tasks':[(t['id'],t['binding']) for t in scoped_tasks]})
            if "layout" in self.state["plan"]:
                # Layout edits can move a whole support tree, but this never expands geometry-edit rights.
                allowed=sorted(i for i in items if items[i]["placement"]!="robot" and items[i]["job"] not in blocked_jobs)
            attempts=[a for a in planning["attempts"] if a.get('scope_key')==scope_key]
            if any(a.get("scope_deferred") for a in attempts):continue
            current_assets,images={},[]
            for key in scoped_jobs:
                asset=self.state["jobs"][key]
                current_assets[key]={"model_py":safe_path(self.root,asset["source"]).read_text(encoding="utf-8"),
                    "manifest":asset.get("manifest"),"geometry":asset.get("geometry_summary"),"version":asset["stage"]}
                for image_path in asset.get("preview_paths",[])[:1]:
                    images.append(image_record(safe_path(self.root,image_path),self.root,self.generation))
            for image_path in latest.get("preview_paths",[])[:3]:
                images.append(image_record(safe_path(self.root,image_path),self.root,self.generation))
            images=list({im["sha256"]:im for im in images}.values())
            for attempt in range(len(attempts),self.limits.max_repair_plan_attempts):
                self.check_pause()
                from .diagnostics import triage
                context={"plan":self.state["plan"],"feedback":report,"visual":review,
                    "benchmark_environment":(self.catalogue.get('benchmark_public') or {}).get('summary'),
                    "evidence_triage":triage(report),
                    "allowed_jobs":scoped_jobs,"allowed_instances":allowed,
                    "environment":self.env,"robot_joint_contract":self.robot_joints,"contract":self.contract,"table_edit_policy":table_policy(self.env),
                    "edge_mount_policy":EDGE_MOUNT_POLICY,"current_assets":current_assets,
                    "repair_tasks":scoped_tasks,"validation_feedback":planning["attempts"],
                    **history_view,
                    "cycle":cycle,"attempt":attempt,"scope_index":scope_index,
                    "strategy":"joint handling" if scope_index==0 else "narrowed to the current asset; resolve one evidenced problem first and keep the others",
                    "recovery_history":self.state.get("repair_history",[])[-8:]}
                value,call=None,None
                try:
                    value,call=self.model.ask(for_environment(REPAIR,self.env),context,images,"scene_repair")
                    if isinstance(value,dict) and set(value)=={"blocked"}:
                        blocked=value["blocked"]
                        require(isinstance(blocked,dict) and set(blocked)=={"kind","detail"}
                            and blocked["kind"] in ("missing_evidence","unsupported_repair","contract_conflict")
                            and isinstance(blocked["detail"],str) and blocked["detail"].strip(),"A repair blocker must state kind and detail")
                        raise ValidationError("This option is unavailable for now; check local placement, existing interfaces or a local asset redesign, and first handle what can progress within this scope: "+str(blocked))
                    updated=repair_action(value,self.state["plan"],scoped_jobs,allowed,self.limits,self.env)
                    if "layout" in updated and (value.get("layout") or value.get("placements")):
                        # Layout edits choose benches and order; coordinates stay arranged.
                        from .arrangement import arrange, shapes_for
                        updated=arrange(updated,shapes_for(self,updated),self.env)
                    self.validate_prepared_plan(updated)
                    chosen_tasks=action_tasks(scoped_tasks,value,self.state['plan'],updated)
                    require(bool(chosen_tasks),'The change must address a currently authorized target problem')
                    require(all(not t['scheduling']['change_mechanism'] or repair_mechanism(value)
                        not in t['scheduling']['failed_mechanisms'] for t in chosen_tasks),
                        'No improvement twice in a row; change the repair mechanism or first obtain new evidence that distinguishes the causes')
                    validate_model_posture(updated,self.robot_joints)
                    input_key=self.repair_input_key();problem_key=problem_identity(report,review);action_key=repair_signature(value)
                    repeated=repeated_repair(self.state.get("repair_history",[]),input_key,problem_key,action_key,
                        mechanism=repair_mechanism(value))
                    require(not repeated,
                        "This repair was already executed with the same inputs and did not resolve the problem; use the history comparison to choose a different action")
                    planning["attempts"].append({"response":value,"call":call,"scope":scope_index,"scope_key":scope_key,"status":"accepted"})
                    tasks=chosen_tasks;solved=True
                    break
                except (ValidationError,ContextLimitError) as exc:
                    planning["attempts"].append({"response":value,"call":call,"scope":scope_index,"scope_key":scope_key,"error":str(exc)})
                    self.state["repair_error"]=str(exc);self.store.save()
                    # Repeated blockers or an oversized joint request yield to a smaller scope.
                    if isinstance(exc,ContextLimitError) or (isinstance(value,dict) and "blocked" in value and
                        any(a.get("response")==value for a in attempts)):
                        planning["attempts"][-1]["scope_deferred"]=True
                        self.store.save();break
                    attempts.append(planning["attempts"][-1])
            if solved:break
        if not solved:
            self.state.update(status="needs_review",stop_reason={"reason":"repair_tasks_deferred",
                "detail":"Neither the joint plan nor local plans for editable assets can be executed now; keeping the best scene and per-item feedback",
                "attempts":len(planning["attempts"])})
            self.store.save()
            return False
        self.state.pop("repair_error", None)
        self.before_repair(value)
        _, self.state["order"] = validate_plan(updated,self.catalogue["asset_ids"],self.limits,self.env)
        dirty = set(value["jobs"])
        for key in self.state["order"]:
            job = next(j for j in updated["jobs"] if j["id"] == key)
            if dirty.intersection(job["depends_on"]):
                dirty.add(key)
        old_items = {i['id']:i for i in self.state['plan']['instances']}
        moved_jobs = {i['job'] for i in updated['instances'] if i != old_items.get(i['id'])}
        if "layout" in updated and layout_revision(updated) != layout_revision(self.state["plan"]):
            moved_jobs |= {j["id"] for j in updated["jobs"] if j["role"]!="asset"}
        revalidate = set()
        for key in self.state['order']:
            job = next(j for j in updated['jobs'] if j['id']==key)
            if job['role'] != 'asset' and (key in moved_jobs or (moved_jobs|revalidate).intersection(job['depends_on'])):
                revalidate.add(key)
        destination = self.begin("repair")
        atomic_json(destination/"action.json",value)
        self.state.setdefault('repair_history',[]).append({'input_key':input_key,'action':value,'cycle':cycle,
            'problem_key':problem_key,'action_key':action_key,'mechanism':repair_mechanism(value),
            'status':'applied',
            'baseline':{'scene':latest['directory'],'jobs':{k:v['stage'] for k,v in self.state['jobs'].items()}}
                if self.limits.validation_mode=='static' and latest.get('assessed') else None,
            'target_checks':[t.get('check_id',t['id']) for t in tasks],
            'task_bindings':{t.get('check_id',t['id']):t['binding'] for t in tasks},
            'call_start':planning.setdefault('call_start',max(0,len(self.store.state['calls'])-1))})
        self.state["plan"] = updated
        self.state["repair_directive"] = value
        self.state["repair_tasks"]={key:[t for t in tasks if any(items[i]["job"]==key for i in t["instances"] if i in items)] for key in value["jobs"]}
        for addition in value.get("tool_additions",[]):
            self.state["repair_tasks"][addition["job"]["id"]]=tasks
        # Dependency propagation triggers re-verification; only geometry edits of the direct jobs are authorized.
        for key in set(value['jobs']):
            if key in self.state["jobs"]:
                self.state["jobs"][key]["dirty"] = True
                self.state["jobs"][key]["regenerate"] = True
        revalidate |= dirty-set(value['jobs'])
        dirty = set(value['jobs'])
        pending_generation = (revalidate | dirty) - set(self.state['jobs'])
        revalidate &= set(self.state['jobs'])
        for key in revalidate-dirty:
            self.state['jobs'][key].update(dirty=True,status='review_pending',layout_revalidation={
                'reason':'The actual instance pose of a dependency or of this device changed; check whether the existing geometry still mates, and request a local redesign if needed',
                'previous_layout':self.state['jobs'][key].get('layout_revision'),
                'current_layout':layout_revision(updated)})
        self.commit("repair",{"call":call,"dirty_jobs":sorted(dirty),"layout_revalidation_jobs":sorted(revalidate-dirty),"pending_generation_jobs":sorted(pending_generation),"plan":updated})
        self.state.pop('repair_planning',None)
        self.state["repair_pending"] = False
        self.store.save()
        return True

    def result(self):
        latest = self.state.get("published_scene") if self.limits.validation_mode == "static" else self.state.get("latest")
        # In static mode only a successfully admitted publication is an artifact.
        # `latest` can describe an assembly failure with a half-written directory.
        current=self.state.get("current_assessment",{})
        if latest and current and current.get("assembly_key")==latest.get("assembly_key") and current.get("artifact_sha256")==latest.get("artifact_sha256") and not latest.get("assessed"):
            latest={**latest,"report":current["report"]}
        artifact = safe_path(self.root,latest.get("artifact",latest["directory"]+"/scene/artifact")) if latest else None
        scene_hash = digest(snapshot(artifact)) if artifact and artifact.is_dir() else None
        if latest and latest.get("artifact_sha256") and scene_hash!=latest["artifact_sha256"]:
            raise StaleEvidence("The published scene digest changed; rebinding old evidence is forbidden")
        gaps = list((latest or {}).get("report",{}).get("coverage",[]))
        gaps.extend((self.catalogue.get("protocol") or {}).get("warnings",[]))
        for job in self.state["jobs"].values():
            gaps.extend(job.get("coverage",[]))
            gaps.extend(job.get("feedback",{}).get("visual",{}).get("unverified",[]))
        for row in check_gaps((latest or {}).get('report',{})):
            gaps.append('Main check incomplete: '+row['id'])
        if latest and latest.get("review"):
            gaps.extend(latest["review"]["unverified"])
        current_evidence = (latest and latest["plan"] == self.state["plan"]
                            and latest.get("job_versions") == {k:v["stage"] for k,v in self.state["jobs"].items()}
                            and not any(v.get("dirty") for v in self.state["jobs"].values()))
        resolutions = {r["id"]:r for r in latest["review"].get("deferred_results",[])} if current_evidence and latest.get("review") else {}
        deferred = []
        for item in self.deferred_checks():
            resolution = resolutions.get(item["id"],{})
            row = {**item,"status":resolution.get("status","pending"),"evidence":resolution.get("evidence")}
            if row["status"] == "out_of_scope":
                row.update(status="pending",stage=resolution["target_stage"],deferred_from=item["stage"])
            deferred.append(row)
            if row["status"] != "resolved":
                gaps.append(f"Pending verification at {row['stage']} stage: {item['detail']}")
        static = self.limits.validation_mode == "static"
        if static:
            gaps.append("The static preview did not verify dynamics, button/latch operation, joint opening clearance or robot reachability")
        status = "static_candidate_ready" if static and self.state["status"] == "candidate_ready" else self.state["status"]
        # Remaining measurement/probe gaps keep a static result at preview level.
        preview_only = static and (any(row.get("deferred_from") for row in deferred)
                                   or bool(check_gaps((latest or {}).get('report',{}))))
        if preview_only and status == "static_candidate_ready":
            status = "static_preview_ready"
        result = {"activity":self.state.get("activity"),"status":status,"provenance":self.store.state["identity"]["provenance"],
                  "validation_mode":self.limits.validation_mode,
                  "budget":self.store.totals(),"scene_rounds":len(self.state["rounds"]),
                  "repair_rounds":len(self.state.get("repair_history", [])),
                  "requests":request_summary(self.store.state["calls"]),
                  "scene_artifact":str(artifact.relative_to(self.root)) if artifact and artifact.is_dir() else None,
                  "scene_sha256":scene_hash,"unverified":list(dict.fromkeys(gaps)),
                  "qualification":"Static scene preview candidate; operation capability not certified" if static else "Scene engineering candidate; does not imply a complete trajectory or a completed biochemical experiment",
                  "deferred_checks":deferred,
                  "blocking_reasons":[reason for job in self.state["jobs"].values()
                                      for reason in job.get("feedback",{}).get("visual",{}).get("blocking",[])],
                  "repair_error":self.state.get("repair_error"),"jobs":{k:v["status"] for k,v in self.state["jobs"].items()}}
        rows=(latest or {}).get("report",{}).get("checks",[])
        result["quality_progress"]={"scene_version":(latest or {}).get("directory"),"evidence_current":bool(current_evidence),
            "assessment_scope":(latest or {}).get('assessment_scope'),
            "confirmed_failed_checks":sum(r.get("status")=="fail" for r in rows),
            "incomplete_checks":sum(incomplete(r) for r in rows),
            "execution_error_checks":sum(r.get("status")=="error" for r in rows),
            "review_pending":not bool(current_evidence and latest.get("review") and latest["review"].get("verdict")!="uncertain")}
        if preview_only:
            result["qualification"] = "Current static layout preview; the listed protocol initial states, other assembly mating poses and operation verification are not yet complete"
        result["verification_scope"] = "initial_static_layout" if static else "scene_engineering"
        if self.state.get("request_failure"):
            result["request_failure"] = self.state["request_failure"]
        if self.state.get("local_failure"):
            result["local_failure"] = self.state["local_failure"]
        if self.state.get("assembly_failure"):
            result["assembly_failure"] = self.state["assembly_failure"]
        result['evidence_outcomes']=self.state.get('evidence_outcomes',{})
        if self.limits.validation_mode == "static":
            missing=(latest or {}).get("missing_jobs",[j["id"] for j in self.state.get("plan",{}).get("jobs",[])])
            result["assembly"]={"available":bool(artifact and artifact.is_dir()),
                "complete":bool(latest and not latest.get("missing_jobs")),
                "missing_jobs":missing,
                "ready_unassembled_jobs":{key:self.state["jobs"][key]["artifact"] for key in missing
                    if self.state["jobs"].get(key,{}).get("assembly_readiness",{}).get("loadable")},
                "quality_accepted":status == "static_candidate_ready",
                "scene_version":(latest or {}).get("directory"),
                "issues":((latest or {}).get("report") or self.state.get("assembly_failure",{})).get("checks",[]),
                "attempts":self.state.get("candidate_attempts",{})}
        result['check_recovery']=[{k:a[k] for k in a if k not in {'report','actions'}}
                                  | {'evidence_source':'state.json:scene.check_recovery.'+key}
                                  for key,a in self.state.get('check_recovery',{}).items()]
        result['repair_progress']=[{k:v for k,v in h.items() if k in {'cycle','status','outcome','comparison','selection','scene','target_checks','calls_consumed'}}
                                   for h in self.state.get('repair_history',[])]
        result['repair_schedule']=self.state.get('repair_schedule',[])
        result['phase_budget']=self.store.state.get('phase_budget')
        from .handoff import quality_handoff, operation_obligations
        result['operation_handoff']=quality_handoff((latest or {}).get('report',{}),scene_hash,self.state.get('repair_schedule',[]))
        result['stop_reason']=self.state.get('stop_reason')
        if not result['stop_reason'] and status not in {'candidate_ready','static_candidate_ready','static_preview_ready'}:
            result['stop_reason']={'status':status,'reason':self.state.get('repair_error') or
                ('This stage\'s attempt budget is exhausted; artifacts and attempt records are kept' if 'exhausted' in status else
                 'Acceptance not yet met; resume after handling the check, review or pause records'),
                'required_action':'Inspect check_failure, evidence_outcomes and per-job feedback; do not treat incomplete as passed'}
        if self.state.get('check_failure'):
            result['check_failure']=self.state['check_failure']
        checkpoint = self.state.get("pending",{}).get("review_checkpoint")
        if checkpoint:
            result["paused_stage"] = checkpoint["kind"] + "_review"
            if checkpoint["kind"] == "asset":
                result["jobs"][checkpoint["record"]["job"]] = "review_pending"
        result["blocking_reasons"].extend((latest or {}).get("review",{}).get("blocking",[]) if (latest or {}).get("review") else [])
        result["scene_document"]="scene_document.md"
        from .document import write as write_scene_document
        write_scene_document(self,result,latest)
        atomic_json(self.root/"result.json",result)
        from .preview import write_preview_index
        write_preview_index(self.root, result)
        if self.limits.validation_mode == "static":
            from .scene_first import write_status_page
            write_status_page(self,result)
        if artifact and artifact.is_dir():
            atomic_json(self.root/"planning_handoff.json",{"scene_sha256":scene_hash,"artifact":result["scene_artifact"],
                "candidate_status":result["status"],"environment":self.env,"plan":latest["plan"],
                "requirements":self.contract,"qualification":result["qualification"],"unverified":result["unverified"],
                "validation_mode":self.limits.validation_mode,"deferred_checks":deferred,
                "operation_obligations":operation_obligations(self.state),
                "quality":result['operation_handoff']})
        return result

    def run(self, feedback=None):
        require(feedback is None or self.limits.validation_mode != "static", "A static scene does not accept operation planning feedback")
        try:
            if not (self.root/"scene_document.md").exists():
                from .document import write
                write(self, {"status":self.state["status"],
                             "qualification":"Scene in preparation; no quality acceptance verdict yet."}, None)
            self.state.pop("request_failure", None)
            self.state.pop("local_failure", None)
            self.state.pop('check_failure',None)
            self.state.pop('stop_reason',None)
            self.initialize()
            if feedback is not None and digest(feedback) not in self.state["feedback_receipts"]:
                checked = planning_feedback(feedback,self.result()["scene_sha256"],self.state["plan"])
                self.state.update(feedback=checked,repair_pending=True,status="running")
                self.state["feedback_receipts"].append(digest(feedback))
                self.store.save()
            elif self.state["status"] == "candidate_ready":
                return self.result()
            if self.limits.validation_mode == "static":
                from .scene_first import run_static
                return run_static(self)
            while len(self.state["rounds"]) < self.limits.max_iterations:
                if self.state.get("repair_pending") and not self.route_repair():
                    break
                self.state["status"] = "running"
                jobs = {j["id"]:j for j in self.state["plan"]["jobs"]}
                for key in self.state["order"]:
                    if key not in self.state["jobs"] or self.state["jobs"][key].get("dirty"):
                        if not self.generate_job(jobs[key],self.state.get("feedback")):
                            self.store.save()
                            return self.result()
                report, review = self.build_scene()
                active = self.state.get('check_recovery',{}).get(self.state.get('active_check_recovery'),{})
                if active.get('status')=='budget_exhausted':
                    self.state['status']='check_recovery_budget_exhausted'
                    self.state['stop_reason']={'reason':'Check recovery budget exhausted; keeping the scene and the executed actions',
                                              'exhaustion':active['exhaustion']}
                    break
                action = decide(report,review)
                self.state['last_decision'] = {'action':action,'scene':self.state['latest']['directory']}
                if action in {'needs_check_support','needs_environment'}:
                    self.state['status']=action
                    self.state['check_failure']={'kind':report.get('error_kind','environment') if action=='needs_environment' else 'measurement_error',
                        'checks':check_gaps(report) or [r for r in report.get('checks',[]) if r.get('status')=='error'],
                        'failures':[f for f in report.get('recovery_failures',[]) if f['status']=='open']}
                    self.state['stop_reason']={'reason':'The main check still has unresolved evidence gaps, and the available supplementary checks have run or are unavailable; unknowns were not judged as passed',
                                              'required_action':'Add check capability or handle the tool faults in the report',
                                              'recovery_keys':list(self.state.get('check_recovery',{})),
                                              'capability_search':[h.get('exhaustion') for h in self.state.get('check_recovery',{}).values()]}
                    break
                if review and review["verdict"] == "uncertain" and action != "repair":
                    self.state["status"] = evidence_stop_status(self.state.get('evidence_outcomes',{}).get('scene_review'))
                    break
                if action == 'candidate_ready':
                    self.state["status"] = "candidate_ready"
                    break
                if action == 'needs_review':
                    self.state['status'] = 'needs_review'
                    self.state['stop_reason'] = {'reason':'No consumable review verdict; no confirmed basis for a geometry repair',
                                                'required_action':'Check the review evidence and the review budget'}
                    break
                self.state["repair_pending"] = True
                self.store.save()
            else:
                self.state["status"] = "scene_repair_budget_exhausted"
            self.store.save()
            return self.result()
        except (PrepError,KeyboardInterrupt,OSError) as original:
            from asset_prep.phase_budget import PhaseBudgetExceeded
            if isinstance(original,PhaseBudgetExceeded):
                self.state.update(status='phase_budget_exhausted',stop_reason={'reason':'phase_budget',
                    'detail':str(original),'next_phase':'operation'})
                self.store.save()
                return self.result()
            self.state["status"] = "paused"
            if isinstance(original, implementation.ImplementationDrift):
                self.state['stop_reason'] = {'reason':'implementation_drift', 'detail':str(original)}
            if isinstance(original, OSError):
                self.state["local_failure"] = {"kind":"local_io", "error_type":type(original).__name__,
                    "errno":original.errno,"winerror":getattr(original,"winerror",None),
                    "path":str(original.filename) if original.filename else None}
            if isinstance(original, (ProviderError, UncertainRequest)):
                calls = self.store.state["calls"]
                last = calls[-1] if calls else {}
                self.state["request_failure"] = {
                    "kind": "provider_request", "call_index": last.get("index"),
                    "phase": last.get("phase"), "diagnostics": last.get("diagnostics", {}),
                    "provider_error": last.get("provider_error", {}),
                    "outcome_unknown": last.get("status") == "pending"}
            for name, operation in (("saving state",self.store.save),("generating the pause report",self.result)):
                try:
                    operation()
                except Exception as secondary:
                    original.add_note(name+" also raised: "+type(secondary).__name__)
            raise
