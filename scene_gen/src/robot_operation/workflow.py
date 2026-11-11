from asset_prep.errors import PrepError
"""Offline task requests, bounded operation graphs, feedback routing and frozen repeated validation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import traceback
import mujoco
import numpy as np
from asset_prep.storage import file_hash, safe_path
from .common import OperationError, Settings, atomic_json, digest, identity, need, new_directory, read
from .contracts import validate_graph, validate_task, PARAMETERS, PREDICATE_FIELDS
from .recording import Recorder
from .runtime import Session
from .scene import load


def request_bundle(task_root,scene,scene_hash,profile,xml_reason=None,xml_fragments=()):
    """Read only the allowlist and send no requests; images keep file references and digests, and PDFs reuse the project parser."""
    task_root,scene=Path(task_root).resolve(),Path(scene).resolve()
    listing=read(task_root/"input_manifest.json")
    need(identity(scene)==scene_hash,"Scene version is stale","stale_scene")
    public=read(profile)
    from asset_prep.task_contracts import public_environment
    normalized_public = public_environment(public)
    prepared = None
    if listing.get('kind') == 'prepared_task_input':
        from asset_prep.task_delivery import load_delivery
        prepared, listing = load_delivery(task_root)
        need(prepared['public_environment'] == normalized_public, 'Operation environment does not match the prepared task')
    entries=[]
    for row in listing["files"]:
        name=row["path"]
        need(not any(word in name.lower() for word in ("rubric","curation")),"Scoring or curation notes must not be used as model input")
        path=safe_path(task_root,name)
        need(file_hash(path)==row["sha256"],"Task input digest mismatch")
        if path.suffix.lower()==".xml" and not xml_reason: continue
        item={"path":str(path),"sha256":row["sha256"],"relative_path":name}
        if path.suffix.lower()==".xml":
            need(xml_fragments and sum(map(len,xml_fragments))<=20000,"XML supplementary evidence must supply bounded selected fragments")
            text=path.read_text(encoding="utf-8-sig")
            need(all(isinstance(fragment,str) and fragment and fragment in text for fragment in xml_fragments),"XML fragment is not part of the source text")
            item["selected_fragments"]=list(xml_fragments)
        elif path.suffix.lower() in {".md",".txt",".json"}:
            item["text"]=path.read_text(encoding="utf-8-sig")
        elif path.suffix.lower()==".pdf":
            from asset_prep.documents import load_documents
            item["document"]=load_documents(path,None,1_000_000)
        entries.append(item)
    bundle={"version":1,"task_id":listing["task_id"],"files":entries,"xml_reason":xml_reason,
            "public_profile":public,"public_profile_sha256":digest(public),
            "scene_sha256":scene_hash,"scene":read(scene/"scene.json"),"interfaces":read(scene/"interfaces.json"),
            "operation_required_parameters":PARAMETERS,"predicate_required_fields":PREDICATE_FIELDS,
            "contract_document":(Path(__file__).resolve().parents[2]/"docs"/"robot_operation.md").read_text(encoding="utf-8")}
    bundle["input_sha256"]=digest({"manifest":listing,"files":[{"path":x["relative_path"],"sha256":x["sha256"]} for x in entries]})
    if prepared:
        from asset_prep.task_delivery import validate_role_bindings
        validate_role_bindings(bundle['scene'].get('plan',{}), prepared)
        bundle['prepared_task'] = {'definition':prepared['task_definition'], 'sha256':prepared['task_sha256'],
                                   'revision':prepared['task_revision'], 'instance_roles':prepared['instance_roles']}
        bundle['role_bindings'] = bundle['scene'].get('plan', {}).get('role_bindings', [])
    return bundle


def request_task(task_root,scene,scene_hash,profile,output,xml_reason=None,xml_fragments=()):
    """Legacy v1 exchange, retained for archived callers; improve uses two frozen stages."""
    bundle=request_bundle(task_root,scene,scene_hash,profile,xml_reason,xml_fragments)
    out=new_directory(output,[scene,task_root])
    atomic_json(out/"request.json",{"status":"awaiting_model","kind":"compile_task_and_operation_graph",
        "network_calls":0,"bundle":bundle,"instructions":"Output a source-backed task contract and a bounded operation graph; do not read scoring materials, rewrite physical state or claim detector success. List missing capabilities explicitly."})
    return {"status":"awaiting_model","request":str(out/"request.json"),"network_calls":0}


class GraphRunner:
    def __init__(self,session,graph):
        self.session,self.graph=session,graph
        self.nodes=validate_graph(graph,session.task,graph["scene_sha256"],session.settings)
        self.current=graph["entry"]; self.visits={}; self.completed=[]; self.history=[]
        self.total=0
        session.evidence.register([n[key] for n in self.nodes.values() for key in ("precondition","completion")])

    def checkpoint(self):
        return {"session":self.session.snapshot(),"current":self.current,"visits":deepcopy(self.visits),
                "completed":list(self.completed),"history":deepcopy(self.history),"total":self.total}

    def restore(self,state):
        self.session.restore(state["session"])
        self.current=state["current"]; self.visits=deepcopy(state["visits"])
        self.completed=list(state["completed"]); self.history=deepcopy(state["history"]); self.total=state["total"]

    def log(self,row):
        self.history.append(row)
        if self.session.recorder: self.session.recorder.event(row)

    def run(self):
        s=self.session
        need(s.evidence.evaluate(s.task["initial"]),"Task initial condition not met","initial_condition")
        while self.current not in {"success","failed"}:
            self.total+=1
            need(self.total<=s.settings.max_total_nodes,"Operation graph total step budget exhausted","budget_exhausted")
            node=self.nodes[self.current]
            self.visits[self.current]=self.visits.get(self.current,0)+1
            need(self.visits[self.current]<=node.get("max_visits",1),"Node visit budget exhausted","budget_exhausted")
            self.log({"type":"node_start","node":node["id"],"stage":node["stage"],"time":float(s.data.time)})
            failure=None
            for attempt in range(s.settings.local_retries+1):
                before_steps=s.sim_steps
                try:
                    s.begin(node)
                    if s.recorder:
                        atomic_json(s.recorder.root/"node_start_state.json",s.snapshot())
                    need(s.evidence.evaluate(node["precondition"]),"Node precondition not met","precondition")
                    s.execute(node)
                    while not s.evidence.evaluate(node["completion"]): s.advance(s.settings.control_period)
                    failure=None; break
                except OperationError as exc:
                    failure=exc
                    self.log({"type":"node_failure","node":node["id"],"attempt":attempt,"code":exc.code,
                              "detail":exc.detail,"evidence":exc.evidence,"time":float(s.data.time)})
                    # Only search failures without physical side effects may be re-searched in place; real contact failures take the explicit recovery branch.
                    if exc.code not in {"path_not_found","ik_not_found"} or s.sim_steps!=before_steps or attempt==s.settings.local_retries: break
            if failure:
                s.failure_snapshot=s.snapshot()
                s.stop()
                self.current=node["on_failure"]
                if self.current=="failed": raise failure
                continue
            for stage in s.task["stages"]:
                if stage["id"]==node["stage"] and stage["id"] not in self.completed:
                    if set(stage.get("depends_on",[]))<=set(self.completed) and s.evidence.evaluate(stage["goal"]):
                        self.completed.append(stage["id"])
                        self.log({"type":"stage_complete","stage":stage["id"],"time":float(s.data.time),
                                  "evidence":deepcopy(s.evidence.last)})
            self.log({"type":"node_complete","node":node["id"],"time":float(s.data.time)})
            self.current=node["on_success"]
        need(self.current=="success","Operation graph ended in failure","graph_failed")
        need(set(self.completed)=={v["id"] for v in s.task["stages"]} and s.evidence.evaluate(s.task["final"]),
             "Node execution finished, but task stages or the final state are not satisfied","task_incomplete")
        return {"status":"success","completed_stages":self.completed,"sim_time":float(s.data.time),
                "logical_time":s.logical_time,"evaluation":"task_contract_only_not_benchmark_rubric"}


def repair_request(error,scene_hash,task,graph,settings,history):
    # A collision or missed grasp by itself does not prove an asset defect. Without an independent re-check, only an operation-graph diagnosis may be requested.
    return {"status":"awaiting_model","kind":"operation_graph_revision","scene_sha256":scene_hash,
        "task_sha256":digest(task),"graph_sha256":digest(graph),"failure":{"code":error.code,"detail":error.detail,"evidence":error.evidence},
        "history":history[-8:],"history_reference":"events.jsonl","limits":{"graph_revisions":settings.graph_revisions,"scene_repairs":settings.scene_repairs},
        "scene_repair_authorized":False,"reason":"Asset/layout defects must first be re-checked with the fixed checker; a failed search is not a proof of unreachability","network_calls":0}


def run(scene,scene_hash,task,graph,output,settings=None,mode="final",images=True,perturbation=None,cancel_guard=None,quality_report=None):
    settings=settings or Settings()
    validate_graph(graph,task,scene_hash,settings)
    m,d,meta,manifest=load(scene,scene_hash)
    out=new_directory(output,[scene])
    admission=read(Path(scene)/"admission.json")
    if admission["status"]!="pass":
        result={"status":"blocked","reason":"Dynamics scene admission not passed","admission":admission}
        atomic_json(out/"task.json",task); atomic_json(out/"graph.json",graph)
        atomic_json(out/"recording.json",{"version":1,"scene_path":str(Path(scene).resolve()),
            "scene_sha256":scene_hash,"settings":asdict(settings),"dataset_complete":False,"physics_started":False})
        atomic_json(out/"result.json",result)
        from asset_generation.pipeline import snapshot
        atomic_json(out/"files.json",snapshot(out))
        return result
    if graph["provenance"]=="model_generated":
        need(task["provenance"]=="model_generated" and graph.get("model_provenance"),"Formal model result lacks provider and request/response provenance")
        for key in ("provider","requested_model","response_model","request_sha256","response_sha256"):
            need(graph["model_provenance"].get(key),"Model provenance field missing: "+key)
    session=Session(m,d,meta,manifest,task,settings)
    session.quality_report=quality_report or {}
    if perturbation is not None:
        # Apply the public perturbation only during this initialization, before the recorder starts; never change state mid-trajectory.
        delta=np.asarray(perturbation,dtype=float)
        need(delta.shape==(3,) and np.isfinite(delta).all() and np.all(np.abs(delta)<=np.array([.005,.005,np.deg2rad(.5)])+1e-12),"Perturbation exceeds the public range")
        q=int(session.base.qposadr[0]); d.qpos[q:q+2]+=delta[:2]
        rotate=np.array([np.cos(delta[2]/2),0,0,np.sin(delta[2]/2)]); quat=np.zeros(4)
        mujoco.mju_mulQuat(quat,rotate,d.qpos[q+3:q+7].copy()); d.qpos[q+3:q+7]=quat
        mujoco.mj_forward(m,d)
    session.cancel_guard=cancel_guard or (lambda:None)
    runner=GraphRunner(session,graph)
    recorder=Recorder(out,mode,images); session.recorder=recorder
    recorder.start(session,scene_hash,task,graph,scene)
    if perturbation is not None: recorder.event({"type":"initial_perturbation","delta":list(perturbation)})
    try:
        session.guard()
        result=runner.run()
    except OperationError as exc:
        atomic_json(out/"failure_state.json",getattr(session,"failure_snapshot",session.snapshot()))
        session.stop()
        request=repair_request(exc,scene_hash,task,graph,settings,runner.history)
        atomic_json(out/"feedback_request.json",request)
        result={"status":"blocked" if exc.code in {"capability_missing","initial_condition"} else "failed",
            "code":exc.code,"detail":exc.detail,"completed_stages":runner.completed,"sim_time":float(d.time),
            "feedback":"feedback_request.json","network_calls":0}
    except PrepError as exc:
        atomic_json(out/"failure_state.json",session.snapshot())
        session.stop()
        result={"status":"interrupted","code":type(exc).__name__,"detail":str(exc),"completed_stages":runner.completed}
        raise
    except Exception as exc:
        atomic_json(out/"failure_state.json",session.snapshot())
        session.stop()
        result={"status":"error","code":"execution_error","detail":f"{type(exc).__name__}: {exc}",
                "traceback":traceback.format_exc(),"completed_stages":runner.completed}
    finally:
        if 'result' not in locals(): result={"status":"interrupted","completed_stages":runner.completed}
        recorder.close(result)
    return result


def repeat(scene,scene_hash,task,graph,output,settings=None,images=True):
    settings=settings or Settings(); out=new_directory(output,[scene])
    rng=np.random.default_rng(settings.seed)
    cases=[("canonical",None)]+[(f"repeat_{i:02d}",None) for i in range(3)]
    cases += [(f"perturbed_{i:02d}",rng.uniform([-0.005,-0.005,-np.deg2rad(.5)],[.005,.005,np.deg2rad(.5)]).tolist()) for i in range(10)]
    results=[]
    atomic_json(out/"schedule.json",{"cases":cases,"settings":asdict(settings),"scene_sha256":scene_hash,"graph_sha256":digest(graph)})
    for name,delta in cases:
        result=run(scene,scene_hash,task,graph,out/name,settings,images=images,perturbation=delta)
        results.append({"case":name,"perturbation":delta,**result})
        atomic_json(out/"summary.json",{"runs":results,"successes":sum(r["status"]=="success" for r in results),"scheduled":14,"completed":len(results)})
    return {"runs":results,"success_rate":sum(r["status"]=="success" for r in results)/14}
