"""Versioned joint repair coordinator; no API client exists without --execute."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
from asset_prep import implementation
from asset_prep.config import Config
from asset_prep.errors import PrepError,BudgetExceeded,UncertainRequest,ValidationError
from asset_prep.storage import Store,atomic_json,digest,run_lock,safe_path
from asset_generation.contracts import GenerationConfig
from .common import Settings,OperationError,need,read,identity
from .contracts import validate_task,validate_graph
from .requests import Exchange,AwaitingResponse,TASK_PROMPT,GRAPH_PROMPT
from .joint_state import Budget,compare,duplicate,mechanism,LIMITS
from .scene import prepare,load
from .diagnosis import inspect_scene,diagnose_run,packet

ROUTE_PROMPT="""根据固定诊断选择下一动作。只返回一种：
{operation:{reason:说明}} 修订操作图；
{asset:{reason:说明,instances:[实际实例],evidence_ids:[证据编号],checks:[初态复查查询]}} 修正真实仪器模型；
{fixture:{reason:说明,instances:[实际夹具或工具实例],evidence_ids:[证据编号],checks:[初态复查查询]}} 有证据的夹具设计试验；
{scene:{reason:说明,instances:[实际实例],evidence_ids:[证据编号],checks:[初态复查查询]}} 已确认的布局或资产缺陷；
{diagnose:{queries:[通用工具查询]}} 有界补查，查询含kind/state/question/hypotheses及工具参数；state为initial、node_start或failure。
根据diagnostic_tools选择和组合量测，不按故障名称编造工具。先读已有证据；重复查询会复用，连续无增益会停止补查。
probe仅允许同一完整状态下已有节点的一个参数变化；局部试验不能修改任务、成功判据或接触许可。
资产返修需要固定缺陷证据；夹具设计可以使用接触异常或保持失败证据作为待验证假设。
检查unknown/invalid/interrupted均不算通过，工具失败不是几何缺陷。允许原因未确定或多个原因共同作用。
{deferred:{reason:原因}} 当前没有有效动作。不得把搜索失败当作不可达证明；工程故障不能修改几何。
先推进可独立完成的阶段。保留必要开口、行程、真实规格、原厂机器人参数和固定任务条件。"""

class Paused(PrepError):pass


def open_run(root,scene,scene_sha,task_root,profile,project,*,resume=False,execute=False,
             expected_implementation=None,seed_run=None,package=None,protocol=None,backend="docker",developer=False,
             provider=None,generation=None,settings=None):
    root,project=Path(root).resolve(),Path(project).resolve()
    for source in (scene,task_root,seed_run,package):
        if source is not None:
            source=Path(source).resolve()
            need(root!=source and source not in root.parents,"联合运行目录不得写入输入场景、任务或来源运行目录")
    from .workflow import request_bundle
    bundle=request_bundle(task_root,scene,scene_sha,profile)
    source_hash=identity(scene)
    code=implementation.implementation_digest(implementation.source_manifest(project))
    if execute:
        import sys
        need(not developer,"开发者样例运行不能调用模型")
        need(getattr(sys,"_harness_runtime",None) and expected_implementation==code,"在线联合运行必须通过 run.ps1 并绑定已验证实现摘要")
        implementation.verify(code,project=project,where="联合执行启动")
    provider=replace(provider or Config(),max_model_calls=800,max_reserved_tokens=240000000)
    generation=replace(generation or GenerationConfig(),max_model_calls=800,max_reserved_tokens=240000000,max_transport_retries=0)
    settings=settings or Settings()
    config={"version":1,"scene":str(Path(scene).resolve()),"scene_sha256":source_hash,
        "task_root":str(Path(task_root).resolve()),"profile":str(Path(profile).resolve()),
        "seed_run":str(Path(seed_run).resolve()) if seed_run else None,"package":str(Path(package).resolve()) if package else None,
        "protocol":str(Path(protocol).resolve()) if protocol else None,"backend":backend,"developer":developer,
        "provider":asdict(provider),"generation":asdict(generation),"settings":asdict(settings),
        "implementation":code,"input_sha256":bundle["input_sha256"],"profile_sha256":bundle["public_profile_sha256"]}
    if seed_run:need(package is not None,"接通资产修复必须提供原资料包")
    lineage={"workflow_contract":"joint_operation_v1","configuration":config,"implementation":code,
             "provenance":"developer_fixture" if developer else "model_generated"}
    if resume:
        saved=read(root/"configuration.json")
        need(saved==config,"联合运行输入、源码或固定配置变化；须新建版本，不原地迁移")
    else:
        need(not root.exists() or not any(p.name!=".lock" for p in root.iterdir()),"运行目录非空，需显式恢复或使用新名称")
        atomic_json(root/"configuration.json",config)
    store=Store(root,lineage,provider,resume=resume)
    if seed_run and "source_identity" not in store.state:
        from .scene_bridge import seed
        copied=seed(seed_run,root)
        source=copied["source_identity"]
        need(developer or (source["provider"]==provider.provider and source["model"]==provider.model),"来源模型不同；禁止静默混合实验")
        original_scene=Path(scene).resolve()
        if "operation_conversion" in read(original_scene/"scene.json"):
            original_sha=read(original_scene/"scene.json")["operation_conversion"]["source_sha256"]
        else:original_sha=source_hash
        records=[*copied["scene"]["stages"],copied["scene"].get("latest",{}),copied["scene"].get("published_scene",{})]
        artifacts={r.get("artifact",r.get("directory","")+"/scene/artifact") for r in records if r}
        need(any((Path(seed_run)/a).is_dir() and identity(Path(seed_run)/a)==original_sha for a in artifacts),"待操作场景不属于提供的原场景运行")
        store.state.update(copied)
        store.state["joint_backend"]=source["backend"]
        budget=Budget(store)
        budget.state["rounds"]=[{"key":digest(["inherited",i,r]),"kind":"inherited_scene"} for i,r in enumerate(store.state["scene"].get("repair_history",[]))]
        for stage in store.state["scene"]["stages"]:
            if stage["kind"]=="asset" and stage.get("source"):
                budget.state["assets"].setdefault(stage["job"],[]).append(stage["directory"])
        # Preserve conservative legacy counters even when early attempts have no completed source.
        for job,count in copied["scene"].get("candidate_attempts",{}).items():
            count=len(count) if isinstance(count,list) else count
            inherited_rows=budget.state["assets"].setdefault(job,[])
            inherited_rows.extend("inherited:"+str(i) for i in range(len(inherited_rows),count))
    store.state.setdefault("joint",{"phase":"prepare","status":"ready","history":[],"candidates":[],"queries":[],"format_errors":[],"diagnostic_actions":[]})
    from asset_prep.phase_budget import enter
    enter(store,'final' if store.state['joint']['phase'] in {'final','complete'} else 'operation')
    store.save()
    return Coordinator(store,bundle,project,provider,generation,settings,execute,developer,config)


class Coordinator:
    def __init__(self,store,bundle,project,provider,generation,settings,execute,developer,config):
        self.store,self.state,self.bundle=store,store.state["joint"],bundle
        self.project,self.settings,self.generation,self.developer=project,settings,generation,developer
        self.root,self.config=store.root,config
        self.guard_impl=implementation.Guard(project,store.state["identity"]["implementation"])
        self.budget=Budget(store)
        self.state.setdefault("diagnostic_ledger", {})
        if "static_report" not in self.state:
            self.state["static_report"]=deepcopy(store.state.get("scene",{}).get("latest",{}).get("report",{}))
        online=None
        if execute:
            from asset_prep.config import credentials
            from asset_generation.model import GenerationModel
            from asset_generation.transport import transport_for
            online=GenerationModel(provider,generation,credentials(project),store,transport_for(provider))
        self.exchange=Exchange(store,generation,online,self.guard)
        self.bridge=None
        if config["seed_run"]:
            from .scene_bridge import SceneBridge
            self.bridge=SceneBridge(store,self.exchange,self.budget,project,config["package"],config["backend"],generation,provider,self.guard,config["protocol"])

    def guard(self):
        self.guard_impl.check()
        self.cancel()

    def cancel(self):
        # Per-step cancellation is cheap; full source verification brackets execution.
        if (self.root/"pause.request").exists():raise Paused("已请求暂停；保留现场、调用预留和已有产物")
        if any(c["status"]=="pending" or c.get("outcome_unknown") for c in self.store.state["calls"]):
            raise UncertainRequest("未知请求保留全额预留；禁止自动重新发送")

    def save(self,**changes):
        self.state.update(changes);self.store.save();self.publish()

    def publish(self):
        current=self.state.get("current",{})
        best=self.state.get("baseline",{})
        document={"version":1,"status":self.state["status"],"phase":self.state["phase"],"reason":self.state.get("reason"),
            "scene":current.get("scene",self.config["scene"]),"scene_sha256":current.get("scene_sha256",self.config["scene_sha256"]),
            "last_verified_baseline":best.get("directory"),"scene_completeness":self.state.get("scene_completeness","source_declared_unknown"),
            "static_quality":self.state.get("static_quality","not_recertified"),"dynamics_admission":(self.state.get("admission") or {}).get("status","unknown"),
            "operation_result":self.state.get("final_result",self.state.get("last_result",{})),
            "evidence_gaps":self.state.get("unverified",[])+self.state.get("diagnosis",{}).get("coverage_gaps",[]),"remaining":self.budget.remaining(),"consumption":self.store.totals(),
            "rounds":len(self.budget.state["rounds"]),"provenance":"developer_fixture" if self.developer else "model_generated",
            "pending_exchange":self.store.state.get("pending_exchange"),"model_effectiveness_verified":False}
        document['phase_budget']=self.store.state.get('phase_budget')
        ledger=self.state.get("diagnostic_ledger",{})
        spent=sum("rebound_from" not in r for r in ledger.get("records",{}).values())
        document["diagnostics"]={"measurements":spent,"evidence_records":len(ledger.get("records",{})),
            "remaining_measurements":max(0,self.settings.diagnostic_queries-spent),
            "sim_seconds_reserved":ledger.get("sim_seconds_reserved",0),
            "invalid_or_unknown":sum(r["status"]!="valid" for r in ledger.get("records",{}).values()),
            "last_batch":self.state.get("diagnosis",{}).get("last_diagnostic_batch"),
            "attribution":self.state.get("diagnosis",{}).get("attribution"),
            "repair_verification":self.state.get("diagnosis",{}).get("repair_verification")}
        atomic_json(self.root/"progress.json",document)
        (self.root/"scene_document.md").write_text("# 联合运行场景文档\n\n"+"\n\n".join(str(k)+"："+str(v) for k,v in document.items()),encoding="utf-8")
        return document

    def ask(self,prompt,context,phase):
        errors=self.state["format_errors"]
        value,call=self.exchange.ask(prompt,{**context,"validation_feedback":[e for e in errors if e["phase"]==self.state["phase"]][-4:]},self.images(),phase)
        need(self.developer or call.get("provenance")!="developer_fixture","正式联合运行拒绝开发者预置响应")
        return value,call

    def images(self):
        from asset_generation.inputs import image_record
        rows=[]
        for item in self.bundle["files"]:
            path=Path(item["path"])
            if path.suffix.lower() in {".png",".jpg",".jpeg",".webp"}:
                rows.append(image_record(path,path.parent,self.generation))
        for item in self.state.get("scene_images",[]):
            rows.append(image_record(self.root/item["path"],self.root,self.generation,expected=item["sha256"]))
        # Scene/failed-pose image paths are immutable recording artifacts.
        run=self.state.get("baseline",{}).get("directory") if self.state.get("baseline_restored") else self.state.get("last_run")
        if run:
            for path in sorted(safe_path(self.root,run).glob("images/*.png"))[-3:]:
                rows.append(image_record(path,self.root,self.generation))
        return rows

    def current_task(self):
        task=read(self.root/"task.json")
        need(digest(task)==self.state["task_sha256"],"冻结任务被修改","stale_task")
        from .bindings import resolve_task
        current=self.state["current"]
        model,_,_,manifest=load(current["scene"],current["scene_sha256"])
        bound,receipt=resolve_task(task,self.bundle["interfaces"],manifest,model)
        atomic_json(self.root/"bindings"/(current["scene_sha256"]+".json"),receipt)
        return bound

    def prepare(self):
        source=self.state.get("next_scene",self.config["scene"])
        src_hash=identity(source)
        index=len(self.state.get("scene_versions",[]))
        destination=self.root/"scenes"/f"{index:04d}"
        if "operation_conversion" in read(Path(source)/"scene.json"):
            need(src_hash==self.config["scene_sha256"] or source!=self.config["scene"],"来源变化")
            load(source,src_hash)
            result={"scene":str(source),"scene_sha256":src_hash,"admission":read(Path(source)/"admission.json")}
        elif destination.exists():
            meta=read(destination/"scene.json");admission=read(destination/"admission.json")
            need(meta.get("operation_conversion",{}).get("source_sha256")==src_hash,"未完成派生场景，须保留现场并修复工程故障")
            result={"scene":str(destination),"scene_sha256":identity(destination),"admission":admission}
        else:result=prepare(source,destination,src_hash,self.settings)
        # Read-only previews live outside scene identity, so rendering does not invalidate evidence.
        from .views import render_current
        images,error=render_current(result["scene"],result["scene_sha256"],self.root/"views"/result["scene_sha256"])
        self.state["scene_images"]=images
        if error:self.state.setdefault("unverified",[]).append(error)
        self.state.setdefault("scene_versions",[]).append(result)
        self.save(current={k:result[k] for k in ("scene","scene_sha256")},admission=result["admission"],phase="graph" if "task_sha256" in self.state else "task")

    def task(self):
        # No graph, graph history, candidate trajectory or graph-derived conditions enter compilation.
        inputs={k:self.bundle[k] for k in ("task_id","files","public_profile","public_profile_sha256","input_sha256","interfaces","predicate_required_fields","contract_document")}
        prepared = self.bundle.get('prepared_task')
        prompt = TASK_PROMPT
        if prepared:
            inputs.update(prepared_task=prepared, role_bindings=self.bundle.get('role_bindings', []))
            prompt += '\n已选流程、准备边界和初终态以上游 prepared_task 为准，不重选流程。额外返回 coverage 列表：每条含 requirement、targets、unresolved。requirement 为 stage:阶段id、initial:对象id、final:终态id；targets 分别为编译阶段id列表、["initial"]或["final"]。无法落实填空列表并说明 unresolved；每条上游要求恰好出现一次。'
        value,call=self.ask(prompt,{"source":inputs},"task_compile")
        expected = {'task','unverified','coverage'} if prepared else {'task','unverified'}
        need(isinstance(value,dict) and set(value)==expected,"任务编译响应字段不完整")
        task=deepcopy(value["task"]);validate_task(task)
        need(task["input_sha256"]==self.bundle["input_sha256"] and task["public_profile_sha256"]==self.bundle["public_profile_sha256"],"任务必须绑定实际输入与公共环境摘要")
        need(task["provenance"]==("developer_fixture" if self.developer else "model_generated"),"任务来源不一致")
        need(isinstance(value["unverified"],list) and all(isinstance(x,str) for x in value["unverified"]),"未知要求必须明确列出")
        if prepared:
            from asset_prep.task_delivery import validate_task_coverage
            value['unverified'].extend(validate_task_coverage(value['coverage'], prepared['definition'], task))
            atomic_json(self.root/'upstream_coverage.json', {'task_sha256':prepared['sha256'], 'coverage':value['coverage']})
        from .bindings import resolve_task
        current=self.state["current"];model,_,_,manifest=load(current["scene"],current["scene_sha256"])
        resolve_task(task,self.bundle["interfaces"],manifest,model)
        atomic_json(self.root/"task.json",task);atomic_json(self.root/"task_receipt.json",{"task_sha256":digest(task),"call":call})
        self.save(task_sha256=digest(task),unverified=value["unverified"],phase="graph")

    def graph(self):
        task=self.current_task();current=self.state["current"]
        _,_,meta,manifest=load(current["scene"],current["scene_sha256"])
        value,call=self.ask(GRAPH_PROMPT,{"contract":task,"current":current,"interfaces":manifest,
            "static_quality":self.state.get('static_report',{}),
            "deferred_repairs":self.store.state.get('scene',{}).get('repair_schedule',[]),
            "environment":meta["environment"],"operation_required_parameters":self.bundle["operation_required_parameters"],
            "contract_document":self.bundle["contract_document"],"diagnosis":self.state.get("diagnosis"),
            "rejected_approaches":self.state["history"][-12:],"baseline_restored":self.state.get("baseline_restored",False)},"operation_graph")
        if isinstance(value,dict) and set(value)=={"deferred"}:
            self.save(status="needs_input",reason=value["deferred"],phase="deferred");return
        need(isinstance(value,dict) and set(value)=={"graph"},"操作规划只允许返回 graph，不得修改任务")
        graph=value["graph"];validate_graph(graph,task,current["scene_sha256"],self.settings)
        need(graph["provenance"]==task["provenance"],"图与任务的生成来源不同")
        if not self.developer:
            provenance={"provider":self.config["provider"]["provider"],"requested_model":self.config["provider"]["model"],
                "response_model":call.get("response_model",call.get("model")),
                "request_sha256":call.get("request_id",call.get("request_key",digest(call))),"response_sha256":call.get("response_sha256",digest(value))}
            need(isinstance(provenance["response_model"],str) and provenance["response_model"],"缺少实际响应模型，不能伪造来源")
            graph["model_provenance"]=provenance
        evidence_hash=digest(self.state.get("diagnosis",{}))
        need(not duplicate(self.state["history"],graph,current["scene_sha256"],evidence_hash),"该输入与证据下已失败的机制重复，须选择不同方向")
        number=len(self.state["candidates"])
        relative=f"candidates/{number:04d}/graph.json";atomic_json(safe_path(self.root,relative),graph)
        candidate={"graph":relative,"graph_sha256":digest(graph),"current":deepcopy(current),"evidence_sha256":evidence_hash,
            "directory":f"executions/{number:04d}","call":call}
        self.state["candidates"].append(candidate)
        self.save(active_candidate=number,phase="execute",baseline_restored=False)

    def execute(self,final=False):
        from .workflow import run
        candidate=self.state["candidates"][self.state["active_candidate"]]
        graph=read(safe_path(self.root,candidate["graph"]));need(digest(graph)==candidate["graph_sha256"],"候选图被改写")
        current=candidate["current"];task=self.current_task()
        relative=("final/"+str(self.state["active_candidate"])) if final else candidate["directory"]
        destination=safe_path(self.root,relative)
        if not final:self.budget.round(digest(["graph",candidate["directory"],candidate["graph_sha256"]]),"operation")
        self.state["inflight"]={"kind":"final" if final else "trial","directory":relative};self.store.save()
        if (destination/"result.json").exists():
            files=read(destination/"files.json")
            from asset_prep.storage import file_hash
            need(all(file_hash(safe_path(destination,n))==sha for n,sha in files.items()),"执行记录被修改")
            result=read(destination/"result.json")
        else:
            need(not destination.exists(),"执行已中断且没有完整回执；不自动重放有副作用的操作","interrupted_execution")
            result=run(current["scene"],current["scene_sha256"],task,graph,destination,self.settings,
                       mode="final" if final else "trial",cancel_guard=self.cancel,
                       quality_report=self.state.get('static_report',{}))
        self.guard()
        self.state.pop("inflight",None)
        if final:
            self.save(final_result=result,phase="complete" if result["status"]=="success" else "diagnose",
                status="verified_task" if result["status"]=="success" and not self.state["unverified"] else "partial",
                last_run=relative,last_result=result)
        else:self.save(last_run=relative,last_result=result,phase="diagnose")

    def diagnose(self):
        current=self.state["current"]
        from .diagnostic_service import inspect_with_ledger, collect, merge, check_obligations
        from .diagnostic_tools import DiagnosticContext
        report=inspect_with_ledger(current["scene"],current["scene_sha256"],self.settings,self.state["diagnostic_ledger"],
            run_root=safe_path(self.root,self.state["last_run"]),evidence_root=self.root/"diagnostic_trials",
            save=self.store.save,guard=self.guard,cancel_guard=self.cancel)
        ticket=self.state.get("repair_ticket")
        if ticket and ticket["scene_sha256"]!=current["scene_sha256"]:
            context=DiagnosticContext(current["scene"],current["scene_sha256"],self.settings,
                evidence_root=self.root/"diagnostic_trials",guard=self.guard,cancel_guard=self.cancel)
            records,_=collect(context,self.state["diagnostic_ledger"],ticket["checks"],self.store.save,count_stall=False)
            report=merge(report,records)
            report["repair_verification"]=check_obligations(report,ticket,records)
        atomic_json(self.root/"diagnoses"/(str(self.state["active_candidate"])+".json"),report)
        candidate=self.state["candidates"][self.state["active_candidate"]]
        evaluated={**candidate,"result":self.state["last_result"],"diagnosis":report,
                   "scene_state":deepcopy(self.store.state.get("scene")),"admission":deepcopy(self.state.get("admission")),"static_quality":self.state.get("static_quality"),"static_report":deepcopy(self.state.get("static_report",{}))}
        outcome=compare(self.state.get("baseline"),evaluated)
        graph=read(safe_path(self.root,candidate["graph"]))
        self.state["history"].append({"input_sha256":current["scene_sha256"],"evidence_sha256":candidate["evidence_sha256"],
            "action_sha256":digest(graph),"mechanism":digest(mechanism(graph)),"outcome":outcome,"directory":candidate["directory"]})
        if outcome in {"baseline","improved","improved_partial"}:self.state["baseline"]=evaluated
        else:
            baseline=self.state["baseline"]
            self.state["current"]=deepcopy(baseline["current"])
            self.state["static_report"]=deepcopy(baseline.get("static_report",{}))
            self.state["admission"]=deepcopy(baseline.get("admission"))
            self.state["static_quality"]=baseline.get("static_quality")
            if self.bridge and baseline.get("scene_state"):self.bridge.restore(baseline["scene_state"])
            self.state["baseline_restored"]=True
        self.state.pop("scene_work",None)
        self.state["diagnosis"]=report if not self.state.get("baseline_restored") else self.state["baseline"]["diagnosis"]
        if self.state.get("baseline_restored"):
            from .views import render_current
            current=self.state["current"]
            self.state["scene_images"],_=render_current(current["scene"],current["scene_sha256"],self.root/"views"/current["scene_sha256"])
        success=(self.state["last_result"]["status"]=="success" and outcome not in {"exploration","ineffective"}
                 and report.get("repair_verification",{}).get("coverage_complete",True)
                 and report.get("repair_verification",{}).get("original_problem_resolved",True))
        self.save(phase="final" if success else "route")

    def route(self):
        report=self.state["diagnosis"]
        value,call=self.ask(ROUTE_PROMPT,{"contract":self.current_task(),"current":self.state["current"],"diagnosis":report,
            "history":self.state["history"][-12:],"queries_already_run":self.state["diagnostic_actions"],
            "diagnostic_tools":report.get("diagnostic_tools",[]),"scene_repair_available":bool(self.bridge)},"joint_repair_route")
        need(isinstance(value,dict) and len(value)==1 and set(value)<={"operation","asset","fixture","scene","diagnose","deferred"},"修复路由格式无效")
        kind=next(iter(value))
        if kind=="operation":self.save(phase="graph")
        elif kind in {"asset","fixture","scene"}:
            need(self.bridge is not None,"没有来源资产工作区；须补充原运行与资料包，不能直接改装配产物")
            from .diagnostic_service import repair_ticket,collect,merge
            from .diagnostic_tools import DiagnosticContext
            ticket=repair_ticket(report,value[kind],kind,self.state["current"]["scene_sha256"])
            current=self.state["current"]
            context=DiagnosticContext(current["scene"],current["scene_sha256"],self.settings,
                evidence_root=self.root/"diagnostic_trials",guard=self.guard,cancel_guard=self.cancel)
            records,_=collect(context,self.state["diagnostic_ledger"],ticket["checks"],self.store.save,count_stall=False)
            need(all(r["status"]=="valid" for r in records),"返修复查无法量测；先补充有效检查能力")
            ticket["baseline_checks"]=records
            report=merge(report,records)
            self.state.setdefault("repair_history",[]).append(ticket)
            self.save(phase="scene",repair_ticket=ticket,diagnosis=report)
        elif kind=="diagnose":
            from .diagnostic_service import collect,merge,attribution
            from .diagnostic_tools import DiagnosticContext
            current=self.state["current"]
            previous=self.state.get("baseline",{}).get("directory",self.state.get("last_run"))
            context=DiagnosticContext(current["scene"],current["scene_sha256"],self.settings,
                run_root=safe_path(self.root,previous) if previous else None,evidence_root=self.root/"diagnostic_trials",guard=self.guard,cancel_guard=self.cancel)
            records,receipt=collect(context,self.state["diagnostic_ledger"],value[kind]["queries"],self.store.save)
            report=merge(report,records);report["last_diagnostic_batch"]=receipt;report["attribution"]=attribution(report)
            baseline=self.state.get("baseline")
            if baseline and baseline["current"]==current:
                baseline["diagnosis"]=deepcopy(report)
            self.state["diagnostic_actions"].append(receipt)
            self.save(diagnosis=report,phase="deferred" if receipt["exhausted"] else "route",
                      status="needs_input" if receipt["exhausted"] else "running",
                      reason="连续补查未增加有效信息；需不同诊断方法或补充能力" if receipt["exhausted"] else None)
        else:self.save(phase="deferred",status="needs_input",reason=value[kind])

    def scene(self):
        result=self.bridge.repair(self.state["diagnosis"])
        if result["status"]=="deferred":
            # Preserve latest loadable baseline and continue trying operation strategies.
            self.state.setdefault("unverified",[]).append(result["reason"])
            if result.get("partial_scene"):
                self.state["published_partial_scene"]=result["partial_scene"]
                self.state["scene_completeness"]={"missing":result["missing"],"partial_scene":result["partial_scene"]}
            baseline=self.state["scene_work"]["baseline"]
            self.bridge.restore(baseline);self.state.pop("scene_work",None)
            self.save(phase="graph",baseline_restored=True);return
        self.save(next_scene=result["scene"],static_quality=result["static_quality"],static_report=result["report"],scene_completeness="complete",phase="prepare")

    def advance(self):
        try:
            while self.state["phase"] not in {"complete","deferred"}:
                self.guard()
                if len(self.state["format_errors"])>=800:raise BudgetExceeded("格式恢复记录达到共享调用预算边界")
                self.state["status"]="running";self.state.pop("reason",None);self.store.save()
                phase=self.state["phase"]
                from asset_prep.phase_budget import enter
                budget_phase='final' if phase=='final' else 'operation'
                if self.store.state['phase_budget']['phase']!=budget_phase:
                    enter(self.store,budget_phase)
                try:
                    {"prepare":self.prepare,"task":self.task,"graph":self.graph,"execute":self.execute,
                     "diagnose":self.diagnose,"route":self.route,"scene":self.scene,"final":lambda:self.execute(True)}[phase]()
                except (OperationError,ValidationError,KeyError,TypeError) as exc:
                    if phase not in {"task","graph","route"}:raise
                    self.state["format_errors"].append({"phase":phase,"error":str(exc),"number":len(self.state["format_errors"])+1})
                    self.store.save()
            return self.publish()
        except AwaitingResponse as exc:self.save(status="awaiting_response",reason=str(exc))
        except Paused as exc:self.save(status="paused_user",reason=str(exc))
        except (PrepError,OperationError,OSError,ValueError,KeyError,TypeError) as exc:self.save(status="needs_input",reason={"kind":type(exc).__name__,"detail":str(exc)})
        except KeyboardInterrupt:self.save(status="paused_user",reason="用户中断；未重放未完成执行")
        return self.publish()
