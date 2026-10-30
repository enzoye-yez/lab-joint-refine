"""Shared context projection for offline exchanges and explicitly enabled model calls."""
from dataclasses import asdict
from pathlib import Path
from asset_generation.request_context import RequestContext, messages_for, measure
from asset_generation.contracts import GenerationConfig
from asset_prep.errors import PrepError
from asset_prep.storage import atomic_json, digest, encoded, safe_path
from .common import need, read

TASK_PROMPT = """只根据规程与资产身份编译任务契约 task，不生成操作图、不读取候选操作计划。
使用已注册检测条件，阶段必须有输入依据。未知能力写 unverified，不伪造可观测条件。
任务来源、身份、初态、终态及功能要求不可为方便执行而降低。返回 {task:任务,unverified:列表}。"""
GRAPH_PROMPT = """为已冻结 task 生成有限操作图 graph；不得修改任务条件或宣称执行成功。
优先使用真实接口局部坐标，检查接近、运动、退离、另一臂及携带物。需要时改变停靠与升降。
失败修订只针对证据，未知搜索结果不是不可达证明。返回 {graph:操作图}，无法推进返回 {deferred:{reason:原因}}。
允许接触只是计划声明，不绕过穿透和接触力检查。"""


class AwaitingResponse(PrepError):
    pass


def export_request(root, instruction, context, images, phase, generation=None, binding=None):
    generation=generation or GenerationConfig()
    root=Path(root)
    view=RequestContext(root,binding or {},instruction,context,images,phase,generation)
    batches=[]
    for index in range(view.delivery_count):
        view.delivery_index=index; view.set_delivery()
        messages=messages_for(view.instruction,view.context,view.images)
        need(len(encoded(messages))+512<=generation.max_request_bytes,"离线请求超过实际字节限制")
        budget=measure(view.instruction,view.context,view.images,generation)
        relative=view.binding+"/"+f"{index:04d}.json"
        batch={"index":index,"messages":messages,"budget":budget}
        atomic_json(root/"requests"/relative,batch)
        batches.append({"index":index,"file":relative,"sha256":digest(batch),"budget":budget})
    request={"version":2,"request_id":view.binding,"phase":phase,"status":"awaiting_model","network_calls":0,
        "batches":batches,"generation":asdict(generation),"source_ref":view.source_ref,
        "context_root":str(view.root.resolve()),"allowed_references":sorted(view.refs)}
    path=root/"requests"/(view.binding+".json")
    atomic_json(path,request)
    return path


def read_page(request_path, query):
    request_path=Path(request_path).resolve()
    request=read(request_path); root=Path(request["context_root"]).resolve()
    need(root==request_path.parent.parent/"request_context"/request["request_id"],"请求资料根目录不匹配")
    need(query.get("reference") in request["allowed_references"],"详情引用不属于请求")
    # Reconstruct only the read-only reader state; all reads remain hash checked.
    view=object.__new__(RequestContext); view.root=root; view.refs=set(request["allowed_references"])
    page=view.read(query)
    request["allowed_references"]=sorted(view.refs)
    atomic_json(request_path,request)
    return page


class Exchange:
    """No transport or credentials are constructed in offline mode."""
    manages_context=True
    def __init__(self, store, generation, online=None, guard=lambda:None):
        self.store,self.generation,self.online,self.guard=store,generation,online,guard
        self.execution_guard=guard
        self.cancel_guard=guard
        self.progress=lambda *_:None

    def ask(self,instruction,context,images,phase):
        self.guard()
        if any(c["status"]=="pending" or c.get("outcome_unknown") for c in self.store.state["calls"]):
            from asset_prep.errors import UncertainRequest
            raise UncertainRequest("存在未知请求；保留全部预留，不自动重发")
        if self.online is not None:
            self.online.execution_guard=self.guard; self.online.cancel_guard=self.guard
            return self.online.ask(instruction,context,images,phase)
        path=export_request(self.store.root,instruction,context,images,phase,self.generation,self.store.state["identity"])
        request=read(path); key=request["request_id"]
        response_path=self.store.root/"imports"/(key+".json")
        self.store.state["pending_exchange"]={"request_id":key,"path":str(path),"phase":phase}
        self.store.save()
        if not response_path.exists():
            pending=self.store.state.get("scene",{}).get("pending")
            if pending is not None:
                pending["awaiting_exchange"]=True;self.store.save()
            raise AwaitingResponse("已导出预算内请求，等待显式导入响应："+str(path))
        response=read(response_path)
        need(response.get("request_id")==key,"响应属于其他请求")
        for batch in request["batches"]:
            need(digest(read(safe_path(self.store.root/"requests",batch["file"])))==batch["sha256"],"实际请求批次被修改")
        need(response.get("delivered_batches")==list(range(len(request["batches"]))),"响应未声明处理全部资料/图片批次")
        source=response.get("provenance")
        need(source in {"developer_fixture","model_generated"},"导入响应必须声明来源")
        need(source!="developer_fixture" or self.store.state["identity"].get("provenance")=="developer_fixture","正式联合运行及子流程拒绝开发者预置响应")
        call={"provenance":source,"request_id":key,"response_sha256":digest(response),"transport":"externally_supplied_response"}
        if source=="model_generated":
            provenance=response.get("model_provenance",{})
            need(all(isinstance(provenance.get(k),str) and provenance[k] for k in ("provider","requested_model","response_model")),"外部模型响应缺少来源")
            configured=self.store.state["identity"].get("configuration",{}).get("provider")
            if configured:
                need(provenance["provider"]==configured["provider"] and provenance["requested_model"]==configured["model"],"外部调用与联合运行选择的提供商或模型不同")
            call.update(provenance)
            accounted=self.store.state.setdefault("imported_responses",{})
            if key not in accounted:
                # Each delivered external request consumes a call; importing again never refunds it.
                accounted[key]={"response_sha256":digest(response),"calls":[]}; self.store.save()
            entry=accounted[key]
            need(entry["response_sha256"]==digest(response),"已记账响应不得替换")
            for index in range(len(request["batches"])):
                ledger_key=digest([key,index])
                row=next((c for c in self.store.state["calls"] if c["key"]==ledger_key),None)
                if row is None:
                    row=self.store.reserve(ledger_key,"model",request["batches"][index]["budget"]["required"],defer_send=True)
                row.update(phase=phase,**provenance,transport="externally_supplied_response")
                if row["index"] not in entry["calls"]:entry["calls"].append(row["index"])
                if row["status"]!="done":self.store.finish(row,{"external_response_sha256":digest(response)})
            call["ledger_indices"]=entry["calls"]
        self.store.state.pop("pending_exchange",None); self.store.save()
        return response["value"],call


def import_exchange(root, response_path):
    root=Path(root); response=read(response_path)
    key=response.get("request_id","")
    need(isinstance(key,str) and len(key)==64 and all(c in "0123456789abcdef" for c in key),"请求编号无效")
    request=read(safe_path(root,"requests/"+key+".json"))
    need(response.get("delivered_batches")==list(range(len(request["batches"]))),"必须确认全部资料批次")
    path=safe_path(root,"imports/"+key+".json")
    if path.exists(): need(digest(read(path))==digest(response),"已导入响应不可覆盖")
    else: atomic_json(path,response)
    return {"status":"imported","request_id":key,"network_calls":0}
