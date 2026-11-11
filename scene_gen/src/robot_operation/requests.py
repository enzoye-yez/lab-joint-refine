"""Shared context projection for offline exchanges and explicitly enabled model calls."""
from dataclasses import asdict
from pathlib import Path
from asset_generation.request_context import RequestContext, messages_for, measure
from asset_generation.contracts import GenerationConfig
from asset_prep.errors import PrepError
from asset_prep.storage import atomic_json, digest, encoded, safe_path
from .common import need, read

TASK_PROMPT = """Compile the task contract `task` from the protocol and asset identities only; do not generate an operation graph or read candidate operation plans.
Use only registered detector conditions; every stage must cite its input source. List unknown capabilities under unverified; never fabricate observable conditions.
Never weaken the task provenance, identities, initial state, final state or functional requirements to ease execution. Return {task:<task>,unverified:<list>}."""
GRAPH_PROMPT = """Generate a bounded operation graph `graph` for the frozen task; do not modify task conditions or claim execution success.
Prefer the real interfaces' local frames, and check approach, motion, retreat, the other arm and carried objects. Change base docking pose and lift height when needed.
Revisions after failure address the evidence only; an inconclusive search result is not a proof of unreachability. Return {graph:<operation graph>}, or {deferred:{reason:<reason>}} if you cannot make progress.
Allowing contact is only a plan declaration; it does not bypass the penetration and contact-force checks."""


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
        need(len(encoded(messages))+512<=generation.max_request_bytes,"Offline request exceeds the actual byte limit")
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
    need(root==request_path.parent.parent/"request_context"/request["request_id"],"Request material root mismatch")
    need(query.get("reference") in request["allowed_references"],"Detail reference does not belong to the request")
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
            raise UncertainRequest("An unknown request exists; all reservations are kept and nothing is resent automatically")
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
            raise AwaitingResponse("Exported an in-budget request; waiting for an explicitly imported response: "+str(path))
        response=read(response_path)
        need(response.get("request_id")==key,"Response belongs to a different request")
        for batch in request["batches"]:
            need(digest(read(safe_path(self.store.root/"requests",batch["file"])))==batch["sha256"],"Actual request batch was modified")
        need(response.get("delivered_batches")==list(range(len(request["batches"]))),"Response does not declare that all material/image batches were processed")
        source=response.get("provenance")
        need(source in {"developer_fixture","model_generated"},"An imported response must declare its provenance")
        need(source!="developer_fixture" or self.store.state["identity"].get("provenance")=="developer_fixture","Formal joint runs and their sub-workflows reject developer fixture responses")
        call={"provenance":source,"request_id":key,"response_sha256":digest(response),"transport":"externally_supplied_response"}
        if source=="model_generated":
            provenance=response.get("model_provenance",{})
            need(all(isinstance(provenance.get(k),str) and provenance[k] for k in ("provider","requested_model","response_model")),"External model response is missing provenance")
            configured=self.store.state["identity"].get("configuration",{}).get("provider")
            if configured:
                need(provenance["provider"]==configured["provider"] and provenance["requested_model"]==configured["model"],"External call used a different provider or model than the joint run selected")
            call.update(provenance)
            accounted=self.store.state.setdefault("imported_responses",{})
            if key not in accounted:
                # Each delivered external request consumes a call; importing again never refunds it.
                accounted[key]={"response_sha256":digest(response),"calls":[]}; self.store.save()
            entry=accounted[key]
            need(entry["response_sha256"]==digest(response),"An accounted response must not be replaced")
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
    need(isinstance(key,str) and len(key)==64 and all(c in "0123456789abcdef" for c in key),"Invalid request id")
    request=read(safe_path(root,"requests/"+key+".json"))
    need(response.get("delivered_batches")==list(range(len(request["batches"]))),"All material batches must be acknowledged")
    path=safe_path(root,"imports/"+key+".json")
    if path.exists(): need(digest(read(path))==digest(response),"An imported response cannot be overwritten")
    else: atomic_json(path,response)
    return {"status":"imported","request_id":key,"network_calls":0}
