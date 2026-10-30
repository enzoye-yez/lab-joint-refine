"""Bounded, deterministic review projections with immutable, paginated full evidence.

The projection is a presentation, never an acceptance report. Omitted data remains
addressable by a hash scoped to this artifact/requirements/implementation binding.
"""
from pathlib import Path
from collections import Counter
from asset_prep.storage import encoded, digest, atomic_json
from asset_generation.contracts import require, read_json
from asset_prep.errors import ContextLimitError

POLICY = """上下文中的 evidence_ref 是完整原文的只读引用，不是已经审查通过的证据。
省略内容明确标注 omitted；需要这些细节才能判断时保持 uncertain，并在补证规划中用
{"kind":"details","reference":"evidence_ref的值","offset":0,"limit":16}
分页获取。支持 contains 文本筛选，返回总数与 next_offset。不得把摘要或缺少细节视作通过。
"""

class ContextArchive:
    def __init__(self, root, binding, budget=120000):
        self.root=Path(root)/"context_evidence"/binding
        self.budget=budget

    def reference(self, value):
        key=digest(value)
        path=self.root/(key+".json")
        if not path.exists():atomic_json(path,value)
        return key

    def summary(self,value):
        row={"evidence_ref":self.reference(value),"omitted":True,
             "type":type(value).__name__,"count":len(value) if hasattr(value,"__len__") else 1}
        if isinstance(value,dict):
            for key in ("id","status","complete","instances","error_kind","component_scope","tolerance_m",
                        "measurement_error_count","unexamined_pairs","collision_count","disconnected_count"):
                if key in value and len(encoded(value[key]))<1024:row[key]=value[key]
            row["fields"]=list(value)[:32]
        elif isinstance(value,list) and all(isinstance(v,dict) and "status" in v for v in value):
            row["status_counts"]=dict(Counter(v["status"] for v in value))
        return row

    def fold(self,value,budget):
        if len(encoded(value))<=budget:return value
        if isinstance(value,dict) and len(value)<=64:
            weights={k:16 if k in {"checks","current_checks"} else 4 if k in {"plan","fixed_contract","protocol","input","source_requirements"} else 1 for k in value}
            unit=max(1,(budget-512)//max(1,sum(weights.values())))
            result={k:self.fold(v,max(200,unit*weights[k])) for k,v in value.items()}
        elif isinstance(value,list) and len(value)<=256:
            share=max(200,(budget-128)//max(1,len(value)))
            result=[self.fold(v,share) for v in value]
        else:result=self.summary(value)
        if len(encoded(result))>budget:result=self.summary(value)
        return result

    def project(self,value):
        # A bounded top-level shape is retained; full data is never written over.
        report=value.get("checks",value.get("current_checks",{})) if isinstance(value,dict) else {}
        rows=report.get("checks",[]) if isinstance(report,dict) else []
        index=[{**{k:r[k] for k in ("id","status","complete","instances","measurement_error_count","unexamined_pairs") if k in r},
                "evidence_ref":self.reference(r)} for r in rows if isinstance(r,dict) and "id" in r]
        if len(encoded(index))>self.budget//2:index=self.summary(index)
        result=self.fold(value,self.budget-2048-len(encoded(index)))
        if isinstance(result,dict):
            result={**result,"evidence_access":POLICY}
            if rows:result["check_index"]=index
        if len(encoded(result))>self.budget:
            raise ContextLimitError("审查摘要仍超过上下文投影预算；未发送请求")
        return result

    @staticmethod
    def validate(request):
        import re
        require(set(request)<={"kind","reference","offset","limit","contains","question"},"invalid evidence detail request")
        require(isinstance(request.get("reference"),str) and re.fullmatch(r"[0-9a-f]{64}",request["reference"]),"invalid evidence reference")
        require(type(request.get("offset",0)) is int and request.get("offset",0)>=0,"invalid detail offset")
        require(type(request.get("limit",16)) is int and 1<=request.get("limit",16)<=32,"invalid detail limit")
        require(isinstance(request.get("contains",""),str) and len(request.get("contains",""))<=200,"invalid detail filter")

    def inspect(self,request):
        self.validate(request)
        path=self.root/(request["reference"]+".json")
        require(path.is_file(),"reference is outside current evidence binding")
        value=read_json(path,128000000)
        require(digest(value)==request["reference"],"archived evidence changed")
        if isinstance(value,dict):items=[{"field":k,"value":v} for k,v in value.items()]
        elif isinstance(value,list):items=value
        elif isinstance(value,str):items=[value[i:i+2048] for i in range(0,len(value),2048)]
        else:items=[value]
        needle=request.get("contains","")
        if needle:items=[v for v in items if needle in encoded(v).decode("utf-8")]
        start=request.get("offset",0);end=min(len(items),start+request.get("limit",16))
        return {"kind":"details","request":request,"total":len(items),"offset":start,
                "next_offset":end if end<len(items) else None,"complete":True,
                "data":[self.fold(v,4096) for v in items[start:end]]}

class ProjectedModel:
    def __init__(self,model,archive):self.model,self.archive=model,archive
    def ask(self,instruction,context,images,phase):
        if getattr(self.model, "manages_context", False):
            return self.model.ask(instruction,context,images,phase)
        return self.model.ask(instruction,self.archive.project(context),images,phase)
