"""Durable shared budgets and evidence-based candidate selection."""
from copy import deepcopy
from asset_prep.errors import BudgetExceeded
from asset_prep.storage import digest

LIMITS={"model_calls":800,"reserved_tokens":240000000,"rounds":80,"asset_candidates":60}

class Budget:
    def __init__(self,store):
        self.store=store
        self.state=store.state.setdefault("joint_budget",{"rounds":[],"assets":{}})
    def round(self,key,kind):
        rows=self.state["rounds"]
        if any(r["key"]==key for r in rows):return
        if len(rows)>=LIMITS["rounds"]:raise BudgetExceeded("Joint-repair budget of 80 rounds exhausted")
        rows.append({"key":key,"kind":kind});self.store.save()
    def asset(self,job,stage):
        rows=self.state["assets"].setdefault(job,[])
        if stage in rows:return
        if len(rows)>=LIMITS["asset_candidates"]:raise BudgetExceeded("Asset-candidate budget of 60 exhausted: "+job)
        rows.append(stage);self.store.save()
    def remaining(self):
        totals=self.store.totals()
        return {"calls":max(0,LIMITS["model_calls"]-totals["model_calls"]),
                "tokens":max(0,LIMITS["reserved_tokens"]-totals["accounted_tokens"]),
                "rounds":max(0,LIMITS["rounds"]-len(self.state["rounds"])),
                "assets":{k:max(0,60-len(v)) for k,v in self.state["assets"].items()}}

def facts(report):
    result=set()
    for row in report.get("rows",[]):
        if row.get("repair_authorized") or (row.get("kind")=="collision" and row.get("status")=="confirmed_observation"):
            collisions=[c for sample in row.get("samples",[]) for c in sample.get("collisions",[])]
            if collisions:
                for c in collisions:result.add(digest({"kind":row["kind"],"interface":row.get("interface"),"geoms":sorted(c["geoms"])}))
            else:result.add(digest({k:row[k] for k in ("kind","interface","geoms","instances") if k in row}))
    return result

def compare(baseline,candidate):
    if baseline is None:return "baseline"
    old,new=facts(baseline["diagnosis"]),facts(candidate["diagnosis"])
    old_done,new_done=set(baseline["result"].get("completed_stages",[])),set(candidate["result"].get("completed_stages",[]))
    old_gaps,new_gaps=set(baseline["diagnosis"].get("coverage_gaps",[])),set(candidate["diagnosis"].get("coverage_gaps",[]))
    from scene_generation.progress import repair_comparison
    static=repair_comparison(baseline.get("static_report",{}),candidate.get("static_report",{}))
    safe=(new<=old and new_gaps<=old_gaps and new_done>=old_done
          and candidate["diagnosis"].get("repair_verification",{}).get("coverage_complete",True)
          and not candidate["diagnosis"].get("repair_verification",{}).get("regressions",[])
          and not static["new_defects"] and not static["coverage_lost"])
    improved=static["adopt"] or new<old or new_done>old_done or new_gaps<old_gaps or (candidate["result"]["status"]=="success" and baseline["result"]["status"]!="success")
    if safe and improved:return "improved_partial" if candidate["result"]["status"]!="success" else "improved"
    if not safe:return "exploration"
    return "ineffective"

def mechanism(value):
    # Collapse numeric tuning, retaining target identities, operation order and field categories.
    if isinstance(value,dict):return {k:mechanism(v) for k,v in value.items() if k not in {"model_provenance","scene_sha256","task_sha256"}}
    if isinstance(value,list):return [mechanism(v) for v in value]
    return "number" if type(value) in (int,float) else value

def duplicate(history,value,input_hash,evidence_hash):
    key=digest(mechanism(value))
    rows=[h for h in history if h.get("input_sha256")==input_hash and h.get("evidence_sha256")==evidence_hash and h.get("outcome") in {"ineffective","exploration"}]
    return any(h.get("action_sha256")==digest(value) for h in rows) or sum(h.get("mechanism")==key for h in rows)>=2
