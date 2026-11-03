"""Rebind identifiers only, keeping the independently frozen task requirements unchanged."""
from copy import deepcopy
from .common import need,digest

REFERENCES={"joint":"joint","site_a":"site","site_b":"site","tcp":"site","body":"body","body_a":"body","body_b":"body","support":"body"}

def resolve_task(task,original_manifest,new_manifest,model):
    old={i["id"]:i for i in original_manifest["interfaces"]}
    new={i["id"]:i for i in new_manifest["interfaces"]}
    aliases={}
    for interface in new.values():
        for previous in interface.get("replaces",[]):
            need(previous in old and previous not in new and previous not in aliases,"Interface replacement mapping is unknown, conflicting or duplicated")
            aliases[previous]=interface["id"]
    maps={"body":{},"site":{},"joint":{},"interface":aliases}
    for name,before in old.items():
        after=new.get(aliases.get(name,name))
        if after is None:continue
        need(before["kind"]==after["kind"] and before.get("asset_id")==after.get("asset_id"),"Semantic interface replacement changed the function or asset identity")
        for key in ("body","site"):
            if before.get(key) and after.get(key):
                previous,target=before[key],after[key]
                need(maps[key].get(previous,target)==target,"One old part maps to several new parts; cannot rebind safely")
                maps[key][previous]=target
        a,b=before.get("operation",{}),after.get("operation",{})
        if a.get("joint"):
            need(b.get("joint"),"Repair removed the original interface joint")
            maps["joint"][a["joint"]]=b["joint"]
        if a.get("range"):
            need(b.get("range") and b["range"][0]<=a["range"][0] and b["range"][1]>=a["range"][1],"Repair shortened the required functional travel")
        if a.get("clearance_m") is not None and a.get("basis",{}).get("clearance_m",{}).get("kind")=="source_specification":
            need(b.get("clearance_m",-1)>=a["clearance_m"],"Repair reduced the known fit clearance")
    def reference(kind,name):
        target=maps[kind].get(name,name)
        try:getattr(model,kind)(target)
        except KeyError:need(False,"Task binding lost and no explicit semantic mapping: "+str(name),"binding_missing")
        return target
    def visit(value):
        if isinstance(value,list):return [visit(x) for x in value]
        if not isinstance(value,dict):return value
        out={}
        for key,item in value.items():
            if key in REFERENCES and isinstance(item,str):out[key]=reference(REFERENCES[key],item)
            elif key=="bindings":out[key]={k:reference("body",v) for k,v in item.items()}
            elif key=="interface" and isinstance(item,str):
                target=aliases.get(item,item);need(target in new,"Device interface binding missing");out[key]=target
            else:out[key]=visit(item)
        return out
    bound=visit(deepcopy(task))
    # Check detector ranges against actual mechanisms; mapped names cannot hide shortened travel.
    def ranges(value):
        if isinstance(value,list):
            for row in value:ranges(row)
        elif isinstance(value,dict):
            if value.get("kind")=="joint_range":
                joint=model.joint(value["joint"]);lo,hi=value["range"]
                need(not joint.limited[0] or max(lo,joint.range[0])<=min(hi,joint.range[1]),"Task requirement does not overlap the new mechanism's travel")
            for child in value.values():ranges(child)
    ranges(bound)
    return bound,{"frozen_task_sha256":digest(task),"bound_task_sha256":digest(bound),"mappings":maps,
                  "requirements_changed":False,"validation":"references_and_declared_functional_ranges_only"}
