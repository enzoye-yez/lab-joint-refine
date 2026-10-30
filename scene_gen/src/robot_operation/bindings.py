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
            need(previous in old and previous not in new and previous not in aliases,"接口替换映射未知、冲突或重复")
            aliases[previous]=interface["id"]
    maps={"body":{},"site":{},"joint":{},"interface":aliases}
    for name,before in old.items():
        after=new.get(aliases.get(name,name))
        if after is None:continue
        need(before["kind"]==after["kind"] and before.get("asset_id")==after.get("asset_id"),"语义接口替换改变了功能或资产身份")
        for key in ("body","site"):
            if before.get(key) and after.get(key):
                previous,target=before[key],after[key]
                need(maps[key].get(previous,target)==target,"一个旧部件映射到多个新部件，无法安全重绑定")
                maps[key][previous]=target
        a,b=before.get("operation",{}),after.get("operation",{})
        if a.get("joint"):
            need(b.get("joint"),"修复移除了原接口关节")
            maps["joint"][a["joint"]]=b["joint"]
        if a.get("range"):
            need(b.get("range") and b["range"][0]<=a["range"][0] and b["range"][1]>=a["range"][1],"修复缩短了必要功能行程")
        if a.get("clearance_m") is not None and a.get("basis",{}).get("clearance_m",{}).get("kind")=="source_specification":
            need(b.get("clearance_m",-1)>=a["clearance_m"],"修复降低了已知配合间隙")
    def reference(kind,name):
        target=maps[kind].get(name,name)
        try:getattr(model,kind)(target)
        except KeyError:need(False,"任务绑定丢失且没有明确语义映射："+str(name),"binding_missing")
        return target
    def visit(value):
        if isinstance(value,list):return [visit(x) for x in value]
        if not isinstance(value,dict):return value
        out={}
        for key,item in value.items():
            if key in REFERENCES and isinstance(item,str):out[key]=reference(REFERENCES[key],item)
            elif key=="bindings":out[key]={k:reference("body",v) for k,v in item.items()}
            elif key=="interface" and isinstance(item,str):
                target=aliases.get(item,item);need(target in new,"设备接口绑定缺失");out[key]=target
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
                need(not joint.limited[0] or max(lo,joint.range[0])<=min(hi,joint.range[1]),"任务要求与新机构行程无交集")
            for child in value.values():ranges(child)
    ranges(bound)
    return bound,{"frozen_task_sha256":digest(task),"bound_task_sha256":digest(bound),"mappings":maps,
                  "requirements_changed":False,"validation":"references_and_declared_functional_ranges_only"}
