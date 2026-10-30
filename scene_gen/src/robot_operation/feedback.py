"""独立复核后才导出场景反馈；所有模型修订均显式导入且有谱系预算。"""
from pathlib import Path
from dataclasses import asdict
import mujoco
from asset_generation.checks import load_scene_model
from asset_prep.storage import file_hash
from scene_generation.feedback import planning_feedback
from .common import Settings, atomic_json, digest, identity, need, new_directory, read
from .scene import initial
from .contracts import validate_graph


def source_collisions(operation_scene, expected_hash, settings=None):
    settings=settings or Settings()
    need(identity(operation_scene)==expected_hash,"操作场景版本过期","stale_scene")
    metadata=read(Path(operation_scene)/"scene.json")
    conversion=metadata["operation_conversion"]
    source=Path(conversion["source"])
    need(identity(source)==conversion["source_sha256"],"来源场景版本已变化","stale_scene")
    model=load_scene_model(source,dynamics=True)
    original=read(source/"scene.json"); data=initial(model,original)
    instances={i["id"] for i in original["plan"]["instances"]}
    def owner(geom):
        body=int(model.geom_bodyid[geom])
        while body:
            name=model.body(body).name
            match=next((i for i in instances if name.startswith(i+"__")),None)
            if match: return match
            body=int(model.body_parentid[body])
        return None
    failures=[]
    for index,contact in enumerate(data.contact):
        if contact.dist >= -settings.penetration_m: continue
        involved=sorted({v for v in (owner(contact.geom1),owner(contact.geom2)) if v})
        if not involved: continue
        evidence={"checker":"source_initial_penetration_v1","contact":index,
            "geom_indices":[int(contact.geom1),int(contact.geom2)],"distance_m":float(contact.dist),
            "threshold_m":settings.penetration_m,"source_sha256":conversion["source_sha256"],
            "operation_scene_sha256":expected_hash}
        failures.append({"kind":"collision","instances":involved,"detail":"固定检查器在来源场景初态复核到穿透超出公开工程阈值",
                         "evidence":str(evidence)})
    return conversion, original, failures


def verify_source_collisions(operation_scene, expected_hash, output, settings=None):
    conversion,original,failures=source_collisions(operation_scene,expected_hash,settings)
    out=new_directory(output,[operation_scene,conversion["source"]])
    if not failures:
        result={"status":"unconfirmed","scene_repair_authorized":False,"reason":"该固定检查器未确认初态几何缺陷；不排除其他缺陷"}
    else:
        feedback={"scene_sha256":conversion["source_sha256"],"failures":failures[:100]}
        planning_feedback(feedback,conversion["source_sha256"],original["plan"])
        atomic_json(out/"scene_feedback.json",feedback)
        result={"status":"verified_defect","scene_repair_authorized":True,"feedback":"scene_feedback.json",
            "network_calls":0,"next_action":"交给已有场景反馈入口；无模型接口时保留请求"}
    result.update(operation_scene_sha256=expected_hash,source_sha256=conversion["source_sha256"],
                  settings=asdict(settings or Settings()))
    atomic_json(out/"result.json",result)
    return result


def import_response(request_path,response_path,output,settings=None):
    settings=settings or Settings()
    request,response=read(request_path),read(response_path)
    need(request["kind"]=="compile_task_and_operation_graph","请求类型不匹配")
    bundle=request["bundle"]; task,graph=response["task"],response["graph"]
    need(task["input_sha256"]==bundle["input_sha256"] and task["public_profile_sha256"]==bundle["public_profile_sha256"],"响应改变了任务输入或公共环境版本")
    validate_graph(graph,task,bundle["scene_sha256"],settings)
    if graph["provenance"]=="model_generated":
        provenance=response.get("model_provenance",{})
        for key in ("provider","requested_model","response_model"):
            need(isinstance(provenance.get(key),str) and provenance[key],"导入响应缺少模型来源："+key)
        graph["model_provenance"]={**provenance,"request_sha256":file_hash(request_path),"response_sha256":file_hash(response_path),
                                   "transport":"externally_supplied_response_not_called_by_operation_module"}
    out=new_directory(output)
    atomic_json(out/"task.json",task); atomic_json(out/"graph.json",graph)
    atomic_json(out/"request.json",request); atomic_json(out/"response.json",response)
    return {"status":"accepted","task":str(out/"task.json"),"graph":str(out/"graph.json"),"network_calls":0}


def import_revision(previous_run,graph_path,output,new_scene=None,new_scene_hash=None,defect_report=None):
    root=Path(previous_run); record=read(root/"recording.json")
    old_graph,task=read(root/"graph.json"),read(root/"task.json")
    graph=read(graph_path); settings=Settings(**record["settings"])
    need(read(root/"result.json")["status"]!="success","已完成结果不接受失败修订")
    lineage=old_graph.get("revision_lineage",{"graph_revisions":0,"scene_repairs":0})
    lineage={"graph_revisions":int(lineage["graph_revisions"]),"scene_repairs":int(lineage["scene_repairs"])}
    target_hash=record["scene_sha256"]
    if new_scene is not None:
        need(new_scene_hash and identity(new_scene)==new_scene_hash,"新场景摘要不匹配")
        need(defect_report is not None,"场景修复缺少独立复核记录")
        report=read(defect_report)
        need(report.get("scene_repair_authorized") is True and report.get("status")=="verified_defect","缺陷未经复核")
        need(report.get("operation_scene_sha256")==record["scene_sha256"],"缺陷证据来自其他场景")
        conversion,_,failures=source_collisions(record["scene_path"],record["scene_sha256"],Settings(**report["settings"]))
        need(bool(failures) and conversion["source_sha256"]==report["source_sha256"],"固定检查器不能重复确认缺陷")
        need(lineage["scene_repairs"]<settings.scene_repairs,"场景修复预算耗尽","budget_exhausted")
        lineage["scene_repairs"]+=1; target_hash=new_scene_hash
    else:
        need(lineage["graph_revisions"]<settings.graph_revisions,"操作图修订预算耗尽","budget_exhausted")
        lineage["graph_revisions"]+=1
    validate_graph(graph,task,target_hash,settings)
    graph["revision_lineage"]={**lineage,"previous_graph_sha256":digest(old_graph),"previous_run":str(root.resolve()),
                               "reset_to_initial_required":True}
    out=new_directory(output,[root])
    atomic_json(out/"task.json",task); atomic_json(out/"graph.json",graph)
    atomic_json(out/"revision.json",graph["revision_lineage"])
    return {"status":"accepted","graph":str(out/"graph.json"),"task":str(out/"task.json"),
            "scene_sha256":target_hash,"lineage":graph["revision_lineage"],"network_calls":0}
