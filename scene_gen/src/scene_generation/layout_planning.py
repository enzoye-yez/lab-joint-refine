"""One bounded layout refinement before fixtures use actual asset geometry."""
from asset_generation.contracts import require
from asset_prep.errors import ValidationError, ContextLimitError
from .contracts import repair_action
from .prompts import REPAIR, for_environment
from .candidate import usable
from asset_prep.storage import digest
from .layout import table_policy
from .edge_mounts import POLICY as EDGE_POLICY


def refine(run):
    plan = run.state["plan"]
    if "layout" not in plan:
        return
    assets = [j for j in plan["jobs"] if j["role"]=="asset"]
    ready = [j for j in assets if usable(run,run.state["jobs"].get(j["id"]))]
    if not ready:
        return
    # Do not spend a new refinement allocation on every repair/restart.
    record = run.state.setdefault("layout_refinement", {"attempts":[], "complete":False})
    input_key = digest({j["id"]:run.state["jobs"][j["id"]]["assembly_readiness"] for j in ready})
    if record.get("input_key") != input_key:
        record.update(input_key=input_key, complete=False)
    if record["complete"]:
        return
    attempts = [a for a in record["attempts"] if a.get("input_key")==input_key]
    allowed = [i["id"] for i in plan["instances"] if i["placement"]!="robot"]
    from .layout_metrics import measure
    context = {"layout_metrics":measure(plan,{}),"plan":plan,"environment":run.env,"allowed_jobs":[],"allowed_instances":allowed,
               "robot_joint_contract":run.robot_joints,
               "table_edit_policy":table_policy(run.env),"edge_mount_policy":EDGE_POLICY,
               "assets":{j["id"]:{"manifest":run.state["jobs"][j["id"]].get("manifest"),
                                 "geometry":run.state["jobs"][j["id"]].get("geometry_summary")}
                         for j in ready},
               "missing_assets":[{"job":j["id"],"requirements":j.get("requirements","")} for j in assets if j not in ready],
               "instruction":"使用已可加载资产的真实几何完善桌子及摆放，保持仪器规格。缺件保持身份和尺寸假设，不生成替身；已存在夹具时不能改变其安装关系。"
                             "本步不改资产脚本。无需调整时返回且仅返回{\"keep_layout\":\"具体理由\"}。"
                             "本步不新增工具任务或改变末端安装关系；正式场景审查后才可按证据换装。"
                             "若给出修复，仍须包含jobs:[]、placements和reason。"}
    for _ in range(len(attempts), min(3,run.limits.max_repair_plan_attempts)):
        run.check_pause()
        # Reserve locally before asking; interruption never refills an attempt.
        attempt = {"status":"started", "input_key":input_key}
        record["attempts"].append(attempt)
        run.store.save()
        try:
            value,call = run.model.ask(for_environment(REPAIR,run.env),
                {**context,"validation_feedback":[a for a in record["attempts"] if a.get("input_key")==input_key][-3:]},[],"scene_layout")
            attempt.update(call=call,response=value)
            if isinstance(value,dict) and set(value)=={"keep_layout"}:
                require(isinstance(value["keep_layout"],str) and value["keep_layout"].strip(),
                        "保留布局须给出理由")
            else:
                require(isinstance(value,dict) and value.get("jobs")==[] and not value.get("tool_additions")
                        and value.get("end_effectors",plan.get("end_effectors"))==plan.get("end_effectors"),
                        "布局细化不新增任务或换装末端")
                updated = repair_action(value,plan,[],allowed,run.limits,run.env)
                from .robot_contract import validate_model_posture
                validate_model_posture(updated,run.robot_joints)
                run.state["plan"] = updated
            attempt["status"]="accepted"
            record["complete"]=True
            run.store.save()
            return
        except (ValidationError,ContextLimitError) as exc:
            attempt.update(status="deferred",error=str(exc))
            run.store.save()
    record.update(complete=True,detail="布局细化未得到合法修改，继续原候选；后续场景审查仍可调整")
    run.store.save()
