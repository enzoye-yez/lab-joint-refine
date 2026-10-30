"""小型场景交付约定。来源要求、机器人及验收条件不属于修复工作区。"""

import copy
import re
from dataclasses import dataclass, asdict

from asset_generation.contracts import finite, require, check_profile
from asset_prep.storage import digest
from .table import validate_cuts
from .edge_mounts import validate_mounts


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value), "无效对象编号")
    return value


def vector(value, n=3):
    require(isinstance(value, list) and len(value) == n and all(finite(x) for x in value), "无效有限数值向量")
    return value


@dataclass(frozen=True)
class SceneConfig:
    validation_mode: str = "full"
    max_iterations: int = 8
    max_plan_attempts: int = 5
    max_repair_plan_attempts: int = 5
    max_evidence_rounds: int = 6
    max_evidence_plan_attempts: int = 5
    max_check_actions: int = 256
    max_check_retries: int = 2
    max_model_calls: int = 120
    max_reserved_tokens: int = 12000000
    max_jobs: int = 64
    max_instances: int = 128
    max_repairs_per_round: int = 3
    max_artifact_bytes: int = 256000000
    max_artifact_files: int = 2048
    build_timeout_seconds: int = 300
    collision_tolerance_m: float = 0.0005
    support_tolerance_m: float = 0.003
    settle_steps: int = 150
    ik_iterations: int = 160
    ik_restarts: int = 4
    # Author requests of independent assets sent concurrently; throughput only, not identity.
    parallel_authors: int = 1

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k == "validation_mode":
                require(isinstance(v,str) and v in {"full", "static"}, "验证模式必须为 full（完整）或 static（静态）")
            elif k == 'max_check_retries':
                require(type(v) is int and v >= 0, '恢复工具重试次数必须为非负整数')
            elif k.endswith("_m"):
                require(finite(v) and 0 <= v <= .02, "无效几何公差")
            else:
                require(type(v) is int and v > 0, "场景限制必须为正整数")
        require(self.settle_steps <= 5000 and self.ik_iterations <= 1000 and self.ik_restarts <= 32,
                "场景检查预算过大")
        require(self.parallel_authors <= 8, "并行作者数不超过 8")

    def identity(self):
        return {k: v for k, v in asdict(self).items()
                if k not in {"parallel_authors", "max_iterations", "max_model_calls", "max_reserved_tokens", "max_plan_attempts", "max_repair_plan_attempts", "max_evidence_rounds", "max_evidence_plan_attempts", "max_check_actions", "max_check_retries"}}


def environment(value):
    require(isinstance(value, dict) and set(value) == {"table", "robot"}, "场景环境须声明参考工作台与机器人规格")
    table, robot = value["table"], value["robot"]
    require(isinstance(table,dict) and set(table) == {"size", "height"}, "桌面字段无效")
    require(all(x > 0 for x in vector(table["size"])) and finite(table["height"])
            and table["height"] > table["size"][2], "桌面尺寸无效")
    from .robot_contract import validate_robot
    validate_robot(robot)
    if "mobile" in robot:
        from .layout import validate_mobile
        validate_mobile(robot["mobile"])
    return copy.deepcopy(value)


def validate_plan(value, asset_ids, limits, env=None):
    require(isinstance(value, dict) and set(value)-{"table_modifications","edge_mounts","layout","end_effectors","robot_posture","role_bindings"} == {"jobs", "instances", "assumptions"}, "场景计划字段无效")
    require(isinstance(value["jobs"], list) and 1 <= len(value["jobs"]) <= limits.max_jobs, "生成任务数超限")
    require(isinstance(value["instances"], list) and 1 <= len(value["instances"]) <= limits.max_instances, "场景实例数超限")
    jobs, covered = {}, []
    for job in value["jobs"]:
        require(isinstance(job, dict) and set(job)-{"configurations"} == {"id", "asset_ids", "role", "depends_on", "requirements"}, "任务字段无效")
        identifier(job["id"])
        require(job["id"] not in jobs and isinstance(job['role'],str) and job["role"] in {"asset", "tool", "fixture"}, "任务重复或类型无效")
        require(isinstance(job["asset_ids"], list) and job["asset_ids"] and all(isinstance(a,str) for a in job['asset_ids'])
                and len(set(job["asset_ids"])) == len(job["asset_ids"]), "任务资产编号无效")
        require(all(isinstance(x, str) and x in asset_ids for x in job["asset_ids"]), "任务引用未知资产")
        if job["role"] == "asset":
            covered.extend(job["asset_ids"])
        require(isinstance(job["depends_on"], list) and all(isinstance(d,str) for d in job['depends_on'])
                and len(set(job["depends_on"])) == len(job["depends_on"]), "依赖无效")
        require(isinstance(job["requirements"], str) and len(job["requirements"]) <= 24000, "任务说明过长")
        if "configurations" in job:
            variants = job["configurations"]
            require(isinstance(variants,list) and 1 <= len(variants) <= 16 and all(isinstance(n,str) for n in variants)
                    and len(set(variants))==len(variants), "invalid job configurations")
            for name in variants: identifier(name)
        jobs[job["id"]] = job
    require(sorted(covered) == sorted(asset_ids), "资料包每项资产须恰好分配到一个资产生成任务")
    multi = env is not None and "arms" in env["robot"]
    if not multi:
        require(sum(j["role"] == "tool" for j in jobs.values()) == 1, "须设计一套统一末端装置")
    else:
        require("end_effectors" in value, "计划须声明每臂末端配置")
    if not multi:
        require("end_effectors" not in value, "旧环境使用原有单臂末端约定")
    from .robot_contract import validate_posture
    validate_posture(value.get("robot_posture",{}),env)
    order = []
    while len(order) < len(jobs):
        ready = [k for k, j in jobs.items() if k not in order and all(x in order for x in j["depends_on"])]
        require(ready, "依赖不存在或形成循环")
        order.extend(ready)
    for job in jobs.values():
        if job["role"] != "asset":
            require(all(k in job["depends_on"] for k, j in jobs.items() if j["role"] == "asset"),
                    "操作装置须依赖所有资产，避免只看单个仪器")
        if job["role"] == "tool":
            require(all(k in job["depends_on"] for k,j in jobs.items() if j["role"] == "fixture"),
                    "末端装置须读取已生成的桌面固定装置，保持协同设计一致")
    instances = {}
    for item in value["instances"]:
        require(isinstance(item, dict) and set(item) - {"support_instance","configuration","table_id"} == {"id", "job", "placement", "position", "yaw"}, "实例字段无效")
        identifier(item["id"])
        require(item["id"] not in instances and isinstance(item['job'],str) and item["job"] in jobs, "实例重复或引用无效")
        require(isinstance(item['placement'],str) and item["placement"] in {"table", "floor", "robot"}, "摆放面无效")
        require('support_instance' not in item or isinstance(item['support_instance'],str) and bool(item['support_instance']), '支撑实例编号无效')
        require(all(abs(x) <= (100 if "layout" in value else 10) for x in vector(item["position"])) and finite(item["yaw"]), "实例姿态无效")
        require((jobs[item["job"]]["role"] == "tool") == (item["placement"] == "robot"), "末端装置必须安装在机械臂上")
        if item["placement"] == "robot":
            require(item["position"] == [0, 0, 0] and item["yaw"] == 0, "工具安装原点固定于机器人法兰")
        variants = jobs[item["job"]].get("configurations")
        require((variants is not None and item.get("configuration") in variants) or
                (variants is None and "configuration" not in item), "instance configuration must match its planned job")
        instances[item["id"]] = item
    require(set(jobs) == {i["job"] for i in instances.values()}, "存在没有实例的生成任务")
    if not multi:
        require(sum(i["placement"] == "robot" for i in instances.values()) == 1, "只允许一套装在机械臂上的装置")
    else:
        from .robot_contract import validate_bindings
        validate_bindings(value,env)
        require(sum(j["role"]=="tool" for j in jobs.values()) <= len(env["robot"]["arms"]), "每臂最多一套当前定制总成")
    for item in instances.values():
        seen = {item["id"]}
        parent = item.get("support_instance")
        require(not parent or item["placement"] != "robot", "机器人安装工具不能额外声明支撑实例")
        while parent:
            require(isinstance(parent,str) and parent in instances and parent not in seen, "支撑引用缺失或成环")
            require(instances[parent]["placement"] != "robot", "初始支撑不能指向机械臂末端工具")
            seen.add(parent)
            parent = instances[parent].get("support_instance")
    require(isinstance(value["assumptions"], list) and all(isinstance(x, str) for x in value["assumptions"]), "布局假设无效")
    from .layout import validate_layout
    validate_layout(value)
    validate_cuts(value.get("table_modifications", []), value)
    validate_mounts(value.get('edge_mounts',[]),value)
    return copy.deepcopy(value), order


def numerical_contract(value):
    """由用户固定的验收；模型不能改写。缺项在报告中显式保留。"""
    require(isinstance(value, dict) and not set(value) - {"jobs", "fits", "targets", "payload_mass_kg", "arm_payloads"}, "验收字段无效")
    result = {"jobs": {}, "fits": [], "targets": [], "payload_mass_kg": 0, **value}
    require(isinstance(result["jobs"], dict), "任务验收须为映射")
    result["jobs"] = {identifier(k): check_profile(v) for k, v in result["jobs"].items()}
    require(finite(result["payload_mass_kg"]) and result["payload_mass_kg"] >= 0, "工作负载无效")
    require(isinstance(result.get("arm_payloads",{}),dict) and
            all(isinstance(k,str) and finite(v) and v>=0 for k,v in result.get("arm_payloads",{}).items()), "逐臂工作负载无效")
    for key in ("fits", "targets"):
        require(isinstance(result[key], list) and len(result[key]) <= 256, "场景验收条目过多")
    for fit in result["fits"]:
        require(isinstance(fit,dict) and set(fit) == {"id", "outer", "inner", "axis", "clearance_m", "evidence"}, "配合检查字段无效")
        identifier(fit["id"])
        require(type(fit["axis"]) is int and fit["axis"] in (0, 1, 2), "配合轴无效")
        vector(fit["clearance_m"], 2)
        require(0 <= fit["clearance_m"][0] <= fit["clearance_m"][1], "配合间隙无效")
        for end in ("outer", "inner"):
            require(isinstance(fit[end], list) and len(fit[end]) == 2 and all(isinstance(x, str) and x for x in fit[end]), "配合尺寸须由两个实际site标记")
        require(isinstance(fit["evidence"], str) and fit["evidence"], "配合要求须记录来源")
    for target in result["targets"]:
        require(isinstance(target,dict) and set(target) == {"id", "site", "tool_site", "offset", "quaternion", "position_tolerance_m", "angle_tolerance_rad", "allowed_contacts"}, "操作目标字段无效")
        identifier(target["id"])
        require(all(isinstance(target[k], str) and target[k] for k in ("site", "tool_site")), "操作接口名称无效")
        vector(target["offset"]); vector(target["quaternion"], 4)
        require(abs(sum(x*x for x in target["quaternion"])-1) < 1e-5, "目标旋转必须为单位四元数")
        require(all(finite(target[k]) and 0 < target[k] <= .2 for k in ("position_tolerance_m", "angle_tolerance_rad")), "目标容差无效")
        require(isinstance(target["allowed_contacts"], list) and all(isinstance(p, list) and len(p) == 2
                and all(isinstance(g, str) and g for g in p) for p in target["allowed_contacts"]), "允许接触须为精确几何对")
    for key in ('fits','targets'):
        require(len({r['id'] for r in result[key]})==len(result[key]), '验收检查编号不能重复')
    return result


def repair_action(value, plan, allowed_jobs, allowed_instances, limits, env=None):
    require(isinstance(value, dict) and set(value)-{"table_modifications","edge_mounts","geometry_changes","layout","end_effectors","tool_additions","robot_posture","room_revision_reason"} == {"jobs", "placements", "reason"}, "修复指令字段无效")
    require("room_revision_reason" not in value or "layout" in value,"房间修订说明须对应实际布局修改")
    additions=value.get("tool_additions",[])
    require(isinstance(additions,list) and (not additions or env is not None and "arms" in env["robot"]), "新增工具需要多安装点环境")
    new_jobs=[]; new_instances=[]
    for addition in additions:
        require(isinstance(addition,dict) and set(addition)=={"job","instance"}, "新增工具须提供完整任务与实例")
        job,item=addition["job"],addition["instance"]
        require(isinstance(job,dict) and job.get("role")=="tool" and isinstance(job.get("id"),str) and job.get("id") not in {j["id"] for j in plan["jobs"]}
                and isinstance(item,dict) and item.get("job")==job.get("id") and item.get("placement")=="robot", "新增范围仅限专用工具")
        new_jobs.append(job);new_instances.append(item)
    for job,item in zip(new_jobs,new_instances):
        identifier(job.get("id")); identifier(item.get("id"))
    require(isinstance(value.get("jobs"),list) and all(isinstance(j,str) for j in value["jobs"]), "修复任务编号无效")
    added_ids={j["id"] for j in new_jobs}
    require(added_ids <= set(value.get("jobs",[])), "新增工具必须列入本轮修复任务")
    allowed_jobs=set(allowed_jobs)|added_ids
    require(isinstance(value["jobs"], list) and all(isinstance(j,str) for j in value['jobs']) and len(set(value["jobs"])) == len(value["jobs"])
            and set(value["jobs"]) <= set(allowed_jobs), "资产修复越出允许范围")
    require(len(value["jobs"]) <= limits.max_repairs_per_round, "本轮资产修复数量超限")
    if 'geometry_changes' in value:
        require(isinstance(value['geometry_changes'],list) and 0 < len(value['geometry_changes']) <= 64,
                '具体几何修复动作无效')
        for change in value['geometry_changes']:
            require(isinstance(change,dict) and set(change)=={'job','operation','parameters'}
                    and change['job'] in value['jobs'] and isinstance(change['operation'],str) and change['operation']
                    and isinstance(change['parameters'],dict) and change['parameters'], '具体几何修复参数缺失或越界')
    require(isinstance(value["placements"], list) and len(value["placements"]) <= limits.max_instances, "布局修复无效")
    require("layout" not in value or "layout" in plan, "旧版场景不能接受移动布局修改")
    updated = copy.deepcopy(plan)
    updated["jobs"].extend(new_jobs)
    updated["instances"].extend(new_instances)
    if "end_effectors" in value:
        require(env is not None and "arms" in env["robot"], "旧环境不接受新的末端配置")
        from .robot_contract import bindings
        old_bindings=bindings(plan,env)
        item_jobs={i["id"]:i["job"] for i in updated["instances"]}
        require(isinstance(value["end_effectors"],dict), "末端配置须为映射")
        for arm,binding in value["end_effectors"].items():
            if binding!=old_bindings.get(arm):
                changed=[b.get("instance") for b in (binding,old_bindings.get(arm,{})) if isinstance(b,dict) and b.get("instance")]
                require(all(item_jobs.get(i) in allowed_jobs for i in changed), "换装超出允许工具范围")
        updated["end_effectors"]=copy.deepcopy(value["end_effectors"])
    if "robot_posture" in value:
        from .robot_contract import validate_posture
        validate_posture(value["robot_posture"],env)
        updated["robot_posture"]=copy.deepcopy(value["robot_posture"])
    items = {i["id"]: i for i in updated["instances"]}
    seen = set()
    for edit in value["placements"]:
        require(isinstance(edit,dict) and set(edit)-{"table_id","placement","support_instance"} == {"id", "position", "yaw"} and isinstance(edit['id'],str)
                and edit["id"] in allowed_instances and edit["id"] not in seen, "布局修复越界或重复")
        seen.add(edit["id"])
        require(items[edit["id"]]["placement"] != "robot", "不能移动固定工具安装接口")
        vector(edit["position"])
        require(finite(edit["yaw"]) and all(abs(x) <= (100 if "layout" in plan else 10) for x in edit["position"]), "布局修复姿态无效")
        items[edit["id"]].update(position=edit["position"], yaw=edit["yaw"])
        for field in ("placement", "table_id", "support_instance"):
            if field in edit:
                require("layout" in plan, "旧版摆放修复不接受新布局字段")
                if edit[field] is None:
                    items[edit["id"]].pop(field, None)
                else:
                    items[edit["id"]][field] = edit[field]
    if "layout" in plan:
        from .layout import edit_layout, follow_supports, validate_layout
        edit_layout(value, plan, updated, allowed_instances, env)
        follow_supports(plan, updated, seen)
        for item in updated["instances"]:
            old_item = next((i for i in plan["instances"] if i["id"] == item["id"]),item)
            require(item == old_item or item["id"] in allowed_instances, "关联移动越出允许实例范围")
        validate_layout(updated, env)
    if "table_modifications" in value:
        cuts = validate_cuts(value["table_modifications"], updated, env)
        old = {c["id"]:c for c in plan.get("table_modifications", [])}
        new = {c["id"]:c for c in cuts}
        for key in old.keys() | new.keys():
            if old.get(key) != new.get(key):
                require(all(c["instance"] in allowed_instances for c in (old.get(key), new.get(key)) if c), "桌面修改越出关联实例范围")
        updated["table_modifications"] = copy.deepcopy(cuts)
    if 'edge_mounts' in value:
        mounts=validate_mounts(value['edge_mounts'],updated)
        old_other=[r for r in plan.get('edge_mounts',[]) if r['instance'] not in allowed_instances]
        new_other=[r for r in mounts if r['instance'] not in allowed_instances]
        require(old_other==new_other,'桌边安装修改越出允许实例')
        updated['edge_mounts']=copy.deepcopy(mounts)
    require(value["jobs"] or digest(updated) != digest(plan), "修复没有产生变化")
    validate_cuts(updated.get("table_modifications", []), updated, env)
    if "layout" in plan or env and "arms" in env["robot"]:
        validate_plan(updated, [a for j in plan["jobs"] if j["role"]=="asset" for a in j["asset_ids"]], limits, env)
    require(isinstance(value["reason"], str) and value["reason"], "修复须说明依据")
    return updated
