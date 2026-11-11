"""Small scene delivery contract. Source requirements, the robot and acceptance conditions are outside the repair workspace."""

import copy
import re
from dataclasses import dataclass, asdict

from asset_generation.contracts import finite, require, check_profile
from asset_prep.storage import digest
from .table import validate_cuts
from .edge_mounts import validate_mounts


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value), "Invalid object id")
    return value


def vector(value, n=3):
    require(isinstance(value, list) and len(value) == n and all(finite(x) for x in value), "Invalid finite numeric vector")
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
                require(isinstance(v,str) and v in {"full", "static"}, "Validation mode must be full or static")
            elif k == 'max_check_retries':
                require(type(v) is int and v >= 0, 'Recovery tool retry count must be a non-negative integer')
            elif k.endswith("_m"):
                require(finite(v) and 0 <= v <= .02, "Invalid geometry tolerance")
            else:
                require(type(v) is int and v > 0, "Scene limits must be positive integers")
        require(self.settle_steps <= 5000 and self.ik_iterations <= 1000 and self.ik_restarts <= 32,
                "Scene check budget too large")
        require(self.parallel_authors <= 8, "At most 8 parallel authors")

    def identity(self):
        return {k: v for k, v in asdict(self).items()
                if k not in {"parallel_authors", "max_iterations", "max_model_calls", "max_reserved_tokens", "max_plan_attempts", "max_repair_plan_attempts", "max_evidence_rounds", "max_evidence_plan_attempts", "max_check_actions", "max_check_retries"}}


def environment(value):
    require(isinstance(value, dict) and set(value) == {"table", "robot"}, "Scene environment must declare the reference workbench and robot specification")
    table, robot = value["table"], value["robot"]
    require(isinstance(table,dict) and set(table) == {"size", "height"}, "Invalid table fields")
    require(all(x > 0 for x in vector(table["size"])) and finite(table["height"])
            and table["height"] > table["size"][2], "Invalid table dimensions")
    from .robot_contract import validate_robot
    validate_robot(robot)
    if "mobile" in robot:
        from .layout import validate_mobile
        validate_mobile(robot["mobile"])
    return copy.deepcopy(value)


def validate_plan(value, asset_ids, limits, env=None):
    require(isinstance(value, dict) and set(value)-{"table_modifications","edge_mounts","layout","end_effectors","robot_posture","role_bindings"} == {"jobs", "instances", "assumptions"}, "Invalid scene plan fields")
    require(isinstance(value["jobs"], list) and 1 <= len(value["jobs"]) <= limits.max_jobs, "Too many generation jobs")
    require(isinstance(value["instances"], list) and 1 <= len(value["instances"]) <= limits.max_instances, "Too many scene instances")
    jobs, covered = {}, []
    for job in value["jobs"]:
        require(isinstance(job, dict) and set(job)-{"configurations"} == {"id", "asset_ids", "role", "depends_on", "requirements"}, "Invalid job fields")
        identifier(job["id"])
        require(job["id"] not in jobs and isinstance(job['role'],str) and job["role"] in {"asset", "tool", "fixture"}, "Duplicate job or invalid job role")
        require(job["role"] != "fixture", "This stage does not generate tabletop fixtures (fixture); only asset and tool jobs are allowed")
        require(isinstance(job["asset_ids"], list) and job["asset_ids"] and all(isinstance(a,str) for a in job['asset_ids'])
                and len(set(job["asset_ids"])) == len(job["asset_ids"]), "Invalid job asset ids")
        require(all(isinstance(x, str) and x in asset_ids for x in job["asset_ids"]), "Job references an unknown asset")
        if job["role"] == "asset":
            covered.extend(job["asset_ids"])
        require(isinstance(job["depends_on"], list) and all(isinstance(d,str) for d in job['depends_on'])
                and len(set(job["depends_on"])) == len(job["depends_on"]), "Invalid dependencies")
        require(isinstance(job["requirements"], str) and len(job["requirements"]) <= 24000, "Job requirements too long")
        if "configurations" in job:
            variants = job["configurations"]
            require(isinstance(variants,list) and 1 <= len(variants) <= 16 and all(isinstance(n,str) for n in variants)
                    and len(set(variants))==len(variants), "invalid job configurations")
            for name in variants: identifier(name)
        jobs[job["id"]] = job
    require(sorted(covered) == sorted(asset_ids), "Every asset in the input package must be assigned to exactly one asset generation job")
    multi = env is not None and "arms" in env["robot"]
    if not multi:
        require(sum(j["role"] == "tool" for j in jobs.values()) == 1, "Exactly one unified end-effector must be designed")
    else:
        require("end_effectors" in value, "The plan must declare the end-effector configuration of every arm")
    if not multi:
        require("end_effectors" not in value, "Legacy environments use the original single-arm end-effector convention")
    from .robot_contract import validate_posture
    validate_posture(value.get("robot_posture",{}),env)
    order = []
    while len(order) < len(jobs):
        ready = [k for k, j in jobs.items() if k not in order and all(x in order for x in j["depends_on"])]
        require(ready, "Dependency missing or cyclic")
        order.extend(ready)
    for job in jobs.values():
        if job["role"] != "asset":
            require(all(k in job["depends_on"] for k, j in jobs.items() if j["role"] == "asset"),
                    "The operating tool must depend on all assets, not on a single instrument only")
        if job["role"] == "tool":
            require(all(k in job["depends_on"] for k,j in jobs.items() if j["role"] == "fixture"),
                    "The end-effector must read the generated tabletop fixtures to keep the co-design consistent")
    instances = {}
    for item in value["instances"]:
        require(isinstance(item, dict) and set(item) - {"support_instance","configuration","table_id","arrangement"} == {"id", "job", "placement", "position", "yaw"}, "Invalid instance fields")
        identifier(item["id"])
        if "arrangement" in item:
            # Optional per-instance spacing overrides (arrangement.OVERRIDES), e.g. from a verified directed repair.
            from .arrangement import overrides
            require("layout" in value and item.get("placement") in {"table", "floor"}, "Arrangement overrides apply to arranged table or floor instances only")
            try: overrides(item)
            except ValueError as exc: require(False, str(exc))
        require(item["id"] not in instances and isinstance(item['job'],str) and item["job"] in jobs, "Duplicate instance or invalid job reference")
        require(isinstance(item['placement'],str) and item["placement"] in {"table", "floor", "robot"}, "Invalid placement surface")
        require('support_instance' not in item or isinstance(item['support_instance'],str) and bool(item['support_instance']), 'Invalid support instance id')
        require(all(abs(x) <= (100 if "layout" in value else 10) for x in vector(item["position"])) and finite(item["yaw"]), "Invalid instance pose")
        require((jobs[item["job"]]["role"] == "tool") == (item["placement"] == "robot"), "End-effectors must be mounted on the robot arm")
        if item["placement"] == "robot":
            require(item["position"] == [0, 0, 0] and item["yaw"] == 0, "The tool mounting origin is fixed at the robot flange")
        variants = jobs[item["job"]].get("configurations")
        require((variants is not None and item.get("configuration") in variants) or
                (variants is None and "configuration" not in item), "instance configuration must match its planned job")
        instances[item["id"]] = item
    require(set(jobs) == {i["job"] for i in instances.values()}, "A generation job has no instance")
    if not multi:
        require(sum(i["placement"] == "robot" for i in instances.values()) == 1, "Only one device mounted on the robot arm is allowed")
    else:
        from .robot_contract import validate_bindings
        validate_bindings(value,env)
        require(sum(j["role"]=="tool" for j in jobs.values()) <= len(env["robot"]["arms"]), "At most one current custom assembly per arm")
    for item in instances.values():
        seen = {item["id"]}
        parent = item.get("support_instance")
        require(not parent or item["placement"] != "robot", "A robot-mounted tool cannot also declare a support instance")
        while parent:
            require(isinstance(parent,str) and parent in instances and parent not in seen, "Support reference missing or cyclic")
            require(instances[parent]["placement"] != "robot", "The initial support cannot be the robot arm end-effector tool")
            seen.add(parent)
            parent = instances[parent].get("support_instance")
    require(isinstance(value["assumptions"], list) and all(isinstance(x, str) for x in value["assumptions"]), "Invalid layout assumptions")
    from .layout import validate_layout
    validate_layout(value)
    validate_cuts(value.get("table_modifications", []), value)
    validate_mounts(value.get('edge_mounts',[]),value)
    return copy.deepcopy(value), order


def numerical_contract(value):
    """Acceptance fixed by the user; the model cannot rewrite it. Missing items stay explicit in the report."""
    require(isinstance(value, dict) and not set(value) - {"jobs", "fits", "targets", "payload_mass_kg", "arm_payloads"}, "Invalid acceptance fields")
    result = {"jobs": {}, "fits": [], "targets": [], "payload_mass_kg": 0, **value}
    require(isinstance(result["jobs"], dict), "Job acceptance must be a mapping")
    result["jobs"] = {identifier(k): check_profile(v) for k, v in result["jobs"].items()}
    require(finite(result["payload_mass_kg"]) and result["payload_mass_kg"] >= 0, "Invalid payload")
    require(isinstance(result.get("arm_payloads",{}),dict) and
            all(isinstance(k,str) and finite(v) and v>=0 for k,v in result.get("arm_payloads",{}).items()), "Invalid per-arm payloads")
    for key in ("fits", "targets"):
        require(isinstance(result[key], list) and len(result[key]) <= 256, "Too many scene acceptance entries")
    for fit in result["fits"]:
        require(isinstance(fit,dict) and set(fit) == {"id", "outer", "inner", "axis", "clearance_m", "evidence"}, "Invalid fit check fields")
        identifier(fit["id"])
        require(type(fit["axis"]) is int and fit["axis"] in (0, 1, 2), "Invalid fit axis")
        vector(fit["clearance_m"], 2)
        require(0 <= fit["clearance_m"][0] <= fit["clearance_m"][1], "Invalid fit clearance")
        for end in ("outer", "inner"):
            require(isinstance(fit[end], list) and len(fit[end]) == 2 and all(isinstance(x, str) and x for x in fit[end]), "Fit dimensions must be marked by two actual sites")
        require(isinstance(fit["evidence"], str) and fit["evidence"], "Fit requirements must record their source")
    for target in result["targets"]:
        require(isinstance(target,dict) and set(target) == {"id", "site", "tool_site", "offset", "quaternion", "position_tolerance_m", "angle_tolerance_rad", "allowed_contacts"}, "Invalid operation target fields")
        identifier(target["id"])
        require(all(isinstance(target[k], str) and target[k] for k in ("site", "tool_site")), "Invalid operation interface names")
        vector(target["offset"]); vector(target["quaternion"], 4)
        require(abs(sum(x*x for x in target["quaternion"])-1) < 1e-5, "Target rotation must be a unit quaternion")
        require(all(finite(target[k]) and 0 < target[k] <= .2 for k in ("position_tolerance_m", "angle_tolerance_rad")), "Invalid target tolerances")
        require(isinstance(target["allowed_contacts"], list) and all(isinstance(p, list) and len(p) == 2
                and all(isinstance(g, str) and g for g in p) for p in target["allowed_contacts"]), "Allowed contacts must be exact geom pairs")
    for key in ('fits','targets'):
        require(len({r['id'] for r in result[key]})==len(result[key]), 'Acceptance check ids must be unique')
    return result


def repair_action(value, plan, allowed_jobs, allowed_instances, limits, env=None):
    require(isinstance(value, dict) and set(value)-{"table_modifications","edge_mounts","geometry_changes","layout","end_effectors","tool_additions","robot_posture","room_revision_reason"} == {"jobs", "placements", "reason"}, "Invalid repair instruction fields")
    require("room_revision_reason" not in value or "layout" in value,"A room revision reason requires an actual layout change")
    additions=value.get("tool_additions",[])
    require(isinstance(additions,list) and (not additions or env is not None and "arms" in env["robot"]), "Adding tools requires a multi-mount environment")
    new_jobs=[]; new_instances=[]
    for addition in additions:
        require(isinstance(addition,dict) and set(addition)=={"job","instance"}, "An added tool must provide both a complete job and an instance")
        job,item=addition["job"],addition["instance"]
        require(isinstance(job,dict) and job.get("role")=="tool" and isinstance(job.get("id"),str) and job.get("id") not in {j["id"] for j in plan["jobs"]}
                and isinstance(item,dict) and item.get("job")==job.get("id") and item.get("placement")=="robot", "Additions are limited to dedicated robot-mounted tools")
        new_jobs.append(job);new_instances.append(item)
    for job,item in zip(new_jobs,new_instances):
        identifier(job.get("id")); identifier(item.get("id"))
    require(isinstance(value.get("jobs"),list) and all(isinstance(j,str) for j in value["jobs"]), "Invalid repair job ids")
    added_ids={j["id"] for j in new_jobs}
    require(added_ids <= set(value.get("jobs",[])), "Added tools must be listed in this round's repair jobs")
    allowed_jobs=set(allowed_jobs)|added_ids
    require(isinstance(value["jobs"], list) and all(isinstance(j,str) for j in value['jobs']) and len(set(value["jobs"])) == len(value["jobs"])
            and set(value["jobs"]) <= set(allowed_jobs), "Asset repair exceeds the allowed scope")
    if 'geometry_changes' in value:
        require(isinstance(value['geometry_changes'],list) and 0 < len(value['geometry_changes']) <= 64,
                'Invalid concrete geometry repair action')
        for change in value['geometry_changes']:
            require(isinstance(change,dict) and set(change)=={'job','operation','parameters'}
                    and change['job'] in value['jobs'] and isinstance(change['operation'],str) and change['operation']
                    and isinstance(change['parameters'],dict) and change['parameters'], 'Concrete geometry repair parameters missing or out of bounds')
    require(isinstance(value["placements"], list) and len(value["placements"]) <= limits.max_instances, "Invalid layout repair")
    require("layout" not in value or "layout" in plan, "Legacy scenes cannot accept mobile layout changes")
    updated = copy.deepcopy(plan)
    updated["jobs"].extend(new_jobs)
    updated["instances"].extend(new_instances)
    if "end_effectors" in value:
        require(env is not None and "arms" in env["robot"], "Legacy environments do not accept new end-effector configurations")
        from .robot_contract import bindings
        old_bindings=bindings(plan,env)
        item_jobs={i["id"]:i["job"] for i in updated["instances"]}
        require(isinstance(value["end_effectors"],dict), "End-effector configuration must be a mapping")
        for arm,binding in value["end_effectors"].items():
            if binding!=old_bindings.get(arm):
                changed=[b.get("instance") for b in (binding,old_bindings.get(arm,{})) if isinstance(b,dict) and b.get("instance")]
                require(all(item_jobs.get(i) in allowed_jobs for i in changed), "End-effector swap exceeds the allowed tool scope")
        updated["end_effectors"]=copy.deepcopy(value["end_effectors"])
    if "robot_posture" in value:
        from .robot_contract import validate_posture
        validate_posture(value["robot_posture"],env)
        updated["robot_posture"]=copy.deepcopy(value["robot_posture"])
    items = {i["id"]: i for i in updated["instances"]}
    seen = set()
    for edit in value["placements"]:
        require(isinstance(edit,dict) and set(edit)-{"table_id","placement","support_instance"} == {"id", "position", "yaw"} and isinstance(edit['id'],str)
                and edit["id"] in allowed_instances and edit["id"] not in seen, "Layout repair out of scope or duplicated")
        seen.add(edit["id"])
        require(items[edit["id"]]["placement"] != "robot", "The fixed tool mounting interface cannot be moved")
        vector(edit["position"])
        require(finite(edit["yaw"]) and all(abs(x) <= (100 if "layout" in plan else 10) for x in edit["position"]), "Invalid layout repair pose")
        items[edit["id"]].update(position=edit["position"], yaw=edit["yaw"])
        for field in ("placement", "table_id", "support_instance"):
            if field in edit:
                require("layout" in plan, "Legacy placement repairs do not accept the new layout field")
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
            require(item == old_item or item["id"] in allowed_instances, "Linked move exceeds the allowed instance scope")
        validate_layout(updated, env)
    if "table_modifications" in value:
        cuts = validate_cuts(value["table_modifications"], updated, env)
        old = {c["id"]:c for c in plan.get("table_modifications", [])}
        new = {c["id"]:c for c in cuts}
        for key in old.keys() | new.keys():
            if old.get(key) != new.get(key):
                require(all(c["instance"] in allowed_instances for c in (old.get(key), new.get(key)) if c), "Table modification exceeds the linked instance scope")
        updated["table_modifications"] = copy.deepcopy(cuts)
    if 'edge_mounts' in value:
        mounts=validate_mounts(value['edge_mounts'],updated)
        old_other=[r for r in plan.get('edge_mounts',[]) if r['instance'] not in allowed_instances]
        new_other=[r for r in mounts if r['instance'] not in allowed_instances]
        require(old_other==new_other,'Edge mount change exceeds the allowed instances')
        updated['edge_mounts']=copy.deepcopy(mounts)
    require(value["jobs"] or digest(updated) != digest(plan), "Repair produced no change")
    validate_cuts(updated.get("table_modifications", []), updated, env)
    if "layout" in plan or env and "arms" in env["robot"]:
        validate_plan(updated, [a for j in plan["jobs"] if j["role"]=="asset" for a in j["asset_ids"]], limits, env)
    require(isinstance(value["reason"], str) and value["reason"], "Repair must state its rationale")
    return updated
