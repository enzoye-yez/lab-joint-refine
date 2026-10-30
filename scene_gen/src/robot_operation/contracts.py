"""有限操作图和任务契约；不解释任意代码或模型成功断言。"""
from .common import digest, need, number

OPERATIONS = {"observe", "check", "navigate", "dock", "joint_move", "cartesian", "grasp", "place",
              "insert", "press", "rotate", "dual_arm", "device", "wait"}
PREDICATES = {"all", "any", "not", "joint_range", "contact", "relative_pose", "body_stable",
              "held", "supported", "device_state", "base_stopped"}
PARAMETERS = {
    "observe": [], "check": [], "wait": [], "joint_move": ["arms","positions"],
    "navigate": ["transport","goal","bounds"], "dock": ["transport","bounds","arm","interface","reach_targets"],
    "cartesian": ["targets"], "dual_arm": ["targets"], "grasp": ["arm","body","controls"],
    "place": ["arm","body","support","controls"], "insert": ["arm","axis","distance"],
    "press": ["arm","axis","distance"], "rotate": ["arm","axis","angle"], "device": ["device"]}
PREDICATE_FIELDS = {"all":["items"],"any":["items"],"not":["item"],"joint_range":["joint","range"],
    "contact":["body_a","body_b"],"relative_pose":["site_a","site_b"],"body_stable":["body"],
    "held":["arm","body","tcp"],"supported":["body","support"],"device_state":["device","state"],"base_stopped":[]}


def predicate(value, depth=0):
    need(depth <= 12 and isinstance(value, dict), "条件树无效或过深")
    kind = value.get("kind")
    need(kind in PREDICATES, "未知状态检测器")
    need(set(PREDICATE_FIELDS[kind])<=set(value), "检测条件缺少必要字段")
    number(value.get("duration", 0), 0, 60)
    if kind in {"all", "any"}:
        need(isinstance(value.get("items"), list) and 0 < len(value["items"]) <= 100, "组合条件不能为空")
        for item in value["items"]: predicate(item, depth+1)
    elif kind == "not":
        predicate(value["item"], depth+1)
    return value


def validate_task(task):
    need(isinstance(task, dict) and task.get("version") == 1, "任务契约版本无效")
    need(task.get("input_sha256") and task.get("public_profile_sha256"), "缺少输入或公共环境版本")
    need(task.get("provenance") in {"developer_fixture", "model_generated"}, "缺少任务来源")
    need(isinstance(task.get("stages"), list) and task["stages"], "任务阶段不能为空")
    ids = set()
    for stage in task["stages"]:
        need(isinstance(stage.get("id"), str) and stage["id"] not in ids, "阶段编号重复")
        need(stage.get("source"), "阶段必须附输入依据")
        need(set(stage.get("depends_on", [])) <= ids, "阶段依赖须引用已有阶段")
        predicate(stage["goal"])
        ids.add(stage["id"])
    predicate(task["initial"])
    predicate(task["final"])
    need(isinstance(task.get("bindings"), dict), "任务必须提供对象身份绑定")
    for rule in task.get("invariants", []): predicate(rule)
    for event in task.get("mechanical_detectors", []):
        need(event.get("event") and event.get("source"), "机械检测器须有事件名与输入依据")
        predicate(event["condition"])
        need(event["condition"].get("duration", 0) > 0, "机械事件须持续证据")
    for device in task.get("devices", []):
        need(device.get("interface") and device.get("id"), "设备必须绑定软件接口")
        predicate(device["start_condition"])
        predicate(device["running_condition"])
        number(device["duration_seconds"], 0, 1e9)
    need(len({d["id"] for d in task.get("devices",[])})==len(task.get("devices",[])), "设备编号重复")
    need(len({d["interface"] for d in task.get("devices",[])})==len(task.get("devices",[])), "软件启动接口重复绑定")
    need(len({d["event"] for d in task.get("mechanical_detectors",[])})==len(task.get("mechanical_detectors",[])), "机械事件检测器重复")
    return task


def validate_graph(graph, task, scene_hash, settings):
    validate_task(task)
    need(graph.get("version") == 1 and graph.get("scene_sha256") == scene_hash, "操作图场景版本过期", "stale_scene")
    need(graph.get("task_sha256") == digest(task), "操作图任务版本过期")
    need(graph.get("provenance") in {"developer_fixture", "model_generated"}, "操作图必须标明来源")
    nodes = graph.get("nodes")
    need(isinstance(nodes, list) and 0 < len(nodes) <= settings.max_total_nodes, "节点数超限")
    by_id = {}
    for node in nodes:
        need(isinstance(node.get("id"), str) and node["id"] not in by_id and node["id"] not in {"success", "failed"}, "节点编号无效")
        need(node.get("operation") in OPERATIONS, "不支持的操作类型", "capability_missing")
        need(isinstance(node.get("parameters"), dict), "节点参数必须是对象")
        need(set(PARAMETERS[node["operation"]])<=set(node["parameters"]), "操作缺少必要参数")
        if "stop_when" in node["parameters"]:
            need(node["operation"] in {"press","insert","cartesian","dual_arm"},"此动作不支持接触停止条件")
            need(node["operation"]!="cartesian" or node["parameters"].get("contact") is True,"停止条件仅用于有界接触路径")
            predicate(node["parameters"]["stop_when"])
        number(node.get("timeout"), .001, settings.max_sim_seconds)
        visits = node.get("max_visits", 1)
        need(type(visits) is int and 1 <= visits <= settings.max_node_visits, "循环次数无效")
        predicate(node["precondition"])
        predicate(node["completion"])
        need(node.get("stage") in {s["id"] for s in task["stages"]}, "节点未绑定任务阶段")
        by_id[node["id"]] = node
    need(graph.get("entry") in by_id, "操作图入口缺失")
    for node in nodes:
        for edge in ("on_success", "on_failure"):
            need(node.get(edge) in set(by_id)|{"success", "failed"}, "转移目标不存在")
    reachable, queue = set(), [graph["entry"]]
    while queue:
        key = queue.pop()
        if key in reachable or key not in by_id: continue
        reachable.add(key)
        queue.extend([by_id[key]["on_success"], by_id[key]["on_failure"]])
    need(reachable == set(by_id), "存在不可达操作节点")
    return by_id
