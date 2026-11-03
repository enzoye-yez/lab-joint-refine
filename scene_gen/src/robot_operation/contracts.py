"""Bounded operation graphs and task contracts; never interprets arbitrary code or model success claims."""
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
    need(depth <= 12 and isinstance(value, dict), "Condition tree is invalid or too deep")
    kind = value.get("kind")
    need(kind in PREDICATES, "Unknown state detector")
    need(set(PREDICATE_FIELDS[kind])<=set(value), "Detector condition is missing required fields")
    number(value.get("duration", 0), 0, 60)
    if kind in {"all", "any"}:
        need(isinstance(value.get("items"), list) and 0 < len(value["items"]) <= 100, "Composite condition must not be empty")
        for item in value["items"]: predicate(item, depth+1)
    elif kind == "not":
        predicate(value["item"], depth+1)
    return value


def validate_task(task):
    need(isinstance(task, dict) and task.get("version") == 1, "Invalid task contract version")
    need(task.get("input_sha256") and task.get("public_profile_sha256"), "Missing input or public environment version")
    need(task.get("provenance") in {"developer_fixture", "model_generated"}, "Missing task provenance")
    need(isinstance(task.get("stages"), list) and task["stages"], "Task stages must not be empty")
    ids = set()
    for stage in task["stages"]:
        need(isinstance(stage.get("id"), str) and stage["id"] not in ids, "Duplicate stage id")
        need(stage.get("source"), "Each stage must cite its input source")
        need(set(stage.get("depends_on", [])) <= ids, "Stage dependencies must reference existing stages")
        predicate(stage["goal"])
        ids.add(stage["id"])
    predicate(task["initial"])
    predicate(task["final"])
    need(isinstance(task.get("bindings"), dict), "The task must provide object identity bindings")
    for rule in task.get("invariants", []): predicate(rule)
    for event in task.get("mechanical_detectors", []):
        need(event.get("event") and event.get("source"), "A mechanical detector requires an event name and an input source")
        predicate(event["condition"])
        need(event["condition"].get("duration", 0) > 0, "A mechanical event requires sustained evidence")
    for device in task.get("devices", []):
        need(device.get("interface") and device.get("id"), "A device must be bound to a software interface")
        predicate(device["start_condition"])
        predicate(device["running_condition"])
        number(device["duration_seconds"], 0, 1e9)
    need(len({d["id"] for d in task.get("devices",[])})==len(task.get("devices",[])), "Duplicate device id")
    need(len({d["interface"] for d in task.get("devices",[])})==len(task.get("devices",[])), "Software start interface is bound more than once")
    need(len({d["event"] for d in task.get("mechanical_detectors",[])})==len(task.get("mechanical_detectors",[])), "Duplicate mechanical event detector")
    return task


def validate_graph(graph, task, scene_hash, settings):
    validate_task(task)
    need(graph.get("version") == 1 and graph.get("scene_sha256") == scene_hash, "Operation graph scene version is stale", "stale_scene")
    need(graph.get("task_sha256") == digest(task), "Operation graph task version is stale")
    need(graph.get("provenance") in {"developer_fixture", "model_generated"}, "Operation graph must declare its provenance")
    nodes = graph.get("nodes")
    need(isinstance(nodes, list) and 0 < len(nodes) <= settings.max_total_nodes, "Too many nodes")
    by_id = {}
    for node in nodes:
        need(isinstance(node.get("id"), str) and node["id"] not in by_id and node["id"] not in {"success", "failed"}, "Invalid node id")
        need(node.get("operation") in OPERATIONS, "Unsupported operation type", "capability_missing")
        need(isinstance(node.get("parameters"), dict), "Node parameters must be an object")
        need(set(PARAMETERS[node["operation"]])<=set(node["parameters"]), "Operation is missing required parameters")
        if "stop_when" in node["parameters"]:
            need(node["operation"] in {"press","insert","cartesian","dual_arm"},"This operation does not support a contact stop condition")
            need(node["operation"]!="cartesian" or node["parameters"].get("contact") is True,"A stop condition is allowed only on a bounded contact path")
            predicate(node["parameters"]["stop_when"])
        number(node.get("timeout"), .001, settings.max_sim_seconds)
        visits = node.get("max_visits", 1)
        need(type(visits) is int and 1 <= visits <= settings.max_node_visits, "Invalid visit count")
        predicate(node["precondition"])
        predicate(node["completion"])
        need(node.get("stage") in {s["id"] for s in task["stages"]}, "Node is not bound to a task stage")
        by_id[node["id"]] = node
    need(graph.get("entry") in by_id, "Operation graph entry is missing")
    for node in nodes:
        for edge in ("on_success", "on_failure"):
            need(node.get(edge) in set(by_id)|{"success", "failed"}, "Transition target does not exist")
    reachable, queue = set(), [graph["entry"]]
    while queue:
        key = queue.pop()
        if key in reachable or key not in by_id: continue
        reachable.add(key)
        queue.extend([by_id[key]["on_success"], by_id[key]["on_failure"]])
    need(reachable == set(by_id), "Unreachable operation nodes exist")
    return by_id
