"""What the repair skills know about a scene: the plan (support chain, tables, docks) and the compiled,
settled model (geometry, interfaces, joints), with small helpers the skills share."""
import json
from pathlib import Path

import numpy as np

from ..diagnose import Measure, instance_of


class SceneView:
    def __init__(self, cache, source):
        self.cache, self.source = Path(cache), Path(source)
        self.measure = Measure(cache, source)
        self.model, self.data = self.measure.model, self.measure.data
        self.interfaces = self.measure.interfaces
        plan = json.loads((self.source / "scene.json").read_text())["plan"]
        self.instances = {i["id"]: i for i in plan["instances"]}
        self.tables = {t["id"]: t for t in plan.get("layout", {}).get("tables", [])}
        self.robot_home = np.asarray(plan.get("layout", {}).get("robot_home", [0, 0, 0])[:2], float)

    # -- plan ------------------------------------------------------------------
    def base_of(self, instance):
        """The instance at the bottom of the support chain (what stands on the bench or floor)."""
        seen = set()
        while instance in self.instances and self.instances[instance].get("support_instance") and instance not in seen:
            seen.add(instance)
            instance = self.instances[instance]["support_instance"]
        return instance

    def carried_by(self, base):
        """Every instance whose support chain reaches `base` (not `base` itself)."""
        return sorted(i for i in self.instances if i != base and self.base_of(i) == base)

    def front(self, instance):
        """Unit floor vector from the instance's bench toward the aisle (the side the robot docks from)."""
        inst = self.instances.get(self.base_of(instance), {})
        table = self.tables.get(inst.get("table_id"))
        position = np.asarray(inst.get("position", [0, 0])[:2], float)
        if table is None:
            v = self.robot_home - position
            return v / (np.linalg.norm(v) or 1.0)
        yaw = float(table.get("yaw", 0.0))
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        if normal @ (self.robot_home - np.asarray(table["position"][:2], float)) < 0:
            normal = -normal
        return normal

    def along(self, instance):
        f = self.front(instance)
        return np.array([-f[1], f[0]])

    # -- model -----------------------------------------------------------------
    def interface(self, task_id):
        return self.interfaces.get(task_id)

    def joint_name(self, joint):
        return self.model.joint(int(joint)).name if joint is not None and joint >= 0 else None

    def body_name(self, body):
        return self.model.body(int(body)).name

    def geom_instance(self, name):
        return instance_of(name)

    def is_robot(self, name):
        return str(name).startswith("robot__")

    def mechanism_joint(self, task_id):
        i = self.interface(task_id)
        return None if i is None or i.mechanism is None else self.joint_name(i.mechanism.joint)
