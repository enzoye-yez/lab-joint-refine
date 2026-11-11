"""Lightweight entry point for downstream planners: loads the default pose and contact events; does not plan motions."""

from pathlib import Path
import mujoco

from asset_generation.checks import load_scene_model
from asset_generation.contracts import read_json, require
from asset_generation.pipeline import snapshot
from asset_generation.runtime import InteractionRuntime
from asset_prep.storage import digest


def load_scene(artifact, expected_sha256=None):
    artifact = Path(artifact)
    actual = digest(snapshot(artifact))
    require(expected_sha256 is None or actual == expected_sha256, "planning scene artifact does not match the delivered digest")
    model = load_scene_model(artifact, dynamics=True)
    data = mujoco.MjData(model)
    meta = read_json(artifact/"scene.json",8000000)
    data.qpos[:] = meta["home_qpos"]
    data.ctrl[:] = meta["home_ctrl"]
    mujoco.mj_forward(model,data)
    manifest = read_json(artifact/"interfaces.json",8000000)
    groups = actor_groups(model, meta)
    actors = sorted({g for ids in groups.values() for g in ids})
    runtime = InteractionRuntime(model,manifest,actor_geom_ids=actors)
    return model,data,runtime,{"scene_sha256":actual,"manifest":manifest,"metadata":meta,"actor_groups":groups,
        "actor_group_reference":"geom indices in this loaded scene; bound to scene_sha256"}


def actor_groups(model, meta):
    """Resolve actual installed hands; absent planned tools never become stock hands."""
    from .geometry import owner
    from .robot_contract import arms, validate_bindings
    plan, env = meta["plan"], meta["environment"]
    selected = validate_bindings(plan, env, partial=True)
    items = {i["id"]: i for i in plan["instances"]}
    result = {}
    for arm in arms(env):
        binding = selected[arm["id"]]
        names = []
        if binding["mode"] != "stock" and binding["instance"] not in items:
            result[arm["id"]] = []
            continue
        bodies = set()
        if binding["mode"] in {"stock", "adapter"} and arm["stock"]:
            # The moving finger subtrees are contact actors. A wrist/adapter site
            # can live on the forearm; including its entire subtree mislabels arm collisions.
            for joint in arm["stock"]["joints"]:
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot__"+joint)
                require(j >= 0, "stock gripper joint missing: "+joint)
                bodies.add(int(model.jnt_bodyid[j]))
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]): continue
            body = int(model.geom_bodyid[g])
            stock = False
            while body:
                if body in bodies: stock = True; break
                body = int(model.body_parentid[body])
            custom = binding["mode"] != "stock" and owner(model,g) == binding["instance"]
            if stock or custom:
                names.append(g)
        require(names or (binding["mode"] == "stock" and arm["stock"] is None),
                "installed end-effector has no bindable contact geometry: "+arm["id"])
        result[arm["id"]] = names
    return result
