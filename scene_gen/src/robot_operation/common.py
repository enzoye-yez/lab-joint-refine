"""Public engineering parameters, bounded inputs and version binding."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from asset_prep.storage import atomic_json, digest, file_hash, safe_path


class OperationError(ValueError):
    def __init__(self, code, detail, evidence=None):
        super().__init__(detail)
        self.code, self.detail, self.evidence = code, detail, evidence or {}


def copy_data(target, model, source):
    """The 3.3.7 Python bindings copy the full integration state; no dependence on unexported C functions."""
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, source, state, spec)
    mujoco.mj_setState(model, target, state, spec)
    mujoco.mj_forward(model, target)


def need(condition, message, code="invalid_contract"):
    if not condition:
        raise OperationError(code, message)


def read(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size <= 32_000_000, "Input missing or exceeds the size limit")
    def reject(value):
        raise OperationError("invalid_contract", "Non-finite numbers are not allowed: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def vector(value, size):
    out = np.asarray(value, dtype=float)
    need(out.shape == (size,) and np.isfinite(out).all(), "Invalid vector length or values")
    return out


def number(value, minimum=0, maximum=1e8):
    need(type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum,
         "Value outside engineering bounds")
    return value


@dataclass(frozen=True)
class Settings:
    # Engineering defaults, not vendor-certified values or hidden scoring thresholds; saved in full with every delivery.
    settle_seconds: float = 2.0
    control_period: float = .02
    image_period: float = .1
    penetration_m: float = .003
    contact_force_n: float = 150.0
    position_tolerance_m: float = .003
    rotation_tolerance_rad: float = .04
    joint_tolerance: float = .025
    stopped_speed: float = .025
    stable_seconds: float = .2
    base_tilt_rad: float = .18
    base_speed_mps: float = .12
    base_yaw_rate: float = .25
    joint_speed: float = .4
    cartesian_speed_mps: float = .025
    grid_m: float = .05
    path_resolution: float = .04
    rrt_iterations: int = 2000
    ik_iterations: int = 180
    ik_restarts: int = 4
    local_retries: int = 3
    graph_revisions: int = 2
    scene_repairs: int = 3
    max_node_visits: int = 10
    max_total_nodes: int = 500
    max_sim_seconds: float = 1800.0
    seed: int = 0
    diagnostic_queries: int = 256
    diagnostic_batch: int = 32
    diagnostic_stalls: int = 3
    diagnostic_probe_seconds: float = 5.0
    diagnostic_total_sim_seconds: float = 120.0
    diagnostic_trace_steps: int = 1000000
    diagnostic_trace_window: int = 200

    @classmethod
    def load(cls, path=None):
        values = read(path) if path else {}
        need(isinstance(values, dict) and set(values) <= set(cls.__dataclass_fields__), "Unknown engineering parameter")
        result = cls(**values)
        for key, value in asdict(result).items():
            number(value, 0 if key in {"seed", "local_retries", "graph_revisions", "scene_repairs"} else 1e-9)
            if type(cls.__dataclass_fields__[key].default) is int:
                need(type(value) is int, "Counts and seeds must be integers")
        return result


def identity(root):
    from asset_generation.pipeline import snapshot
    return digest(snapshot(Path(root)))


def new_directory(path, sources=()):
    target = Path(path).resolve()
    need(not target.exists(), "Output directory already exists; use a new run name", "output_exists")
    for source in sources:
        parent = Path(source).resolve()
        need(not target.is_relative_to(parent) and not parent.is_relative_to(target), "Input and output directories must not be nested")
    target.mkdir(parents=True)
    return target
