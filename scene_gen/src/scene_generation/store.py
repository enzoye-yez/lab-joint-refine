"""Run-to-run copying of a planned scene's archived stages and call ledger.

seed() is the one way a new run inherits a source run (external_repair, directed_repair and the
robot-operation joint runs): every archived stage is copied after its snapshot is checked, call
receipts are copied byte-identical, and the scene state is returned for the caller to adopt.
"""
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

from asset_generation.contracts import require
from asset_generation.pipeline import snapshot
from asset_prep.storage import digest, run_lock, safe_path


def seed(source, destination):
    """Copy the source run's stages and ledger into destination (idempotent) and return its scene."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
            "A joint run must not overwrite or nest with the source run")
    with run_lock(source):
        from asset_prep.state_archive import read_state
        state = read_state(source/"state.json")
        if (source/"state_reports").is_dir():
            shutil.copytree(source/"state_reports", destination/"state_reports", dirs_exist_ok=True)
        require("scene" in state and state["scene"].get("plan"), "Source is not a planned scene run")
        for stage in state["scene"]["stages"]:
            origin = safe_path(source, stage["directory"])
            require(snapshot(origin) == stage["files"], "An archived source stage was modified")
            target = safe_path(destination, stage["directory"])
            present = snapshot(target) if target.exists() else {}
            require(all(stage["files"].get(n) == sha for n, sha in present.items()),
                    "External modification during an unfinished source copy")
            for name in set(stage["files"])-set(present):
                dest = safe_path(target, name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = destination/(".seed-"+uuid.uuid4().hex+".tmp")
                try:
                    shutil.copyfile(safe_path(origin, name), temp)
                    temp.replace(dest)
                finally:
                    temp.unlink(missing_ok=True)
        for folder in ("records", "responses"):
            if (source/folder).exists():
                for path in (source/folder).glob("*.json"):
                    target = destination/folder/path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        require(target.read_bytes() == path.read_bytes(), "Inherited call receipt mismatch")
                    else:
                        shutil.copyfile(path, target)
        scene = deepcopy(state["scene"])
        # Interrupted sources remain preserved in the source tree; no claim of a completed candidate.
        scene.pop("pending", None)
        return {"calls": deepcopy(state["calls"]), "scene": scene, "source_state_sha256": digest(state),
                "source_identity": state["identity"],
                **({"phase_budget": deepcopy(state["phase_budget"])} if "phase_budget" in state else {})}
