"""Explicit, bounded variant selection shared by validation, inspection and assembly."""
from pathlib import Path
import re
from asset_prep.storage import safe_path
from .contracts import read_json, require

def catalogue(root):
    root = Path(root)
    manifest = read_json(root / "interfaces.json")
    variants = manifest.get("configurations")
    if variants is None:
        require("selected_configuration" not in manifest, "selected_configuration requires configurations")
        return {"default": {"file": "model.xml", "interfaces_file": "interfaces.json"}}
    require(isinstance(variants, dict) and 1 <= len(variants) <= 16, "configurations must contain 1..16 variants")
    for name, entry in variants.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name), "invalid configuration id")
        require(isinstance(entry, dict) and {"file", "interfaces_file"} <= set(entry) <= {"file", "interfaces_file", "root_body"}, "invalid configuration entry")
        for key in ("file", "interfaces_file"):
            path = entry[key]
            require(isinstance(path, str) and "/" not in path and "\\" not in path, "configuration files must be artifact-local")
            target = safe_path(root, path)
            require(target.is_file() and target.suffix == (".xml" if key == "file" else ".json"), "missing configuration file")
    selected = manifest.get("selected_configuration")
    require(isinstance(selected, str) and selected in variants, "explicit default configuration required")
    return variants

def selection(root, configuration=None):
    root = Path(root)
    variants = catalogue(root)
    if configuration is None:
        require(set(variants) == {"default"}, "multi-configuration artifact requires explicit instance selection")
        configuration = "default"
    require(isinstance(configuration,str) and configuration in variants, "unknown configuration: " + str(configuration))
    entry = variants[configuration]
    return configuration, entry["file"], read_json(root / entry["interfaces_file"])
