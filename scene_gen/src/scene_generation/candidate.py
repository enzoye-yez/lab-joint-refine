"""Compile-only admission for diagnostic assembly, never quality acceptance."""
from pathlib import Path
import mujoco
from asset_generation.checks import load_model, validate_manifest, render_views
from asset_generation.configurations import catalogue, selection
from asset_generation.contracts import read_json, require
from asset_generation.static_state import geometry_state
from asset_generation.pipeline import snapshot
from asset_prep.storage import digest
from asset_prep.errors import ValidationError



def assembly_manifest(value, model, asset_ids, *, diagnostic=True):
    """Preserve geometry for diagnostics when interaction-only validation fails.

    Scope, parts and interface references remain mandatory for safe assembly.
    The original manifest is retained in the artifact and fully checked for quality.
    """
    from copy import deepcopy
    try:
        return validate_manifest(value,model,asset_ids), []
    except ValidationError as exc:
        if not diagnostic: raise  # Tool installation depends on the complete connection contract.
        if not isinstance(value,dict) or not isinstance(value.get("interfaces"),list):raise
        diagnostic=deepcopy(value)
        for key in ("buttons","connections","mechanical_events"):
            diagnostic[key]=[]
        # Operation declarations are not geometric attachment references. Never
        # expose invalid motion claims to downstream planning, or invent their basis.
        for interface in diagnostic.get("interfaces", []):
            if isinstance(interface,dict):
                interface.pop("operation",None)
                interface.pop("replaces",None)
        # First establish the structural contract. Do not let malformed metadata
        # raise an unclassified exception while preparing a diagnostic projection.
        validate_manifest(diagnostic,model,asset_ids)
        from asset_generation.operation_interface import validate_operation
        interface_ids={i["id"] for i in diagnostic["interfaces"]}
        for original,interface in zip(value["interfaces"],diagnostic["interfaces"]):
            try:
                validate_operation(original,model,interface_ids)
            except (ValidationError,KeyError,ValueError,TypeError):
                continue
            for key in ("operation","replaces"):
                if key in original:interface[key]=deepcopy(original[key])
        issue={"id":"interaction_contract","status":"fail","detail":str(exc),"complete":True}
        diagnostic["approximations"].append("诊断装配未采用无效交互声明；原始接口清单保留，交互契约未通过："+str(exc))
        return diagnostic,[issue]

def inspect_candidate(root, asset_ids, profile, render_dir=None):
    root = Path(root)
    rows, renders, summaries = [], [], {}
    admissible = True
    variants = catalogue(root)
    expected = profile.get("assembly_configurations")
    require(expected is None or set(variants) == set(expected), "Delivered configurations must match assembly plan")
    for name in sorted(variants):
        _, filename, manifest = selection(root, name)
        model = load_model(root, model_file=filename)
        rows.append({"id":name+":assembly_compile", "status":"pass", "configuration":name,
                     "detail":"Model independently loaded; interfaces and quality assessed separately"})
        try:
            manifest, issues = assembly_manifest(manifest, model, asset_ids, diagnostic=profile.get("assembly_role")!="tool")
        except ValidationError as exc:
            if profile.get("assembly_role")=="tool":raise
            admissible = False
            rows.append({"id":name+":assembly_interfaces", "status":"fail", "configuration":name,"detail":str(exc)})
            summaries[name] = {}
            continue
        rows.extend({**r,"id":name+":"+r["id"],"configuration":name} for r in issues)
        if profile.get("assembly_role") == "tool":
            from .tool_contract import tool_roots
            tool_roots(model, manifest)
        data = mujoco.MjData(model)
        geometry_state(model, data)
        summaries[name] = {"unit":"m", "interfaces":[{"id":i["id"],"site":i["site"],
            "position":data.site_xpos[model.site(i["site"]).id].tolist(),
            "rotation":data.site_xmat[model.site(i["site"]).id].reshape(3,3).tolist()}
            for i in manifest["interfaces"] if i.get("site")],
            "bodies":[{"id":b["id"],"body":b["body"],"position":data.xpos[model.body(b["body"]).id].tolist(),
                       "mass_kg":float(model.body_mass[model.body(b["body"]).id])} for b in manifest["parts"]]}
        from .geometry import bounds
        boxes=[bounds(model,data,g) for g in range(model.ngeom) if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE]
        if boxes:
            summaries[name]["bounds_m"]=[[float(min(b[0][k] for b in boxes)) for k in range(3)],
                                          [float(max(b[1][k] for b in boxes)) for k in range(3)]]
        if render_dir is not None:
            try:
                paths = render_views(model,[("initial",data.qpos.copy())],Path(render_dir)/name)
                renders.extend(name+"/"+s for s in paths)
            except Exception as exc:
                rows.append({"id":name+":candidate_render", "status":"partial", "detail":str(exc)[:1000]})
    rows.append({"id":"asset_quality_pending", "status":"partial", "detail":"Quality checks deferred until diagnostic scene is saved"})
    from asset_generation.check_report import report_status
    return {"status":report_status(rows), "checks":rows,"renders":renders,
        "manifest":read_json(root/"interfaces.json"),
        "geometry_summary":summaries["default"] if set(summaries)=={"default"} else {"configurations":summaries},
        "coverage":["Assembly admission only; no quality acceptance"],"validation_mode":"static",
        "assembly_readiness":{"loadable":admissible,"geometry_loadable":True,
            "interfaces_valid":not any(r["status"]=="fail" for r in rows),
            "interaction_verified":False,"artifact_sha256":digest(snapshot(root))}}


def usable(run, record):
    from asset_prep.storage import safe_path
    from asset_prep.errors import StaleEvidence
    admission = (record or {}).get("assembly_readiness",{})
    if not admission.get("loadable") or not record.get("artifact"): return False
    actual = digest(snapshot(safe_path(run.root,record["artifact"])))
    if actual != admission["artifact_sha256"]:
        raise StaleEvidence("Assembly candidate changed after admission")
    return True
