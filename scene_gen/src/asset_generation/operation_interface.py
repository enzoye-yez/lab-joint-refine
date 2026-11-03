"""Optional, versioned operation geometry; declarations remain unverified until measured."""
import math
import numpy as np
from .contracts import require, finite

FIELDS={"axis","joint","range","clearance_m","grasp_geoms","mate_interface","basis","status"}


def validate_operation(interface, model, interface_ids):
    replacements=interface.get("replaces",[])
    require(isinstance(replacements,list) and len(replacements)<=32 and all(isinstance(x,str) and x for x in replacements)
            and len(set(replacements))==len(replacements),"Semantic id mapping must be a list of unique old ids")
    value=interface.get("operation")
    if value is None: return  # Older assets remain readable, with no invented capability.
    require(isinstance(value,dict) and set(value)<=FIELDS,"Invalid operating interface fields")
    require(value.get("status") in {"declared","unknown"},"An asset cannot certify its own operating interface")
    basis=value.get("basis",{})
    require(isinstance(basis,dict) and set(basis)<=FIELDS-{"basis","status"},"Invalid interface parameter basis")
    for key in sorted(set(value)-{"basis","status"}):
        entry=basis.get(key)
        require(isinstance(entry,dict) and set(entry)=={"kind","reference"} and
                entry["kind"] in {"source_specification","measurement","assumption"} and
                isinstance(entry["reference"],str) and 0<len(entry["reference"])<=2048,
                f"Interface {interface.get('id')}: operation.{key} must state its source or assumption (basis.{key})")
    if "axis" in value:
        a=value["axis"]
        require(isinstance(a,list) and len(a)==3 and all(finite(x) for x in a) and abs(sum(x*x for x in a)-1)<1e-6,"Operation axis must be a local unit vector")
        require(bool(interface.get("site")),"Operation axis needs a local site")
    if "range" in value:
        limits=value["range"]
        require("joint" in value and isinstance(limits,list) and len(limits)==2 and all(finite(x) for x in limits) and limits[0]<limits[1],
                f"Interface {interface.get('id')}: operation travel must be bound to this asset's joint that drives the part; motion of external objects must be listed separately as a requirement to verify, never faked with the asset's own joint")
    if "joint" in value:
        j=model.joint(value["joint"]).id
        require(model.jnt_type[j] in (2,3) and model.jnt_limited[j],"Operation joint must be a bounded slide or hinge joint")
        body=model.body(interface["body"]).id; ancestor=body
        while ancestor and ancestor!=model.jnt_bodyid[j]: ancestor=int(model.body_parentid[ancestor])
        require(ancestor==model.jnt_bodyid[j],"Operation joint does not drive the interface part")
        if "range" in value:
            lo,hi=value["range"]
            require(lo>=model.jnt_range[j,0]-1e-9 and hi<=model.jnt_range[j,1]+1e-9,"Declared travel exceeds the model's limits")
        if "axis" in value:
            # Model-local axis and site-local declaration must describe the same world line direction.
            import mujoco
            data=mujoco.MjData(model); mujoco.mj_forward(model,data)
            axis=data.site_xmat[model.site(interface["site"]).id].reshape(3,3)@np.asarray(value["axis"])
            require(abs(float(axis@data.xaxis[j]))>1-1e-5,"Interface operation axis disagrees with the actual joint axis")
    if "clearance_m" in value: require(finite(value["clearance_m"]) and value["clearance_m"]>=0,"Clearance must be a non-negative finite number")
    if "grasp_geoms" in value:
        names=value["grasp_geoms"]
        require(isinstance(names,list) and names and len(names)<=64 and len(set(names))==len(names),"Invalid grasp geom list")
        body=model.body(interface["body"]).id
        for name in names:
            geom=model.geom(name).id; ancestor=int(model.geom_bodyid[geom])
            while ancestor and ancestor!=body: ancestor=int(model.body_parentid[ancestor])
            require(ancestor==body,"Grasp region does not belong to the interface part")
    if "mate_interface" in value: require(value["mate_interface"] in interface_ids,"Mating interface does not exist")


def namespace_operation(interface,prefix):
    if "replaces" in interface:interface["replaces"]=[prefix+x for x in interface["replaces"]]
    value=interface.get("operation")
    if value is None: return
    for key in ("joint","mate_interface"):
        if key in value: value[key]=prefix+value[key]
    if "grasp_geoms" in value: value["grasp_geoms"]=[prefix+n for n in value["grasp_geoms"]]
