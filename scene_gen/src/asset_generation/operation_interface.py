"""Optional, versioned operation geometry; declarations remain unverified until measured."""
import math
import numpy as np
from .contracts import require, finite

FIELDS={"axis","joint","range","clearance_m","grasp_geoms","mate_interface","basis","status"}


def validate_operation(interface, model, interface_ids):
    replacements=interface.get("replaces",[])
    require(isinstance(replacements,list) and len(replacements)<=32 and all(isinstance(x,str) and x for x in replacements)
            and len(set(replacements))==len(replacements),"语义编号映射须为唯一的旧编号列表")
    value=interface.get("operation")
    if value is None: return  # Older assets remain readable, with no invented capability.
    require(isinstance(value,dict) and set(value)<=FIELDS,"操作接口字段无效")
    require(value.get("status") in {"declared","unknown"},"资产不能自行认证操作接口")
    basis=value.get("basis",{})
    require(isinstance(basis,dict) and set(basis)<=FIELDS-{"basis","status"},"接口参数依据无效")
    for key in sorted(set(value)-{"basis","status"}):
        entry=basis.get(key)
        require(isinstance(entry,dict) and set(entry)=={"kind","reference"} and
                entry["kind"] in {"source_specification","measurement","assumption"} and
                isinstance(entry["reference"],str) and 0<len(entry["reference"])<=2048,
                f"接口 {interface.get('id')} 的 operation.{key} 须标明来源或假设（basis.{key}）")
    if "axis" in value:
        a=value["axis"]
        require(isinstance(a,list) and len(a)==3 and all(finite(x) for x in a) and abs(sum(x*x for x in a)-1)<1e-6,"操作轴须为局部单位向量")
        require(bool(interface.get("site")),"操作轴需要局部坐标点")
    if "range" in value:
        limits=value["range"]
        require("joint" in value and isinstance(limits,list) and len(limits)==2 and all(finite(x) for x in limits) and limits[0]<limits[1],
                f"接口 {interface.get('id')} 的操作行程须绑定本资产驱动该部件的关节；外部物体运动须另列待验证要求，不能伪造自身关节")
    if "joint" in value:
        j=model.joint(value["joint"]).id
        require(model.jnt_type[j] in (2,3) and model.jnt_limited[j],"操作关节必须为有界滑动或转动关节")
        body=model.body(interface["body"]).id; ancestor=body
        while ancestor and ancestor!=model.jnt_bodyid[j]: ancestor=int(model.body_parentid[ancestor])
        require(ancestor==model.jnt_bodyid[j],"操作关节不驱动接口部件")
        if "range" in value:
            lo,hi=value["range"]
            require(lo>=model.jnt_range[j,0]-1e-9 and hi<=model.jnt_range[j,1]+1e-9,"声明行程超过模型限位")
        if "axis" in value:
            # Model-local axis and site-local declaration must describe the same world line direction.
            import mujoco
            data=mujoco.MjData(model); mujoco.mj_forward(model,data)
            axis=data.site_xmat[model.site(interface["site"]).id].reshape(3,3)@np.asarray(value["axis"])
            require(abs(float(axis@data.xaxis[j]))>1-1e-5,"接口操作轴与实际关节轴不一致")
    if "clearance_m" in value: require(finite(value["clearance_m"]) and value["clearance_m"]>=0,"间隙须为非负有限数值")
    if "grasp_geoms" in value:
        names=value["grasp_geoms"]
        require(isinstance(names,list) and names and len(names)<=64 and len(set(names))==len(names),"抓取几何清单无效")
        body=model.body(interface["body"]).id
        for name in names:
            geom=model.geom(name).id; ancestor=int(model.geom_bodyid[geom])
            while ancestor and ancestor!=body: ancestor=int(model.body_parentid[ancestor])
            require(ancestor==body,"抓取区域不属于接口部件")
    if "mate_interface" in value: require(value["mate_interface"] in interface_ids,"配合接口不存在")


def namespace_operation(interface,prefix):
    if "replaces" in interface:interface["replaces"]=[prefix+x for x in interface["replaces"]]
    value=interface.get("operation")
    if value is None: return
    for key in ("joint","mate_interface"):
        if key in value: value[key]=prefix+value[key]
    if "grasp_geoms" in value: value["grasp_geoms"]=[prefix+n for n in value["grasp_geoms"]]
