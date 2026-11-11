"""Trusted read-only capability registry shared by asset and scene evidence planning."""
REGISTRY={}
def register(capability):
    if capability.id in REGISTRY: raise ValueError("duplicate inspection capability")
    REGISTRY[capability.id]=capability
    return capability

class MaterialInspection:
    id="material"
    description="Actual material connectivity and internal intersections, same artifact and selected configuration."
    def propose(self,report,scope):
        from .check_report import incomplete
        if scope!="asset":return
        for row in report.get("checks",[]):
            if row["id"].split(":")[-1]=="material_integrity" and incomplete(row):
                yield {"kind":self.id,"tolerance_m":row.get("tolerance_m",.0005),**({"configuration":row["configuration"]} if row.get("configuration") else {})}
    def validate(self,request,scope):
        from .contracts import require
        require(set(request) <= {"kind","instance","configuration","tolerance_m"},"invalid material request")
        import math
        require(type(request.get("tolerance_m",.0005)) in (int,float) and math.isfinite(request.get("tolerance_m",.0005)) and 0<=request.get("tolerance_m",.0005)<=.02,"invalid material tolerance")
        require(scope=="asset" and "instance" not in request or scope=="scene" and isinstance(request.get("instance"),str),
                "material inspection needs a scene instance")
    def inspect(self,model,data,request,root,scope,output,number):
        from .material import integrity
        from scene_generation.recovery_geometry import RecoveryQuery
        from .contracts import require
        if scope=="scene":
            require(any((model.body(b).name or "").startswith(request["instance"]+"__") for b in range(1,model.nbody)),
                    "unknown material inspection instance")
        result=integrity(model,data,instance=request.get("instance"),tolerance=request.get("tolerance_m",.0005),query=RecoveryQuery(model,data))
        return {"kind":self.id,"request":request,"checks":[result],"complete":result["complete"],
                "evidence_status":result["status"]}

register(MaterialInspection())
