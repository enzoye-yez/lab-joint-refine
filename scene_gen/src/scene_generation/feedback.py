"""Planner failures are version-bound data, not executable repair commands."""

from asset_generation.contracts import require
from asset_prep.storage import digest


def planning_feedback(value, scene_hash, plan):
    require(isinstance(value,dict) and set(value) == {"scene_sha256","failures"}, "Invalid planner feedback fields")
    require(value["scene_sha256"] == scene_hash, "Planner feedback comes from a stale scene; refusing to repair the new version")
    require(isinstance(value["failures"],list) and 1 <= len(value["failures"]) <= 100, "Invalid failures list")
    known = {i["id"] for i in plan["instances"]}
    rows = []
    for number, failure in enumerate(value["failures"]):
        require(isinstance(failure,dict) and set(failure) == {"kind","instances","detail","evidence"}, "Invalid failure item fields")
        require(isinstance(failure['kind'],str) and failure["kind"] in {"collision","unreachable","grasp","mechanism","fit","support"}, "Unknown failure kind")
        require(isinstance(failure["instances"],list) and failure["instances"] and all(isinstance(i,str) for i in failure['instances'])
                and set(failure["instances"]) <= known, "Failure instances do not belong to the scene")
        require(all(isinstance(failure[k],str) and 0 < len(failure[k]) <= 12000 for k in ("detail","evidence")), "Feedback must provide a bounded failure detail and evidence")
        rows.append({"id":"planner:"+failure["kind"]+':'+str(number),"status":"fail",**failure})
    return {"status":"fail","checks":rows,"coverage":["external planner feedback, not re-certified by the framework"],"feedback_sha256":digest(value)}
