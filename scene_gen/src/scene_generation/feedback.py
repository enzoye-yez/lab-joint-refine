"""规划器失败是有版本约束的数据，不是可以执行的修复命令。"""

from asset_generation.contracts import require
from asset_prep.storage import digest


def planning_feedback(value, scene_hash, plan):
    require(isinstance(value,dict) and set(value) == {"scene_sha256","failures"}, "规划反馈字段无效")
    require(value["scene_sha256"] == scene_hash, "规划反馈来自过期场景，拒绝修复新版本")
    require(isinstance(value["failures"],list) and 1 <= len(value["failures"]) <= 100, "失败列表无效")
    known = {i["id"] for i in plan["instances"]}
    rows = []
    for number, failure in enumerate(value["failures"]):
        require(isinstance(failure,dict) and set(failure) == {"kind","instances","detail","evidence"}, "失败项字段无效")
        require(isinstance(failure['kind'],str) and failure["kind"] in {"collision","unreachable","grasp","mechanism","fit","support"}, "未知失败类型")
        require(isinstance(failure["instances"],list) and failure["instances"] and all(isinstance(i,str) for i in failure['instances'])
                and set(failure["instances"]) <= known, "失败对象不属于场景")
        require(all(isinstance(failure[k],str) and 0 < len(failure[k]) <= 12000 for k in ("detail","evidence")), "反馈须提供有限的失败描述和证据")
        rows.append({"id":"planner:"+failure["kind"]+':'+str(number),"status":"fail",**failure})
    return {"status":"fail","checks":rows,"coverage":["外部规划器反馈，未经框架重复认证"],"feedback_sha256":digest(value)}
