"""开发者合成演示：不读取基准、不调用模型、不作为正式实验结果。"""
import argparse
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"tests"))
from operation_fixtures import fixture,task_graph
from robot_operation.common import Settings,atomic_json,digest,identity,new_directory
from robot_operation.scene import prepare
from robot_operation.workflow import run
from robot_operation.recording import replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=new_directory(args.output)
    source=fixture(root/"source")
    settings=Settings(settle_seconds=.5)
    converted=prepare(source,root/"dynamic",identity(source),settings)
    task,graph=task_graph(converted["scene_sha256"])
    initial=task["initial"]
    goal={"kind":"joint_range","joint":"robot__left_x","range":[.07,.09]}
    task["stages"]=[{"id":"navigate","source":"developer synthetic demonstration","depends_on":[],
        "goal":{"kind":"base_stopped","duration":.1}},
        {"id":"move","source":"developer synthetic demonstration","depends_on":["navigate"],"goal":goal}]
    task["final"]=goal
    graph["nodes"]=[{"id":"navigate","operation":"navigate","stage":"navigate","parameters":{
        "goal":[.04,0,0],"bounds":[[-.5,-.5],[.5,.5]],"transport":{"arms":["left","right"],
        "positions":{"robot__left_x":0.,"robot__right_x":0.}}},"precondition":initial,
        "completion":task["stages"][0]["goal"],"timeout":8,"on_success":"move","on_failure":"failed"},
        {"id":"move","operation":"joint_move","stage":"move","parameters":{"arms":["left"],
        "positions":{"robot__left_x":.08}},"precondition":initial,"completion":goal,"timeout":4,
        "on_success":"success","on_failure":"failed"}]
    graph["entry"]="navigate"; graph["task_sha256"]=digest(task)
    atomic_json(root/"task.json",task); atomic_json(root/"graph.json",graph)
    result=run(root/"dynamic",converted["scene_sha256"],task,graph,root/"run",settings)
    report={"provenance":"developer_fixture","network_calls":0,"result":result}
    if result["status"]=="success": report["replay"]=replay(root/"run")
    atomic_json(root/"demo_report.json",report)
    print(str(root/"demo_report.json"))
    return 0 if result["status"]=="success" else 2


if __name__=="__main__": raise SystemExit(main())
