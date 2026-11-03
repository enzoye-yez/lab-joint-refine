"""Small durable scene description, independent of a full diagnostic report."""
import json
import os


def write(run, result, latest):
    plan = (latest or {}).get("plan") or run.state.get("plan", {})
    requested = run.state.get("plan", {})
    text = ["# Scene delivery notes", "", "Status: "+result["status"], "",
            result["qualification"], "", "This file is updated at each checkpoint; having a candidate does not mean quality passed."]
    artifact = result.get("scene_artifact")
    text += ["", "## Entry points", "", f"- Scene model: [{artifact}/model.xml]({artifact}/model.xml)" if artifact
             else "- No loadable model yet; this note, the budget and failure records are kept.",
             "- [Lightweight preview index](preview.json)", "- [Full result](result.json)", "",
             "## Currently published layout", ""]
    layout = plan.get("layout")
    if layout:
        text += ["Room, workbenches and initial docking poses (meters, radians):", "```json",
                 json.dumps(layout,ensure_ascii=False,indent=2),"```"]
        platform = run.env["robot"]["mobile"]
        text += ["", "Mobile platform source: "+platform["source"], "", *("- "+s for s in platform["approximations"])]
    else:
        text += ["No multi-table plan generated yet." if not plan else "Legacy fixed-workbench layout."]
    if plan.get("end_effectors"):
        text += ["", "## Actual end-effector configuration", "",
                 "Stock grippers come from the robot model resources and are not counted as model-generated assets."]
        for arm,binding in plan["end_effectors"].items():
            present_tool=any(i["id"]==binding["instance"] for i in plan.get("instances",[]))
            status=("stock kept" if binding["mode"]=="stock" else "custom assembled" if present_tool else
                    "adapter missing, stock gripper kept" if binding["mode"]=="adapter" else "custom part missing, stock gripper not restored")
            text.append(f"- {arm}: {binding['mode']}; instance {binding['instance']}; "
                        f"{status}; {binding['reason']}.")
        text += ["", "Declared robot posture: "+json.dumps(plan.get("robot_posture",{}),ensure_ascii=False)]
    text += ["", "## Assets and missing parts", ""]
    present = {i["id"] for i in plan.get("instances",[])} if artifact else set()
    for i in requested.get("instances",[]):
        record = run.state["jobs"].get(i["job"],{})
        text.append(f"- {i['id']}: {'assembled' if i['id'] in present else 'not assembled'}; job {i['job']}; "
                    f"generation status {record.get('status','pending')}; support {i.get('support_instance',i.get('table_id',i['placement']))}.")
    staged=result.get("assembly",{}).get("ready_unassembled_jobs",{})
    if staged:
        text += ["", "Assets saved and passing the load check, waiting for the next assembly (not yet in the current scene):", ""]
        text += [f"- {key}: [{path}/model.xml]({path}/model.xml)" for key,path in staged.items()]
    if run.state.get("blocked_jobs"):
        text += ["", "## Blocked jobs and resume conditions", ""]
        for key,entry in run.state["blocked_jobs"].items():
            text.append("- %s: %s; resume condition: %s." % (key,entry.get("reason",entry.get("kind")),entry.get("recovery_condition","new repair instruction or dependency change; budget is not reset")))
    if requested != plan:
        text += ["", "The currently proposed repair layout differs from the published version; the layout in this note matches the model above. The pending plan is stored in the run state."]
    advice = (latest or {}).get("report",{}).get("layout_assessment")
    text += ["", "## Operability and check scope", ""]
    if advice:
        metrics=advice.get("compactness",{})
        if metrics:
            text += ["Layout compactness: %s workbenches; total tabletop area %.2f m^2; room area %.2f m^2." %
                     (metrics["table_count"],metrics["table_area_m2"],metrics["room_area_m2"]),metrics["qualification"]]
            text += ["- %s: occupancy upper bound %.1f%%; purpose: %s" % (r["table"],100*r["occupancy_upper_bound"],r["reason"]) for r in metrics["tables"]]
        text += [advice["qualification"],f"Conservative whole-robot envelope clear at the initial pose: {advice['home_clear']}; connected sampled cells: {advice['reachable_cells']}."]
        for row in advice["instances"]:
            text.append(f"- {row['instance']}: {row['status']} (rough advice, not operation acceptance).")
        text += ["", "Kinematic spot check of modeled movable joints (the full opening sweep is not checked):"]
        for row in advice.get("articulations",[]):
            text.append(f"- {row['joint']}: kinematic_motion={row['kinematic_motion']}, range={row['range']}.")
        if not advice.get("articulations"):
            text.append("- No evidence of non-robot movable joints yet; fixed assets need no invented joints, and structures that should move but are not modeled are still flagged by review.")
    else:
        text += ["No mobile corridor or rough operating-distance check results yet; this cannot be treated as passing."]
    text += ["", *("- "+str(v) for v in result.get("unverified",[])),
             "", "## Provenance and budget", "", "```json",
             json.dumps({"provenance":result.get("provenance"),"budget":result.get("budget"),
                         "stop_reason":result.get("stop_reason")},ensure_ascii=False,indent=2),"```"]
    target = run.root/"scene_document.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(text)+"\n",encoding="utf-8")
    os.replace(temporary,target)
