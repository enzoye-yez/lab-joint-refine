"""Small durable scene description, independent of a full diagnostic report."""
import json
import os


def write(run, result, latest):
    plan = (latest or {}).get("plan") or run.state.get("plan", {})
    requested = run.state.get("plan", {})
    text = ["# 场景交付说明", "", "状态："+result["status"], "",
            result["qualification"], "", "本文件随检查点更新；有候选不等于质量通过。"]
    artifact = result.get("scene_artifact")
    text += ["", "## 查看入口", "", f"- 场景模型：[{artifact}/model.xml]({artifact}/model.xml)" if artifact
             else "- 暂无可加载模型；保留本说明、预算与失败记录。",
             "- [轻量预览索引](preview.json)", "- [完整结果](result.json)", "",
             "## 当前发布布局", ""]
    layout = plan.get("layout")
    if layout:
        text += ["房间、工作台与初始停靠位置（米、弧度）：", "```json",
                 json.dumps(layout,ensure_ascii=False,indent=2),"```"]
        platform = run.env["robot"]["mobile"]
        text += ["", "移动平台来源："+platform["source"], "", *("- "+s for s in platform["approximations"])]
    else:
        text += ["尚未生成多桌计划。" if not plan else "旧版固定工作台布局。"]
    if plan.get("end_effectors"):
        text += ["", "## 实际末端配置", "",
                 "原厂夹爪来自整机资源，不计作模型生成资产。"]
        for arm,binding in plan["end_effectors"].items():
            present_tool=any(i["id"]==binding["instance"] for i in plan.get("instances",[]))
            status=("原厂保留" if binding["mode"]=="stock" else "定制已装配" if present_tool else
                    "适配件缺件，原厂夹爪保留" if binding["mode"]=="adapter" else "定制缺件，未恢复原厂夹爪")
            text.append(f"- {arm}：{binding['mode']}；实例 {binding['instance']}；"
                        f"{status}；{binding['reason']}。")
        text += ["", "声明的整机姿态："+json.dumps(plan.get("robot_posture",{}),ensure_ascii=False)]
    text += ["", "## 资产与缺件", ""]
    present = {i["id"] for i in plan.get("instances",[])} if artifact else set()
    for i in requested.get("instances",[]):
        record = run.state["jobs"].get(i["job"],{})
        text.append(f"- {i['id']}：{'已装配' if i['id'] in present else '未装配'}；任务 {i['job']}；"
                    f"生成状态 {record.get('status','待生成')}；支撑 {i.get('support_instance',i.get('table_id',i['placement']))}。")
    staged=result.get("assembly",{}).get("ready_unassembled_jobs",{})
    if staged:
        text += ["", "已保存且通过加载检查、等待下一次装配的资产（当前场景尚未包含）：", ""]
        text += [f"- {key}：[{path}/model.xml]({path}/model.xml)" for key,path in staged.items()]
    if run.state.get("blocked_jobs"):
        text += ["", "## 暂挂与恢复条件", ""]
        for key,entry in run.state["blocked_jobs"].items():
            text.append("- %s：%s；恢复条件：%s。" % (key,entry.get("reason",entry.get("kind")),entry.get("recovery_condition","新修复指令或依赖变化；预算不重置")))
    if requested != plan:
        text += ["", "当前提出的修复布局与已发布版本不同；本说明中的布局对应上方模型。待执行计划保存在运行状态中。"]
    advice = (latest or {}).get("report",{}).get("layout_assessment")
    text += ["", "## 操作性与检查范围", ""]
    if advice:
        metrics=advice.get("compactness",{})
        if metrics:
            text += ["布局紧凑性：工作台数量 %s；桌面总面积 %.2f 平方米；房间面积 %.2f 平方米。" %
                     (metrics["table_count"],metrics["table_area_m2"],metrics["room_area_m2"]),metrics["qualification"]]
            text += ["- %s：占用率上界 %.1f%%；用途：%s" % (r["table"],100*r["occupancy_upper_bound"],r["reason"]) for r in metrics["tables"]]
        text += [advice["qualification"],f"初始整机保守包络无障碍：{advice['home_clear']}；可连通采样单元：{advice['reachable_cells']}。"]
        for row in advice["instances"]:
            text.append(f"- {row['instance']}：{row['status']}（粗略建议，不是操作验收）。")
        text += ["", "已建模活动关节的运动学抽查（不检查完整开合空间）："]
        for row in advice.get("articulations",[]):
            text.append(f"- {row['joint']}：运动学变化={row['kinematic_motion']}，限位={row['range']}。")
        if not advice.get("articulations"):
            text.append("- 尚无非机器人活动关节证据；固定资产无需凭空增加关节，需活动却未建模的结构仍由审查指出。")
    else:
        text += ["尚无移动通道及粗略操作距离检查结果；不能视为通过。"]
    text += ["", *("- "+str(v) for v in result.get("unverified",[])),
             "", "## 来源与预算", "", "```json",
             json.dumps({"provenance":result.get("provenance"),"budget":result.get("budget"),
                         "stop_reason":result.get("stop_reason")},ensure_ascii=False,indent=2),"```"]
    target = run.root/"scene_document.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(text)+"\n",encoding="utf-8")
    os.replace(temporary,target)
