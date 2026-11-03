"""视觉审查格式重试与几何修复分离，共享原调用账本和预算。"""

from pathlib import Path

from asset_prep.errors import ValidationError, ContextLimitError
from asset_prep.storage import atomic_json
from .contracts import visual_result


def retain_batch_defects(previous, current):
    """The final page decides, having been shown every earlier page's findings.

    A final repair keeps the union of all pages' findings; a final accept stands, and the
    earlier findings it overruled are returned for the call receipt. Final stage validation
    still checks ids, units and the full response; this grants no new edit rights.
    """
    if current.get("verdict") == "accept":
        return current, [f for v in previous for f in v.get("findings", [])]
    result = dict(current)
    reviews = previous + [current]
    result["verdict"] = "repair"
    for field in ("findings", "unverified", "affected_instances", "affected_layout"):
        values = [v[field] for v in reviews if field in v]
        if any(not isinstance(rows,list) or any(not isinstance(r,str) for r in rows) for rows in values):
            raise ValidationError("跨图片批次审查字段无效："+field)
        if values:
            result[field] = list(dict.fromkeys(r for rows in values for r in rows))
    blockers = result.pop("blocking", [])
    if not isinstance(blockers,list) or any(not isinstance(r,str) for r in blockers):
        raise ValidationError("跨图片批次审查字段无效：blocking")
    result.setdefault("unverified", []).extend(blockers)
    return result, []


def ask_review(model, instruction, context, images, phase, destination, validator=visual_result, *, session=None, save=lambda:None):
    session = session if session is not None else {}
    correction = session.get("correction")
    last_call = session.get("call")
    if "review" in session:
        return validator(session["review"]), last_call
    for attempt in range(session.get("attempts",0),3):
        request = context if correction is None else {**context, "review_format_correction": correction}
        value = candidate = None
        last_call = None
        try:
            value, last_call = model.ask(instruction, request, images, phase)
            candidate=value
            if isinstance(value,dict) and session.get("accepted_results"):
                # A small patch changes only answers supplied in this response.
                patch_only=set(value)=={"deferred_results"}
                candidate={**session.get("draft",{}),**value} if patch_only else dict(value)
                rows=value.get("deferred_results",[])
                if isinstance(rows,list):
                    replaced={r.get("id") for r in rows if isinstance(r,dict) and isinstance(r.get("id"),str)}
                    candidate["deferred_results"]=[r for r in session["accepted_results"] if r["id"] not in replaced]+rows
            review = validator(candidate)
        except ContextLimitError as exc:
            session["deferred_reason"] = str(exc); save()
            result = {"verdict":"uncertain","findings":[],"unverified":[str(exc)]}
            if validator is not visual_result:
                result["blocking"] = [str(exc)]
            return result, {**(last_call or {}),"task_deferred":str(exc)}
        except ValidationError as exc:
            session["attempts"]=attempt+1
            if hasattr(exc,"accepted_results"):
                session["accepted_results"]=exc.accepted_results
                session["draft"]=candidate
            correction = {"retained_results":session.get("accepted_results",[]),
                          "item_feedback":getattr(exc,"review_feedback",None),
                          "patch_rule":"已保留同一输入版本的合法事项；可仅返回deferred_results补丁纠正缺失或错误编号。其他字段错误仍须返回完整审查。",
                          "attempt": attempt + 1, "error": str(exc), "previous_response": value,
                          "instruction": "只重新给出视觉审查，不修改资产。findings只列需要修复的缺陷；"
                          "正面观察不要写入findings。accept时findings必须为空；"
                          "若确有缺陷请给repair；证据不足时遵守当前阶段的结论及延后验证规则，未验证项放unverified。"}
            atomic_json(Path(destination)/f"review_format_{attempt + 1}.json",
                        {"call": last_call, "response": value, "error": str(exc)})
            session.update(correction=correction,call=last_call); save()
            continue
        session.update(review=review,call=last_call,attempts=attempt+1); save()
        return review, last_call
    # 不把矛盾响应直接当作通过，也不将格式问题发送给建模器。
    reason = "视觉审查连续三次格式无效，暂挂此审查并继续其他可靠任务：" + (correction or {}).get("error","审查约定尚未恢复")
    result = {"verdict": "uncertain", "findings": [], "unverified": [reason]}
    if validator is not visual_result:
        result["blocking"] = [reason]
    if context.get("deferred_checks"):
        accepted={r["id"]:r for r in session.get("accepted_results",[])}
        result["deferred_results"]=[accepted.get(d["id"],{"id":d["id"],"status":"unresolved",
            "evidence":"框架记录：审查响应尚未合法完成，未认证解决"}) for d in context["deferred_checks"] if d.get("stage")=="scene"]
    return result, last_call
