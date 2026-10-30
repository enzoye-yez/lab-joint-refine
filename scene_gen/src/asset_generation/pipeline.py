"""有界单资产/配合组件组闭环；不运行正式机器人任务。"""

import shutil
from dataclasses import replace
from pathlib import Path

from asset_prep.errors import PrepError, StaleEvidence, UncertainRequest, ValidationError
from asset_prep.storage import Store, atomic_json, digest, file_hash, safe_path
from .contracts import author_result, require
from .review import ask_review
from .inputs import image_record, public_bundle
from .model import GenerationModel
from .prompts import AUTHOR, REVIEW


def implementation_identity(project):
    files = {}
    for package in ("asset_generation", "asset_prep"):
        for path in sorted((Path(project)/"src"/package).glob("*.py")):
            files[f"{package}/{path.name}"] = file_hash(path)
    return digest(files)


def snapshot(root):
    require(not Path(root).is_symlink(), "资产版本目录不能为链接")
    result = {}
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "运行产物不能包含符号链接")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def clean_pending(root, pending):
    """只移除账本明确标记、尚未提交的本轮目录，保留所有已提交版本。"""
    relative = pending["directory"]
    require(relative == f"revisions/{pending['iteration']:04d}", "待恢复工作区记录无效")
    target = safe_path(root, relative)
    require(target.resolve().is_relative_to(Path(root).resolve()) and not target.is_symlink(), "清理范围越界")
    if target.exists():
        require(not any(p.is_symlink() for p in target.rglob("*")), "待恢复目录含链接，拒绝清理")
        if pending.get("source_sha256"):
            script = target/"model.py"
            if not script.is_file() or file_hash(script) != pending["source_sha256"]:
                raise StaleEvidence("未完成脚本被外部修改；保留现场，不自动清理")
        shutil.rmtree(target)


def execute(bundle, profile, config, generation, keys, run_root, transport, sandbox, project,
            *, resume=False, retry_uncertain=False, provenance="model_generated"):
    # 必须先验证所选执行环境，避免付费生成后才发现完全无法构建。
    backend = sandbox.preflight()
    identity = {"input": bundle["identity"], "profile": profile, "generation": generation.identity(),
                "provider": config.provider, "base_url": config.base_url, "model": config.model,
                "output_token_parameter": config.output_token_parameter, "backend": backend,
                "implementation": implementation_identity(project), "provenance": provenance}
    ledger_config = replace(config, max_model_calls=generation.max_model_calls,
                            max_reserved_tokens=generation.max_reserved_tokens)
    store = Store(run_root, identity, ledger_config, resume=resume, retry_uncertain=retry_uncertain)
    if any(call["status"] == "pending" for call in store.state["calls"]) and not retry_uncertain:
        raise UncertainRequest("账本存在结果不明的请求；核对后显式重试，不能通过改变上下文绕过")
    state = store.state.setdefault("generation", {"status": "running", "revisions": []})
    for entry in state["revisions"]:
        if snapshot(safe_path(run_root, entry["directory"])) != entry["files"]:
            raise StaleEvidence("已提交的资产版本被修改，拒绝复用检查证据")
    for relative, expected in store.state.get("generation_outputs", {}).items():
        path = safe_path(run_root, relative)
        if not path.is_file() or file_hash(path) != expected:
            raise StaleEvidence("生成输入快照或验收配置被修改，拒绝复用")
    resume_numerical=bool(state["revisions"] and (state["status"] in {"needs_check_support","needs_environment"} or
        state.get("pending",{}).get("numerical_recovery")))
    resume_review = (state.get("pending") or {}).get("review_checkpoint")
    if resume_review:
        actual = snapshot(safe_path(run_root,state["pending"]["directory"]))
        expected = resume_review["files"]
        if any(actual.get(name) != sha for name,sha in expected.items()) or any(
                name not in expected and name != "visual_review.json" and not name.startswith("review_format_")
                for name in actual):
            raise StaleEvidence("待审查资产检查点发生变化，保留现场并停止恢复")
    elif state.get("pending"):
        clean_pending(run_root, state["pending"])
        state.pop("pending")
        store.save()
    if state["status"] == "candidate_ready":
        return summary(store)
    model = GenerationModel(config, generation, keys, store, transport)
    previous = None
    feedback = None
    preview_images = []
    for entry in state["revisions"]:
        if entry.get("source"):
            previous = safe_path(run_root, entry["source"]).read_text(encoding="utf-8")
        feedback = entry["feedback"]
        preview_images = load_previews(run_root, entry, generation)
    atomic_json(Path(run_root)/"input.json", public_bundle(bundle))
    atomic_json(Path(run_root)/"checks.json", profile)
    store.state["generation_outputs"] = {name: file_hash(Path(run_root)/name) for name in ("input.json", "checks.json")}
    store.save()
    try:
        for iteration in range(len(state["revisions"])+1, generation.max_iterations+1):
            state["status"] = "running"
            relative = f"revisions/{iteration:04d}"
            destination = safe_path(run_root, relative)
            if resume_review:
                require(state["pending"]["directory"] == relative,"审查检查点轮次与账本不一致")
                entry = dict(resume_review["entry"])
                report = resume_review["report"]
                previous = safe_path(run_root,entry["source"]).read_text(encoding="utf-8")
                preview_images = load_previews(run_root,entry,generation)
            else:
                require(not destination.exists(), "发现未归档的版本目录；保留现场，请使用新运行名称")
                state["pending"] = {"iteration": iteration, "directory": relative, "numerical_recovery":resume_numerical}
                store.save()
                destination.mkdir(parents=True)
                entry = {"iteration": iteration, "directory": relative, "source": None, "preview_paths": []}
            try:
                if resume_numerical and not resume_review:
                    from .check_recovery import recover_checks
                    latest=state["revisions"][-1]
                    entry.update(source=latest["source"],preview_paths=list(latest["preview_paths"]),rechecked_from=latest["directory"])
                    artifact=safe_path(run_root,latest["source"]).parent/"artifact"
                    report=latest["feedback"]["numerical"]
                    if report["status"]=="error" and hasattr(sandbox,"recheck_asset"):
                        from .check_report import checked_report,checked_previews
                        report,render_root=sandbox.recheck_asset(artifact,destination/"recheck",bundle["asset_ids"],profile)
                        report=checked_report(report,domain="asset",mode=profile.get("validation_mode","full"),require_renders=True)
                        report,preview_images=checked_previews(report,render_root,
                            lambda directory,name:image_record(safe_path(directory,name),Path(run_root),generation))
                        entry["preview_paths"]=[im["path"] for im in preview_images]
                    if report["status"]!="error":
                        report=recover_checks(sandbox,artifact,report,Path(run_root)/"asset_check_recovery",
                            implementation=store.state["identity"].get("implementation"),contract=profile)
                    atomic_json(destination/"checks.json",report)
                elif not resume_review:
                    from .authoring import capabilities
                    value, call = model.ask(AUTHOR, {"authoring_capabilities":capabilities(generation,backend),"input": public_bundle(bundle), "profile": profile,
                        "current_model_py": previous, "feedback": feedback, "iteration": iteration},
                        bundle["images"]+preview_images, "author")
                    source = author_result(value, previous, generation.max_script_bytes)
                    script = destination/"model.py"
                    script.write_text(source, encoding="utf-8")
                    state["pending"]["source_sha256"] = file_hash(script)
                    store.save()
                    previous = source
                    entry.update(source=f"{relative}/model.py", author=call)
                    report, render_root = sandbox.build_and_check(script, destination, bundle["asset_ids"], profile)
                    from .check_report import checked_report, checked_previews
                    report = checked_report(report,domain='asset',mode=profile.get('validation_mode','full'),require_renders=True)
                    from .check_report import incomplete
                    if any(incomplete(r) for r in report.get("checks",[])) and report["status"]!="error":
                        from .check_recovery import recover_checks
                        report=recover_checks(sandbox,destination/"artifact",report,Path(run_root)/"asset_check_recovery",
                            implementation=store.state["identity"].get("implementation"),contract=profile)
                    report,preview_images = checked_previews(report,render_root,
                        lambda directory,name:image_record(safe_path(directory,name),Path(run_root),generation))
                    atomic_json(destination/"checks.json", report)
                    entry["preview_paths"] = [im["path"] for im in preview_images]
                feedback = {"numerical": report}
                # Partial rows without a confirmed failure are measurement/probe gaps the
                # author cannot fix; review the asset instead of stopping, keep the gaps.
                unverified = [r["id"] for r in report.get("checks", []) if r.get("status") == "partial"]
                reviewable = report["status"] == "pass" or (report["status"] == "partial" and not any(
                    r.get("status") in {"fail", "error"} for r in report.get("checks", [])))
                if reviewable and preview_images:
                    if not resume_review:
                        state["pending"]["review_checkpoint"] = {
                            "entry":dict(entry),"report":report,"files":snapshot(destination)}
                        store.save()
                    review, review_call = ask_review(model, REVIEW, {"input": public_bundle(bundle), "checks": report},
                                                  bundle["images"]+preview_images, "visual_review", destination)
                    atomic_json(destination/"visual_review.json", review)
                    entry["review_call"] = review_call
                    feedback["visual"] = review
                    if review["verdict"] == "accept":
                        state["status"] = "candidate_ready"
                        entry["unverified_checks"] = unverified
                    elif review["verdict"] == "uncertain":
                        state["status"] = "needs_review"
                elif report["status"] == "error":
                    state["status"] = "needs_environment"
                elif report["status"] == "partial":
                    state["status"] = "needs_check_support"
            except ValidationError as exc:
                feedback = {"generation_error": str(exc)}
            entry["feedback"] = feedback
            resume_review = None
            resume_numerical = False
            atomic_json(destination/"revision.json", {**entry, "provenance": provenance})
            entry["files"] = snapshot(destination)
            state["revisions"].append(entry)
            state.pop("pending", None)
            store.save()
            if state["status"] in {"candidate_ready", "needs_review", "needs_environment", "needs_check_support"}:
                break
        else:
            state["status"] = "repair_budget_exhausted"
        store.save()
        result = summary(store)
        atomic_json(Path(run_root)/"result.json", result)
        return result
    except (PrepError, KeyboardInterrupt, OSError) as original:
        state["status"] = "paused"
        try:
            store.save()
        except Exception as secondary:
            original.add_note("保存暂停状态也发生异常："+type(secondary).__name__)
        raise


def load_previews(root, entry, generation):
    return [image_record(safe_path(root, relative), Path(root), generation) for relative in entry["preview_paths"]]


def summary(store):
    state = store.state["generation"]
    latest = state["revisions"][-1] if state["revisions"] else None
    feedback = (latest or {}).get("feedback", {})
    unverified = (feedback.get("numerical", {}).get("coverage", []) + feedback.get("visual", {}).get("unverified", [])
                  + ["检查未完成（量测或探测能力不足）："+c for c in (latest or {}).get("unverified_checks", [])])
    return {"status": state["status"], "provenance": store.state["identity"]["provenance"],
            "iterations": len(state["revisions"]), "latest_revision": latest["directory"] if latest else None,
            "budget": store.totals(), "asset_ids": store.state["identity"]["input"]["asset_ids"],
            "qualification": "候选资产；通过已列检查不等于完整尺寸、动力学或机器人任务验收",
            "unverified": unverified, "last_feedback": feedback}
