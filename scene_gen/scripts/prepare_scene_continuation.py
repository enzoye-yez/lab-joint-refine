"""阶段审查升级的离线复制工具：复核旧证据，保留账本，重新审查已有几何。"""

import argparse
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asset_generation.contracts import require
from asset_generation.pipeline import snapshot
from asset_prep import implementation
from asset_prep.storage import atomic_json, digest, file_hash, run_directory, run_lock, safe_path


ALLOWED_CHANGES = {"src/asset_generation/review.py", "src/scene_generation/pipeline.py",
                   "src/scene_generation/review.py"}

BINDING_CHANGES = {'src/runtime_bootstrap.py', 'src/asset_prep/implementation.py',
    'src/asset_generation/local.py', 'src/asset_generation/sandbox.py', 'src/asset_generation/worker.py',
    'src/asset_generation/model.py', 'src/asset_generation/distance.py', 'src/asset_generation/convex.py',
    'src/scene_generation/pipeline.py', 'src/scene_generation/cli.py', 'src/scene_generation/__main__.py', 'src/scene_generation/capabilities.py',
    'src/scene_generation/dispatch.py', 'src/scene_generation/progress.py',
    'src/scene_generation/scene_first.py','src/scene_generation/candidate.py',
    'src/asset_generation/rendering.py','src/asset_generation/checks.py','src/asset_generation/observation.py',
    'src/scene_generation/checks.py','src/scene_generation/preview.py'}


def copy_context_archive(source, target):
    """Preserve every original byte without reproducing hundreds of thousands of files."""
    if not source.exists():return []
    target.mkdir(parents=True,exist_ok=True)
    entries=list(source.iterdir())
    for entry in entries:
        if entry.is_file():shutil.copy2(entry,target/entry.name)
    def pack(entry):
        archive=target/(entry.name+'.zip')
        if (source/(entry.name+'.zip')).exists():
            shutil.copytree(entry,target/entry.name,dirs_exist_ok=True)
            return None  # Preserve overlays over an existing immutable pack.
        temp=archive.with_suffix('.zip.tmp')
        count=size=0
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as packed:
            for path in entry.rglob('*'):
                if path.is_file():
                    packed.write(path,path.relative_to(entry).as_posix())
                    count+=1;size+=path.stat().st_size
        with zipfile.ZipFile(temp) as packed:
            require(packed.testzip() is None,'历史上下文压缩校验失败')
        temp.replace(archive)
        return {'directory':entry.name,'files':count,'original_bytes':size,
                'archive_bytes':archive.stat().st_size,'sha256':file_hash(archive),'crc_verified':True}
    receipts=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(pack,entry) for entry in entries if entry.is_dir()]
        for future in as_completed(futures):
            record=future.result()
            if record is not None:
                receipts.append(record)
                print(f"上下文归档 {len(receipts)}：{record['files']} 个原文件已无损保存",flush=True)
    return receipts


def prepare(project, source_name, target_name, before_file=None, kind="stage_review", *, resume_copy=False):
    source, target = run_directory(project, source_name), run_directory(project, target_name)
    require((source / "state.json").is_file(), "源运行不存在")
    require(resume_copy or not target.exists(), "目标运行已存在；禁止覆盖")
    with run_lock(source), (run_lock(target) if resume_copy else nullcontext()):
        if resume_copy:
            require((target/'state.json').is_file() and file_hash(target/'state.json')==file_hash(source/'state.json'),
                    '仅能恢复账本尚未改变的复制中断；禁止覆盖已经升级或执行的运行')
        from asset_prep.state_archive import read_state, write_state
        state = read_state(source / "state.json")
        from asset_prep import implementation as manifest_source
        archive_only = kind == 'archive_revalidation'
        if archive_only:
            require(before_file is None, '归档重验不冒充旧源码升级；不得提供未采用的旧源码清单')
            before = None
        else:
            require(before_file is not None, '源码升级需要 --before-sources；缺失时不能自动跳过核验')
            before = json.loads(Path(before_file).read_text(encoding="utf-8"))
            legacy = digest(before)
            normalized = manifest_source.implementation_digest(before)
            require(state["identity"]["implementation"] in {legacy, normalized},
                    "升级前源码清单与旧运行身份不匹配")
            before = manifest_source.normalize_manifest(before)
        after = manifest_source.source_manifest(project)
        prefixed = lambda m: {("src/" + key): value for key, value in m.items()}
        changed = None if archive_only else {p for p in prefixed(before).keys() | prefixed(after).keys()
                   if prefixed(before).get(p) != prefixed(after).get(p)}
        options = {"evidence_pipeline":BINDING_CHANGES | ALLOWED_CHANGES | {
                       "src/asset_prep/storage.py", "src/asset_prep/state_archive.py", "src/robot_operation/scene_bridge.py",
                       "src/scene_generation/assembly.py", "src/scene_generation/checkpoints.py",
                       "src/scene_generation/recovery.py","src/scene_generation/prompts.py",
                       "src/scene_generation/gallery.py",
                       "src/asset_generation/contracts.py","src/asset_generation/model.py","src/asset_generation/transport.py",
                       "src/asset_generation/request_context.py","src/asset_generation/check_recovery.py",
                       "src/asset_generation/surface_contact.py","src/asset_generation/material.py","src/asset_prep/errors.py",
                       "src/scene_generation/geometry.py","src/asset_generation/evidence.py",
                       "src/scene_generation/context.py","src/scene_generation/evidence.py","src/scene_generation/evidence_state.py",
                       "src/scene_generation/inspection.py","src/scene_generation/support.py"},
                   "stage_review":ALLOWED_CHANGES,
                   "archive_revalidation":set(),
                   "implementation_binding":BINDING_CHANGES,
                   "main_check_recovery":{"src/asset_generation/distance.py","src/asset_generation/evidence.py",
                       "src/asset_generation/local.py","src/asset_generation/sandbox.py","src/asset_generation/worker.py",
                       "src/scene_generation/geometry.py","src/scene_generation/checks.py","src/scene_generation/inspection.py",
                       "src/scene_generation/contracts.py","src/scene_generation/pipeline.py","src/scene_generation/prompts.py",
                       "src/scene_generation/checkpoint.py","src/scene_generation/recovery.py","src/scene_generation/edge_mounts.py",
                       "src/scene_generation/capabilities.py","src/scene_generation/dispatch.py","src/scene_generation/recovery_geometry.py",
                       "src/scene_generation/faults.py"},
                   "evidence_checks":{"src/asset_generation/distance.py", "src/asset_generation/evidence.py",
                       "src/scene_generation/geometry.py", "src/scene_generation/checks.py", "src/scene_generation/inspection.py",
                       "src/scene_generation/assembly.py", "src/scene_generation/evidence.py", "src/scene_generation/contracts.py",
                       "src/scene_generation/cli.py", "src/scene_generation/pipeline.py"},
                   "local_repair":{"src/asset_prep/storage.py", "src/asset_generation/model.py",
                       "src/scene_generation/contracts.py", "src/scene_generation/table.py", "src/scene_generation/assembly.py",
                       "src/scene_generation/checks.py", "src/scene_generation/prompts.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/cli.py"},
                   "transport_recovery":{"src/asset_generation/contracts.py", "src/asset_generation/model.py",
                       "src/asset_generation/provider_diagnostics.py", "src/asset_generation/transport.py",
                       "src/asset_prep/network.py", "src/scene_generation/gallery.py", "src/scene_generation/pipeline.py"},
                   "scene_capacity":{"src/asset_generation/checks.py", "src/scene_generation/checks.py",
                                     "src/scene_generation/preview.py", "src/scene_generation/runtime.py"},
                   "scene_repair":{"src/scene_generation/geometry.py", "src/scene_generation/pipeline.py",
                                   "src/scene_generation/review.py"},
                   "static_preview":{"src/scene_generation/checks.py", "src/scene_generation/pipeline.py",
                                     "src/scene_generation/review.py", "src/scene_generation/cli.py"},
                   "evidence_recovery":{"src/asset_generation/evidence.py", "src/asset_generation/transport.py",
                       "src/asset_generation/model.py", "src/asset_generation/cli.py", "src/asset_generation/local.py",
                       "src/asset_generation/sandbox.py", "src/asset_generation/worker.py",
                       "src/scene_generation/evidence.py", "src/scene_generation/pipeline.py",
                       "src/scene_generation/cli.py", "src/scene_generation/preview.py"}}
        require(kind in options, "升级类型无效")
        allowed = options[kind]
        require(archive_only or changed and {p.replace("\\", "/") for p in changed} <= allowed,
                "存在阶段审查以外的源码变化，不支持本次升级")
        scene = state["scene"]
        require(not scene.get("pending") or kind in {"evidence_recovery", "transport_recovery", "evidence_pipeline"}, "存在未归档工作区")
        if kind in {'main_check_recovery', 'implementation_binding', 'archive_revalidation', 'evidence_pipeline'}:
            require(state['identity']['limits']['validation_mode']=='static'
                    and scene['status'] in {'needs_check_support','needs_environment','needs_review','paused',
                                            'scene_repair_budget_exhausted'},
                    '主检查升级仅支持尚未通过验收的静态运行')
            old_recovery = scene.pop('check_recovery',None)
            if old_recovery:
                scene.setdefault('check_recovery_history',[]).append(old_recovery)
            # 动作、调用和旧证据保留；旧运行没有范围收据，不能复用为当前通过依据。
            scene.pop('scene_review_resume',None)
        elif kind in {"evidence_recovery", "transport_recovery", "local_repair", "evidence_checks"}:
            require(state["identity"]["limits"]["validation_mode"] == "static"
                    and scene["status"] in {"needs_review","paused"}, "只允许续接静态证据不足或网络暂停运行")
        elif kind == "stage_review":
            require(not scene.get("rounds") and not scene.get("latest"), "阶段审查升级只支持尚未装配的运行")
        else:
            failures = scene.get("latest", {}).get("report", {}).get("checks", [])
            expected_error = "模型规模超限" if kind == "scene_capacity" else "近邻碰撞检查对数超限"
            if kind == "static_preview":
                require(state["identity"]["limits"]["validation_mode"] == "static"
                        and scene["latest"]["report"]["status"] == "pass"
                        and scene["latest"]["review"]["verdict"] == "uncertain",
                        "静态预览升级仅适用于工程检查通过、视觉证据不足的静态运行")
            else:
                require(any(row.get("id") == "scene_build" and expected_error in row.get("detail", "") for row in failures),
                        "场景升级只适用于已确认的对应工程限制")
            require(all(j["status"] in {"candidate_ready", "assembly_ready"} and not j.get("dirty")
                        for j in scene["jobs"].values()), "场景容量升级不改变任何资产审查结果")
        unknown = any(c["status"] == "pending" for c in state["calls"])
        require(not unknown or kind in {"evidence_recovery", "transport_recovery", "evidence_pipeline"}, "存在结果不明请求，须先核对账单")
        for call in state["calls"]:
            if call["status"] == "done":
                require(file_hash(safe_path(source, call["file"])) == call["sha256"], "调用记录摘要不一致")
        stages = {}
        for stage in scene["stages"]:
            require(snapshot(safe_path(source, stage["directory"])) == stage["files"], "归档证据摘要不一致")
            stages[stage["directory"].replace("\\", "/")] = stage
        if archive_only:
            require(scene.get('plan') and set(scene['jobs']) == {j['id'] for j in scene['plan']['jobs']}
                    and all(j['status'] in {'candidate_ready','assembly_ready'} and not j.get('dirty')
                            for j in scene['jobs'].values()),
                    '归档重验只接受全部资产已归档、尚未通过完整场景验收的静态运行')
        recheck = []
        for key, job in scene["jobs"].items():
            if kind=="evidence_pipeline":
                owner=job.get("artifact_stage",job["stage"]).replace("\\", "/")
                require(owner in stages and job.get("artifact"),"待重验资产缺少归档版本")
                artifact=safe_path(source,job["artifact"])
                require(artifact.is_relative_to(safe_path(source,owner)) and (artifact/"model.xml").is_file(),"待重验产物无效")
                for field in ("quality_key","numeric_audit_key","assembly_readiness"):job.pop(field,None)
                job.update(status="review_pending",dirty=True,needs_implementation_recheck=True)
                recheck.append(key)
                continue
            if job["status"] not in {"candidate_ready", "assembly_ready", "needs_review", "failed"}:
                continue
            if job["status"] == "failed" and job.get("feedback", {}).get("numerical", {}).get("status") != "pass":
                continue
            owner = job.get("artifact_stage", job["stage"]).replace("\\", "/")
            require(owner in stages, "资产版本无归档")
            require(job.get("feedback", {}).get("numerical", {}).get("status") == "pass", "待重审资产没有通过工程检查")
            require(job.get("source") and job.get("artifact") and job.get("preview_paths"), "待重审资产缺少模型或图像")
            for relative in [job["source"], *job["preview_paths"]]:
                path = safe_path(source, relative)
                stage_root = safe_path(source, owner)
                belongs = path.is_relative_to(stage_root) and path.relative_to(stage_root).as_posix() in stages[owner]["files"]
                # 只读复查可为原几何产生新图；须由已校验的阶段记录明确绑定原资产。
                if not belongs and relative in job['preview_paths']:
                    for archived in stages.values():
                        evidence_root = safe_path(source, archived['directory'])
                        if (not path.is_relative_to(evidence_root)
                                or path.relative_to(evidence_root).as_posix() not in archived['files']
                                or 'record.json' not in archived['files']):
                            continue
                        evidence = json.loads((evidence_root/'record.json').read_text(encoding='utf-8'))
                        belongs = (evidence.get('kind') in {'asset_review','review_pending'}
                                   and evidence.get('job') == key and evidence.get('artifact_stage') == owner
                                   and evidence.get('artifact') == job['artifact'])
                        if belongs:
                            break
                require(belongs, "待重审模型或图像不属于已验证归档")
            artifact = safe_path(source, job["artifact"])
            require(artifact.is_relative_to(safe_path(source, owner)) and (artifact / "model.xml").is_file(),
                    "待重审资产产物无效")
            layout_recheck=(kind=='main_check_recovery' and
                            next(j['role'] for j in scene['plan']['jobs'] if j['id']==key)!='asset')
            if kind == "stage_review" or kind in {'implementation_binding','archive_revalidation'} or (kind == "evidence_recovery" and job["status"] == "needs_review") or layout_recheck:
                job.update(status="review_pending", dirty=True, artifact_stage=owner)
                if kind in {'implementation_binding','archive_revalidation'}:
                    job['needs_implementation_recheck'] = True
                if kind=='main_check_recovery':
                    job['layout_revalidation']={'reason':'旧版本没有传递权威布局，使用实际位置重新核对原装置；不手工修改几何',
                                                'current_layout':digest(scene['plan']['instances'])}
                recheck.append(key)
        if kind == "scene_repair":
            # 只继承已归档模型决策，不从自由文本重新猜测应修资产。
            actions = [json.loads(safe_path(source, s["directory"]+"/action.json").read_text(encoding="utf-8"))
                       for s in scene["stages"] if s["kind"] == "repair"]
            directive = next((a for a in reversed(actions) if a["jobs"]), None)
            require(directive is not None, "没有已归档的资产修复决策可继承")
            dirty = set(directive["jobs"])
            for key in scene["order"]:
                job = next(j for j in scene["plan"]["jobs"] if j["id"] == key)
                if dirty.intersection(job["depends_on"]):
                    dirty.add(key)
            for key in dirty:
                scene["jobs"][key]["dirty"] = True
            for key in directive["jobs"]:
                scene["jobs"][key]["status"] = "review_pending"
                recheck.append(key)
            scene["repair_directive"] = directive
        require(recheck or kind != "stage_review", "没有可复用的待重审资产")
        # 拒绝复制链接；原状态文件与所有归档均不被修改。
        for path in source.rglob("*"):
            require(not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction()), "运行目录包含链接")
        receipt = {"kind":kind+"_upgrade", "at":time.time(), "source_run":source_name,
                   "source_state_sha256":file_hash(source / "state.json"),
                   "from":state["identity"]["implementation"], "to":digest(after),
                   "changed_files":None if archive_only else sorted(changed),
                   "legacy_source_verified":not archive_only,
                   "before_sources":before, "after_sources":after,
                   "review_pending":recheck, "model_calls_added":0,
                   "unknown_request_preserved":unknown,
                   "pending_workspace_files":snapshot(safe_path(source,scene["pending"]["directory"])) if scene.get("pending") else {},
                   "reason":("旧源码清单未保存，旧实现无法完整核验；仅续接经摘要验证的原资产、账本及预算，"
                             "全部重新量测、复审并装配检查，不继承旧通过证据；这是工程干预后的续接，不是原冻结实验成功"
                             if archive_only else
                             "保留资产、调用账本和历史动作；绑定新实现，重新量测已有资产并复审、重建检查场景；旧通过结论不继承"
                             if kind == 'implementation_binding' else
                             "保留原始资产、场景和调用记录；升级本地保存、实际支撑检查及受限桌面修复，重新构建检查后由模型决策，不手工修改实验模型"
                             if kind == "local_repair" else
                             "原运行保持不变；在副本启用主检查恢复、实际布局传递及重新检查，旧场景结论不继承为通过"
                             if kind == "main_check_recovery" else
                             "保留原模型与账本；启用距离证据一致性核验、专项检查和有界补证重试，重建复核场景，旧结论不继承为通过"
                             if kind == "evidence_checks" else
                             "保留资产、审查、待完成阶段与调用账本；仅升级传输恢复及资产展示，不改变验收结论"
                             if kind == "transport_recovery" else
                             "保留已有几何及账本，启用只读补证据后复审；未知请求保持pending且禁止自动重发"
                             if kind == "evidence_recovery" else
                             "保留已有几何和工程检查，重新执行按阶段审查；旧结论不自动升级为通过"
                             if kind == "stage_review" else
                             "保留所有未验证事项；重新执行工程检查和近景审查，只交付当前静态布局预览"
                             if kind == "static_preview" else
                             "保留旧报告并重新检查；修正场景工程容量或粗筛，并完整继承已归档修复决策")}
        if resume_copy:
            for stage in scene['stages']:
                require(snapshot(safe_path(target,stage['directory']))==stage['files'],'中断副本的阶段不完整或被修改')
            for call in state['calls']:
                if call.get('file'):
                    require(file_hash(safe_path(target,call['file']))==call['sha256'],'中断副本的响应不完整或被修改')
        else:
            shutil.copytree(source, target, ignore=lambda directory,names:
                [name for name in names if name=='.lock' or (Path(directory)==source and name=='request_context')])
        receipt['request_context_archives']=copy_context_archive(source/'request_context',target/'request_context')
        if kind=="evidence_pipeline":
            from asset_generation.contracts import GenerationConfig
            state["identity"]["generation"]=GenerationConfig(**state["identity"].get("generation",{})).identity()
            scene.pop("current_assessment",None)
            receipt["reason"]="保留原资产、场景、账本和执行历史；副本绑定新证据数据流与上下文预算，重新量测复审，不继承旧通过结论；未知请求仍阻断发送"
        state["identity"]["implementation"] = manifest_source.implementation_digest(after)
        state['identity']['binding'] = implementation.from_identity(project, state['identity'])
        implementation.archive_sources(project, target, state['identity']['implementation'])
        state.setdefault("implementation_history", []).append(receipt)
        scene["status"] = "paused"
        if kind not in {"stage_review", "transport_recovery"}:
            scene.update(repair_pending=False, repair_error=None)
            scene.pop("feedback", None)
        write_state(target / "state.json", state)
        atomic_json(target / "stage_review_upgrade.json", receipt)
        result = json.loads((target / "result.json").read_text(encoding="utf-8"))
        result.update(status="paused", jobs={k:v["status"] for k,v in scene["jobs"].items()}, repair_error=None,
                      continuation_note=receipt["reason"])
        if kind == "evidence_pipeline" and "assembly" in result:
            result["assembly"]["quality_accepted"]=False
            result.setdefault("unverified",[]).append("工程版本已变化，旧检查结论仅保留为历史；须重新量测复审")
        if kind == "stage_review":
            result.update(deferred_checks=[], blocking_reasons=[])
        atomic_json(target / "result.json", result)
    return {"run":target_name, "review_pending":recheck, "model_calls_added":0, "unknown_request_preserved":unknown}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--before-sources", type=Path)
    parser.add_argument("--kind", choices=("evidence_pipeline", "archive_revalidation", "implementation_binding", "stage_review", "scene_capacity", "scene_repair", "static_preview", "evidence_recovery", "transport_recovery", "local_repair", "evidence_checks", "main_check_recovery"), default="stage_review")
    args = parser.parse_args()
    print(json.dumps(prepare(PROJECT, args.source, args.target, args.before_sources, args.kind), ensure_ascii=False))
