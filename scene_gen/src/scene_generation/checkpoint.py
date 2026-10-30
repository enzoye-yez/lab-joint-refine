"""审查前写入检查点；恢复只复用摘要匹配的构建结果，不替模型宣布通过。"""

from copy import deepcopy

from asset_generation.pipeline import snapshot
from asset_prep.errors import StaleEvidence
from asset_prep.storage import safe_path


def save_review_checkpoint(run, kind, record):
    pending = run.state["pending"]
    pending["review_checkpoint"] = {
        "kind": kind, "record": deepcopy(record),
        "files": snapshot(safe_path(run.root, pending["directory"])),
        "job_versions": {key: job["stage"] for key, job in run.state["jobs"].items()},
    }
    run.store.save()


def recover_review_checkpoint(run):
    pending = run.state["pending"]
    checkpoint = pending.get("review_checkpoint")
    if not checkpoint:
        # 旧版本或尚未构建完成的现场不猜测为有效资产。
        run.commit("abandoned", {"reason": "中断工作区保留，未完成构建不进入场景"})
        return
    actual = snapshot(safe_path(run.root, pending["directory"]))
    expected = checkpoint["files"]
    if any(actual.get(name) != sha for name, sha in expected.items()):
        raise StaleEvidence("审查检查点的模型、检查或图像被修改，禁止复用")
    extras = set(actual) - set(expected)
    if any(not (name == "review.json" or name == "record.json" or
                name.startswith(("review_format_", "evidence_")) or
                checkpoint['kind']=='scene_check' and (name in {'checks.json','checks_before_recovery.json'}
                                                       or name.startswith('main_check_recovery/')))
           for name in extras):
        raise StaleEvidence("审查检查点出现未登记的构建产物，禁止复用")
    if checkpoint["job_versions"] != {key: job["stage"] for key, job in run.state["jobs"].items()}:
        raise StaleEvidence("审查中断后依赖资产版本变化，不能复用旧检查点")
    record = deepcopy(checkpoint["record"])
    directory = pending["directory"]
    if checkpoint["kind"] == "asset":
        record.update(status="review_pending", stage=directory, dirty=True, interrupted_review=True)
        run.state["jobs"][record["job"]] = record
    elif checkpoint["kind"] in {"scene","scene_check"}:
        run.state["scene_review_resume"] = {**record, "stage": directory}
    else:
        raise StaleEvidence("未知审查检查点类型")
    # 与解除 pending 同一次持久化，避免已归档而资产状态尚未登记的窗口。
    run.commit("review_pending", record)
