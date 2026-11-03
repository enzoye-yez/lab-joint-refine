"""Write a checkpoint before review; recovery reuses only build results whose digests match and never declares a pass on the model's behalf."""

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
        # An old-version or not-yet-built workspace is never assumed to be a valid asset.
        run.commit("abandoned", {"reason": "interrupted workspace kept; an unfinished build does not enter the scene"})
        return
    actual = snapshot(safe_path(run.root, pending["directory"]))
    expected = checkpoint["files"]
    if any(actual.get(name) != sha for name, sha in expected.items()):
        raise StaleEvidence("review checkpoint model, checks or images were modified; reuse forbidden")
    extras = set(actual) - set(expected)
    if any(not (name == "review.json" or name == "record.json" or
                name.startswith(("review_format_", "evidence_")) or
                checkpoint['kind']=='scene_check' and (name in {'checks.json','checks_before_recovery.json'}
                                                       or name.startswith('main_check_recovery/')))
           for name in extras):
        raise StaleEvidence("review checkpoint contains unregistered build outputs; reuse forbidden")
    if checkpoint["job_versions"] != {key: job["stage"] for key, job in run.state["jobs"].items()}:
        raise StaleEvidence("dependency asset version changed after review was interrupted; the old checkpoint cannot be reused")
    record = deepcopy(checkpoint["record"])
    directory = pending["directory"]
    if checkpoint["kind"] == "asset":
        record.update(status="review_pending", stage=directory, dirty=True, interrupted_review=True)
        run.state["jobs"][record["job"]] = record
    elif checkpoint["kind"] in {"scene","scene_check"}:
        run.state["scene_review_resume"] = {**record, "stage": directory}
    else:
        raise StaleEvidence("unknown review checkpoint type")
    # Persist together with clearing pending, avoiding a window where the stage is archived but the asset state is not yet recorded.
    run.commit("review_pending", record)
