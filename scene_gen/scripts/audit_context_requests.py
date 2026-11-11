"""Offline request replay. Never calls a provider, builder, checker or experiment runner."""
import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asset_prep.storage import atomic_json, file_hash, digest
from asset_generation.contracts import GenerationConfig
from asset_generation.request_context import RequestContext, measure, PINNED, unpack_tables
from scene_generation.contracts import SceneConfig
from scene_generation.inputs import catalog
from scene_generation.pipeline import SceneRun


class Captured(BaseException):
    pass


class Capture:
    def ask(self, instruction, context, images, phase):
        self.request = instruction, context, images, phase
        raise Captured()


def replay(project, batch_name, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    batch_root = project / "runs" / "batches" / batch_name
    batch = json.loads((batch_root / "batch.json").read_text("utf-8"))
    if output.is_relative_to(project / "runs") or not output.is_relative_to(project):
        raise ValueError("Replay output must be inside project and outside formal runs")
    generation = GenerationConfig.load(batch_root / "config" / "generation.json")
    monitor_path = project / ".cache" / batch_name / "monitor.json"
    monitor = json.loads(monitor_path.read_text("utf-8")) if monitor_path.exists() else {}
    summaries = []
    for entry in batch["entries"]:
        if entry["protocol"] not in {"NP-001", "NP-043", "NP-088", "NP-002"}:
            continue
        root = project / "runs" / entry["run"]
        state_path = root / "state.json"
        before = file_hash(state_path)
        from asset_prep.state_archive import read_state
        original = read_state(state_path)
        identity = original["identity"]
        run = SceneRun.__new__(SceneRun)
        run.root, run.state = root, deepcopy(original["scene"])
        run.env, run.contract = identity["environment"], identity["contract"]
        run.generation, run.limits = generation, SceneConfig(**identity["limits"])
        run.check_pause = lambda: None
        # The live construction path may begin a stage; this replay never writes it.
        run.begin = lambda kind: output / entry["protocol"] / "unexecuted_stage"
        run.sandbox = SimpleNamespace()
        run.model = Capture()
        try:
            if run.state.get("repair_pending"):
                run.route_repair()
            else:
                run.catalogue = catalog(entry["package"], generation, entry["pdf"])
                assert run.catalogue["identity"] == identity["catalogue"], "Archived inputs changed"
                jobs = {j["id"]: j for j in run.state["plan"]["jobs"]}
                missing = next(k for k in run.state["order"] if k not in run.state["jobs"])
                run.generate_job(jobs[missing], draft=True)
        except Captured:
            pass
        else:
            raise AssertionError("No pending request captured")
        instruction, context, images, phase = run.model.request
        destination = output / entry["protocol"]
        view = RequestContext(destination, identity, instruction, context, images, phase, generation)
        assert view.after["required"] <= generation.max_context_tokens
        reason = monitor.get("protocol_outcomes", {}).get(entry["protocol"], {}).get("reason", "")
        match = re.search(r"estimated\s*(\d+)", reason)
        if match:
            assert view.before["required"] == int(match.group(1)), "Reconstructed request differs from failed request"
        if "dependencies" in context:
            restored = unpack_tables(view.context["dependencies"])
            for key, dep in context["dependencies"].items():
                assert restored[key]["manifest"] == dep["manifest"]
                assert restored[key]["instances"] == dep["instances"]
                assert restored[key]["geometry"].get("interfaces") == dep["geometry"].get("interfaces")
                assert restored[key]["source_requirements"] == dep["source_requirements"]
        for k in PINNED & context.keys():
            assert view.context[k] == context[k], "Mandatory field changed: " + k
        assert {digest(i["data_url"]) for i in images} == {digest(i["data_url"]) for i in view.images}
        assert file_hash(state_path) == before, "Formal state mutated"
        atomic_json(destination / "captured_context.json", context)
        atomic_json(destination / "projected_context.json", view.context)
        atomic_json(destination / "request_metadata.json", {
            "instruction": instruction, "phase": phase,
            "images": [{k: v for k, v in im.items() if k != "data_url"} for im in images]})
        rows = view.context.get("_context", {}).get("check_index", [])
        summary = {"protocol": entry["protocol"], "phase": phase,
                   "before": view.before, "after": view.after,
                   "binding": view.binding, "check_index_count": len(rows),
                   "mandatory_fields_unchanged": sorted(PINNED & context.keys()),
                   "all_image_pixels_retained": True, "matches_recorded_failure": bool(match),
                   "dependency_interfaces_and_constraints_lossless": True, "formal_state_sha256": before,
                   "formal_calls": len(original["calls"]), "formal_state_unchanged": True,
                   "model_calls_added": 0, "scope": "Exact request-building path, intercepted before model.ask"}
        summaries.append(summary)
        print(entry["protocol"], phase, view.before["required"], "->", view.after["required"], flush=True)
    atomic_json(output / "report.json", {"source_batch": batch_name, "results": summaries,
                "formal_experiment_executed": False,
                "limitation": "No real model decision or acceptance was tested"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay(Path(__file__).resolve().parents[1], args.batch, args.output)

