"""Parallel author prefetch; developer-authored fixtures, never formal model output."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from asset_generation.contracts import GenerationConfig
from asset_prep.config import Config
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.inputs import catalog
from scene_generation.pipeline import SceneRun
from generation_fixtures import PROVENANCE
from scene_fixtures import environment
from test_scene_first import ASSET, TOOL, ACCEPT, Backend


def two_asset_plan():
    return {"jobs": [{"id": "left", "asset_ids": ["A01"], "role": "asset", "depends_on": [], "requirements": "合成"},
                     {"id": "right", "asset_ids": ["A02"], "role": "asset", "depends_on": [], "requirements": "合成"},
                     {"id": "tool", "asset_ids": ["A01", "A02"], "role": "tool", "depends_on": ["left", "right"],
                      "requirements": "合成装置"}],
            "instances": [{"id": "left_station", "job": "left", "placement": "table", "position": [.4, 0, .8], "yaw": 0},
                          {"id": "right_station", "job": "right", "placement": "table", "position": [.6, .2, .8], "yaw": 0},
                          {"id": "gripper", "job": "tool", "placement": "robot", "position": [0, 0, 0], "yaw": 0}],
            "assumptions": []}


class TwoAssetBackend(Backend):
    """The shared fixture writes A01 manifests; relabel them with the job's own assets."""

    def build_candidate(self, script, destination, ids, profile):
        from scene_fixtures import SceneSandbox
        from scene_generation.candidate import inspect_candidate
        self.events.append("generate")
        report, renders = SceneSandbox.build_and_check(self, script, destination, ids, profile)
        path = Path(destination) / "artifact" / "interfaces.json"
        data = json.loads(path.read_text(encoding="utf-8").replace('"A01"', json.dumps(ids[0])))
        data["asset_ids"] = ids
        path.write_text(json.dumps(data), encoding="utf-8")
        admission = inspect_candidate(Path(destination) / "artifact", ids, profile)
        admission["renders"] = report["renders"]
        return admission, renders


class PhaseTransport:
    """Answers by request phase and records concurrency; thread-safe, no network."""

    def __init__(self, delay=0.3):
        self.delay, self.lock, self.phase = delay, threading.Lock(), None
        self.requests, self.active, self.peak = [], 0, 0

    def __copy__(self):
        clone = PhaseTransport.__new__(PhaseTransport)
        clone.__dict__.update(self.__dict__)
        clone.phase = None
        clone.parent = self
        return clone

    def post(self, url, payload, key, timeout):
        owner = getattr(self, "parent", self)
        phase = self.phase
        with owner.lock:
            owner.requests.append(phase)
            owner.active += 1
            owner.peak = max(owner.peak, owner.active)
        try:
            if phase and phase.startswith("asset:"):
                time.sleep(self.delay)
                reply = TOOL if phase == "asset:tool" else ASSET
            elif phase == "scene_plan":
                reply = two_asset_plan()
            else:
                reply = ACCEPT
            return {"model": "developer-fixture-model", "choices": [{"finish_reason": "stop",
                    "message": {"content": json.dumps(reply)}}], "usage": {"total_tokens": 100, "completion_tokens": 80}}
        finally:
            with owner.lock:
                owner.active -= 1


class ParallelAuthorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Path(__file__).resolve().parents[1]
        source = self.root / "assets.md"
        source.write_text("# Test\n### A01｜Synthetic left\n100 mm\n### A02｜Synthetic right\n80 mm\n", encoding="utf-8")
        self.generation = GenerationConfig()
        self.catalogue = catalog(source, self.generation)
        self.env = environment(self.root / "robot")

    def run_scene(self, workers, name):
        transport = PhaseTransport()
        run = SceneRun(self.catalogue, self.env, numerical_contract({}), Config(), self.generation,
                       SceneConfig(validation_mode="static", parallel_authors=workers), {}, self.root / name,
                       transport, TwoAssetBackend(), self.root / "robot", self.project, provenance=PROVENANCE)
        return run.run(), run, transport

    def test_independent_assets_are_authored_concurrently_once(self):
        result, run, transport = self.run_scene(4, "parallel")
        serial, _, serial_transport = self.run_scene(1, "serial")
        self.assertEqual(result["status"], serial["status"])
        self.assertEqual(sorted(transport.requests), sorted(serial_transport.requests))
        # Each author request is sent once: the serial sweep consumed the prefetched responses.
        self.assertEqual(transport.requests.count("asset:left"), 1)
        self.assertEqual(transport.requests.count("asset:right"), 1)
        self.assertGreaterEqual(transport.peak, 2)
        self.assertEqual(serial_transport.peak, 1)
        prefetched = run.state["prefetch"][0]
        self.assertEqual(sorted(j["job"] for j in prefetched["jobs"]), ["left", "right"])
        self.assertTrue(all(j["outcome"] == "cached" for j in prefetched["jobs"]))
        self.assertEqual(len({c["index"] for c in run.store.state["calls"]}), len(run.store.state["calls"]))
        self.assertFalse(run.store.inflight)

    def test_parallel_setting_is_not_part_of_run_identity(self):
        self.assertEqual(SceneConfig(parallel_authors=4).identity(), SceneConfig().identity())
        with self.assertRaises(Exception):
            SceneConfig(parallel_authors=9)


if __name__ == "__main__":
    unittest.main()
