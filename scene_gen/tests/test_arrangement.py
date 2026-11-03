"""Deterministic room arrangement: benches on the walls, one row of instruments facing the aisle."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scene_generation.arrangement import arrange, front, AISLE_M, SETBACK_M, WALL_M
from scene_generation.layout import local_xy, rotate, validate_layout

ENV = {"robot": {"mobile": {"base_size": [.63, .602, .68]}}}


def box(w, d, target=None, normal=None):
    """Local bounds centred on the origin; an operation target on the -Y face unless given."""
    target = np.array(target if target is not None else [0., -d/2, .05])
    return {"lo": np.array([-w/2, -d/2, 0.]), "hi": np.array([w/2, d/2, .2]),
            "targets": [(target, None if normal is None else np.array(normal, float))]}


def plan():
    tables = [{"id": "bench1", "size": [3, 1.2, .05], "height": .8, "position": [-1, 2], "yaw": 0, "reason": "r"},
              {"id": "bench2", "size": [3, 1.2, .05], "height": .8, "position": [1, -2], "yaw": 0, "reason": "r"}]
    instances = [{"id": "a", "job": "a", "placement": "table", "table_id": "bench1", "position": [-1.5, 2.3, .8], "yaw": 0},
                 {"id": "b", "job": "b", "placement": "table", "table_id": "bench1", "position": [-.5, 1.9, .8], "yaw": 1.},
                 {"id": "tube", "job": "t", "placement": "table", "table_id": "bench1", "support_instance": "b",
                  "position": [-.45, 1.95, .9], "yaw": 1.},
                 {"id": "c", "job": "c", "placement": "table", "table_id": "bench2", "position": [1, -2.2, .8], "yaw": 0},
                 {"id": "fridge", "job": "f", "placement": "floor", "position": [3, 0, 0], "yaw": 0}]
    return {"jobs": [], "instances": instances,
            "layout": {"room": {"size": [9, 9], "reason": "r"}, "tables": tables, "robot_home": [0, 0, 0], "docks": []}}


SHAPES = {"a": box(.4, .5), "b": box(.3, .3), "c": box(.6, .4), "fridge": box(.7, .7), "tube": box(.02, .02)}


class ArrangementTests(unittest.TestCase):
    def test_instruments_stand_in_one_row_at_the_bench_edge_and_get_docks(self):
        new = arrange(plan(), SHAPES, ENV)
        validate_layout(new, None)
        tables = {t["id"]: t for t in new["layout"]["tables"]}
        items = {i["id"]: i for i in new["instances"]}
        spans = []
        for key in ("a", "b"):
            table = tables["bench1"]
            x, y = local_xy(items[key]["position"], table)
            # Front face (local -Y of the bench) SETBACK_M behind the edge.
            self.assertAlmostEqual(y-SHAPES[key]["hi"][1], -table["size"][1]/2+SETBACK_M, places=6)
            spans.append((x-.2, x+.2))
        self.assertLess(spans[0][1], spans[1][0]+.1)
        room = new["layout"]["room"]["size"]
        for table in new["layout"]["tables"]:
            # Back of every bench against a wall.
            back = abs(table["position"][1])+table["size"][1]/2
            self.assertAlmostEqual(back, room[1]/2-WALL_M, places=3)
        fronts = sorted(abs(t["position"][1])-t["size"][1]/2 for t in new["layout"]["tables"])
        self.assertGreaterEqual(2*min(fronts), AISLE_M-1e-6)
        docks = {d["instance"]: d for d in new["layout"]["docks"]}
        self.assertEqual(set(docks), {"a", "b", "c", "fridge"})
        # The chassis faces its bench, its front 3 cm short of the edge.
        d = docks["a"]; table = tables["bench1"]
        heading = np.array([math.cos(d["pose"][2]), math.sin(d["pose"][2])])
        edge = table["position"][1]-table["size"][1]/2
        self.assertAlmostEqual(d["pose"][1]+heading[1]*.315, edge-.03, places=3)

    def test_supported_parts_keep_their_pose_on_the_support(self):
        before = plan()
        new = arrange(before, SHAPES, ENV)
        old = {i["id"]: i for i in before["instances"]}; now = {i["id"]: i for i in new["instances"]}
        rel = lambda items: rotate([items["tube"]["position"][k]-items["b"]["position"][k] for k in range(2)], -items["b"]["yaw"])
        np.testing.assert_allclose(rel(old), rel(now), atol=1e-6)
        self.assertAlmostEqual(now["tube"]["yaw"]-now["b"]["yaw"], old["tube"]["yaw"]-old["b"]["yaw"])

    def test_front_follows_declared_approach_then_targets(self):
        self.assertEqual(front(box(.4, .4, normal=[1, 0, 0])).tolist(), [1, 0])
        self.assertEqual(front(box(.4, .4, target=[0, .2, .1])).tolist(), [0, 1])
        self.assertEqual(front({"lo": np.zeros(3), "hi": np.ones(3), "targets": []}).tolist(), [0, -1])

    def test_refine_waits_for_every_asset_and_calls_no_model(self):
        from scene_generation import layout_planning
        state = {"plan": {**plan(), "jobs": [{"id": "a", "role": "asset"}, {"id": "b", "role": "asset"}]},
                 "jobs": {"a": {"assembly_readiness": {"artifact_sha256": "1"}}}}
        run = SimpleNamespace(state=state, env=ENV, store=SimpleNamespace(save=lambda: None))
        with patch.object(layout_planning, "usable", side_effect=lambda r, v: bool(v)), \
                patch("scene_generation.arrangement.shapes_for", return_value=SHAPES):
            layout_planning.refine(run)
            self.assertNotIn("layout_refinement", state)
            state["jobs"]["b"] = {"assembly_readiness": {"artifact_sha256": "2"}}
            layout_planning.refine(run)
        self.assertEqual(state["layout_refinement"]["method"], "deterministic_rows")
        self.assertIn("docks", state["plan"]["layout"])


if __name__ == "__main__":
    unittest.main()
