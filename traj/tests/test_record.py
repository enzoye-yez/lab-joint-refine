"""Run directory, deterministic replay, failure-tolerant rendering and worker GL environment (plan §6)."""
import os
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from np002_traj.record.camera import Camera, close_up, worker_environment
from np002_traj.record.run import Recorder, Run, replay
from support import controllers, world


class RecordTests(unittest.TestCase):
    def test_recorded_controls_replay_to_the_identical_state(self):
        model, data, robot = world()
        control = controllers(model, data, robot)
        target = data.qpos[robot.arms["left"].qpos] + 0.2
        control["left"].set(target)
        data.ctrl[robot.base_actuators[0]] = 0.2
        with tempfile.TemporaryDirectory() as root:
            run = Run("replay", {"seed": 0}, root=root)
            recorder = Recorder(model, data, run, watched_geoms=robot.geoms)
            for _ in range(500):
                for c in control.values():
                    c.apply(data)
                mujoco.mj_step(model, data)
                recorder.tick()
            recorder.finish()
            run.event("done", data.time)
            run.close()
            result = replay(model, run.directory)
            self.assertTrue(result["identical"])
            self.assertEqual(result["steps"], 500)
            self.assertEqual(len(np.load(run.directory / "trace.npz")["time"]), 10)   # 1 s at 10 Hz
            with self.assertRaises(FileExistsError):
                Run("replay", {}, root=root)

    def test_render_failure_is_per_frame(self):
        model, data, robot = world()
        camera = Camera(model, 64, 48)
        view = close_up(model, data, [data.site_xpos[robot.arms["left"].tcp]], 150, -25)
        image, error = camera.render(data, view)
        self.assertTrue(image is not None or error)
        camera.close()
        image, error = camera.render(data, view)       # renderer gone: an error value, not an exception
        self.assertIsNone(image)

    def test_worker_keeps_gl_settings_and_drops_credentials(self):
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
        os.environ["SOME_API_KEY"] = "secret"
        try:
            env = worker_environment()
            out = subprocess.run([sys.executable, "-c", "import os;print(os.environ.get('MUJOCO_GL'),os.environ.get('SOME_API_KEY'))"],
                                 env=env, capture_output=True, text=True).stdout.split()
            self.assertEqual(out, [os.environ["MUJOCO_GL"], "None"])
        finally:
            del os.environ["SOME_API_KEY"]


if __name__ == "__main__":
    unittest.main()
