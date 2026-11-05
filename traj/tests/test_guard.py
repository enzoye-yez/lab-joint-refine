import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from np002_traj import guard

ROOT = Path(__file__).resolve().parents[1]


class GuardTest(unittest.TestCase):
    def test_limits_and_workers(self):
        limits = guard.default_limits()
        self.assertGreater(limits["tree_bytes"], 0)
        self.assertLessEqual(guard.safe_workers(10_000, limits=limits), os.cpu_count())

    def test_trips_on_run_memory_and_kills_children(self):
        # A child process tree grows past a 300 MB budget: the guard must kill it and exit with 3.
        out = Path(tempfile.mkdtemp())
        script = textwrap.dedent(f"""
            import multiprocessing as mp, time
            from np002_traj import guard
            def hog(_):
                block = bytearray(150 * 1024 * 1024)
                for i in range(0, len(block), 4096):
                    block[i] = 1
                time.sleep(60)
            if __name__ == "__main__":
                limits = guard.default_limits()
                limits["tree_bytes"] = 300 * 1024 * 1024
                g = guard.Guard({str(out)!r}, limits, interval=0.2)
                with mp.get_context("fork").Pool(4) as pool:
                    g.start(on_trip=pool.terminate)
                    pool.map(hog, range(4))
        """)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-c", script], env=env, timeout=60, capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        record = json.loads((out / "guard.json").read_text())
        self.assertIn("run memory", record["reason"])

    def test_worker_check(self):
        with self.assertRaises(MemoryError):
            guard.worker_check(1)


if __name__ == "__main__":
    unittest.main()
