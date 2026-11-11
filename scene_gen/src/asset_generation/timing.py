"""Flush stage boundaries to the existing worker log, including interrupted work."""
from contextlib import contextmanager
import json
import time


@contextmanager
def stage(name):
    started = time.perf_counter()
    print(json.dumps({"event":"stage_started", "stage":name}), flush=True)
    outcome = "completed"
    try:
        yield
    except BaseException:
        outcome = "failed"
        raise
    finally:
        print(json.dumps({"event":"stage_finished", "stage":name, "outcome":outcome,
                          "seconds":round(time.perf_counter()-started, 6)}), flush=True)
