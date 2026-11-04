"""One directory per run: config, code identity, seed, initial state, controls, events, replay check."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from ..robot import ROOT

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def code_identity():
    """Git commit of the package and whether the working tree had uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src", "scripts", "data"))}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    return state


def set_state(model, data, state):
    mujoco.mj_setState(model, data, state, STATE)
    mujoco.mj_forward(model, data)


class Run:
    def __init__(self, name, config, root=None, overwrite=False):
        self.directory = Path(root or ROOT / "runs") / name
        if self.directory.exists() and not overwrite:
            raise FileExistsError(f"run {self.directory} exists; use a new name")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = {**config, "code": code_identity(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.write_json("config.json", self.config)
        self._events = (self.directory / "events.jsonl").open("a", encoding="utf-8")

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=1, default=float), encoding="utf-8")

    def event(self, kind, time_s, **detail):
        self._events.write(json.dumps({"t": round(float(time_s), 4), "kind": kind, **detail}, default=float) + "\n")
        self._events.flush()

    def close(self):
        self._events.close()


class Recorder:
    """Records every control and a 10 Hz state trace; contacts only for the watched geoms."""

    def __init__(self, model, data, run, watched_geoms=(), trace_hz=10.0):
        self.model, self.data, self.run = model, data, run
        self.every = max(1, int(round(1.0 / (trace_hz * model.opt.timestep))))
        self.watched = np.asarray(sorted(set(int(g) for g in watched_geoms)), int)
        self.initial = get_state(model, data)
        np.save(run.directory / "initial_state.npy", self.initial)
        self.controls, self.times, self.qpos, self.contacts = [], [], [], []
        self.ticks = 0
        self._force = np.zeros(6)

    def tick(self):
        """Call after each mj_step."""
        d = self.data
        self.controls.append(d.ctrl.copy())
        if self.ticks % self.every == 0:
            self.times.append(d.time)
            self.qpos.append(d.qpos.copy())
            self.contacts.append(self._watched_contacts())
        self.ticks += 1

    def _watched_contacts(self):
        d, n = self.data, self.data.ncon
        if not n or not len(self.watched):
            return []
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        rows = []
        for i in np.flatnonzero(np.isin(g1, self.watched) | np.isin(g2, self.watched)):
            mujoco.mj_contactForce(self.model, d, int(i), self._force)
            rows.append([int(g1[i]), int(g2[i]), round(float(self._force[0]), 3), round(float(d.contact.dist[i]), 5)])
        return rows

    def finish(self):
        directory = self.run.directory
        np.save(directory / "controls.npy", np.array(self.controls))
        np.savez_compressed(directory / "trace.npz", time=np.array(self.times), qpos=np.array(self.qpos))
        with (directory / "contacts.jsonl").open("w", encoding="utf-8") as out:
            for t, rows in zip(self.times, self.contacts):
                out.write(json.dumps({"t": round(t, 4), "contacts": rows}) + "\n")
        np.save(directory / "final_state.npy", get_state(self.model, self.data))


def replay(model, directory):
    """Re-run the recorded controls from the recorded initial state and compare the final state."""
    directory = Path(directory)
    data = mujoco.MjData(model)
    set_state(model, data, np.load(directory / "initial_state.npy"))
    for ctrl in np.load(directory / "controls.npy"):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    error = float(np.max(np.abs(get_state(model, data) - np.load(directory / "final_state.npy"))))
    result = {"steps": int(len(np.load(directory / "controls.npy"))), "max_state_error": error, "identical": error == 0.0}
    (directory / "replay.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
