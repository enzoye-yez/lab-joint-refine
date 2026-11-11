"""Exact-input check checkpoints, never a substitute for a completed report."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid
import os
import mujoco
from asset_generation.pipeline import snapshot
from asset_prep import implementation
from asset_prep.storage import atomic_json, digest, safe_path, file_hash


class Checkpoints:
    def __init__(self, directory, artifact, limits, contract, recovery):
        self.directory = None
        self.hits = self.completed = 0
        self.write_warning = False
        if directory is None: return
        from asset_generation.contracts import require
        require(not Path(directory).resolve().is_relative_to(Path(artifact).resolve()), "checkpoints must not be written into the input asset directory")
        identity = {"artifact":snapshot(artifact), "limits":asdict(limits), "contract":contract,
                    "recovery":recovery, "implementation":implementation.package_manifest(Path(__file__).resolve().parents[1]),
                    "dependencies":implementation.dependency_versions()}
        self.key = digest(identity)
        self.directory = safe_path(Path(directory),self.key)
        # Numerical findings remain implementation-bound. A compile depends on the
        # exact files, loader policy and engine, not planner prompts or report storage.
        self.compile_identity = self.compile_inputs(identity)
        self.model_key = digest(self.compile_identity)
        self.cache_root = Path(directory)
        self.model_directory = safe_path(Path(directory)/'compiled', self.model_key)
        self.artifact, self.artifact_snapshot = Path(artifact), identity["artifact"]
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            self.model_directory.mkdir(parents=True,exist_ok=True)
        except OSError:
            self.directory=None
            print('{"event":"checkpoint_unavailable"}',flush=True)
            return
        self.save(self.directory/"identity.json",identity)

    @staticmethod
    def compile_inputs(identity):
        sources = identity['implementation']
        return {'artifact':identity['artifact'], 'dependencies':identity['dependencies'],
                'loader':{name:sources[name] for name in
                          ('asset_generation/checks.py','asset_generation/contracts.py')},
                'static':identity['limits']['validation_mode']=='static'}

    def previous_compiles(self):
        """Read-only compatibility with previously validated XML compiles."""
        for path in self.cache_root.glob('*/identity.json'):
            try:
                identity=json.loads(path.read_text(encoding='utf-8'))
                if self.compile_inputs(identity)==self.compile_identity:
                    yield digest({k:v for k,v in identity.items() if k!='recovery'})
            except (OSError,ValueError,KeyError,TypeError):
                continue

    def save(self, path, value):
        try:atomic_json(path,value)
        except OSError:
            if not self.write_warning:print('{"event":"checkpoint_write_unavailable"}',flush=True)
            self.write_warning=True

    def verify_input(self):
        if hasattr(self,"artifact_snapshot") and snapshot(self.artifact)!=self.artifact_snapshot:
            from asset_prep.errors import StaleEvidence
            raise StaleEvidence("input scene changed during checks; mixed-version evidence cannot be published")

    def load_model(self, compute):
        """Reuse only a successfully validated compile of this entire exact input."""
        if self.directory is None:return compute()
        self.verify_input()
        from itertools import chain
        for key in chain((self.model_key,),self.previous_compiles()):
            directory=safe_path(self.cache_root/'compiled',key)
            binary,receipt=directory/'model.mjb',directory/'model.json'
            try:
                value=json.loads(receipt.read_text(encoding="utf-8"))
                if value["identity"]==key and file_hash(binary)==value["sha256"]:
                    model=mujoco.MjModel.from_binary_path(str(binary))
                    if all(getattr(model,k)==value["scale"][k] for k in ("ngeom","nbody","nv")):
                        self.verify_input()
                        if key != self.model_key:self.store_model(model,value['scale'])
                        print(json.dumps({"event":"compiled_model_reused","identity":self.model_key,
                                          "source_identity":key}),flush=True)
                        return model,value["scale"]
            except (OSError,ValueError,KeyError,TypeError):pass
        model,scale=compute()
        self.store_model(model,scale)
        return model,scale

    def store_model(self, model, scale):
        self.verify_input()
        binary=safe_path(self.model_directory,'model.mjb')
        receipt=safe_path(self.model_directory,'model.json')
        temp=self.model_directory/("model-"+uuid.uuid4().hex+".mjb")
        try:
            mujoco.mj_saveModel(model,str(temp),None)
            sha=file_hash(temp)
            os.replace(temp,binary)
            self.save(receipt,{"identity":self.model_key,"sha256":sha,"scale":scale})
        except OSError:
            print('{"event":"compiled_cache_write_unavailable"}',flush=True)
        finally:
            try:temp.unlink(missing_ok=True)
            except OSError:pass

    def run(self, name, compute, query):
        from asset_generation.timing import stage
        path = safe_path(self.directory,digest(name)+".json") if self.directory else None
        if path and path.is_file():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                payload = entry["payload"]
                if (entry["identity"] == self.key and entry["stage"] == name
                        and digest(payload) == entry["sha256"]):
                    query.recoveries.extend(payload["recoveries"])
                    self.hits += 1
                    print(json.dumps({"event":"checkpoint_reused","stage":name}),flush=True)
                    return payload["value"]
            except (OSError,ValueError,KeyError,TypeError):
                pass  # Incomplete/corrupt cache is recomputed, never treated as a pass.
        if path: self.save(self.directory/"progress.json",{"stage":name,"status":"running","time":time.time()})
        offset = len(query.recoveries)
        try:
            with stage(name): value = compute()
        except BaseException:
            if path: self.save(self.directory/"progress.json",{"stage":name,"status":"failed","time":time.time()})
            raise
        if path:
            payload={"value":value,"recoveries":query.recoveries[offset:]}
            self.save(path,{"identity":self.key,"stage":name,"payload":payload,"sha256":digest(payload)})
            self.save(self.directory/"progress.json",{"stage":name,"status":"completed","time":time.time()})
        self.completed += 1
        return value
