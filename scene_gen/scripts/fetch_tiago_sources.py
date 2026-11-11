"""Acquire pinned PAL description data only; do not execute downloaded programs."""
from pathlib import Path
import concurrent.futures
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]/".cache/tiago_omni_sources"
REPOS={
    "tiago_robot":"f1c33c92bdde7c1dd79f0c3e739e98a233dbd30b",
    "omni_base_robot":"49ecf709e3f42ba0613ccaeb8a07105be24a6c36",
    "pal_urdf_utils":"e8600a339a3a9b7ffe67d1030190b97b67fb577b",
    "pal_gripper":"485b535532c04b0ce31a00ad431d99e2714c0d54",
    "pal_hey5":"344de441fa3c378573dd7f6e04dc7ec4f8cb0cab",
    "pal_robotiq_gripper":"966485faa4c367eb17848deb44d9b7bbb3555605"}

def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"harness-tiago-integration"})
    with urllib.request.urlopen(req,timeout=120) as response:
        data=response.read(250_000_001)
    if len(data)>250_000_000:
        raise ValueError("Source archive too large")
    return data

def fetch(repo):
    target=ROOT/repo
    if (target/"source.json").exists() and (repo != "pal_urdf_utils" or (target/"urdf").exists()):
        record=json.loads((target/"source.json").read_text(encoding="utf-8"))
        if record["revision"]!=REPOS[repo]:
            raise ValueError("Cached source revision differs: "+repo)
        for relative, expected in record["files"].items():
            path=(target/relative).resolve()
            if not path.is_relative_to(target.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError("Cached source changed: "+repo+"/"+relative)
        return record
    commit=REPOS[repo]
    archive=get(f"https://codeload.github.com/pal-robotics/{repo}/zip/{commit}")
    record={"repository":f"https://github.com/pal-robotics/{repo}","revision":commit,
            "archive_sha256":hashlib.sha256(archive).hexdigest(),"files":{}}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for member in z.infolist():
            if member.is_dir():continue
            rel=Path(*Path(member.filename).parts[1:])
            if not rel.parts or rel.is_absolute() or ".." in rel.parts:raise ValueError("Invalid archive path")
            if repo != "pal_urdf_utils" and not any("description" in p for p in rel.parts) and not rel.name.startswith(("LICENSE","COPYING")):continue
            dest=(target/rel).resolve()
            if not dest.is_relative_to(target.resolve()):raise ValueError("Archive path escape")
            content=z.read(member)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
            record["files"][rel.as_posix()]=hashlib.sha256(content).hexdigest()
    (target/"source.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record

if __name__=="__main__":
    ROOT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch,REPOS):
            print(result["repository"],result["revision"],len(result["files"]),flush=True)
