"""Real execution-backend engineering self-test: runs only the repository synthetic samples, zero model calls."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/"src"),str(PROJECT/"tests")]

from asset_generation.contracts import GenerationConfig, check_profile, read_json, require
from asset_generation.sandbox import DockerSandbox
from asset_generation.local import LocalBackend
from asset_prep.storage import atomic_json
from generation_fixtures import XML, MANIFEST, PROVENANCE
from scene_fixtures import ASSET, TOOL, manifest, plan
from scene_generation.contracts import SceneConfig, numerical_contract
from scene_generation.preview import export_preview


def source(xml, interfaces, check_isolation=False):
    # Content is repository-authored test data, never a model response.
    checks = '''
    import os
    import socket
    assert not any(k in os.environ for k in ("HARNESS_MODEL_API_KEY", "OPENAI_API_KEY"))
    assert not Path("/input/.env.local").exists()
    for path in ("/app/__container_probe__", "/input/__container_probe__", "/__container_probe__"):
        try:
            Path(path).write_text("unexpected write")
        except OSError:
            pass
        else:
            raise AssertionError("Read-only mount unexpectedly writable")
    with socket.socket() as sock:
        sock.settimeout(1)
        try:
            sock.connect(("1.1.1.1", 443))
        except OSError:
            pass
        else:
            raise AssertionError("External network unexpectedly reachable")
''' if check_isolation else ""
    return ("from pathlib import Path\nimport json\ndef build(output_dir):\n"+checks+
            f"    (output_dir/'model.xml').write_text({xml!r}, encoding='utf-8')\n"+
            f"    (output_dir/'interfaces.json').write_text({json.dumps(interfaces,ensure_ascii=False)!r}, encoding='utf-8')\n")


def run(output, kind="docker"):
    output = Path(output)
    require(not output.exists(),"self-test output directory already exists; use a new directory")
    output.mkdir(parents=True)
    backend = (LocalBackend if kind == "local" else DockerSandbox)(GenerationConfig.load(PROJECT/"config/generation.json"),PROJECT)
    report = {"provenance":PROVENANCE,"model_calls":0,"status":"running","checks":[]}
    try:
        report["backend"] = backend.preflight()
        components = {}
        for name, xml, interfaces, mode in (("button",XML,MANIFEST,"full"),
                ("instrument",ASSET,manifest(),"static"),("tool",TOOL,manifest(True),"static")):
            destination = output/name
            destination.mkdir()
            script = destination/"model.py"
            script.write_text(source(xml,interfaces,name == "button" and kind == "docker"),encoding="utf-8")
            checked, renders = backend.build_and_check(script,destination,["A01"],check_profile({"validation_mode":mode}))
            atomic_json(destination/"checks.json",checked)
            report["checks"].append({"id":name,"status":checked["status"],"renders":checked.get("renders",[])})
            require(checked["status"] == "pass",f"asset self-test failed: {name}; see {destination/'checks.json'}")
            require(renders and len(checked["renders"]) >= 3,"missing actual renders")
            components[name] = destination/"artifact"
        destination = output/"assembly"
        destination.mkdir()
        checked, renders = backend.assemble_and_check(plan(),read_json(PROJECT/"config/scene_environment.json"),
            {k:components[k] for k in ("instrument","tool")},PROJECT/"data/robots/fr3",destination,
            SceneConfig(validation_mode="static"),numerical_contract({}))
        atomic_json(destination/"checks.json",checked)
        report["checks"].append({"id":"public_fr3_static_scene","status":checked["status"],"renders":checked.get("renders",[])})
        require(checked["status"] == "pass","real-robot static assembly check did not pass")
        report.update(status="pass",validation_mode="static",qualification="developer synthetic engineering self-test, not a model-generated experiment",
                      artifact=str(destination/"scene/artifact"))
        report["gallery"] = str(export_preview(destination/"scene/artifact",output/"preview",report))
    except Exception as exc:
        report.update(status="fail",error=type(exc).__name__+": "+str(exc))
    atomic_json(output/"verification.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("docker","local"),default="docker")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT/".cache"/(args.backend+"-verification-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    raise SystemExit(run(output,args.backend))
