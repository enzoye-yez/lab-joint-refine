"""Joint workflow commands, offline by default."""
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
from asset_prep.storage import run_lock,atomic_json
from asset_prep.config import Config
from asset_generation.contracts import GenerationConfig
from asset_prep.errors import PrepError
from .common import Settings,read,need,OperationError
from .improve import open_run


def main(argv):
    p=argparse.ArgumentParser(description="Joint asset and operation repair; offline request and import only by default")
    p.add_argument("command",choices=("improve","improve-import","improve-page","improve-status","improve-pause","diagnose-run","diagnose-tools","diagnose-query"))
    p.add_argument("--run",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--scene",type=Path);p.add_argument("--expected-sha")
    p.add_argument("--task-root",type=Path);p.add_argument("--public-profile",type=Path)
    p.add_argument("--seed-run",type=Path);p.add_argument("--package",type=Path);p.add_argument("--protocol-pdf",type=Path)
    p.add_argument("--backend",choices=("local","docker"),default="docker")
    p.add_argument("--provider-config",type=Path);p.add_argument("--generation-config",type=Path);p.add_argument("--settings",type=Path)
    p.add_argument("--resume",action="store_true");p.add_argument("--continue-paused",action="store_true")
    p.add_argument("--execute",action="store_true");p.add_argument("--expected-implementation")
    p.add_argument("--developer-fixture",action="store_true")
    p.add_argument("--response",type=Path);p.add_argument("--request",type=Path);p.add_argument("--query",type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=="diagnose-tools":
            from .diagnostic_tools import CATALOG
            print(json.dumps({"tools":CATALOG,"states":["initial","node_start","failure"]},ensure_ascii=False,indent=2))
            return 0
        need(args.run is not None,"--run is required to select the run")
        if args.command=="improve-status":result=read(args.run/"progress.json")
        elif args.command=="diagnose-query":
            from .diagnostic_service import query_run
            need(args.query is not None and args.output is not None,"A follow-up query requires --query and a separate --output")
            result=query_run(args.run,args.output,read(args.query),resume=args.resume)
        elif args.command=="improve-pause":
            atomic_json(args.run/"pause.request",{"reason":"Joint run paused explicitly by the user"})
            result={"status":"pause_requested"}
        elif args.command=="diagnose-run":
            from .diagnosis import diagnose_run
            result=diagnose_run(args.run,queries=read(args.query) if args.query else [])
        else:
            with run_lock(args.run):
                if args.command=="improve-import":
                    from .requests import import_exchange
                    need(args.response is not None,"--response is required")
                    result=import_exchange(args.run,args.response)
                elif args.command=="improve-page":
                    from .requests import read_page
                    need(args.request is not None and args.query is not None,"A request file and a page query are required")
                    need(args.request.resolve().parent==args.run.resolve()/"requests","Request does not belong to this run")
                    result=read_page(args.request,read(args.query))
                else:
                    project=Path(__file__).resolve().parents[2]
                    if args.resume:
                        saved=read(args.run/"configuration.json")
                        scene,sha,task,profile=saved["scene"],saved["scene_sha256"],saved["task_root"],saved["profile"]
                        options={k:saved[k] for k in ("seed_run","package","protocol","backend","developer")}
                        options.update(provider=Config(**saved["provider"]),generation=GenerationConfig(**saved["generation"]),settings=Settings(**saved["settings"]))
                    else:
                        need(args.scene and args.expected_sha and args.task_root and args.public_profile,"A new joint run requires the scene and its expected digest, task materials and the public environment")
                        need(not args.continue_paused,"--continue-paused applies only to an explicit resume")
                        scene,sha,task,profile=args.scene,args.expected_sha,args.task_root,args.public_profile
                        options=dict(seed_run=args.seed_run,package=args.package,protocol=args.protocol_pdf,backend=args.backend,developer=args.developer_fixture,
                            provider=Config.load(args.provider_config) if args.provider_config else None,
                            generation=GenerationConfig.load(args.generation_config) if args.generation_config else None,settings=Settings.load(args.settings))
                    run=open_run(args.run,scene,sha,task,profile,project,resume=args.resume,execute=args.execute,
                                 expected_implementation=args.expected_implementation,**options)
                    if args.continue_paused:
                        # Only acknowledge an existing pause after all immutable input checks succeed.
                        pause=args.run/"pause.request"
                        if pause.exists():
                            import uuid
                            pause.rename(args.run/("pause.acknowledged."+uuid.uuid4().hex+".json"))
                    result=run.advance()
        print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
        return 0 if result.get("status") not in {"needs_input","error"} else 2
    except (PrepError,OperationError,OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({"status":"error","kind":type(exc).__name__,"detail":str(exc)},ensure_ascii=False));return 2
