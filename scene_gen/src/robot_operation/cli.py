"""Offline operation entry point; no implicit model calls, environment installation or formal experiments."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .common import OperationError, Settings, identity, read


def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and (argv[0].startswith("improve") or argv[0] in {"diagnose-run","diagnose-tools","diagnose-query"}):
        from .improve_cli import main as joint_main
        return joint_main(argv)
    parser=argparse.ArgumentParser(description="Mobile dual-arm operation, evidence and replay (offline by default)")
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect",help="Read the scene summary")
    inspect.add_argument("--scene",type=Path,required=True)
    sub.add_parser("defaults",help="Print the public engineering defaults")
    view=sub.add_parser("view",help="View joint poses; does not step dynamics or produce passing evidence")
    view.add_argument("--scene",type=Path,required=True);view.add_argument("--expected-sha",required=True)
    prepare=sub.add_parser("prepare",help="Derive a free-base operation scene and run offline admission checks")
    request=sub.add_parser("request",help="Prepare an allowlisted task request; no network access")
    execute=sub.add_parser("run",help="Execute an explicitly supplied operation graph")
    repeat=sub.add_parser("repeat",help="Run the frozen 14-trial validation; invoked explicitly by the user")
    replay=sub.add_parser("replay",help="Replay from the recorded control inputs")
    diagnose=sub.add_parser("diagnose",help="Independently re-check initial-state penetration in the source scene and export feedback")
    accept=sub.add_parser("accept",help="Import an explicitly supplied model or developer response; no network access")
    revise=sub.add_parser("revise",help="Import a new operation graph or repaired scene within budget")
    accept.add_argument("--request",type=Path,required=True)
    accept.add_argument("--response",type=Path,required=True)
    accept.add_argument("--output",type=Path,required=True)
    revise.add_argument("--previous-run",type=Path,required=True)
    revise.add_argument("--graph",type=Path,required=True)
    revise.add_argument("--output",type=Path,required=True)
    revise.add_argument("--new-scene",type=Path)
    revise.add_argument("--new-scene-sha")
    revise.add_argument("--defect-report",type=Path)
    for p in (prepare,request,execute,repeat,diagnose):
        p.add_argument("--scene",type=Path,required=True)
        p.add_argument("--expected-sha",required=True)
        p.add_argument("--output",type=Path,required=True)
    for p in (prepare,execute,repeat,diagnose,accept): p.add_argument("--settings",type=Path)
    for p in (execute,repeat):
        p.add_argument("--task",type=Path,required=True)
        p.add_argument("--graph",type=Path,required=True)
        p.add_argument("--no-images",action="store_true",help="For debugging; output is not marked as a complete dataset")
    execute.add_argument("--mode",choices=("trial","final"),default="final")
    request.add_argument("--task-root",type=Path,required=True)
    request.add_argument("--public-profile",type=Path,required=True)
    replay.add_argument("--run",type=Path,required=True)
    replay.add_argument("--scene",type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=="inspect":
            result={"scene":str(args.scene.resolve()),"scene_sha256":identity(args.scene)}
        elif args.command=="view":
            from .views import show_joints
            show_joints(args.scene,args.expected_sha);result={"status":"view_closed","qualification":"diagnostic_only"}
        elif args.command=="defaults": result=asdict(Settings())
        elif args.command=="prepare":
            from .scene import prepare as convert
            result=convert(args.scene,args.output,args.expected_sha,Settings.load(args.settings))
        elif args.command=="request":
            from .workflow import request_task
            result=request_task(args.task_root,args.scene,args.expected_sha,args.public_profile,args.output)
        elif args.command=="replay":
            from .recording import replay as replay_run
            result=replay_run(args.run,args.scene)
        elif args.command=="diagnose":
            from .feedback import verify_source_collisions
            result=verify_source_collisions(args.scene,args.expected_sha,args.output,Settings.load(args.settings))
        elif args.command=="accept":
            from .feedback import import_response
            result=import_response(args.request,args.response,args.output,Settings.load(args.settings))
        elif args.command=="revise":
            from .feedback import import_revision
            result=import_revision(args.previous_run,args.graph,args.output,args.new_scene,args.new_scene_sha,args.defect_report)
        else:
            from .workflow import run,repeat as repeat_run
            method=run if args.command=="run" else repeat_run
            extra={"mode":args.mode} if args.command=="run" else {}
            result=method(args.scene,args.expected_sha,read(args.task),read(args.graph),args.output,
                          Settings.load(args.settings),images=not args.no_images,**extra)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
        return 0 if result.get("status") not in {"failed","blocked","error"} else 2
    except (OperationError,ValueError,OSError,KeyError) as exc:
        print(json.dumps({"status":"error","code":getattr(exc,"code","invalid_input"),"detail":str(exc)},ensure_ascii=False))
        return 2
