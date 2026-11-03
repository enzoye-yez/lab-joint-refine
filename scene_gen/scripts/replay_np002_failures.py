"""Offline replay of archived NP-002 outputs, in a new directory; no provider calls."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from asset_generation.contracts import GenerationConfig, author_result, check_profile
from asset_generation.checks import load_model
from asset_generation.local import LocalBackend
from asset_generation.pipeline import snapshot
from asset_prep.implementation import source_manifest, implementation_digest
from asset_prep.errors import ValidationError
from asset_prep.storage import atomic_json, file_hash, digest
from scene_generation.candidate import inspect_candidate
from scene_generation.contracts import SceneConfig


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists():raise ValueError('Use a new output directory')
    root.mkdir(parents=True)
    for key in list(os.environ):
        if 'API_KEY' in key:os.environ.pop(key)
    # The subprocess backend remains the existing local backend, with its existing
    # credential stripping, time limits and artifact checks. It is NOT OS isolation.
    corpus=PROJECT/'tests/data/np002_1027'
    for name,record in read(corpus/'provenance.json')['files'].items():
        if file_hash(corpus/name)!=record['sha256']:raise ValueError('Archived fixture changed: '+name)
    old=PROJECT/'runs/np002_tatu_sol_1027_01_np002'
    before=snapshot(old);code=implementation_digest(source_manifest(PROJECT))
    state=read(old/'state.json');gen=GenerationConfig.load(PROJECT/'config/generation.json')
    backend=LocalBackend(gen,PROJECT);rows={};artifacts={}
    profile=check_profile({'validation_mode':'static'})
    for n in range(1,17):
        asset=f'A{n:02d}';destination=root/asset;destination.mkdir()
        original=corpus/asset/'artifact'
        if original.is_dir():
            shutil.copytree(original,destination/'artifact')
            try:
                report=inspect_candidate(destination/'artifact',[asset],profile)
            except (ValueError,ValidationError) as exc:
                # Capture the replay evidence but never convert it to acceptance.
                report={'status':'fail','checks':[{'id':'assembly_compile','status':'fail','detail':str(exc)}],
                        'renders':[],'assembly_readiness':{'loadable':False}}
        else:
            source=((corpus/asset/'model.py').read_text(encoding='utf-8') if n!=16 else
                    author_result(json.loads(read(corpus/'A16/response.json')['choices'][0]['message']['content']),None,gen.max_script_bytes))
            script=destination/'model.py';script.write_text(source,encoding='utf-8')
            report,_=backend.build_and_check(script,destination,[asset],profile,admission=True)
        atomic_json(destination/'replay_checks.json',report)
        rows[asset]={'status':report['status'],'checks':report['checks'],
                     'assembly_readiness':report.get('assembly_readiness',{'loadable':False})}
        if rows[asset]['assembly_readiness']['loadable']:artifacts['job_'+asset]=destination/'artifact'
    # A minimal developer correction demonstrates iterative error discovery. It is
    # deliberately separate from the unmodified archived-model assembly.
    corrected=root/'developer_A04';corrected.mkdir()
    source=(corpus/'A04/model.py').read_text(encoding='utf-8')
    edited=author_result({'edits':[{'old':'"irreversible": false','new':'"irreversible": False'}]},source,gen.max_script_bytes)
    script=corrected/'model.py';script.write_text(edited,encoding='utf-8')
    report,_=backend.build_and_check(script,corrected,['A04'],profile,admission=True)
    atomic_json(corrected/'replay_checks.json',report)
    plan=deepcopy(state['scene']['plan']);missing=[j['id'] for j in plan['jobs'] if j['id'] not in artifacts]
    plan['jobs']=[j for j in plan['jobs'] if j['id'] in artifacts]
    plan['instances']=[i for i in plan['instances'] if i['job'] in artifacts]
    ids={i['id'] for i in plan['instances']}
    if 'layout' in plan:plan['layout']['docks']=[d for d in plan['layout']['docks'] if d['instance'] in ids]
    for field in ('table_modifications','edge_mounts'):
        if field in plan:plan[field]=[v for v in plan[field] if v['instance'] in ids]
    destination=root/'assembly';destination.mkdir()
    env=state['identity']['environment'];robot=PROJECT/env['robot']['directory']
    robot_before=snapshot(robot)
    scene,_=backend.assemble_candidate(plan,env,artifacts,robot,destination,
        SceneConfig(**read(PROJECT/'config/scene.json')),state['identity']['contract'])
    artifact=destination/'scene/artifact'
    if not scene.get('assembly_readiness',{}).get('loadable'):raise RuntimeError(scene)
    model=load_model(artifact)
    quality,_=backend.check_scene(artifact,root/'quality_check',
        SceneConfig(**read(PROJECT/'config/scene.json')),state['identity']['contract'])
    atomic_json(root/'quality_replay.json',{'implementation':code,'artifact_sha256':digest(snapshot(artifact)),
        'model_api_calls':0,'report':quality})
    if quality['status']=='error':raise RuntimeError('Quality report contract or worker failure: '+str(quality.get('detail')))
    from scene_generation.preview import export_preview
    export_preview(artifact,root/'preview',{
        'status':'incomplete_scene','provenance':'archived_model_output_replay',
        'validation_mode':'static','qualification':'offline diagnostic assembly of archived models; has missing parts and interface defects; quality and operation are not verified',
        'assembly':{'available':True,'complete':False,'missing_jobs':missing,'quality_accepted':False}})
    unchanged=before==snapshot(old) and robot_before==snapshot(robot)
    stable=code==implementation_digest(source_manifest(PROJECT))
    result={'status':'pass' if unchanged and stable else 'failed','model_api_calls':0,
        'provenance':'archived_model_output_replay','model_autonomous_repair_success':False,
        'original_run_unchanged':before==snapshot(old),'original_robot_unchanged':robot_before==snapshot(robot),
        'implementation':code,'source_unchanged':stable,'assets':rows,
        'original_responses':len(state['calls']),'reported_total_tokens':sum(c.get('reported_total_tokens',0) for c in state['calls']),
        'accounted_tokens':sum(max(c['reserved_tokens'],c.get('reported_total_tokens',0)) for c in state['calls']),
        'a16_response_usage':read(corpus/'A16/response.json')['usage'],
        'developer_A04':{'provenance':'developer_authored_edit','report':report},
        'assembly':{'artifact':str(artifact),'artifact_sha256':digest(snapshot(artifact)),
            'loadable':True,'quality_accepted':False,'operation_verified':False,
            'included_jobs':sorted(artifacts),'missing_jobs':missing,'instances':len(plan['instances']),
            'ngeom':model.ngeom,'report':scene,'quality_report':'quality_replay.json'},
        'backend':backend.preflight()}
    atomic_json(root/'replay.json',result)
    print(json.dumps({k:result[k] for k in ('status','implementation','model_api_calls','source_unchanged')},ensure_ascii=False))
    print('loadable archived assets: '+', '.join(artifacts)+'; missing jobs: '+str(len(missing)))
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
