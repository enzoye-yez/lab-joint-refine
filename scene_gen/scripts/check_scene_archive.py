"""Read-only replay of archived scene checks; does not run model scripts, call the provider or change old conclusions."""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
if __name__ == '__main__' and not getattr(sys, '_harness_runtime', None):
    import os
    os.execv(sys.executable, [sys.executable, '-B', '-P', str(PROJECT/'src/runtime_bootstrap.py'),
                             '--script', __file__, '--', *sys.argv[1:]])
sys.path.insert(0,str(PROJECT/'src'))

from asset_generation.pipeline import snapshot
from asset_prep import implementation
from asset_prep.storage import atomic_json, digest
from asset_generation.contracts import require
from scene_generation.checks import scene_checks
from scene_generation.contracts import SceneConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recovery',action='store_true')
    parser.add_argument('--checkpoints',type=Path,help='separate checkpoint directory; completed stages are reused automatically for identical inputs')
    parser.add_argument('--expected-implementation')
    args=parser.parse_args()
    stage=args.stage.resolve();output=args.output.resolve()
    require(not output.exists(),'output directory already exists; refusing to overwrite')
    require(not output.is_relative_to(stage.parent.parent),'replay output must be separate from the original run')
    if args.checkpoints:
        require(not args.checkpoints.resolve().is_relative_to(stage.parent.parent),'checkpoints must be separate from the original run')
    job=json.loads((stage/'scene_job.json').read_text(encoding='utf-8'))
    binding = implementation.runtime_record(PROJECT, inputs={'scene_job':digest(job)}, configuration=job['limits'])
    identity = args.expected_implementation or binding['implementation']
    guard = implementation.Guard(PROJECT, identity)
    artifact=stage/'scene/artifact'
    before=snapshot(stage)
    started=time.monotonic()
    print('Starting read-only check of the archived scene; the model is not called.',flush=True)
    report=scene_checks(artifact,SceneConfig(**job['limits']),job['contract'],recovery=args.recovery,checkpoint_dir=args.checkpoints)
    guard.check()
    require(snapshot(stage)==before,'the read-only replay modified archived files')
    output.mkdir(parents=True)
    atomic_json(output/'report.json',report)
    manifest = binding['manifest']
    implementation.archive_sources(PROJECT, output, identity)
    receipt={'status':report['status'],'source_stage':str(stage),'source_unchanged':True,
             'source_sha256':digest(before),'elapsed_seconds':round(time.monotonic()-started,2),
             'model_calls':0,'recovery':args.recovery,
             'implementation':identity,'implementation_manifest':manifest,
             'entrypoint':getattr(sys,'_harness_runtime',{}).get('entrypoint'),
             'dependencies':implementation.dependency_versions(),
             'archived_report_status':json.loads((stage/'scene/report.json').read_text(encoding='utf-8')).get('status')
                                      if (stage/'scene/report.json').is_file() else None,
             'implementation_covers':implementation.COVERS['implementation'],
             'checks':[{'id':r['id'],'status':r['status'],'collision_count':r.get('collision_count'),
                        'gap_count':len(r.get('coverage_gaps',[]))} for r in report['checks']]}
    atomic_json(output/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    main()
