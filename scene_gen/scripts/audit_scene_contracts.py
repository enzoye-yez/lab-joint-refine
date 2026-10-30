"""离线审核归档报告及补查收据；不调用模型、不执行建模脚本、不改原运行。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from asset_generation.pipeline import snapshot
from asset_generation.contracts import read_json, require
from asset_generation.check_report import validate_report
from asset_prep.storage import safe_path, digest, atomic_json, file_hash
from scene_generation.capabilities import REGISTRY, validate_result
from scene_generation.dispatch import consume
from scene_generation.recovery import decide
from scene_generation.faults import initialize, add_failure


def audit(root, output):
    root,output=Path(root).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(root) and not root.is_relative_to(output),
            '审计输出须为原运行之外的新目录')
    before=snapshot(root)
    from asset_prep.state_archive import read_state
    state=read_state(root/'state.json')
    for call in state.get('calls',[]):
        if call['status']=='done':
            require(file_hash(safe_path(root,call['file']))==call['sha256'], '已完成调用的归档摘要不匹配')
    scene=state['scene']; mode=state['identity']['limits']['validation_mode']
    records=[]; replays=[]
    for stage in scene['stages']:
        path=safe_path(root,stage['directory'])
        require(snapshot(path)==stage['files'], '原归档摘要不匹配，拒绝审计：'+stage['directory'])
        check=path/'checks.json'
        if not check.is_file():
            continue
        report=read_json(check,16000000)
        try:
            validate_report(report,domain='scene' if stage['kind']=='scene' else 'asset',mode=mode)
            records.append({'path':str(check.relative_to(root)),'contract_valid':True,'status':report['status']})
        except Exception as exc:
            records.append({'path':str(check.relative_to(root)),'contract_valid':False,'error':str(exc)})
    for identity,session in scene.get('check_recovery',{}).items():
        actions=session['actions']
        if not actions:
            continue
        first=safe_path(root,actions[0]['receipt'])
        stage=next(p for p in first.parents if p.parent.name=='stages')
        report=read_json(stage/'scene/report.json',16000000)
        validate_report(report,domain='scene',mode=mode)
        report=initialize(report,identity)
        applied=[]
        for action in actions:
            receipt=safe_path(root,action['receipt'])
            result=read_json(receipt,16000000)
            try:
                validate_result(result,action['parameters'],identity,mode)
                if 'failure' not in result:
                    report=consume(report,result,action['parameters'],REGISTRY[action['capability']],action['key'],identity,mode)
                else:
                    request=action['parameters']
                    report=add_failure(report,result['failure'],input_key=identity,action_key=action['key'],
                        attempt=action['attempts'],capability=action['capability'],request=request,
                        scope='scene' if request.get('full_scene') else 'pairs' if 'pairs' in request else
                              getattr(REGISTRY[action['capability']],'failure_scope','checks'))
                applied.append({'capability':action['capability'],'original_action_status':action['status'],
                                'contract_valid':True,'execution_failure':result.get('failure')})
            except Exception as exc:
                applied.append({'capability':action['capability'],'contract_valid':False,'error':str(exc)})
        replays.append({'identity':identity,'actions':applied,'report':report,'decision':decide(report)})
    require(snapshot(root)==before,'审计期间原运行发生变化，拒绝发布结果')
    valid=bool(records) and all(r['contract_valid'] for r in records) and all(a['contract_valid'] for r in replays for a in r['actions'])
    result={'status':'pass' if valid else 'fail','qualification':'仅认证所读取记录的契约；不认证场景质量或实验成功',
            'source_run':str(root),'source_sha256':digest(before),'source_unchanged':True,
            'model_calls':0,'generated_code_executed':False,'reports':records,'replays':replays}
    output.mkdir(parents=True)
    atomic_json(output/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.run,args.output)
    print(json.dumps({'status':result['status'],'reports':len(result['reports']),
        'replay_decisions':[r['decision'] for r in result['replays']],
        'source_unchanged':result['source_unchanged'],'output':str(args.output/'audit.json')},ensure_ascii=False))
    raise SystemExit(0 if result['status']=='pass' else 1)
