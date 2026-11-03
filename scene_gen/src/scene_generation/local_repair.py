"""Reject proven local regressions before assembling a large scene."""
from asset_prep.storage import digest, safe_path
from asset_generation.check_report import complete
from asset_generation.contracts import check_profile
from asset_generation.sandbox import WorkerTimeout
from .progress import repair_comparison


def check_key(run,record,job,profile):
    return digest({'artifact':record['assembly_readiness']['artifact_sha256'],'profile':profile,
        'ids':job['asset_ids'],'implementation':run.implementation_guard.expected})


def cached_check(run,binding):
    record=run.state.get('local_checks',{}).get(binding)
    if record:
        from asset_generation.pipeline import snapshot
        from asset_prep.errors import StaleEvidence
        stage=next(s for s in run.state['stages'] if s['directory']==record['directory'])
        if snapshot(safe_path(run.root,stage['directory']))!=stage['files']:
            raise StaleEvidence('Local check archive was modified; cannot be reused')
    return record


def precheck(run):
    entry=next((h for h in reversed(run.state.get('repair_history',[]))
        if h.get('status')=='applied' and h.get('baseline')),None)
    if entry is None:return False
    action=entry['action']
    if not action['jobs']:return False
    stages={s['directory']:s for s in run.state['stages']}
    jobs={j['id']:j for j in run.state['plan']['jobs']}
    cache=run.state.setdefault('local_checks',{})
    comparisons=[]
    for key in action['jobs']:
        if key not in entry['baseline']['jobs']:continue
        old=stages[entry['baseline']['jobs'][key]]
        new=run.state['jobs'][key]
        reports=[]
        profile=check_profile({**run.contract['jobs'].get(key,{}),'validation_mode':'static'})
        for record in (old,new):
            readiness=record.get('assembly_readiness',{})
            if not readiness.get('loadable'):break
            artifact=record.get('artifact',record.get('directory',record.get('stage',''))+'/artifact')
            from .candidate import usable
            usable(run,{**record,'artifact':artifact})
            binding=check_key(run,record,jobs[key],profile)
            if binding not in cache:
                from time import monotonic
                started=monotonic()
                destination=run.begin('local_repair_check')
                try:
                    report,renders=run.sandbox.recheck_asset(safe_path(run.root,artifact),destination,jobs[key]['asset_ids'],profile)
                except WorkerTimeout as exc:
                    # Only this asset's recheck stalled (e.g. an unstable model); the scene goes on.
                    report,renders={'status':'timeout','checks':[],'detail':str(exc)},None
                render_directory=str(renders.relative_to(run.root)) if renders else None
                stage=run.commit('local_repair_check',{'job':key,'binding':binding,'report':report,
                    'render_directory':render_directory,'seconds':monotonic()-started})
                cache[binding]={'report':report,'directory':stage['directory'],'render_directory':render_directory}
                run.store.save()
            reports.append(cached_check(run,binding)['report'])
        if len(reports)==2 and reports[1].get('status')=='timeout' and reports[0].get('status')!='timeout':
            # The baseline finished the same recheck; the repaired model cannot, so it is not adopted.
            comparisons.append({'job':key,'identical':False,'checks':[],'complete':False,
                'comparison':{'outcome':'regressed','new_defects':[{'check':'local_validate','added':['worker_timeout'],
                    'detail':reports[1]['detail'][:500]}]}})
            continue
        if len(reports)!=2 or any(r.get('status') in {'error','timeout'} for r in reports):continue
        comparison=repair_comparison(*reports)
        comparisons.append({'job':key,'comparison':comparison,
            'identical':old['assembly_readiness']['artifact_sha256']==new['assembly_readiness']['artifact_sha256'],
            'checks':[r['id'] for r in reports[0].get('checks',[]) if r.get('status')=='fail'],
            'complete':all(r.get('checks') and all(complete(c) or c.get('status')=='skipped' for c in r['checks']) for r in reports)})
    entry['local_verification']=comparisons
    regression=any(c['comparison']['new_defects'] for c in comparisons)
    # Unchanged local geometry cannot disprove a placement/cross-asset repair.
    checked={f"asset:{c['job']}:{key}" for c in comparisons for key in c['checks']}
    for c in comparisons:
        if 'material_integrity' in c['checks']:
            checked.update('material_integrity:'+i['id'] for i in run.state['plan']['instances'] if i['job']==c['job'])
    local_targets=bool(entry['target_checks']) and set(entry['target_checks'])<=checked
    geometry_only=not any(action.get(k) for k in ('placements','layout','table_modifications','edge_mounts','end_effectors','robot_posture','tool_additions'))
    ineffective=(geometry_only and bool(comparisons) and len(comparisons)==len(action['jobs'])
        and (all(c['identical'] for c in comparisons) or local_targets
             and all(c['complete'] and c['comparison']['outcome']=='ineffective' for c in comparisons)))
    if not (regression or ineffective):
        run.store.save();return False
    comparison={'outcome':'regressed' if regression else 'ineffective','adopt':False,
        'new_defects':[d for c in comparisons for d in c['comparison']['new_defects']],
        'coverage_lost':[],'resolved_checks':[],'remaining_checks':entry['target_checks'],
        'rule':'Local recheck proved a regression or no improvement on the target; scene quality is not declared passed'}
    entry.update(status='evaluated',comparison=comparison,outcome=comparison['outcome'],
        output_key=run.repair_input_key(),scene=entry['baseline']['scene'],
        calls_consumed=len(run.store.state['calls'])-entry['call_start'])
    run.select_repair_candidate()
    return True
