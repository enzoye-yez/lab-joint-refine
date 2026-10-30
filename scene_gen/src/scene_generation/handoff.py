"""Quality facts for operation planning; scheduling never grants execution."""
def quality_handoff(report, scene_sha256, schedule=()):
    deferred={t.get('check_id',t['id']):t for t in schedule if t['scheduling']['status']=='deferred'}
    rows=[]
    if report.get('status')=='error' and not any(r.get('status')=='error' for r in report.get('checks',[])):
        rows.append({'check':'report_execution','status':'error','instances':[],
            'execution_requirement':True,'global_fault':True,'deferred':False,'detail':report.get('detail')})
    for fault in report.get('recovery_failures',[]):
        if fault.get('status')=='open' and (fault.get('scope')!='checks' or fault.get('kind')=='program'):
            rows.append({'check':fault['id'],'status':'error','instances':[],
                'execution_requirement':True,'global_fault':True,'deferred':False,'detail':fault.get('diagnostic')})
    for row in report.get('checks',[]):
        if row.get('status') in {'pass','skipped'}:continue
        key=row['id']
        required=any(s in key for s in ('interaction','interface','material_integrity','support:','collision','assembly_missing'))
        rows.append({'check':key,'status':row['status'],'instances':row.get('instances',[]),
            'execution_requirement':required,'global_fault':row.get('status')=='error' and not row.get('instances'),
            'deferred':key in deferred,'detail':row.get('detail'),
            'evidence':{k:row[k] for k in ('coverage_gaps','missing_jobs','complete') if k in row}})
    return {'scene_sha256':scene_sha256,'issues':rows,
        'rule':'规划可继续；执行须核对所用对象的前置条件。未知、暂挂与静态可接近性不等于操作通过。'}
