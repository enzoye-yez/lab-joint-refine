"""Quality facts for operation planning; scheduling never grants execution."""


def operation_obligations(scene):
    """What the static stages handed to the robot-operation stage, as one checklist.

    Asset reviews defer cross-instance handling, grasps, loads and sequences to stage="operation";
    the scene review moves items it cannot judge statically to target_stage="operation"; the
    layout assessment lists operation interfaces no dock reaches. Operation planning must verify
    each one with a trajectory (or report it unresolved), and a failure found there is a concrete
    scene or asset defect to route back for repair."""
    rows=[]
    for key,job in sorted(scene.get("jobs",{}).items()):
        for item in job.get("deferred_checks",[]):
            if item.get("stage")=="operation" and item.get("status")=="pending":
                rows.append({"id":item["id"],"source":"asset_review","job":key,"detail":item["detail"]})
    latest=scene.get("latest") or {}
    for item in (latest.get("review") or {}).get("deferred_results",[]):
        if item.get("target_stage")=="operation":
            rows.append({"id":item["id"],"source":"scene_review","detail":item.get("evidence","")})
    assessment=(latest.get("report") or {}).get("layout_assessment") or {}
    for row in assessment.get("instances",[]):
        for target in row.get("targets",[]):
            if target.get("status")!="plausible":
                rows.append({"id":"reach:"+row["instance"]+":"+str(target.get("target")),"source":"layout_assessment",
                             "instance":row["instance"],"detail":"No declared dock reaches this operation interface "
                             "(shoulder to gripper tip); verify with a trajectory or report it unreachable"})
    return rows
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
