"""Data contract shared by check producers and consumers; contains no asset design rules or acceptance tolerances."""

from asset_prep.errors import ValidationError


ROW_STATUSES = frozenset({'pass', 'fail', 'partial', 'error', 'skipped'})
REPORT_STATUSES = ROW_STATUSES - {'skipped'}
STATIC_SKIPS = {
    # Asset motion, settling and probes run in every mode; an asset stage finishes its own checks.
    'asset': frozenset(),
    'scene': frozenset({'scene_joint_sweeps', 'settling', 'robot_reachability'}),
}


class CheckContractError(ValidationError):
    """A report contract violation by a trusted tool; must not be turned into geometry repair feedback for the model."""
    component = 'asset_generation.check_report'


def require_report(condition, message):
    if not condition:
        raise CheckContractError(message)


def incomplete(row):
    """The single authority on coverage diagnostics; a truncated summary or missing error details does not mean complete."""
    return (row.get('status') in ('partial', 'error') or row.get('complete') is False or row.get('component_complete') is False
            or bool(row.get('coverage_gaps')) or bool(row.get('measurement_errors'))
            or bool(row.get('measurement_error_count', 0)) or bool(row.get('unexamined_pairs', 0)))


def complete(row):
    # skipped only means excluded by mode; it never means evidence was obtained for that question.
    return row.get('status') in ('pass', 'fail') and not incomplete(row)


def validate_rows(rows, *, domain='scene', mode=None):
    require_report(isinstance(rows, list), 'Check entries must be a list')
    seen = set()
    for row in rows:
        require_report(isinstance(row, dict), 'Check entry must be an object')
        name, status = row.get('id'), row.get('status')
        require_report(isinstance(name, str) and bool(name) and name not in seen, 'Check id missing or duplicated')
        seen.add(name)
        require_report(isinstance(status, str) and status in ROW_STATUSES, 'Invalid check status')
        for key in ('coverage_gaps', 'measurement_errors', 'collisions', 'moving_bodies', 'unsupported_components', 'disconnected_components', 'surface_mismatches'):
            if key in row:
                require_report(isinstance(row[key], list) and all(isinstance(v, dict) for v in row[key]), key+' must be a list of diagnostics')
                for diagnostic in row[key]:
                    for field in ('geom1','geom2'):
                        if field in diagnostic:
                            require_report(isinstance(diagnostic[field],str) and diagnostic[field], 'Invalid diagnostic geom name')
                    if 'instances' in diagnostic:
                        require_report(isinstance(diagnostic['instances'],list) and
                                       all(isinstance(i,str) and i for i in diagnostic['instances']), 'Invalid diagnostic instance name')
                    if key == 'unsupported_components':
                        require_report(isinstance(diagnostic.get('body'),str) and bool(diagnostic['body']), 'Unsupported component is missing its actual part name')
        for key in ('collision_count', 'measurement_error_count', 'unexamined_pairs', 'examined_pairs', 'disconnected_count', 'surface_mismatch_count'):
            if key in row:
                require_report(type(row[key]) is int and row[key] >= 0, key+' must be a non-negative integer')
        if 'complete' in row:
            require_report(type(row['complete']) is bool, 'complete must be a boolean')
        if 'connectivity' in row:
            graph=row['connectivity']
            require_report(isinstance(graph,dict) and graph.get('version')==2,'Invalid connectivity evidence version')
            names=graph.get('geoms')
            require_report(isinstance(names,list) and all(isinstance(n,str) and n for n in names)
                           and len(names)==len(set(names)),'Connectivity evidence must list complete and unique geom names')
            groups=graph.get('rigid_groups')
            require_report(isinstance(groups,dict),'Invalid rigid-connection grouping')
            for field in ('confirmed','possible'):
                flattened=[]
                for group in groups.values():
                    require_report(isinstance(group,dict) and isinstance(group.get(field),list),'Invalid connectivity partition')
                    for partition in group[field]:
                        require_report(isinstance(partition,list) and bool(partition)
                            and all(isinstance(n,str) for n in partition),'Connectivity partition is missing geom objects')
                        flattened.extend(partition)
                require_report(sorted(flattened)==sorted(names),'Connectivity partitions lose or duplicate geom objects')
        for key in ('component_complete','all_components_supported'):
            if key in row:
                require_report(type(row[key]) is bool, key+' must be a boolean')
        if 'component_scope' in row:
            require_report(isinstance(row['component_scope'],str) and row['component_scope'] in {'whole_instance','selected_geoms'},
                           'Invalid scope of component support evidence')
        if 'component_support' in row:
            components=row['component_support']
            require_report(isinstance(components,list) and all(isinstance(c,dict) and isinstance(c.get('body'),str)
                           and c['body'] and isinstance(c.get('status'),str) and c['status'] in {'supported','unsupported','unknown'}
                           for c in components), 'Invalid component support details')
            if status=='pass' or row.get('all_components_supported') is True:
                require_report(bool(components) and all(c['status']=='supported' for c in components), 'Unsupported or unknown components cannot be summarized as pass')
        if 'instances' in row:
            require_report(isinstance(row['instances'], list) and all(isinstance(i,str) and i for i in row['instances']),
                           'Check instances must be a list of non-empty names')
        require_report(status != 'pass' or not incomplete(row), 'A check with coverage gaps cannot declare pass')
        if row.get('collisions') or row.get('collision_count'):
            require_report(status in {'fail','error'}, 'A confirmed collision cannot declare pass, unknown or skipped')
        for detail,count in (('disconnected_components','disconnected_count'),('surface_mismatches','surface_mismatch_count')):
            if row.get(detail) or row.get(count):
                require_report(status in {'fail','error'}, 'confirmed material defect cannot pass or remain unknown')
            if count in row:
                require_report(row[count]>=len(row.get(detail,[])), 'material defect count omits evidence')
        if row.get('unsupported_components'):
            require_report(status in {'fail','error'}, 'A confirmed unsupported component cannot declare pass, unknown or skipped')
            require_report(row.get('all_components_supported') is not True, 'A confirmed unsupported component cannot be summarized as all components supported')
        if 'collision_count' in row:
            require_report(row['collision_count'] >= len(row.get('collisions',[])), 'Collision summary is smaller than the listed evidence')
        if status == 'skipped':
            require_report(mode == 'static' and (name.split(':',1)[-1] if domain=='asset' and row.get('configuration')==name.split(':',1)[0] else name) in STATIC_SKIPS.get(domain, ()), 'A required check cannot declare itself skipped from its result')
            require_report(not incomplete(row), 'A skipped entry cannot hide an unfinished check')
    return rows


def report_status(rows):
    if not isinstance(rows,list) or any(not isinstance(r,dict) or not isinstance(r.get('status'),str)
                                      or r['status'] not in ROW_STATUSES for r in rows):
        return 'error'
    if any(r.get('status') == 'error' for r in rows):
        return 'error'
    if any(r.get('status') == 'fail' for r in rows):
        return 'fail'
    if not rows or all(r['status']=='skipped' for r in rows) or any(incomplete(r) for r in rows):
        return 'partial'
    return 'pass'


def validate_report(report, *, domain, mode, require_renders=False):
    require_report(isinstance(report, dict), 'Check report must be an object')
    require_report(isinstance(report.get('status'),str) and report['status'] in REPORT_STATUSES, 'Report status missing or invalid')
    require_report(report.get('validation_mode',mode) == mode, 'Check report validation mode does not match the calling contract')
    require_report('checks' in report, 'Check report is missing checks')
    validate_rows(report['checks'], domain=domain, mode=mode)
    if require_renders or 'renders' in report:
        require_report(isinstance(report.get('renders'),list) and all(isinstance(p,str) and p for p in report['renders']),
                       'Check report is missing a valid list of render paths')
        if require_renders and report['status']=='pass':
            require_report(bool(report['renders']), 'A passing report awaiting review is missing render evidence')
    derived = report_status(report['checks'])
    # A report-level execution error may have no entries; any other status must be derived from the actual entries.
    require_report(report['status'] == derived or report['status'] == 'error', 'Overall report status contradicts the check entries')
    return report


def failed_report(exc):
    from scene_generation.faults import exception_failure
    failure = exception_failure(exc)
    return {'status':'error', 'error_kind':failure['error_kind'], 'checks':[], 'renders':[],
            'execution_failure':failure, 'detail':str(exc),
            'coverage':['Check tool or report contract error; not an asset defect',str(exc)]}


def checked_report(report, *, domain, mode, require_renders=False):
    """Callers route via the error report so contract exceptions are not caught by the asset author's format retries."""
    try:
        return validate_report(report, domain=domain, mode=mode,require_renders=require_renders)
    except CheckContractError as exc:
        return failed_report(exc)


def checked_previews(report, directory, loader):
    """Render delivery errors belong to the evidence tool: keep the existing check facts and do not turn them into geometry-edit feedback."""
    from copy import deepcopy
    try:
        names = report.get('renders',[])
        require_report(not names or directory is not None, 'Render report has no corresponding output directory')
        images = [loader(directory,name) for name in names]
        require_report(report['status']!='pass' or bool(images), 'Passing report has no usable preview evidence')
        return report,images
    except (ValidationError,OSError) as exc:
        error = failed_report(exc)
        error['checks'] = deepcopy(report.get('checks',[]))
        error['coverage'] = list(report.get('coverage',[]))+error['coverage']
        return {**report,**error},[]


def inspection_results(report, requests):
    """Additional evidence answers each request individually; an overall pass cannot stand in for missing, duplicated or unrelated results."""
    from asset_prep.storage import digest
    require_report(isinstance(report,dict) and report.get('status') in ('pass','incomplete'), 'Invalid read-only check report status')
    require_report(type(report.get('simulation_steps',0)) is int and report.get('simulation_steps',0)==0 and report.get('source_mutated',False) is False,
                   'Read-only check changed the scene or advanced the simulation')
    require_report(isinstance(report.get('renders',[]),list) and
                   all(isinstance(p,str) and p for p in report.get('renders',[])), 'Invalid read-only image path list')
    requested = {digest(r):r for r in requests}
    require_report(len(requested)==len(requests), 'Duplicate read-only check requests')
    rows = report.get('results',[])
    require_report(isinstance(rows,list), 'Read-only check results must be a list')
    results = {}
    for row in rows:
        require_report(isinstance(row,dict) and isinstance(row.get('request'),dict), 'Check result is missing its request')
        key = digest(row['request'])
        require_report(key in requested and key not in results, 'Check result duplicated or outside the requested scope')
        results[key] = row
        if report['status']=='pass':
            require_report(not incomplete(row) and not row.get('error') and not row.get('error_kind'), 'Unfinished additional evidence cannot declare success')
            if row['request']['kind']=='view':
                require_report(isinstance(row.get('image'),str) and row['image'] in report.get('renders',[]), 'View result is missing its image')
    if report['status']=='pass':
        expected = {k for k,r in requested.items() if r['kind']!='inventory'}
        require_report(expected <= set(results), 'Successful report omits requested results')
        if any(r['kind']=='inventory' for r in requests):
            require_report(isinstance(report.get('inventory'),dict) and bool(report['inventory']), 'Inventory check returned no inventory')
    return results
