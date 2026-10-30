"""检查生产端和消费端共用的数据契约；不包含资产设计规则或验收公差。"""

from asset_prep.errors import ValidationError


ROW_STATUSES = frozenset({'pass', 'fail', 'partial', 'error', 'skipped'})
REPORT_STATUSES = ROW_STATUSES - {'skipped'}
STATIC_SKIPS = {
    'asset': frozenset({'sampled_joint_sweep', 'numerical_stability', 'interaction_probes'}),
    'scene': frozenset({'scene_joint_sweeps', 'settling', 'robot_reachability'}),
}


class CheckContractError(ValidationError):
    """可信工具的报告违约，不能转为模型的几何修复反馈。"""
    component = 'asset_generation.check_report'


def require_report(condition, message):
    if not condition:
        raise CheckContractError(message)


def incomplete(row):
    """覆盖诊断的唯一判定；摘要截断或未提供详细错误不等于完成。"""
    return (row.get('status') in ('partial', 'error') or row.get('complete') is False or row.get('component_complete') is False
            or bool(row.get('coverage_gaps')) or bool(row.get('measurement_errors'))
            or bool(row.get('measurement_error_count', 0)) or bool(row.get('unexamined_pairs', 0)))


def complete(row):
    # skipped 只表示模式排除，永远不是该问题已获得证据。
    return row.get('status') in ('pass', 'fail') and not incomplete(row)


def validate_rows(rows, *, domain='scene', mode=None):
    require_report(isinstance(rows, list), '检查条目须为列表')
    seen = set()
    for row in rows:
        require_report(isinstance(row, dict), '检查条目须为对象')
        name, status = row.get('id'), row.get('status')
        require_report(isinstance(name, str) and bool(name) and name not in seen, '检查编号缺失或重复')
        seen.add(name)
        require_report(isinstance(status, str) and status in ROW_STATUSES, '检查状态无效')
        for key in ('coverage_gaps', 'measurement_errors', 'collisions', 'moving_bodies', 'unsupported_components', 'disconnected_components', 'surface_mismatches'):
            if key in row:
                require_report(isinstance(row[key], list) and all(isinstance(v, dict) for v in row[key]), key+'须为诊断列表')
                for diagnostic in row[key]:
                    for field in ('geom1','geom2'):
                        if field in diagnostic:
                            require_report(isinstance(diagnostic[field],str) and diagnostic[field], '诊断几何名称无效')
                    if 'instances' in diagnostic:
                        require_report(isinstance(diagnostic['instances'],list) and
                                       all(isinstance(i,str) and i for i in diagnostic['instances']), '诊断实例名称无效')
                    if key == 'unsupported_components':
                        require_report(isinstance(diagnostic.get('body'),str) and bool(diagnostic['body']), '未承托组件缺少实际部件名')
        for key in ('collision_count', 'measurement_error_count', 'unexamined_pairs', 'examined_pairs', 'disconnected_count', 'surface_mismatch_count'):
            if key in row:
                require_report(type(row[key]) is int and row[key] >= 0, key+'须为非负整数')
        if 'complete' in row:
            require_report(type(row['complete']) is bool, 'complete须为布尔值')
        if 'connectivity' in row:
            graph=row['connectivity']
            require_report(isinstance(graph,dict) and graph.get('version')==2,'连接证据版本无效')
            names=graph.get('geoms')
            require_report(isinstance(names,list) and all(isinstance(n,str) and n for n in names)
                           and len(names)==len(set(names)),'连接证据须列出完整且唯一的几何名称')
            groups=graph.get('rigid_groups')
            require_report(isinstance(groups,dict),'刚性连接分组无效')
            for field in ('confirmed','possible'):
                flattened=[]
                for group in groups.values():
                    require_report(isinstance(group,dict) and isinstance(group.get(field),list),'连接分区无效')
                    for partition in group[field]:
                        require_report(isinstance(partition,list) and bool(partition)
                            and all(isinstance(n,str) for n in partition),'连接分区缺少几何对象')
                        flattened.extend(partition)
                require_report(sorted(flattened)==sorted(names),'连接分区丢失或重复几何对象')
        for key in ('component_complete','all_components_supported'):
            if key in row:
                require_report(type(row[key]) is bool, key+'须为布尔值')
        if 'component_scope' in row:
            require_report(isinstance(row['component_scope'],str) and row['component_scope'] in {'whole_instance','selected_geoms'},
                           '组件承托证据范围无效')
        if 'component_support' in row:
            components=row['component_support']
            require_report(isinstance(components,list) and all(isinstance(c,dict) and isinstance(c.get('body'),str)
                           and c['body'] and isinstance(c.get('status'),str) and c['status'] in {'supported','unsupported','unknown'}
                           for c in components), '组件承托明细无效')
            if status=='pass' or row.get('all_components_supported') is True:
                require_report(bool(components) and all(c['status']=='supported' for c in components), '未承托或未知组件不能汇总为通过')
        if 'instances' in row:
            require_report(isinstance(row['instances'], list) and all(isinstance(i,str) and i for i in row['instances']),
                           '检查实例须为非空名称列表')
        require_report(status != 'pass' or not incomplete(row), '有覆盖缺口的检查不能声明通过')
        if row.get('collisions') or row.get('collision_count'):
            require_report(status in {'fail','error'}, '已确认碰撞不能声明通过、未知或跳过')
        for detail,count in (('disconnected_components','disconnected_count'),('surface_mismatches','surface_mismatch_count')):
            if row.get(detail) or row.get(count):
                require_report(status in {'fail','error'}, 'confirmed material defect cannot pass or remain unknown')
            if count in row:
                require_report(row[count]>=len(row.get(detail,[])), 'material defect count omits evidence')
        if row.get('unsupported_components'):
            require_report(status in {'fail','error'}, '已确认未承托组件不能声明通过、未知或跳过')
            require_report(row.get('all_components_supported') is not True, '已确认未承托组件不能汇总为全部有承托')
        if 'collision_count' in row:
            require_report(row['collision_count'] >= len(row.get('collisions',[])), '碰撞摘要少于已列出的证据')
        if status == 'skipped':
            require_report(mode == 'static' and (name.split(':',1)[-1] if domain=='asset' and row.get('configuration')==name.split(':',1)[0] else name) in STATIC_SKIPS.get(domain, ()), '必要检查不能由结果自行声明跳过')
            require_report(not incomplete(row), '跳过条目不能掩盖未完成检查')
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
    require_report(isinstance(report, dict), '检查报告须为对象')
    require_report(isinstance(report.get('status'),str) and report['status'] in REPORT_STATUSES, '报告状态缺失或无效')
    require_report(report.get('validation_mode',mode) == mode, '检查报告验证模式与调用约定不一致')
    require_report('checks' in report, '检查报告缺少checks')
    validate_rows(report['checks'], domain=domain, mode=mode)
    if require_renders or 'renders' in report:
        require_report(isinstance(report.get('renders'),list) and all(isinstance(p,str) and p for p in report['renders']),
                       '检查报告缺少有效渲染路径列表')
        if require_renders and report['status']=='pass':
            require_report(bool(report['renders']), '待审查的通过报告缺少渲染证据')
    derived = report_status(report['checks'])
    # 报告级执行错误可以没有条目；其余状态必须从实际条目派生。
    require_report(report['status'] == derived or report['status'] == 'error', '报告总状态与检查条目矛盾')
    return report


def failed_report(exc):
    from scene_generation.faults import exception_failure
    failure = exception_failure(exc)
    return {'status':'error', 'error_kind':failure['error_kind'], 'checks':[], 'renders':[],
            'execution_failure':failure, 'detail':str(exc),
            'coverage':['检查工具或报告契约错误，不能作为资产缺陷',str(exc)]}


def checked_report(report, *, domain, mode, require_renders=False):
    """调用方使用错误报告分流，避免契约异常被资产作者的格式重试捕获。"""
    try:
        return validate_report(report, domain=domain, mode=mode,require_renders=require_renders)
    except CheckContractError as exc:
        return failed_report(exc)


def checked_previews(report, directory, loader):
    """渲染交付错误属于证据工具，保留已有检查事实，不转为几何修改反馈。"""
    from copy import deepcopy
    try:
        names = report.get('renders',[])
        require_report(not names or directory is not None, '渲染报告没有对应输出目录')
        images = [loader(directory,name) for name in names]
        require_report(report['status']!='pass' or bool(images), '通过报告没有可用预览证据')
        return report,images
    except (ValidationError,OSError) as exc:
        error = failed_report(exc)
        error['checks'] = deepcopy(report.get('checks',[]))
        error['coverage'] = list(report.get('coverage',[]))+error['coverage']
        return {**report,**error},[]


def inspection_results(report, requests):
    """补证据逐项回应请求；总pass不能替代缺失、重复或无关的结果。"""
    from asset_prep.storage import digest
    require_report(isinstance(report,dict) and report.get('status') in ('pass','incomplete'), '只读检查报告状态无效')
    require_report(type(report.get('simulation_steps',0)) is int and report.get('simulation_steps',0)==0 and report.get('source_mutated',False) is False,
                   '只读检查改变了场景或推进仿真')
    require_report(isinstance(report.get('renders',[]),list) and
                   all(isinstance(p,str) and p for p in report.get('renders',[])), '只读图像路径列表无效')
    requested = {digest(r):r for r in requests}
    require_report(len(requested)==len(requests), '只读检查请求重复')
    rows = report.get('results',[])
    require_report(isinstance(rows,list), '只读检查结果须为列表')
    results = {}
    for row in rows:
        require_report(isinstance(row,dict) and isinstance(row.get('request'),dict), '检查结果缺少对应请求')
        key = digest(row['request'])
        require_report(key in requested and key not in results, '检查结果重复或越出请求范围')
        results[key] = row
        if report['status']=='pass':
            require_report(not incomplete(row) and not row.get('error') and not row.get('error_kind'), '未完成的补证不能声明成功')
            if row['request']['kind']=='view':
                require_report(isinstance(row.get('image'),str) and row['image'] in report.get('renders',[]), '视图结果缺少图像')
    if report['status']=='pass':
        expected = {k for k,r in requested.items() if r['kind']!='inventory'}
        require_report(expected <= set(results), '成功报告遗漏请求结果')
        if any(r['kind']=='inventory' for r in requests):
            require_report(isinstance(report.get('inventory'),dict) and bool(report['inventory']), '清单检查未返回清单')
    return results
