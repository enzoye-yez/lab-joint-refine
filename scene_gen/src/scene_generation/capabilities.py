"""可信工程能力注册与只读执行。注册项提供选择、执行及覆盖声明，主流程无方法分支。"""

from copy import deepcopy
from dataclasses import dataclass

from asset_generation.contracts import require
from asset_prep.storage import digest
from .recovery import gaps, report_status
from asset_generation.check_report import validate_rows, require_report


REGISTRY = {}


def register(capability):
    require(capability.id not in REGISTRY, '恢复能力编号重复')
    REGISTRY[capability.id] = capability
    return capability


@dataclass
class Context:
    artifact: object
    backend: object
    limits: object
    contract: dict
    plan: dict
    input_key: str


def semantic(value):
    """只移除解释文字；保留实际动作参数、证据数值、目标及未知范围。"""
    if isinstance(value, dict):
        return {k:semantic(v) for k,v in value.items()
                if k not in {'reason','detail','error','recovered_from','meaning','findings'}}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def problem_identity(report, review=None):
    problems = []
    for row in report.get('checks',[]):
        if row.get('status') in {'fail','partial','error'} or row.get('coverage_gaps'):
            problems.append({'check':row['id'],'status':row.get('status'),
                'instances':sorted(row.get('instances',[])), 'defect_names':confirmed_names(row),
                'gaps':sorted((sorted((g.get('geom1',''),g.get('geom2',''))) for g in row.get('coverage_gaps',[])))})
    return digest({'checks':sorted(problems,key=lambda r:r['check']),
                   'visual':{k:sorted((review or {}).get(k,[])) for k in ('affected_instances','affected_layout')} if (review or {}).get('verdict')=='repair' else {}})


def repair_signature(value):
    value = semantic(value)
    for key in ('jobs','placements','table_modifications','edge_mounts','geometry_changes'):
        if key in value:
            value[key] = sorted(value[key],key=digest)
    return digest(value)


def repair_mechanism(value):
    """Advisory grouping, never proof that a whole repair direction is impossible.

    Keep operation, target and field identities across all supported action kinds;
    collapse numerical tuning only. Exact actions retain their separate signature.
    """
    def shape(item):
        if isinstance(item, dict):
            return {k:shape(v) for k,v in item.items()}
        if isinstance(item, list):
            return sorted((shape(v) for v in item), key=digest)
        if type(item) in (int, float):
            return "<number>"
        return item
    action = {k:v for k,v in semantic(value).items() if v}
    return digest(shape(action)) if action else "no_change"


def repeated_repair(history, input_key, problem_key, action_key, *, mechanism=None):
    """Reject only a proven ineffective exact action on an unchanged input/output.

    Coverage failures do not disprove geometry. Similar operations with different
    parameters remain possible; their measured history is provided to the planner.
    """
    for entry in history:
        outcome = (entry.get("comparison") or {}).get("outcome", entry.get("outcome"))
        if entry.get("status") != "evaluated" or outcome not in {"ineffective", "regressed", "mixed"}:
            continue
        if input_key not in {entry.get("input_key"), entry.get("output_key")}:
            continue
        if entry.get("problem_key") == problem_key and entry.get("action_key") == action_key:
            return {"reason":"exact_action_repeated", "entry":entry.get("action_key")}
    return None


def proposal(checks, **parameters):
    return {'checks':sorted(checks), **parameters}


def collision_identity(row):
    if row.get('geom1') and row.get('geom2'):
        return tuple(sorted((row['geom1'],row['geom2'])))
    return digest(semantic(row))


def validate_request(request):
    require_report(isinstance(request,dict) and isinstance(request.get('checks'),list), '补查请求缺少检查编号列表')
    checks = request['checks']
    require_report(all(isinstance(c,str) and c for c in checks) and len(checks)==len(set(checks)), '补查请求检查编号无效或重复')
    require_report(bool(checks) or request.get('full_scene') is True, '空检查范围必须明确请求整场景')
    if 'full_scene' in request:
        require_report(type(request['full_scene']) is bool, '整场景范围须为布尔值')
    if 'pairs' in request:
        require_report(isinstance(request['pairs'],list) and bool(request['pairs']) and
                       all(isinstance(p,list) and len(p)==2 and all(isinstance(n,str) and n for n in p)
                           and p[0]!=p[1] for p in request['pairs']), '补查几何对无效')
    return request


def validate_result(result, request, input_key, mode=None):
    validate_request(request)
    require_report(isinstance(result,dict) and result.get('input_key')==input_key, '补查证据版本不匹配')
    scope = result.get('scope')
    require_report(isinstance(scope,str) and scope in {'checks','pairs','localization','scene'}, '补查未声明有效覆盖范围')
    require_report('checks' in result, '补查结果缺少检查列表')
    validate_rows(result['checks'],domain='scene',mode=mode)
    if scope in {'checks','scene'}:
        require_report(all(row.get('component_scope') != 'selected_geoms' for row in result['checks']),
                       '选定几何子集的承托证据不能替代整项实例检查')
    if 'failure' in result:
        failure=result['failure']
        require_report(not result['checks'] and isinstance(failure,dict), '故障结果不能同时声明检查证据')
        require_report(isinstance(failure.get('kind'),str) and failure['kind'] in {'program','temporary','environment'}, '故障类型无效')
        require_report(type(failure.get('retryable')) is bool and (not failure['retryable'] or failure['kind']=='temporary'), '故障重试资格无效')
        require_report(isinstance(failure.get('components'),list) and bool(failure['components'])
                       and all(isinstance(c,str) and c for c in failure['components']), '故障组件来源无效')
    if scope=='scene':
        require_report(request.get('full_scene') is True, '补查越出整场景授权范围')
        require_report(type(result.get('complete_scene')) is bool, '整场景补查缺少完成声明')
    if scope=='pairs':
        require_report('pairs' in request and isinstance(result.get('covered_pairs'),list), '几何对补查缺少实际覆盖范围')
        require_report(all(isinstance(p,list) and len(p)==2 and all(isinstance(n,str) and n for n in p)
                           and p[0]!=p[1] for p in result['covered_pairs']), '实际覆盖几何对无效')
    return result


def merge(report, result, request, input_key, mode=None):
    """只消费同一输入版本、同一检查、声明覆盖范围内的结果。缺失行永不删除。"""
    mode = mode or report.get('validation_mode')
    validate_rows(report.get('checks',[]),domain='scene',mode=mode)
    validate_result(result,request,input_key,mode)
    require_report('failure' not in result, '失败执行不能合并为检查证据')
    rows = {r['id']:deepcopy(r) for r in report.get('checks', [])}
    incoming = result.get('checks', [])
    require(len({r['id'] for r in incoming}) == len(incoming), '补查检查编号重复')
    whole_scene = result.get('scope')=='scene' and request.get('full_scene') is True
    require(whole_scene or all(r['id'] in request['checks'] and r['id'] in rows for r in incoming), '补查越出指定检查范围')
    for new in incoming:
        old = rows.get(new['id'],{})
        if new['status']=='skipped':
            require_report((whole_scene and not old) or old.get('status')=='skipped', '不能用模式排除覆盖原检查证据')
            require_report(result['scope'] in {'checks','scene'}, '局部补查不能排除必要检查')
        if result.get('scope') == 'pairs':
            require(new['id'] == 'scene_collisions', '几何对补查不能替代承托或其他检查')
            if new['status'] in {'error','partial'} and not new.get('coverage_gaps'):
                continue  # 失败执行或没有范围诊断，不能把请求范围算成已完成。
            if new['status']=='fail' and not new.get('collisions'):
                continue
            covered = {tuple(sorted(p)) for p in result.get('covered_pairs', [])}
            requested = {tuple(sorted(p)) for p in request.get('pairs', [])}
            require(covered <= requested, '补查几何对越出请求范围')
            pair = lambda r: tuple(sorted((r.get('geom1',''),r.get('geom2',''))))
            require(all(pair(r) in covered for r in new.get('collisions',[])+new.get('coverage_gaps',[])+new.get('measurement_errors',[])),
                    '局部补查返回了范围外证据')
            undetailed = old.get('complete') is False and not old.get('coverage_gaps')
            remaining = [g for g in old.get('coverage_gaps',[]) if pair(g) not in covered]
            old['coverage_gaps'] = remaining + deepcopy(new.get('coverage_gaps',[]))
            # 局部结果不认证未展开的诊断计数或未量测候选。
            errors = old.get('measurement_errors',[])
            kept_errors = [g for g in errors if pair(g) not in covered]
            removed = len(errors)-len(kept_errors)
            if errors or new.get('measurement_errors'):
                old['measurement_errors'] = kept_errors+deepcopy(new.get('measurement_errors',[]))
            if 'measurement_error_count' in old or 'measurement_error_count' in new:
                old['measurement_error_count'] = max(0,old.get('measurement_error_count',len(errors))-removed)+new.get('measurement_error_count',0)
            old['unexamined_pairs'] = max(old.get('unexamined_pairs',0),new.get('unexamined_pairs',0))
            old['complete'] = (not undetailed and new.get('complete') is not False and not old['coverage_gaps']
                               and not old.get('unexamined_pairs') and not old.get('measurement_error_count')
                               and not old.get('measurement_errors'))
            old['collisions'] = list({collision_identity(h):deepcopy(h)
                                      for h in old.get('collisions',[])+new.get('collisions',[])}.values())
            old['collision_count'] = max(old.get('collision_count',0),len(old['collisions']))
            old['status'] = 'fail' if old.get('status')=='fail' or old['collisions'] else 'partial' if not old['complete'] else 'pass'
            old['instances'] = sorted(set(old.get('instances',[])) | {i for h in new.get('collisions',[]) for i in h.get('instances',[])})
        elif result.get('scope') == 'localization':
            require(old.get('status') == 'fail', '未知几何不能生成可编辑实例')
            require(set(new) <= {'id','status','instances','localization'}, '定位动作不能改写验收证据')
            old['instances'] = sorted(set(old.get('instances',[])) | set(new.get('instances',[])))
            old['localization'] = deepcopy(new.get('localization',{}))
        else:
            require(result.get('scope') == 'checks' or whole_scene, '补查未声明有效覆盖范围')
            # 同版本的已确认缺陷不能被另一算法的沉默或相反结论洗掉。
            if old.get('status') == 'fail':
                if new.get('status') != 'fail':
                    continue
                known = old.get('collisions',[])
                present = {collision_identity(h) for h in new.get('collisions',[])}
                new = deepcopy(new)
                new['collisions'] = new.get('collisions',[]) + [h for h in known if collision_identity(h) not in present]
                new['collision_count'] = max(old.get('collision_count',0),new.get('collision_count',0),len(new['collisions']))
                if old.get('unsupported_components'):
                    known_components = {c['body']:c for c in old['unsupported_components']}
                    known_components.update({c['body']:c for c in new.get('unsupported_components',[])})
                    new['unsupported_components'] = list(known_components.values())
                    if 'all_components_supported' in new:
                        new['all_components_supported'] = False
                    for component in new.get('component_support',[]):
                        if component['body'] in known_components:
                            component['status'] = 'unsupported'
                for field,count in (('disconnected_components','disconnected_count'),('surface_mismatches','surface_mismatch_count')):
                    if old.get(field):
                        known={digest(semantic(v)):v for v in old[field]}
                        known.update({digest(semantic(v)):v for v in new.get(field,[])})
                        new[field]=list(known.values())
                        new[count]=max(old.get(count,0),new.get(count,0),len(new[field]))
                new['instances'] = sorted(set(old.get('instances',[])) | set(new.get('instances',[])))
            rows[new['id']] = deepcopy(new)
    merged = {**deepcopy(report),'checks':list(rows.values())}
    merged['status'] = report_status(merged['checks'])
    # 报告级程序/环境错误不能被局部通过结论清除。
    if report.get('status') == 'error' and report.get('error_kind') != 'measurement_error':
        merged['status'] = 'error'
    return merged


class FullRecheck:
    id = 'full_recheck_v2'
    dependencies = {'main_check','asset_generation.sandbox','asset_generation.local','asset_generation.checks',
                    'asset_generation.distance','asset_generation.surface_contact','asset_generation.convex','asset_generation.material','asset_generation.static_state','scene_generation.checks','scene_generation.geometry',
                    'scene_generation.table','scene_generation.edge_mounts','scene_generation.support','scene_generation.checkpoints'}

    def propose(self, ctx, report):
        scene_fault = any(f['status']=='open' and f['scope']=='scene' for f in report.get('recovery_failures',[]))
        if (gaps(report) or scene_fault) and hasattr(ctx.backend, 'recheck_scene'):
            yield proposal([r['id'] for r in (report['checks'] if scene_fault else gaps(report))],
                           **({'full_scene':True} if scene_fault else {}))

    def execute(self, ctx, request, destination):
        report = ctx.backend.recheck_scene(ctx.artifact,destination,ctx.limits,ctx.contract)
        if report.get('status') == 'error' and report.get('error_kind')!='measurement_error':
            failure = report.get('execution_failure',{'kind':'program','retryable':False,'components':['*'],
                                                      'detail':report.get('detail','检查器未提供故障类型')})
            return {'input_key':ctx.input_key,'scope':'checks','checks':[], 'failure':failure}
        if request.get('full_scene'):
            return {'input_key':ctx.input_key,'scope':'scene','checks':report.get('checks',[]),
                    'complete_scene':report.get('status') in {'pass','fail'}}
        return {'input_key':ctx.input_key,'scope':'checks',
                'checks':[r for r in report.get('checks',[]) if r['id'] in request['checks']]}


class WorkerCapability:
    """扩展执行器按同一个注册表查找能力，无需修改工作进程的方法分支。"""
    dependencies = {'asset_generation.sandbox','asset_generation.local','asset_generation.checks',
                    'asset_generation.distance','asset_generation.surface_contact','asset_generation.convex','asset_generation.static_state','scene_generation.recovery_geometry','scene_generation.geometry','scene_generation.checkpoints'}

    def execute(self, ctx, request, destination):
        return ctx.backend.recover_scene(ctx.artifact,destination,ctx.limits,ctx.contract,
                                         self.id,request,ctx.input_key)


class CollisionCertificates(WorkerCapability):
    id = 'collision_certificates_v1'

    def propose(self, ctx, report):
        if hasattr(ctx.backend,'recover_scene') and any(r['id']=='scene_collisions' for r in gaps(report)):
            yield proposal(['scene_collisions'], max_pairs=800000)

    def inspect(self, root, limits, contract, request, *, checkpoint_dir=None):
        from .recovery_geometry import load_scene, RecoveryQuery
        from .geometry import collision_scan
        model,data,meta = load_scene(root, limits, contract, checkpoint_dir)
        scan = collision_scan(model,data,limits.collision_tolerance_m,
                              query=RecoveryQuery(model,data),max_pairs=request['max_pairs'])
        hits = scan.pop('collisions')
        row = {'id':'scene_collisions','status':'fail' if hits else 'partial' if scan['coverage_gaps'] else 'pass',
               'collisions':hits[:100],'collision_count':len(hits),
               'instances':sorted({i for h in hits for i in h['instances']}),**scan}
        return {'scope':'checks','checks':[row]}


class SupportCertificates(WorkerCapability):
    id = 'support_certificates_v1'
    dependencies = WorkerCapability.dependencies | {'scene_generation.checks','scene_generation.support','asset_generation.material'}

    def propose(self, ctx, report):
        if hasattr(ctx.backend,'recover_scene'):
            for row in gaps(report):
                if row['id'].startswith('support:'):
                    yield proposal([row['id']], instance=row['id'].split(':',1)[1], max_pairs=800000)

    def inspect(self, root, limits, contract, request, *, checkpoint_dir=None):
        from .recovery_geometry import load_scene, RecoveryQuery
        from .checks import support_evidence
        from .geometry import bounds
        import mujoco
        model,data,meta = load_scene(root, limits, contract, checkpoint_dir)
        item = next(i for i in meta['plan']['instances'] if i['id']==request['instance'])
        geometry = {g:bounds(model,data,g) for g in range(model.ngeom)
                    if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE and (model.geom_contype[g] or model.geom_conaffinity[g])}
        evidence = support_evidence(model,data,item,limits.support_tolerance_m,limits.collision_tolerance_m,
                                    geometry,RecoveryQuery(model,data,support=True),max_pairs=request['max_pairs'])
        supported = evidence.pop('supported')
        return {'scope':'checks','checks':[{'id':request['checks'][0],
            'status':'fail' if evidence.get('unsupported_components') else 'partial' if not evidence['complete'] else 'pass' if supported else 'fail',
            'instances':[item['id']],'coverage_gaps':evidence['measurement_errors'],**evidence}]}


def confirmed_names(row):
    """只读取确定缺陷的几何证据，绝不读取覆盖缺口。"""
    names = set()
    for item in [row,*row.get('collisions',[]),*row.get('moving_bodies',[]),*row.get('unsupported_components',[]),*row.get('disconnected_components',[]),*row.get('surface_mismatches',[])]:
        for key in ('geom1','geom2','body','object_geom','support_geom'):
            if isinstance(item.get(key),str):
                names.add(item[key])
    return sorted(names)


class LocateDefects(WorkerCapability):
    id = 'locate_defects_v1'
    failure_scope = 'localization'
    # 只读取编译场景的几何归属与父体链，不经过数值距离核心；声明必须如实，
    # 否则数值核心的独立故障会被错误地判为"无法解除"。
    dependencies = {'asset_generation.sandbox','asset_generation.local','asset_generation.checks',
                    'asset_generation.contracts','scene_generation.recovery_geometry','scene_generation.geometry','scene_generation.checkpoints'}

    def propose(self, ctx, report):
        editable = {i['id'] for i in ctx.plan['instances']}
        if hasattr(ctx.backend,'recover_scene'):
            for row in report.get('checks',[]):
                if row.get('status')=='fail' and not editable.intersection(row.get('instances',[])):
                    yield proposal([row['id']], names=confirmed_names(row))

    def inspect(self, root, limits, contract, request, *, checkpoint_dir=None):
        from .recovery_geometry import load_scene
        from .geometry import owner
        model,data,meta = load_scene(root, limits, contract, checkpoint_dir)
        instances = {i['id'] for i in meta['plan']['instances']}
        mapping = {model.geom(g).name:owner(model,g) for g in range(model.ngeom)}
        # 几何名称优先；移动部件证据可通过实际编译的父体链定位。
        for b in range(1,model.nbody):
            current = b
            while current:
                name = model.body(current).name
                if name and name.split('__',1)[0] in instances:
                    mapping[model.body(b).name] = name.split('__',1)[0]
                    break
                current = int(model.body_parentid[current])
        located = {n:mapping[n] for n in request['names'] if mapping.get(n) in instances}
        return {'scope':'localization','checks':[{'id':request['checks'][0],'status':'fail',
            'instances':sorted(set(located.values())),
            'localization':{'matched':located,'unresolved':[n for n in request['names'] if n not in located]}}]}


class MaterialCertificates(WorkerCapability):
    id="material_certificates_v1"
    dependencies=WorkerCapability.dependencies | {"asset_generation.material"}
    def propose(self,ctx,report):
        if hasattr(ctx.backend,"recover_scene"):
            for row in gaps(report):
                if row["id"].startswith("material_integrity:"):
                    yield proposal([row["id"]],instance=row["id"].split(":",1)[1],max_pairs=800000)
    def inspect(self,root,limits,contract,request, *, checkpoint_dir=None):
        from .recovery_geometry import load_scene, RecoveryQuery
        from asset_generation.material import integrity
        model,data,meta=load_scene(root, limits, contract, checkpoint_dir)
        return {"scope":"checks","checks":[integrity(model,data,instance=request["instance"],
            tolerance=limits.collision_tolerance_m,max_pairs=request["max_pairs"],query=RecoveryQuery(model,data))]}

for _capability in (FullRecheck(), CollisionCertificates(), SupportCertificates(), MaterialCertificates(), LocateDefects()):
    register(_capability)
