"""已编译场景的专项只读检查；量测成功不等于满足未提供的规格。"""

from pathlib import Path
import numpy as np
import mujoco

from asset_generation.contracts import require
from asset_generation.distance import DistanceQuery, observe
from .geometry import bounds, owner, aggregate_bounds
from .checks import support_evidence


def pairs(model, data, request):
    query = DistanceQuery(model,data)
    results = [{**observe(query,model.geom(a).id,model.geom(b).id,request['distance_m']),
                'geom1':a,'geom2':b,'instances':[owner(model,model.geom(a).id),owner(model,model.geom(b).id)]}
               for a,b in request['pairs']]
    return {'kind':'pairs','results':results,'distance_recoveries':query.recoveries,
            'measurement_errors':[r for r in results if r['outcome']=='unknown'],
            'meaning':'精确指定几何对的有符号距离、世界坐标最近点和从第一几何朝向第二几何的法向；网格按碰撞凸包处理。'}


def support(model, data, meta, request):
    item = next((x for x in meta['plan']['instances'] if x['id']==request['instance']),None)
    require(item is not None and item['placement']!='robot','承托目标不是可检查的场景实例')
    geometry = {g:bounds(model,data,g) for g in range(model.ngeom)
                if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE and (model.geom_contype[g] or model.geom_conaffinity[g])}
    query=DistanceQuery(model,data)
    result=support_evidence(model,data,item,request.get("support_tolerance_m",.003),request.get("collision_tolerance_m",.0005),geometry,query,
                            request.get('object_geoms'),request.get('support_geoms'))
    row={"id":"support:"+item["id"],"instances":[item["id"]],
         "status":"fail" if result.get("unsupported_components") else "partial" if not result["complete"] else "pass" if result["supported"] else "fail",
         "coverage_gaps":result["measurement_errors"],**result}
    return {'checks':[row] if result.get('component_scope')=='whole_instance' else [],'kind':'support','instance':item['id'],'support_instance':item.get('support_instance'),
            **result,'distance_recoveries':query.recoveries,
            'meaning':'初始姿态的实际朝上承托；未指定几何子集时逐独立组件追踪实际运动树、启用焊接与接触链。指定几何子集只回答该范围。正间隙超过10微米仅列为邻近，不计作接触；不证明连接制造适配、承重及动力学稳定。'}


def perimeter(root, model, data, meta, request):
    target=request['instance']; margin=request['distance_m']
    require(target in {i['id'] for i in meta['plan']['instances']},'周边净空目标不是场景实例')
    ids=[g for g in range(model.ngeom) if owner(model,g)==target and model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE]
    require(ids,'目标没有有限几何')
    boxes=[bounds(model,data,g) for g in ids]
    lo=np.min([x[0] for x in boxes],axis=0); hi=np.max([x[1] for x in boxes],axis=0)
    require(hi[2]-lo[2]>1e-6,'目标高度无效')
    # 四个互不重叠的矩形体组成包络外周带，内部和底部接触不属于此周边区。
    regions=[([lo[0]-margin,lo[1]-margin,lo[2]],[lo[0],hi[1]+margin,hi[2]]),
             ([hi[0],lo[1]-margin,lo[2]],[hi[0]+margin,hi[1]+margin,hi[2]]),
             ([lo[0],lo[1]-margin,lo[2]],[hi[0],lo[1],hi[2]]),
             ([lo[0],hi[1],lo[2]],[hi[0],hi[1]+margin,hi[2]])]
    spec=mujoco.MjSpec.from_file(str(Path(root)/'model.xml'))
    names=[]
    for i,(a,b) in enumerate(regions):
        name=f'__inspection_perimeter_{i}'
        require(spec.geom(name) is None,'检查探针名称冲突')
        names.append(name)
        spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=(np.array(a)+b)/2,
                                size=(np.array(b)-a)/2,contype=0,conaffinity=0,rgba=[0,0,0,0])
    probe=spec.compile(); state=mujoco.MjData(probe)
    require(probe.nq==model.nq,'只读探针意外改变关节数')
    state.qpos[:]=data.qpos
    from asset_generation.static_state import geometry_state
    geometry_state(probe,state)
    query=DistanceQuery(probe,state); intrusions=[]; count=0; errors=[]; skipped=0; error_count=0
    for g in range(model.ngeom):
        if owner(model,g) in {target,'environment','unassigned','floor','table'} or model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        a,b=bounds(model,data,g)
        for name,(rl,rh) in zip(names,regions):
            if np.any(b<rl) or np.any(a>rh):
                continue
            if count>=20000:
                skipped+=1
                continue
            count+=1
            result=observe(query,probe.geom(name).id,probe.geom(model.geom(g).name).id,.001)
            if result['outcome']=='unknown':
                error_count+=1
                if len(errors)<128:
                    errors.append({**result,'geom1':name,'geom2':model.geom(g).name,'instances':[target,owner(model,g)]})
                continue
            if result['distance_m'] is not None and result['distance_m'] < -2e-6:
                intrusions.append({**result,'instance':owner(model,g)})
    if skipped:
        errors.append({'error_kind':'resource_limit','unexamined_pairs':skipped})
    return {'kind':'perimeter','instance':target,'width_m':margin,'target_bounds':[lo.tolist(),hi.tolist()],
            'regions':regions,'intrusions':intrusions,'examined_pairs':count,'region_clear':not intrusions and not errors,
            'measurement_errors':errors,'measurement_error_count':error_count,'complete':not errors,
            'distance_recoveries':query.recoveries,
            'meaning':'世界轴对齐外包络之外、目标最低至最高几何高度之间的矩形周边区；排除包络内部和其下方，含视觉及碰撞几何。不是整体最小距离或散热性能认证。'}


def mount(model, data, meta, request):
    target=request['instance']; items=meta['plan']['instances']
    require(any(i['id']==target and i['placement']=='robot' for i in items),'安装检查须指向机器人上的工具实例')
    from .robot_contract import mount_for
    site_name='robot__'+mount_for(meta['plan'],meta['environment'],target)['site']
    sid=model.site(site_name).id; rotation=data.site_xmat[sid].reshape(3,3); origin=data.site_xpos[sid]
    recorded=next((m for m in meta.get('mounts',[]) if m['instance']==target),None)
    expected={r['body']:r for r in recorded['roots']} if recorded else {}
    roots=[]
    for body in range(1,model.nbody):
        name=model.body(body).name
        if not name.startswith(target+'__') or model.body(int(model.body_parentid[body])).name.startswith(target+'__'):
            continue
        local_pos=rotation.T@(data.xpos[body]-origin)
        local_rot=rotation.T@data.xmat[body].reshape(3,3)
        row={'body':name,'parent_body':model.body(int(model.body_parentid[body])).name,
             'position_in_flange_m':local_pos.tolist(),'rotation_in_flange':local_rot.tolist()}
        if name in expected:
            matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,np.array(expected[name]['local_quat']))
            row.update(position_error_m=float(np.linalg.norm(local_pos-expected[name]['local_pos'])),
                       rotation_matrix_error=float(np.linalg.norm(local_rot-matrix.reshape(3,3))))
        roots.append(row)
    query=DistanceQuery(model,data); nearby=[]; count=0; errors=[]; skipped=0; error_count=0
    tools=[g for g in range(model.ngeom) if owner(model,g)==target]
    robot=[g for g in range(model.ngeom) if owner(model,g)=='robot']
    boxes={g:bounds(model,data,g) for g in tools+robot}
    for a in tools:
        for b in robot:
            lo,hi=boxes[a];other_lo,other_hi=boxes[b]
            if np.linalg.norm(np.maximum(np.maximum(lo-other_hi,other_lo-hi),0))>request['distance_m']:
                continue
            if count>=200000:
                skipped+=1
                continue
            count+=1
            evidence=observe(query,b,a,request['distance_m'])
            if evidence['outcome']=='unknown':
                error_count+=1
                if len(errors)<128:
                    errors.append({**evidence,'geom1':model.geom(b).name,'geom2':model.geom(a).name,'instances':['robot',target]})
                continue
            if evidence['distance_m'] is not None:
                nearby.append(evidence)
    nearby.sort(key=lambda r:r['distance_m'])
    if skipped:
        errors.append({'error_kind':'resource_limit','unexamined_pairs':skipped})
    return {'kind':'mount','instance':target,'attachment_site':site_name,'world_position':origin.tolist(),
            'world_rotation':rotation.tolist(),'roots':roots,'expected_transform_available':bool(recorded),
            'attachment_transform_matches':bool(recorded and roots and all(r.get('position_error_m',1)<1e-7 and r.get('rotation_matrix_error',1)<1e-7 for r in roots)),
            'query_distance_m':request['distance_m'],'nearby_pairs':nearby[:64],'nearby_pair_count':len(nearby),
            'truncated':len(nearby)>64,'examined_pairs':count,'distance_recoveries':query.recoveries,
            'measurement_errors':errors,'measurement_error_count':error_count,'complete':not errors,
            'unverified':['变换一致不证明安装面制造尺寸匹配；需指定实体几何对或只读尺寸要求。','没有建模的电缆不能推断为具备净空。'],
            'meaning':'实际固定法兰至工具根部的刚性变换，及当前工具对机器人几何的距离；不排除或豁免负距离接触。'}


def mount_contact(model, data, meta, target, support_tolerance, collision_tolerance, query):
    """A rigid parent alone does not prove that an accessory touches its mount."""
    from .robot_contract import mount_for
    binding=mount_for(meta["plan"],meta["environment"],target)
    sid=model.site("robot__"+binding["site"]).id
    wrist=int(model.site_bodyid[sid])
    def under(body):
        while body:
            if body==wrist:return True
            body=int(model.body_parentid[body])
        return False
    tool=[g for g in range(model.ngeom) if owner(model,g)==target and
          (model.geom_contype[g] or model.geom_conaffinity[g])]
    selected=[g for g in range(model.ngeom) if owner(model,g)=="robot" and under(int(model.geom_bodyid[g]))
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
    boxes={g:bounds(model,data,g) for g in tool+selected}
    contacts=[];gaps=[]
    for a in tool:
        for b in selected:
            lo,hi=boxes[a];other_lo,other_hi=boxes[b]
            if np.linalg.norm(np.maximum(np.maximum(lo-other_hi,other_lo-hi),0))>support_tolerance:
                continue
            result=observe(query,b,a,support_tolerance+.001)
            evidence={**result,"geom1":model.geom(b).name,"geom2":model.geom(a).name}
            if result["outcome"]=="unknown":
                gaps.append(evidence)
            elif result["distance_m"] is not None and -collision_tolerance<=result["distance_m"]<=support_tolerance:
                contacts.append(evidence)
    return {"id":"mount:"+binding["arm"],"status":"pass" if contacts else "partial" if gaps else "fail",
            "instances":[target],"arm":binding["arm"],"mode":binding["mode"],
            "attachment_site":"robot__"+binding["site"],"contacts":contacts[:16],"coverage_gaps":gaps[:32],
            "complete":not gaps,"support_tolerance_m":support_tolerance,"collision_tolerance_m":collision_tolerance,
            "detail":"定制总成与所选腕部/原厂夹爪实体邻接；不把连接到同一运动树当作实体接合，不认证制造配合或紧固强度"}
