"""Targeted read-only checks of the compiled scene; a successful measurement does not mean unstated specifications are met."""

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
            'meaning':'Signed distance of exactly specified geom pairs, nearest points in world coordinates, and the normal from the first geom toward the second; meshes are treated as their collision convex hulls.'}


def support(model, data, meta, request):
    item = next((x for x in meta['plan']['instances'] if x['id']==request['instance']),None)
    require(item is not None and item['placement']!='robot','Support target is not a checkable scene instance')
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
            'meaning':'Actual upward-facing support in the initial pose; without a geom subset, each independent component is traced through the actual kinematic tree, enabled welds and contact chain. A specified geom subset answers only for that scope. A positive gap above 10 micrometers is listed as nearby only, not counted as contact; does not prove manufacturable connection fit, load bearing or dynamic stability.'}


def perimeter(root, model, data, meta, request):
    target=request['instance']; margin=request['distance_m']
    require(target in {i['id'] for i in meta['plan']['instances']},'Perimeter clearance target is not a scene instance')
    ids=[g for g in range(model.ngeom) if owner(model,g)==target and model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE]
    require(ids,'Target has no finite geometry')
    boxes=[bounds(model,data,g) for g in ids]
    lo=np.min([x[0] for x in boxes],axis=0); hi=np.max([x[1] for x in boxes],axis=0)
    require(hi[2]-lo[2]>1e-6,'Invalid target height')
    # Four non-overlapping boxes form the band around the envelope; contacts inside or below it are not part of this perimeter region.
    regions=[([lo[0]-margin,lo[1]-margin,lo[2]],[lo[0],hi[1]+margin,hi[2]]),
             ([hi[0],lo[1]-margin,lo[2]],[hi[0]+margin,hi[1]+margin,hi[2]]),
             ([lo[0],lo[1]-margin,lo[2]],[hi[0],lo[1],hi[2]]),
             ([lo[0],hi[1],lo[2]],[hi[0],hi[1]+margin,hi[2]])]
    spec=mujoco.MjSpec.from_file(str(Path(root)/'model.xml'))
    names=[]
    for i,(a,b) in enumerate(regions):
        name=f'__inspection_perimeter_{i}'
        require(spec.geom(name) is None,'Inspection probe name collision')
        names.append(name)
        spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=(np.array(a)+b)/2,
                                size=(np.array(b)-a)/2,contype=0,conaffinity=0,rgba=[0,0,0,0])
    probe=spec.compile(); state=mujoco.MjData(probe)
    require(probe.nq==model.nq,'Read-only probe unexpectedly changed the joint count')
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
            'meaning':'Rectangular perimeter region outside the world-axis-aligned outer envelope, between the lowest and highest geom heights; excludes the envelope interior and the space below it, includes visual and collision geoms. Not a whole-body minimum distance or a thermal performance certification.'}


def mount(model, data, meta, request):
    target=request['instance']; items=meta['plan']['instances']
    require(any(i['id']==target and i['placement']=='robot' for i in items),'Mount check must target a tool instance on the robot')
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
            'unverified':['A matching transform does not prove the mounting faces match in manufactured size; specify solid geom pairs or read-only dimensional requirements.','Unmodeled cables cannot be inferred to have clearance.'],
            'meaning':'Rigid transform from the actual fixed flange to the tool root, and distances from the current tool to robot geoms; negative-distance contacts are neither excluded nor exempted.'}


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
    # A measured contact answers the question; pairs that could not be measured then leave nothing
    # open (collisions are judged by scene_collisions), so they are listed but are not coverage gaps.
    return {"id":"mount:"+binding["arm"],"status":"pass" if contacts else "partial" if gaps else "fail",
            "instances":[target],"arm":binding["arm"],"mode":binding["mode"],
            "attachment_site":"robot__"+binding["site"],"contacts":contacts[:16],
            **({"unmeasured_pairs":gaps[:32]} if contacts else {"coverage_gaps":gaps[:32]}),
            "complete":bool(contacts) or not gaps,"support_tolerance_m":support_tolerance,"collision_tolerance_m":collision_tolerance,
            "detail":"Custom assembly is in solid contact with the selected wrist/stock gripper; being attached to the same kinematic tree is not treated as a solid joint; does not certify manufactured fit or fastening strength"}
