"""逐个独立组件追踪承托链；世界固定或实例标签本身不是实体承托证据。"""

import numpy as np
import mujoco

from asset_generation.distance import observe


def component_support(model, data, item, geometry, external_contacts, query, tolerance,
                      collision_tolerance, max_pairs, uncertain_objects, upward_resultant):
    """沿实际运动树/启用的焊接及朝上接触传播支撑，不认证连接制造或力矩平衡。"""
    prefix = item['id'] + '__'
    bodies = {b for b in range(1, model.nbody) if model.body(b).name.startswith(prefix)}
    parents = {b: b for b in bodies}

    def find(b):
        while parents[b] != b:
            parents[b] = parents[parents[b]]
            b = parents[b]
        return b

    def join(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    for b in sorted(bodies):
        parent = int(model.body_parentid[b])
        joints = range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
        if parent in bodies and not any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in joints):
            join(b, parent)
    connections = []
    for e in range(model.neq):
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if (model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and data.eq_active[e]
                and a in bodies and b in bodies):
            join(a, b)
            connections.append(model.equality(e).name)
    groups = {}
    for b in sorted(bodies):
        groups.setdefault(find(b), {'bodies': [], 'geoms': []})['bodies'].append(model.body(b).name)
    geom_groups = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in bodies:
            group = find(b)
            groups[group]['geoms'].append(g)
            geom_groups[g] = group
    # 纯坐标节点没有需要承托的实体；独立视觉实体仍须保留，不能凭没有碰撞体通过。
    groups = {key: value for key, value in groups.items() if value['geoms']}
    contacts = {key: [] for key in groups}
    errors, error_count = [], 0
    unassigned = [g for g in geometry if (model.geom(g).name or '').startswith(prefix) and g not in geom_groups]
    if unassigned:
        error_count += len(unassigned)
        errors.extend({'error_kind':'unsupported_check','geom1':model.geom(g).name,'instances':[item['id']],
                       'detail':'实例几何缺少具名物理部件，无法追踪组件连接'} for g in unassigned)
    for row in external_contacts:
        g = model.geom(row['object_geom']).id
        if g in geom_groups:
            contacts[geom_groups[g]].append(row)
    incoming = {key: [] for key in groups}
    links, unknown_links, link_count, examined, skipped = [], set(), 0, 0, 0
    geoms = [g for g in geometry if g in geom_groups]
    epsilon = min(tolerance, 1e-5)
    if len(groups) > 1:
        lower = np.array([geometry[g][0] for g in geoms])
        upper = np.array([geometry[g][1] for g in geoms])
        owners = np.array([geom_groups[g] for g in geoms])
        for i, a in enumerate(geoms):
            near = ((owners[i+1:] != owners[i])
                    & np.all(upper[i] + tolerance >= lower[i+1:], axis=1)
                    & np.all(upper[i+1:] + tolerance >= lower[i], axis=1))
            for index in np.flatnonzero(near) + i + 1:
                b = geoms[index]
                if examined >= max_pairs:
                    skipped += 1
                    unknown_links.add((geom_groups[a],geom_groups[b]))
                    continue
                examined += 1
                row = {**observe(query, a, b, tolerance + .001),
                       'geom1': model.geom(a).name, 'geom2': model.geom(b).name,
                       'instances': [item['id']]}
                distance = row.get('distance_m')
                if (row['outcome'] == 'unknown' or distance is None
                        and (row.get('lower_bound_m') is None or row['lower_bound_m'] <= tolerance)):
                    error_count += 1
                    errors.append(row)
                    unknown_links.add((geom_groups[a],geom_groups[b]))
                    continue
                if distance is None or not -collision_tolerance <= distance <= epsilon or row['normal'] is None:
                    continue
                normal = np.array(row['normal'])
                if abs(normal[2]) <= 1e-7:
                    continue
                from asset_generation.material import visible_witness
                points=row.get("points")
                witnesses=([visible_witness(model,data,query,a,np.array(points[:3])),
                    visible_witness(model,data,query,b,np.array(points[3:]))] if points else [None,None])
                if not all(witnesses):
                    from asset_generation.surface_contact import contact
                    replacement=contact(query,a,b,normal)
                    if replacement is not None:
                        points=replacement["points"]
                        witnesses=[visible_witness(model,data,query,a,np.array(points[:3])),
                                   visible_witness(model,data,query,b,np.array(points[3:]))]
                        if all(witnesses):row={**row,**replacement};normal=np.array(row["normal"])
                if not all(witnesses):
                    error_count+=1;unknown_links.add((geom_groups[a],geom_groups[b]))
                    errors.append({"geom1":model.geom(a).name,"geom2":model.geom(b).name,
                        "error_kind":"unsupported_check","detail":"component contact lacks displayed surface witnesses"})
                    continue
                if abs(normal[2])<=1e-7:continue
                row["visible_witnesses"]=witnesses
                source, target = (a, b) if normal[2] > 0 else (b, a)
                if normal[2] < 0:
                    row = {**row, 'geom1': row['geom2'], 'geom2': row['geom1'],
                           'points': row['points'][3:] + row['points'][:3] if row.get('points') else None}
                directed_normal = (normal if normal[2] > 0 else -normal).tolist()
                incoming[geom_groups[target]].append((geom_groups[source],{'normal':directed_normal}))
                link_count += 1
                if len(links) < 64:
                    links.append({**row, 'normal': directed_normal,
                                  'support_geom': model.geom(source).name, 'object_geom': model.geom(target).name})

    def reachable(seeds, unknown=()):
        reached = set(seeds)
        while True:
            newly = {key for key in groups if key not in reached and upward_resultant(
                contacts[key] + [row for source, row in incoming[key] if source in reached])}
            for a,b in unknown:
                if a in reached:
                    newly.add(b)
                if b in reached:
                    newly.add(a)
            newly -= reached
            if not newly:
                return reached
            reached.update(newly)

    grounded = reachable(())
    missing_geometry = [key for key, value in groups.items() if not any(g in geometry for g in value['geoms'])]
    if missing_geometry:
        error_count += len(missing_geometry)
        errors.extend({'error_kind': 'unsupported_check', 'body': groups[key]['bodies'][0],
                       'instances': [item['id']], 'detail': '独立组件没有可量测的碰撞实体，不能证明其承托'}
                      for key in missing_geometry)
    # 未知只沿可能的承托链传播；另一处未知不能抹掉独立的已确认无承托组件。
    possible = reachable(grounded | {geom_groups[g] for g in uncertain_objects if g in geom_groups}
                         | set(missing_geometry) | (set(groups) if unassigned else set()),unknown_links)
    complete = not uncertain_objects and not error_count and not skipped
    unsupported = [dict(body=value['bodies'][0], bodies=value['bodies']) for key, value in groups.items()
                   if key not in possible]
    summary = [dict(body=value['bodies'][0], bodies=value['bodies'],
                    status='supported' if key in grounded else 'unsupported' if key not in possible else 'unknown')
               for key, value in groups.items()]
    return {'all_components_supported': bool(groups) and not unassigned and len(grounded) == len(groups),
            'component_support': summary, 'unsupported_components': unsupported,
            'component_connections': connections,
            'component_contacts': links, 'component_contact_count': link_count,
            'component_examined_pairs': examined, 'component_unexamined_pairs': skipped,
            'component_errors': errors, 'component_error_count': error_count,
            'component_scope': 'whole_instance', 'component_complete': complete}
