"""Geometric continuity of modeled connections; body membership alone is not evidence.

Primitive unions within one body are one material representation. Distinct bodies are
checked for intersections independently of engine contact filters. Mesh hull contact
is admitted only with witnesses on original triangle surfaces; unsupported overlaps
remain unknown. This is geometric assembly evidence, not strength certification.
"""
import numpy as np
import mujoco
from .distance import DistanceQuery, observe

def _find(parent, value):
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value

def _join(parent, a, b):
    a,b=_find(parent,a),_find(parent,b)
    parent[max(a,b)]=min(a,b)

def visible(model,g):
    return model.geom_rgba[g,3]>0 and model.geom_group[g]<3

def connection_groups(model,data,bodies,articulated=True):
    parent={b:b for b in bodies}
    for b in bodies:
        p=int(model.body_parentid[b])
        joints=range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b])
        if p in parent and (articulated or not model.body_jntnum[b]) and not any(model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE for j in joints):
            _join(parent,b,p)
    for e in range(model.neq):
        a,b=int(model.eq_obj1id[e]),int(model.eq_obj2id[e])
        if model.eq_type[e]==mujoco.mjtEq.mjEQ_WELD and data.eq_active[e] and a in parent and b in parent:
            _join(parent,a,b)
    return {b:_find(parent,b) for b in bodies}

def closed_surface(vertices,faces):
    """Topological prerequisite for a sampled original-surface union witness."""
    _,indices=np.unique(vertices,axis=0,return_inverse=True)
    faces=indices[faces]
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    return bool(len(faces)>=4 and np.all(counts==2) and not np.any(
        np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))))


def union_witness(query,a,b,max_points=4096):
    """One original closed-mesh surface point strictly inside the other solid.

    This proves a constructive union within one rigid body only. It supplies no
    upper penetration bound and must never clear a cross-body collision check.
    """
    model,data=query.model,query.data
    if not hasattr(query,"union_surfaces"):query.union_surfaces={}
    for source,target in ((a,b),(b,a)):
        if model.geom_type[source]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=int(model.geom_dataid[source])
        if mesh not in query.union_surfaces:
            vertices,triangles=query.vertices(source)
            start,count=model.mesh_faceadr[mesh],model.mesh_facenum[mesh]
            query.union_surfaces[mesh]=(np.vstack((vertices,triangles.mean(axis=1)))[:max_points]
                if closed_surface(vertices,model.mesh_face[start:start+count]) else None)
        candidates=query.union_surfaces[mesh]
        if candidates is None:continue
        for local in candidates:
            point=local@data.geom_xmat[source].reshape(3,3).T+data.geom_xpos[source]
            radius=query.interior_radius(target,point)
            if radius is not None and radius>2e-6 and query.surface_error(source,point)<=2e-6:
                return {"method":"original_surface_inside_solid","surface_geom":model.geom(source).name,
                        "solid_geom":model.geom(target).name,"point":point.tolist(),
                        "interior_margin_m":float(radius-2e-6)}
    return None


def convex_solid(vertices,faces):
    """Certify original closed triangle material equals its convex hull.

    No optional graph/SciPy dependency. Bound work; unsupported meshes remain unknown.
    """
    if len(faces)<4 or len(vertices)*len(faces)>10000000:return False
    vertices,indices=np.unique(vertices,axis=0,return_inverse=True)
    faces=indices[faces]
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(counts!=2):return False
    winding=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    if np.any(winding):return False
    triangles=vertices[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths<=1e-14):return False
    normals/=lengths[:,None]
    for start in range(0,len(faces),64):
        n=normals[start:start+64];t=triangles[start:start+64,0]
        projection=vertices@n.T-np.sum(t*n,axis=1)
        if np.any((projection.max(axis=0)>1e-8)&(projection.min(axis=0)<-1e-8)):return False
    return True

def integrity(model,data,*,instance=None,tolerance=.0005,max_pairs=200000,query=None):
    from scene_generation.geometry import bounds
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0
           and model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_PLANE,mujoco.mjtGeom.mjGEOM_HFIELD)
           and (instance is None or (model.geom(g).name or "").startswith(instance+"__"))]
    bodies={int(model.geom_bodyid[g]) for g in geoms}
    # Include pure ancestor frames for a correctly modeled joint tree.
    for b in list(bodies):
        p=int(model.body_parentid[b])
        while p and (instance is None or (model.body(p).name or "").startswith(instance+"__")):
            bodies.add(p);p=int(model.body_parentid[p])
    groups=connection_groups(model,data,bodies)
    rigid=connection_groups(model,data,bodies,articulated=False)
    displayed=[g for g in geoms if visible(model,g)]
    colliders=[g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    query=query or DistanceQuery(model,data,recovery=True)
    boxes={g:bounds(model,data,g) for g in geoms}
    components={g:g for g in displayed};possible={g:g for g in displayed}
    rigid_components=components.copy();rigid_possible=components.copy()
    gaps=[];defects=[];hits=[];examined=0;skipped=0;unknown_count=0
    convex_meshes={};unknown_pairs=[];connection_witnesses=[]
    eps=1e-5
    # Candidate generation is conservative; omitted pairs retain uncertainty.
    # Same-body overlaps can represent one solid, but never bridge a positive gap.
    considered=sorted(set(displayed+colliders))
    lower=np.array([boxes[g][0] for g in considered]).reshape(-1,3)
    upper=np.array([boxes[g][1] for g in considered]).reshape(-1,3)
    for i,a in enumerate(considered):
        lo,hi=boxes[a]
        ba=int(model.geom_bodyid[a])
        near=np.all(upper[i]+eps>=lower[i+1:],axis=1) & np.all(upper[i+1:]+eps>=lower[i],axis=1)
        for index in np.flatnonzero(near)+i+1:
            b=considered[index]
            bb=int(model.geom_bodyid[b])
            same_group=groups[ba]==groups[bb] and a in components and b in components
            if np.any(hi+eps<boxes[b][0]) or np.any(boxes[b][1]+eps<lo): continue
            if ba==bb and (not same_group or _find(components,a)==_find(components,b)): continue
            row={"geom1":model.geom(a).name,"geom2":model.geom(b).name}
            if examined>=max_pairs:
                skipped+=1
                if same_group:_join(possible,a,b)
                if same_group and rigid[ba]==rigid[bb]:_join(rigid_possible,a,b)
                continue
            examined+=1
            result=observe(query,a,b,.01);d=result.get("distance_m")
            mesh=any(model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH for g in (a,b))
            unknown=result["outcome"]=="unknown"
            connected=False
            penetration=result.get("penetration_lower_bound_m",0)
            solid=True
            for g in (a,b):
                if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                mesh_id=int(model.geom_dataid[g])
                if mesh_id not in convex_meshes:
                    vertices,_=query.vertices(g)
                    start,count=model.mesh_faceadr[mesh_id],model.mesh_facenum[mesh_id]
                    convex_meshes[mesh_id]=convex_solid(vertices,model.mesh_face[start:start+count])
                solid &= convex_meshes[mesh_id]
            if d is not None and d<=eps:
                if mesh and not solid:
                    points=result.get("points")
                    if abs(d)<=eps and points is not None:
                        connected=max(query.surface_error(a,np.array(points[:3])),
                                      query.surface_error(b,np.array(points[3:])))<=2e-6
                        unknown=not connected
                    else: unknown=True
                else: connected=True
            elif penetration>0:
                connected=True
            elif d is None and (result.get("lower_bound_m") or 0)<=eps:
                unknown=True
            if unknown and same_group and ba==bb:
                witness=union_witness(query,a,b)
                if witness is not None:
                    connected=True;unknown=False;connection_witnesses.append(witness)
            if unknown and mesh:
                from .surface_contact import contact
                certificate=contact(query,a,b,result.get("normal"))
                if certificate is not None and certificate["penetration_upper_bound_m"]<=tolerance:
                    result=certificate;d=result["distance_m"]
                    connected=d<=eps;unknown=False
            if connected and same_group:
                _join(components,a,b);_join(possible,a,b)
                if rigid[ba]==rigid[bb]:
                    _join(rigid_components,a,b);_join(rigid_possible,a,b)
            if unknown:
                unknown_count+=1
                if same_group:_join(possible,a,b)
                if same_group and rigid[ba]==rigid[bb]:_join(rigid_possible,a,b)
                unknown_pairs.append((a,b,{**row,"error_kind":"measurement_error","detail":result.get("error","original material intersection unresolved")}))
            if ba!=bb and solid and ((d is not None and d < -tolerance) or penetration>tolerance):
                hits.append({**row,**result})
    # A redundant unknown same-body union pair is no longer an open obligation
    # when another measured path proves material connectivity. Cross-body collision
    # obligations remain independent even if connectivity has been established.
    gaps=[row for a,b,row in unknown_pairs if not (model.geom_bodyid[a]==model.geom_bodyid[b]
        and a in components and b in components and _find(components,a)==_find(components,b))]
    unknown_count=len(gaps)
    grouped={};rigid_grouped={}
    for g in displayed:
        grouped.setdefault(groups[int(model.geom_bodyid[g])],[]).append(g)
        rigid_grouped.setdefault(rigid[int(model.geom_bodyid[g])],[]).append(g)
    unresolved=False
    for gs in rigid_grouped.values():
        islands={}
        for g in gs:islands.setdefault(_find(rigid_possible,g),[]).append(g)
        if len(islands)>1:
            for representative in sorted(islands):
                island=islands[representative]
                defects.append({"body":model.body(int(model.geom_bodyid[island[0]])).name,
                                "geom1":model.geom(island[0]).name,
                                "geoms":[model.geom(g).name for g in island],
                                "detail":"modeled rigid/weld connection has geometrically disconnected visible material"})
    # Test articulation between rigid groups independently of defects inside a
    # group. Otherwise repairing a rigid island reveals a previously hidden gap.
    articulation={r:r for r in set(rigid.values())}
    measured={}
    for g in displayed:
        r=rigid[int(model.geom_bodyid[g])]
        c=_find(components,g)
        if c in measured:_join(articulation,r,measured[c])
        measured[c]=r
    for gs in grouped.values():
        rs={rigid[int(model.geom_bodyid[g])] for g in gs}
        if len({_find(articulation,r) for r in rs})>1:
            unresolved=True
            gaps.append({"error_kind":"unsupported_check","obligation":"articulation:"+"|".join(sorted(model.body(r).name for r in rs)),
                         "detail":"articulated connection needs geometric engagement evidence; clearance alone is not a defect",
                         "geoms":[model.geom(g).name for g in gs]})
    # Do not certify invisible components as visible material.
    invisible_groups=set(groups.values())-set(grouped)
    if invisible_groups:
        unresolved=True
        gaps.append({"error_kind":"unsupported_check","detail":"modeled component lacks visible geometry"})
    # A collision envelope protruding beyond all visible material is a proven mismatch.
    mismatches=[]
    collider_groups={}
    for g in colliders:collider_groups.setdefault(groups[int(model.geom_bodyid[g])],[]).append(g)
    for group in set(groups.values()):
        vs=grouped.get(group,[])
        cs=collider_groups.get(group,[])
        if not vs or not cs: continue
        vlo=np.min([boxes[g][0] for g in vs],axis=0);vhi=np.max([boxes[g][1] for g in vs],axis=0)
        for g in cs:
            excess=float(max(np.max(vlo-boxes[g][0]),np.max(boxes[g][1]-vhi)))
            if excess>tolerance:
                mismatches.append({"geom1":model.geom(g).name,"body":model.body(int(model.geom_bodyid[g])).name,
                    "excess_m":excess,"detail":"collision envelope protrudes beyond visible material"})
    # Unknown intersection pairs remain uncovered even when connectivity is independently proven.
    complete=not gaps and not skipped and not unresolved
    status="fail" if defects or hits or mismatches else "pass" if complete else "partial"
    def partitions(parent,gs):
        result={}
        for g in gs:result.setdefault(_find(parent,g),[]).append(model.geom(g).name)
        return sorted(sorted(names) for names in result.values())
    connectivity={"version":2,"geoms":sorted(model.geom(g).name for g in displayed),
        "rigid_groups":{model.body(r).name:{"confirmed":partitions(rigid_components,gs),
            "possible":partitions(rigid_possible,gs)} for r,gs in rigid_grouped.items()}}
    return {"id":"material_integrity"+(":"+instance if instance else ""),"status":status,
            "connectivity":connectivity,
            "obligations":{"rigid_connection":"fail" if defects else "unknown" if skipped or any(
                partitions(rigid_components,gs)!=partitions(rigid_possible,gs) for gs in rigid_grouped.values()) else "pass",
                "articulated_connection":"unknown" if unresolved else "pass",
                "collision":"fail" if hits else "unknown" if unknown_count or skipped else "pass",
                "collision_envelope":"fail" if mismatches else "pass"},
            "instances":[instance] if instance else [],"complete":complete,"tolerance_m":tolerance,
            "connection_witnesses":connection_witnesses,
            "disconnected_components":defects,"disconnected_count":len(defects),
            "surface_mismatches":mismatches,"surface_mismatch_count":len(mismatches),
            "collisions":hits,"collision_count":len(hits),"coverage_gaps":gaps,
            "examined_pairs":examined,"unexamined_pairs":skipped,"measurement_error_count":unknown_count,
            "detail":"Visible material connectivity, distinct-body static intersections, collision-envelope consistency; not strength or full surface equivalence."}

def prepare_surfaces(model,data,query):
    from scene_generation.geometry import owner
    state=(data.geom_xpos,data.geom_xmat,model.geom_group,model.geom_rgba[:,3],
           model.geom_size,model.mesh_vert,model.mesh_face)
    if hasattr(query,"visible_surface_state") and all(np.array_equal(a,b) for a,b in zip(state,query.visible_surface_state)):
        return
    # Custom measurement queries need not implement DistanceQuery's private caches.
    for name in ("meshes","face_normals","solid_halfspaces"):
        cache=getattr(query,name,None)
        if cache is not None:cache.clear()
    query.visible_surface_state=tuple(a.copy() for a in state)
    by_owner={}
    for g in range(model.ngeom):
        if visible(model,g) and model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            by_owner.setdefault(owner(model,g),[]).append(g)
    query.visible_surface_geoms=by_owner
    query.visible_surface_index={}


from contextlib import contextmanager

@contextmanager
def surface_batch(model,data,query):
    """One read-only measurement batch; validate the state once at its boundary."""
    prepare_surfaces(model,data,query)
    previous=getattr(query,"surface_batch_active",False)
    query.surface_batch_active=True
    try: yield
    finally: query.surface_batch_active=previous


def visible_witness(model,data,query,geom,point,tolerance=1e-5):
    """Certify measured support points on displayed surfaces, in original geom order."""
    from scene_generation.geometry import bounds,owner
    from .spatial import BoxIndex
    if not getattr(query,"surface_batch_active",False):prepare_surfaces(model,data,query)
    key=owner(model,geom)
    if key not in query.visible_surface_index:
        geoms=query.visible_surface_geoms.get(key,[])
        planes=[g for g in geoms if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
        ids=[g for g in geoms if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE]
        query.visible_surface_index[key]=(ids,BoxIndex([bounds(model,data,g) for g in ids]),planes)
    entry=query.visible_surface_index[key]
    ids,index,planes=entry
    for g in sorted([ids[i] for i in index.overlap(point,point,tolerance)]+planes):
        error=query.surface_error(g,point)
        if error<=tolerance:
            return {"geom":model.geom(g).name,"error_m":float(error)}
    return None
