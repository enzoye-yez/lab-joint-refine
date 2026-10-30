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

def physical(model,a,b):
    """Whether MuJoCo's contact filter lets geoms a and b collide."""
    ta,aa,tb,ab=(int(x) for x in (model.geom_contype[a],model.geom_conaffinity[a],
                                   model.geom_contype[b],model.geom_conaffinity[b]))
    if not (ta&ab or tb&aa):return False
    ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
    if ba==bb:return False
    parent=lambda x,y:int(model.body_parentid[x])==y and y!=0
    filtered=not model.opt.disableflags&mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    return not (filtered and (parent(ba,bb) or parent(bb,ba)))

CAPTURE_FIT_M=.002

def captured(model,data,query,joint,child,parent,boxes,fit=CAPTURE_FIT_M):
    """Joint-aware engagement: the child is held if every displacement the joint does
    not permit is blocked by parent material within the fit tolerance.

    A slide must be blocked across its axis; a hinge across and along its axis. Each
    probe shifts the child's geoms by `fit` and asks the original-surface test whether
    new material intersection appears. This certifies geometric capture, not strength.
    """
    from .mesh_exact import exact
    axis=data.xmat[model.jnt_bodyid[joint]].reshape(3,3)@model.jnt_axis[joint]
    axis=axis/np.linalg.norm(axis)
    seed=np.eye(3)[int(np.argmin(np.abs(axis)))]
    u=np.cross(axis,seed);u/=np.linalg.norm(u);v=np.cross(axis,u)
    directions=[u,-u,v,-v]+([axis,-axis] if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_SLIDE else [])
    near=[(c,p) for c in child for p in parent
          if not (np.any(boxes[c][1]+fit<boxes[p][0]) or np.any(boxes[p][1]+fit<boxes[c][0]))]
    def state():
        out={}
        for c,p in near:
            e=exact(query,c,p)
            if e is not None:out[(c,p)]=e
        return out
    before=state();blocked=[];free=[]
    if near and not before:
        # No original-surface evidence for any nearby pair: capture stays undecided.
        return {"joint":model.joint(joint).name,"body":model.body(int(model.jnt_bodyid[joint])).name,
                "fit_tolerance_m":fit,"blocked_directions":[],"free_directions":[],
                "clearance_to_parent_m":None,"captured":False,"undecided":True}
    saved=data.geom_xpos[child].copy()
    try:
        for direction in directions:
            data.geom_xpos[child]=saved+fit*direction
            after=state();hit=None
            for key,e in after.items():
                if not e["intersecting"]:continue
                prior=before.get(key)
                grew=(prior is None or not prior["intersecting"] or not e["depth_known"] or not prior["depth_known"]
                      or e["penetration_lower_bound_m"]>prior["penetration_lower_bound_m"]+.25*fit)
                if grew:hit=key;break
            (blocked if hit else free).append(np.round(direction,4).tolist())
    finally:
        data.geom_xpos[child]=saved
    clearance=min((e["distance_m"] for e in before.values()),default=None)
    return {"joint":model.joint(joint).name,"body":model.body(int(model.jnt_bodyid[joint])).name,
            "fit_tolerance_m":fit,"blocked_directions":blocked,"free_directions":free,
            "clearance_to_parent_m":clearance,"captured":bool(near) and not free}

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

def _local(model,data,geom,vector):
    """World direction in the frame where model.py places this geom (its body frame)."""
    rotation=data.xmat[int(model.geom_bodyid[geom])].reshape(3,3)
    return [round(float(x),4) for x in rotation.T@np.asarray(vector,dtype=float)]


def collision_hint(model,data,a,b,result,depth):
    """Actionable separation for the author; derived from the measured normal (geom1 to geom2)."""
    normal=result.get("normal")
    hint={"body1":model.body(int(model.geom_bodyid[a])).name,"body2":model.body(int(model.geom_bodyid[b])).name,
          "depth_mm":round(depth*1000,2)}
    if normal is not None:
        hint.update(move_geom2_local=_local(model,data,b,normal),move_geom1_local=_local(model,data,a,-np.asarray(normal)),
            instruction="沿 move_geom2_local（geom2 所属 body 的局部坐标）移动 geom2 约 depth_mm，"
                        "或沿 move_geom1_local 移动 geom1；若二者应当咬合，则改为声明连接/接口并控制嵌入深度")
    return hint


def island_hint(model,data,query,island,others,boxes,limit=.05):
    """Nearest measured neighbour of a disconnected rigid island within the same rigid group."""
    if not others:return {"nearest":None,"instruction":"该刚体只有这一块可见材料与其余部分分离；检查 body 归属或补连接结构"}
    def box_gap(g,h):
        return float(np.max(np.maximum(boxes[g][0]-boxes[h][1],boxes[h][0]-boxes[g][1]).clip(min=0)))
    candidates=sorted(((box_gap(g,h),g,h) for g in island[:16] for h in others),key=lambda x:x[0])[:24]
    best=None
    for gap,g,h in candidates:
        if gap>limit:break
        result=observe(query,g,h,limit);d=result.get("distance_m")
        if d is not None and (best is None or d<best[0]):best=(d,g,h,result.get("normal"))
    if best is None:
        return {"nearest":None,"instruction":f"{int(limit*1000)} mm 内没有同一刚体的其他材料；该部分悬空，需连接到主体或移入正确的 body"}
    d,g,h,normal=best
    hint={"nearest_geom":model.geom(h).name,"from_geom":model.geom(g).name,"gap_mm":round(d*1000,2)}
    if normal is not None and d>0:
        hint["move_island_local"]=_local(model,data,g,normal)
        hint["instruction"]="沿 move_island_local 移动该孤立部分约 gap_mm 使其与 nearest_geom 接触，或加连接件"
    else:
        hint["instruction"]="与 nearest_geom 已接触或重叠但无法确认材料连续；检查是否为非封闭网格，或改用基本体"
    return hint


# Bodies already joined by a joint or weld may embed by design: a button set into its
# bezel, a hub seated on its pedestal, a latch tongue in its catch. Such engagement up to
# this depth is recorded, not reported as a collision defect. Deeper overlap, and any
# overlap between unrelated bodies, remains a defect.
ENGAGEMENT_ALLOWANCE_M=.003+1e-6

def integrity(model,data,*,instance=None,tolerance=.0005,max_pairs=200000,query=None,
              engagement=ENGAGEMENT_ALLOWANCE_M):
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
    gaps=[];defects=[];hits=[];engagements=[];examined=0;skipped=0;unknown_count=0
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
            # Same-body pairs are skipped only once rigidly confirmed. Articulated
            # connectivity may already link them through another body, which says
            # nothing about this body's own material and used to hide the pair.
            if ba==bb and (not same_group or _find(rigid_components,a)==_find(rigid_components,b)): continue
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
            # The simulator collides mesh colliders by their convex hulls, so a measured
            # hull overlap between colliding geoms is a physical collision whatever the
            # original triangles look like.
            hull_depth=-d if (not solid and d is not None and d<0 and physical(model,a,b)) else 0.
            if unknown:
                from .mesh_exact import exact
                evidence=exact(query,a,b)
                if evidence is not None:
                    result={**row,**evidence};d=evidence["distance_m"]
                    penetration=evidence["penetration_lower_bound_m"] or 0
                    connected=evidence["intersecting"] or d<=eps
                    unknown=False;solid=True
                    if ba!=bb and evidence["crossing"] and not evidence["depth_known"]:
                        # Proven visible crossing of open shells; depth itself is not measurable.
                        target=engagements if groups[ba]==groups[bb] else hits
                        target.append({**row,**evidence,"depth_m":None,"repair_hint":
                            None if target is engagements else collision_hint(model,data,a,b,evidence,tolerance)})
                        if connected and same_group:
                            _join(components,a,b);_join(possible,a,b)
                            if rigid[ba]==rigid[bb]:
                                _join(rigid_components,a,b);_join(rigid_possible,a,b)
                        continue
            if connected and same_group:
                _join(components,a,b);_join(possible,a,b)
                if rigid[ba]==rigid[bb]:
                    _join(rigid_components,a,b);_join(rigid_possible,a,b)
            if unknown:
                unknown_count+=1
                if same_group:_join(possible,a,b)
                if same_group and rigid[ba]==rigid[bb]:_join(rigid_possible,a,b)
                unknown_pairs.append((a,b,{**row,"error_kind":"measurement_error","detail":result.get("error","original material intersection unresolved")}))
            depth=max(-d if d is not None else 0,penetration,hull_depth)
            if hull_depth>0:solid=True
            if ba!=bb and solid and depth>tolerance:
                if groups[ba]==groups[bb] and depth<=engagement:
                    engagements.append({**row,"depth_m":float(depth)})
                else:
                    hits.append({**row,**result,"repair_hint":collision_hint(model,data,a,b,result,depth)})
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
                others=[g for r,members in islands.items() if r!=representative for g in members]
                defects.append({"body":model.body(int(model.geom_bodyid[island[0]])).name,
                                "geom1":model.geom(island[0]).name,
                                "geoms":[model.geom(g).name for g in island],
                                "detail":"modeled rigid/weld connection has geometrically disconnected visible material",
                                "repair_hint":island_hint(model,data,query,island,others,boxes)
                                    if len(defects)<32 else None})
    # Test articulation between rigid groups independently of defects inside a
    # group. Otherwise repairing a rigid island reveals a previously hidden gap.
    articulation={r:r for r in set(rigid.values())}
    measured={}
    for g in displayed:
        r=rigid[int(model.geom_bodyid[g])]
        c=_find(components,g)
        if c in measured:_join(articulation,r,measured[c])
        measured[c]=r
    capture_evidence=[]
    for b in sorted(bodies):
        p=int(model.body_parentid[b])
        if p not in bodies or rigid[b]==rigid[p] or _find(articulation,rigid[b])==_find(articulation,rigid[p]):continue
        joints=[j for j in range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b])
                if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE,mujoco.mjtJoint.mjJNT_BALL)]
        if len(joints)!=1:continue
        child=rigid_grouped.get(rigid[b],[]);parent=rigid_grouped.get(rigid[p],[])
        evidence=captured(model,data,query,joints[0],child,parent,boxes)
        capture_evidence.append(evidence)
        if evidence["captured"]:_join(articulation,rigid[b],rigid[p])
    # A jointed part neither touching nor captured by its parent's material is a decided
    # defect: it visibly floats and only the joint definition holds it in place.
    floating=[];decided={r:_find(articulation,r) for r in articulation}
    for evidence in capture_evidence:
        b=model.body(evidence["body"]).id;p=int(model.body_parentid[b])
        if evidence["captured"] or evidence.get("undecided") or _find(articulation,rigid[b])==_find(articulation,rigid[p]):continue
        _join(decided,rigid[b],rigid[p])
        clearance=evidence["clearance_to_parent_m"]
        floating.append({**evidence,"parent_body":model.body(p).name,
            "detail":"jointed part is neither in contact with nor captured by its parent's material",
            "repair_hint":{"instruction":"让该部件的导向/轴销/铰链叶片与父件材料接触或包覆：沿 free_directions 补导轨、轴肩、"
                "轴套或限位面，使 free_directions 全部被父件材料在 fit_tolerance_m 内挡住",
                "clearance_mm":None if clearance is None else round(clearance*1000,2)}})
    for gs in grouped.values():
        rs={rigid[int(model.geom_bodyid[g])] for g in gs}
        if len({_find(decided,r) for r in rs})>1:
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
    status="fail" if defects or hits or mismatches or floating else "pass" if complete else "partial"
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
                "articulated_connection":"fail" if floating else "unknown" if unresolved else "pass",
                "collision":"fail" if hits else "unknown" if unknown_count or skipped else "pass",
                "collision_envelope":"fail" if mismatches else "pass"},
            "instances":[instance] if instance else [],"complete":complete,"tolerance_m":tolerance,
            "connection_witnesses":connection_witnesses,
            "disconnected_components":defects,"disconnected_count":len(defects),
            "surface_mismatches":mismatches,"surface_mismatch_count":len(mismatches),
            "collisions":hits,"collision_count":len(hits),"coverage_gaps":gaps,
            "engagements":engagements,"engagement_count":len(engagements),"engagement_allowance_m":engagement,
            "capture_evidence":capture_evidence,"floating_articulations":floating,"floating_count":len(floating),
            "examined_pairs":examined,"unexamined_pairs":skipped,"measurement_error_count":unknown_count,
            "detail":"Visible material connectivity, distinct-body static intersections, collision-envelope consistency; "
                     "shallow engagement between jointed or welded bodies is recorded separately. Not strength or full surface equivalence."}

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
