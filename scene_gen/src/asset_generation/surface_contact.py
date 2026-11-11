"""Actual exposed triangle patches, not a convex-hull face filling mesh holes.

A supporting half-space bounds the whole shape; an intersection of its original
surface patches supplies witnesses. Failure to find one remains unknown.
"""
import numpy as np
import mujoco
from .convex import closest_witness


def patches(query,g,normal):
    model,data=query.model,query.data
    if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:
        feature=query.feature_support(g,normal)
        return [] if feature is None else [feature]
    triangles=query.vertices(g)[1]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
    height=query.interval(g,normal)[1]
    # This is the existing witness consistency tolerance, not acceptance tolerance.
    candidates=triangles[np.all(np.abs(triangles@normal-height)<=2e-6,axis=1)]
    result=[]
    for triangle in candidates:
        def feature(direction,t=triangle):return t[int(np.argmax(t@direction))]
        feature.triangle=triangle
        result.append(feature)
    return result


def triangle_overlap(first,second,normal,gap):
    """Clip original triangular patches in their supporting plane.

    This independent construction avoids GJK degeneracy on tiny coplanar faces.
    The caller still validates both original surfaces at the existing tolerance.
    """
    seed=np.eye(3)[int(np.argmin(np.abs(normal)))]
    u=np.cross(normal,seed);u/=np.linalg.norm(u);v=np.cross(normal,u)
    origin=first.mean(axis=0)
    a=np.column_stack(((first-origin)@u,(first-origin)@v))
    b=np.column_stack(((second-origin)@u,(second-origin)@v))
    def cross(x,y):return x[0]*y[1]-x[1]*y[0]
    winding=cross(b[1]-b[0],b[2]-b[0])
    if abs(winding)<1e-22:return None
    if winding<0:b=b[::-1]
    polygon=list(a)
    for i in range(3):
        if not polygon:return None
        start,end=b[i],b[(i+1)%3];edge=end-start
        clipped=[]
        previous=polygon[-1];pd=cross(edge,previous-start)
        for current in polygon:
            cd=cross(edge,current-start)
            if (cd>=0)!=(pd>=0):
                clipped.append(previous+(current-previous)*pd/(pd-cd))
            if cd>=0:clipped.append(current)
            previous,pd=current,cd
        polygon=clipped
    if not polygon:return None
    point=np.mean(polygon,axis=0)
    a=origin+point[0]*u+point[1]*v
    return a,a+gap*normal


def contact(query,g1,g2,normal=None,max_patch_pairs=4096):
    axes=[] if normal is None else [np.asarray(normal,dtype=float)]
    axes.extend(query.separating_faces(g1))
    axes.extend(-n for n in query.separating_faces(g2))
    seen=set();examined=0
    for axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-12:continue
        axis=axis/length;key=tuple(np.round(axis,10))
        if key in seen:continue
        seen.add(key)
        gap=query.interval(g2,axis)[0]-query.interval(g1,axis)[1]
        # Only near-contact witnesses. No claims about deep mesh intersections.
        if not np.isfinite(gap) or abs(gap)>1e-5:continue
        left,right=patches(query,g1,axis),patches(query,g2,-axis)
        origin=(query.data.geom_xpos[g1]+query.data.geom_xpos[g2])/2
        for first in left:
            for second in right:
                if examined>=max_patch_pairs:return None
                examined+=1
                def support(direction):
                    tangent=direction-float(direction@axis)*axis
                    return first(tangent)-origin,second(-tangent)-origin-gap*axis
                seed=np.eye(3)[int(np.argmin(np.abs(axis)))]
                pair=(triangle_overlap(first.triangle,second.triangle,axis,gap)
                      if hasattr(first,"triangle") and hasattr(second,"triangle") else None)
                if pair is None:
                    solved=closest_witness(support,np.cross(axis,seed),tolerance=1e-10)
                    if solved is None or solved["upper"]>1e-9:continue
                    pair=solved["a"]+origin,solved["b"]+origin+gap*axis
                a,b=pair
                if max(query.surface_error(g1,a),query.surface_error(g2,b))>2e-6:continue
                value,_=query.validate(g1,g2,gap,np.r_[a,b],.01,normal_hint=axis)
                if value is not None:
                    return {**value,"method":"original_surface_patches","consistency_checked":True,
                            "outcome":"measured","original_surface_certified":True,
                            "penetration_upper_bound_m":max(0.,-gap)+2e-6,
                            "patch_pairs_examined":examined}
    return None
