"""Get envelopes and distances from actual geometry instead of relying only on engine contact filtering."""

import numpy as np
import mujoco

from asset_generation.contracts import require


def bounds(model, data, g):
    rotation, center = data.geom_xmat[g].reshape(3,3), data.geom_xpos[g]
    kind, size = model.geom_type[g], model.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start+count] @ rotation.T + center
        return points.min(axis=0), points.max(axis=0)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        half = np.abs(rotation) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.repeat(size[0], 3)
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        half = np.linalg.norm(rotation * size, axis=1)
    elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        z = np.abs(rotation[:,2])
        half = z*size[1]+size[0]*(np.ones(3) if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else np.sqrt(np.maximum(0,1-z*z)))
    else:
        raise ValueError("Finite envelope does not support this geom type")
    return center-half, center+half


def owner(model, geom):
    name = model.geom(int(geom)).name
    if name and "__" in name:
        return name.split("__", 1)[0]
    body = model.geom_bodyid[geom]
    while body:
        name = model.body(int(body)).name
        if name and "__" in name:
            return name.split("__",1)[0]
        body = model.body_parentid[body]
    return "unassigned"


def support_candidates(geometry, parents, objects, tolerance):
    """Equivalent AABB filter in original pair order; planes retain all objects.

    Index object lower-x bounds once, then vector-filter the reachable window.
    Original ordering matters because the downstream measurement budget is bounded.
    """
    if not objects: return
    lower=np.array([geometry[g][0] for g in objects])
    upper=np.array([geometry[g][1] for g in objects])
    order=np.argsort(lower[:,0],kind="stable")
    ordered_x=lower[order,0]
    for parent in parents:
        if parent not in geometry:
            candidates=range(len(objects))
        else:
            lo,hi=geometry[parent]
            window=np.searchsorted(ordered_x,hi[0]+tolerance,side="right")
            candidates=order[:window]
            candidates=candidates[np.all(upper[candidates]+tolerance>=lo,axis=1)
                                  & np.all(hi+tolerance>=lower[candidates],axis=1)]
            candidates=np.sort(candidates)
        for index in candidates:
            yield parent,objects[index]


def collision_scan(model, data, tolerance, allowed=(), max_pairs=200000, query=None,
                   only_owners=None):
    """Pairwise cross-instance distances, including static and filtered collisions; mesh distances still use the engine convex hull.

    Broad phase is sweep-and-prune along x instead of one full pass per geom. The
    candidate set is unchanged (AABB overlap plus bounding radii); only the way it is
    found is, so cost follows the number of spatially close pairs rather than ngeom^2.
    ``only_owners`` restricts the scan to pairs touching those instances, which is what
    a joint sweep needs: parts that did not move cannot collide differently.
    """
    allowed = {frozenset(p) for p in allowed}
    from asset_generation.distance import DistanceQuery, observe
    query = query or DistanceQuery(model,data)
    geoms = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    owners = np.array([owner(model,g) for g in geoms])
    centers = data.geom_xpos[geoms]
    radii = model.geom_rbound[geoms]
    planes = model.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_PLANE
    boxes = [(np.full(3,-np.inf),np.full(3,np.inf)) if plane else bounds(model,data,g)
             for g,plane in zip(geoms,planes)]
    lower = np.array([box[0] for box in boxes]).reshape(-1,3)
    upper = np.array([box[1] for box in boxes]).reshape(-1,3)
    examined = skipped = unknown_count = 0
    result, gaps, omitted_owners = [], [], set()
    unknown_owners = set()
    focused_owners = None if only_owners is None else set(only_owners)
    focus = None if focused_owners is None else np.isin(owners, list(focused_owners))
    # Sweep-and-prune: sorted by lower x, only boxes reachable within the x window can
    # overlap. Planes carry an infinite extent and therefore stay in every window.
    order = np.argsort(lower[:,0], kind="stable")
    ordered_lower = lower[order,0]
    for rank, i in enumerate(order):
        window = np.searchsorted(ordered_lower, upper[i,0]+tolerance, side="right")
        candidates = order[rank+1:window]
        candidates = candidates[owners[candidates] != owners[i]]
        if focus is not None:
            # A pair matters when either side is in focus: a moving part can strike a
            # stationary neighbour, and that neighbour is not itself in the moving set.
            if not focus[i]:
                candidates = candidates[focus[candidates]]
            # else: keep every candidate, since the stationary side may be outside focus
        if not len(candidates):
            continue
        other_centers = centers[candidates]
        near = planes[i] | planes[candidates] | (
            np.linalg.norm(other_centers-centers[i],axis=1) <= radii[i]+radii[candidates]+tolerance)
        # Conservative world-frame bounding boxes cull only geoms that are certainly separated; exact distances and penetration thresholds are unchanged.
        near &= np.all(upper[i]+tolerance >= lower[candidates],axis=1) & np.all(upper[candidates]+tolerance >= lower[i],axis=1)
        for index in np.flatnonzero(near):
            g2 = geoms[candidates[index]]
            o1, o2 = str(owners[i]), str(owners[candidates[index]])
            # Internal geometry of the same instrument is handled by the single-asset mechanism check. Robot self-collision is supplemented from actual contacts.
            n1, n2 = model.geom(geoms[i]).name, model.geom(g2).name
            if frozenset((n1,n2)) in allowed:
                continue
            if examined >= max_pairs:
                skipped += 1
                omitted_owners.update((o1,o2))
                continue
            examined += 1
            evidence = {**observe(query,geoms[i],g2,.05),"geom1":n1,"geom2":n2,"instances":[o1,o2]}
            if evidence["outcome"] == "unknown":
                unknown_count += 1
                unknown_owners.update((o1,o2))
                gaps.append(evidence)
                continue
            distance = evidence["distance_m"]
            if (distance is not None and distance < -tolerance
                    or evidence.get("penetration_lower_bound_m",0) > tolerance):
                result.append({**evidence,"instances":[o1,o2]})
            elif evidence.get("penetration_lower_bound_m") is not None:
                unknown_count += 1
                unknown_owners.update((o1,o2))
                gaps.append({**evidence,"outcome":"unknown","error":"Penetration lower bound is insufficient to decide whether the acceptance tolerance is exceeded"})
    for c in data.contact:
        if focused_owners is not None and "robot" not in focused_owners:
            continue
        names = [model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name]
        if c.dist < -tolerance and owner(model,c.geom1) == owner(model,c.geom2) == "robot" and frozenset(names) not in allowed:
            result.append({"geom1":names[0], "geom2":names[1], "instances":["robot"], "distance_m":float(c.dist)})
    if skipped:
        gaps.append({"error_kind":"resource_limit","error":"Geom-pair budget exhausted; remaining scope not checked",
                     "instances":sorted(omitted_owners),"unexamined_pairs":skipped})
    return {"collisions":result,"coverage_gaps":gaps,"examined_pairs":examined,
            "unexamined_pairs":skipped,"complete":not gaps}


def collisions(model, data, tolerance, allowed=(), max_pairs=200000, query=None):
    """Compatibility for legacy strict callers; unknown is never treated as collision-free."""
    from asset_generation.distance import MeasurementError
    result = collision_scan(model,data,tolerance,allowed,max_pairs,query)
    if result["coverage_gaps"]:
        raise MeasurementError(str(result["coverage_gaps"][:3]))
    return result["collisions"]


def aggregate_bounds(model, data, prefix):
    boxes = [bounds(model,data,g) for g in range(model.ngeom)
             if owner(model,g) == prefix and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    require(boxes, "Instance lacks valid collision geometry")
    return np.min([b[0] for b in boxes],axis=0), np.max([b[1] for b in boxes],axis=0)


def site_surface_distance(model, data, site_name):
    """Check that measurement markers lie on the actual collision surface of their owning part; reject floating self-reported measurement points."""
    site = model.site(site_name).id
    body = model.site_bodyid[site]
    distances = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        p = (data.site_xpos[site]-data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3,3)
        kind, size = model.geom_type[g], model.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            q = np.abs(p)-size
            distance = np.linalg.norm(np.maximum(q,0))+min(float(q.max()),0)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            distance = np.linalg.norm(p)-size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            q = np.array([np.linalg.norm(p[:2])-size[0],abs(p[2])-size[1]])
            distance = np.linalg.norm(np.maximum(q,0))+min(float(q.max()),0)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            distance = np.linalg.norm(p-np.array([0,0,np.clip(p[2],-size[1],size[1])]))-size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            from trimesh.triangles import closest_point
            m = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
            fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
            require(fn <= 200000,"Mesh too large for measurement-marker surface verification")
            triangles = model.mesh_vert[va:va+vn][model.mesh_face[fa:fa+fn]]
            nearest = closest_point(triangles,np.repeat(p[None,:],fn,axis=0))
            distance = np.linalg.norm(nearest-p,axis=1).min()
        else:
            continue
        distances.append(abs(float(distance)))
    require(distances,"The part owning the measurement marker has no supported collision surface")
    return min(distances)
