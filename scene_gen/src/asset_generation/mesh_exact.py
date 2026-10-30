"""Exact original-surface evidence for pairs the convex-hull measurement cannot decide.

MuJoCo measures a mesh by its convex hull. For a non-convex or open mesh, hull overlap
says nothing about the triangles a viewer sees. This module answers on the original
triangles with a bounding-volume hierarchy (python-fcl): a triangle crossing or a solid
containment proves intersection; otherwise the exact minimum surface distance is the gap.
Penetration depth is a lower bound from vertices proven inside a closed solid. For an
open shell the depth is estimated from the shell's own face orientation (outward normals,
as MJCF meshes are authored) and marked as an estimate; a shell with inconsistent
orientation leaves the depth unknown while the crossing itself stays proven.
"""
import math
import numpy as np
import mujoco
import fcl

from .distance import triangle_surface_distance
from .material import closed_surface

MESH = mujoco.mjtGeom.mjGEOM_MESH
MAX_DEPTH_SAMPLES = 256


def _shape(query, g):
    """fcl geometry in the geom frame, cached per mesh; None for unsupported types."""
    model = query.model
    kind, size = model.geom_type[g], model.geom_size[g]
    if kind == MESH:
        cache = query.__dict__.setdefault("exact_meshes", {})
        mesh = int(model.geom_dataid[g])
        if mesh not in cache:
            vertices, _ = query.vertices(g)
            start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            faces = model.mesh_face[start:start+count].astype(np.int32)
            bvh = fcl.BVHModel()
            bvh.beginModel(len(vertices), len(faces))
            bvh.addSubModel(vertices.astype(np.float64), faces)
            bvh.endModel()
            cache[mesh] = (bvh, closed_surface(vertices, faces))
        return cache[mesh]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return fcl.Box(*(2*size)), True
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return fcl.Sphere(size[0]), True
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return fcl.Cylinder(size[0], 2*size[1]), True
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return fcl.Capsule(size[0], 2*size[1]), True
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return fcl.Ellipsoid(*size), True
    return None


def _object(query, g, shape):
    data = query.data
    return fcl.CollisionObject(shape, fcl.Transform(data.geom_xmat[g].reshape(3, 3).copy(),
                                                    data.geom_xpos[g].copy()))


def _world_vertices(query, g):
    data = query.data
    vertices, triangles = query.vertices(g)
    rotation, origin = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
    return vertices@rotation.T+origin, triangles@rotation.T+origin


def winding(triangles, points):
    """Generalized winding number of each point w.r.t. a closed triangle surface."""
    total = np.zeros(len(points))
    for start in range(0, len(triangles), 4096):
        t = triangles[start:start+4096]
        a = t[None, :, 0]-points[:, None]; b = t[None, :, 1]-points[:, None]; c = t[None, :, 2]-points[:, None]
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("pij,pij->pi", a, np.cross(b, c))
        div = (la*lb*lc+np.einsum("pij,pij->pi", a, b)*lc+np.einsum("pij,pij->pi", b, c)*la
               + np.einsum("pij,pij->pi", c, a)*lb)
        total += np.arctan2(det, div).sum(axis=1)
    return total/(2*math.pi)


def _primitive_inside(query, g, points):
    """Signed distance (negative inside) of world points to a primitive geom."""
    model, data = query.model, query.data
    local = (points-data.geom_xpos[g])@data.geom_xmat[g].reshape(3, 3)
    kind, size = model.geom_type[g], model.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(local)-size
        return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(axis=1), 0)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(local, axis=1)-size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        q = np.column_stack((np.linalg.norm(local[:, :2], axis=1)-size[0], np.abs(local[:, 2])-size[1]))
        return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(axis=1), 0)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = np.column_stack((np.zeros(len(local)), np.zeros(len(local)), np.clip(local[:, 2], -size[1], size[1])))
        return np.linalg.norm(local-axis, axis=1)-size[0]
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # Conservative: scaled radial distance times the smallest semi-axis.
        return (np.linalg.norm(local/size, axis=1)-1)*float(size.min())
    return np.full(len(points), np.inf)


SHELL_REACH_M = .01


def closest_points(a, b, c, p):
    """Closest points on triangles (a,b,c) to points p, all arrays (..., 3); Ericson's regions."""
    ab, ac, ap = b-a, c-a, p-a
    bp, cp = p-b, p-c
    dot = lambda x, y: np.einsum("...i,...i->...", x, y)
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
    denom = va+vb+vc
    safe = lambda x: np.where(np.abs(x) > 0, x, 1.)
    q = a+ab*(vb/safe(denom))[..., None]+ac*(vc/safe(denom))[..., None]
    for cond, value in reversed((
            ((d1 <= 0) & (d2 <= 0), a),
            ((d3 >= 0) & (d4 <= d3), b),
            ((d6 >= 0) & (d5 <= d6), c),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a+ab*(d1/safe(d1-d3))[..., None]),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a+ac*(d2/safe(d2-d6))[..., None]),
            ((va <= 0) & (d4-d3 >= 0) & (d5-d6 >= 0), b+(c-b)*((d4-d3)/safe((d4-d3)+(d5-d6)))[..., None]))):
        q = np.where(cond[..., None], value, q)
    return q


def _shell_depth(query, solid, points, neighbours=16):
    """Deepest penetration of points behind an open shell's outward faces, or None."""
    from scipy.spatial import cKDTree
    _, triangles = _world_vertices(query, solid)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0
    triangles, normals = triangles[keep], normals[keep]/lengths[keep, None]
    if not len(triangles):
        return None
    k = min(neighbours, len(triangles))
    _, index = cKDTree(triangles.mean(axis=1)).query(points, k=k)
    index = index.reshape(len(points), k)
    t = triangles[index]
    q = closest_points(t[..., 0, :], t[..., 1, :], t[..., 2, :], points[:, None, :])
    dist = np.linalg.norm(points[:, None, :]-q, axis=2)
    best = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    dist, q = dist[rows, best], q[rows, best]
    signed = np.einsum("ij,ij->i", points-q, normals[index[rows, best]])
    near = dist <= SHELL_REACH_M
    # 'Behind' only counts where the closest point is a face interior: past a shell's
    # boundary edge the face normal does not say which side of the material a point is.
    facing = np.abs(np.abs(signed)-dist) <= 1e-7+1e-6*dist
    behind, front = near & facing & (signed < -1e-9), near & (signed >= -1e-9)
    # A sample 'behind' the shell is meaningful only when nearby samples also sit in
    # front of it; otherwise the face orientation does not define an outside.
    if behind.any() and not front.any():
        return None
    return float(dist[behind].max()) if behind.any() else 0.


def _depth_into(query, solid, closed, points):
    """Lower bound on how deep any of `points` lies inside `solid`; None if undecidable."""
    if not len(points):
        return 0.
    if query.model.geom_type[solid] != MESH:
        signed = _primitive_inside(query, solid, points)
        return float(max(0., -signed.min()))
    if not closed:
        return _shell_depth(query, solid, points)
    _, triangles = _world_vertices(query, solid)
    inside = points[winding(triangles, points) > .5]
    if not len(inside):
        return 0.
    return float(max(triangle_surface_distance(triangles, p) for p in inside[:MAX_DEPTH_SAMPLES]))


def _samples(query, g):
    if query.model.geom_type[g] == MESH:
        vertices, triangles = _world_vertices(query, g)
        points = np.vstack((vertices, triangles.mean(axis=1)))
    else:
        points = query.data.geom_xpos[g][None]
    if len(points) > 4*MAX_DEPTH_SAMPLES:
        points = points[np.linspace(0, len(points)-1, 4*MAX_DEPTH_SAMPLES).astype(int)]
    return points


def exact(query, a, b):
    """Original-surface relation of geoms a and b, or None when a type is unsupported."""
    shapes = [_shape(query, g) for g in (a, b)]
    if any(s is None for s in shapes):
        return None
    (sa, ca), (sb, cb) = shapes
    oa, ob = _object(query, a, sa), _object(query, b, sb)
    collision = fcl.CollisionResult()
    crossing = fcl.collide(oa, ob, fcl.CollisionRequest(num_max_contacts=1), collision) > 0
    distance_result = fcl.DistanceResult()
    gap = 0. if crossing else float(fcl.distance(oa, ob, fcl.DistanceRequest(enable_nearest_points=True),
                                                 distance_result))
    depths = [_depth_into(query, b, cb, _samples(query, a)), _depth_into(query, a, ca, _samples(query, b))]
    # An open-shell estimate never proves nesting by itself; only a crossing or a closed solid does.
    nested = not crossing and any(d is not None and d > 0 and closed_side for d, closed_side in
                                  zip(depths, (cb or query.model.geom_type[b] != MESH, ca or query.model.geom_type[a] != MESH)))
    known = [d for d in depths if d is not None]
    result = {"method": "exact_triangle_bvh", "closed": [bool(ca), bool(cb)],
              "intersecting": bool(crossing or nested), "crossing": bool(crossing), "nested": bool(nested)}
    if result["intersecting"]:
        depth = max(known) if known else None
        estimated = not (ca or query.model.geom_type[a] != MESH) or not (cb or query.model.geom_type[b] != MESH)
        result.update(distance_m=-depth if depth else 0., penetration_lower_bound_m=depth,
                      depth_known=depth is not None, points=None, normal=None,
                      depth_method="oriented_shell_estimate" if estimated else "closed_solid_lower_bound")
    else:
        points = np.r_[distance_result.nearest_points[0], distance_result.nearest_points[1]]
        result.update(distance_m=gap, penetration_lower_bound_m=0., depth_known=True,
                      points=points.tolist() if np.isfinite(points).all() else None, normal=None)
    return result
