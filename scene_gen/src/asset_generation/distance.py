"""有几何一致性校验的距离证据；异常结果不能进入验收。"""

import math
import numpy as np
import mujoco

from asset_prep.errors import ValidationError


class MeasurementError(ValidationError):
    pass


def observe(query, g1, g2, limit=.05):
    """仅把预期的量测不确定性转成数据；程序错误仍向上传播。"""
    try:
        value = query.measure(g1, g2, limit)
        return {**value, "outcome": "measured" if value.get("distance_m") is not None else "bounded"}
    except MeasurementError as exc:
        return {"outcome":"unknown", "distance_m":None, "lower_bound_m":None,
                "points":None, "normal":None, "error":str(exc), "error_kind":"measurement_error"}


def triangle_surface_distance(triangles, point):
    """Exact face/edge candidates without an absolute area tolerance.

    Area-like products scale with the fourth power of length. Comparing them to
    a fixed epsilon misclassifies submillimeter face interiors as triangle edges.
    Degenerate faces are handled by their segments, not accepted as plane hits.
    """
    triangles=np.asarray(triangles,dtype=float)
    if not len(triangles):return math.inf
    a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a
    normal=np.cross(u,v);area2=np.sum(normal*normal,axis=1)
    delta=np.asarray(point)-a
    safe=np.where(area2>0,area2,1.)
    beta=np.sum(np.cross(delta,v)*normal,axis=1)/safe
    gamma=np.sum(np.cross(u,delta)*normal,axis=1)/safe
    inside=(area2>0)&(beta>=0)&(gamma>=0)&(beta+gamma<=1)
    distances=np.full(len(triangles),math.inf)
    distances[inside]=np.abs(np.sum(delta[inside]*normal[inside],axis=1))/np.sqrt(area2[inside])
    for i,j in ((0,1),(1,2),(2,0)):
        start=triangles[:,i];edge=triangles[:,j]-start
        length2=np.sum(edge*edge,axis=1)
        t=np.clip(np.sum((point-start)*edge,axis=1)/np.where(length2>0,length2,1.),0,1)
        distances=np.minimum(distances,np.linalg.norm(point-start-t[:,None]*edge,axis=1))
    return float(distances.min())


class DistanceQuery:
    def __init__(self, model, data, recovery=False):
        self.model, self.data = model, data
        self.meshes = {}
        self.face_normals = {}
        self.recoveries = []
        self.recovery = recovery

    def interior_radius(self, g, point):
        """点周围完全位于碰撞凸体内的球半径；负数表示点不在内部。"""
        m, d = self.model, self.data
        p = (point-d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3,3)
        size, kind = m.geom_size[g], m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            # A ball is inside an original closed convex mesh only if it is inside
            # every face half-space. Concave/open meshes receive no certificate.
            from .material import convex_solid
            if not hasattr(self,"solid_halfspaces"):self.solid_halfspaces={}
            mesh=int(m.geom_dataid[g])
            if mesh not in self.solid_halfspaces:
                vertices,triangles=self.vertices(g)
                start,count=m.mesh_faceadr[mesh],m.mesh_facenum[mesh]
                planes=None
                if convex_solid(vertices,m.mesh_face[start:start+count]):
                    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                    normals/=np.linalg.norm(normals,axis=1)[:,None]
                    center=vertices.mean(axis=0)
                    normals[np.sum((center-triangles[:,0])*normals,axis=1)>0]*=-1
                    planes=(normals,np.sum(triangles[:,0]*normals,axis=1))
                self.solid_halfspaces[mesh]=planes
            planes=self.solid_halfspaces[mesh]
            return None if planes is None else float(np.min(planes[1]-planes[0]@p))
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.min(size-np.abs(p)))
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0]-np.linalg.norm(p))
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(min(size[0]-np.linalg.norm(p[:2]), size[1]-abs(p[2])))
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return float(size[0]-np.linalg.norm(p-[0,0,np.clip(p[2],-size[1],size[1])]))
        return None

    def overlap_certificate(self, g1, g2):
        """共同内接球只证明穿透下界，不伪造精确距离或支撑法向。"""
        centers = [self.data.geom_xpos[g] for g in (g1,g2)]
        points = [(centers[0]+centers[1])/2]
        axes = [*self.data.geom_xmat[g1].reshape(3,3).T,
                *self.data.geom_xmat[g2].reshape(3,3).T]
        for axis in axes:
            for normal in (axis,-axis):
                a = self.support_point(g1,normal)
                b = self.support_point(g2,-normal)
                if a is not None and b is not None:
                    # 在共同支撑面之间取点；仍须独立核对点确实在两个实体内部。
                    points.extend(((a+b)/2, b+(float(a@normal)-float(b@normal))*normal/2,
                                   a-(float(a@normal)-float(b@normal))*normal/2))
        best = None
        for point in points:
            radii = [self.interior_radius(g,point) for g in (g1,g2)]
            if any(r is None for r in radii):
                continue
            radius = min(radii)-2e-6  # 保守扣除数值误差，不放宽穿透容差。
            if radius > 0 and (best is None or radius > best[0]):
                best = radius, point
        if best is None:
            return None
        radius, point = best
        return {"distance_m":None,"lower_bound_m":None,"points":None,"normal":None,
                "penetration_lower_bound_m":2*radius, "interior_point":point.tolist(),
                "interior_radius_m":radius,"method":"common_interior_ball",
                "geom1":self.model.geom(g1).name,"geom2":self.model.geom(g2).name,
                "consistency_checked":True}

    def support_point(self, g, direction):
        m,d=self.model,self.data
        r=d.geom_xmat[g].reshape(3,3);axis=r.T@direction
        length=np.linalg.norm(axis)
        if length<1e-15:
            return None
        axis=axis/length;size=m.geom_size[g];kind=m.geom_type[g]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            vertices=self.vertices(g)[0];point=vertices[int(np.argmax(vertices@axis))]
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            point=np.sign(axis)*size
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:
            point=axis*size[0]
        elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            point=size*size*axis/np.linalg.norm(size*axis)
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_CAPSULE):
            point=axis*size[0] if kind==mujoco.mjtGeom.mjGEOM_CAPSULE else np.r_[axis[:2]*size[0]/max(np.linalg.norm(axis[:2]),1e-15),0.]
            point[2]+=np.sign(axis[2])*size[1]
        else:
            return None
        return r@point+d.geom_xpos[g]

    def convex_distance(self, g1, g2, limit):
        """独立GJK凸体距离：上下界收敛后还须通过相同见证点校验；不猜测穿透深度。"""
        from .convex import closest_witness
        def support(direction):
            a=self.support_point(g1,direction);b=self.support_point(g2,-direction)
            if a is None or b is None:
                return None
            return a,b
        solved=closest_witness(support,self.data.geom_xpos[g2]-self.data.geom_xpos[g1],limit=limit)
        if solved is None or solved['upper']<=1e-9:
            return None  # 实体相交不等于接触；须另有支撑平面证书。
        if solved['lower']>=limit:
            return {'distance_m':None,'lower_bound_m':limit,'points':None,'normal':None}
        return self.validate(g1,g2,solved['upper'],np.r_[solved['a'],solved['b']],limit)[0]

    def feature_support(self, g, normal):
        """返回整片支撑区域的极值查询，而不是从中任选一个顶点。

        平面区域可以退化成边或点；圆柱端面保留为圆盘，不作多边形近似。
        """
        from itertools import product
        m,d=self.model,self.data
        rotation=d.geom_xmat[g].reshape(3,3); axis=rotation.T@normal
        size=m.geom_size[g]; kind=m.geom_type[g]
        if kind in (mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_MESH):
            vertices=(np.array(list(product((-1,1),repeat=3)))*size if kind==mujoco.mjtGeom.mjGEOM_BOX
                      else self.vertices(g)[0])
            projection=vertices@axis
            points=vertices[projection>=projection.max()-1e-10]@rotation.T+d.geom_xpos[g]
            return lambda tangent:points[int(np.argmax(points@tangent))]
        if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_CAPSULE):
            def support(tangent):
                t=rotation.T@tangent
                if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
                    p=axis/np.linalg.norm(axis)*size[0]
                else:
                    radial=axis[:2] if np.linalg.norm(axis[:2])>1e-12 else t[:2]
                    p=np.r_[radial*size[0]/max(np.linalg.norm(radial),1e-15),0.]
                p[2]+=np.sign(axis[2] if abs(axis[2])>1e-12 else t[2])*size[1]
                return rotation@p+d.geom_xpos[g]
            return support
        point=self.support_point(g,normal)
        return None if point is None else lambda tangent:point

    def support_witness(self, g1, g2, normal, gap, limit):
        """平移支撑区域到同一平面求交，再回到原位核对两端实际表面。"""
        from .convex import closest_witness
        first,second=self.feature_support(g1,normal),self.feature_support(g2,-normal)
        if first is None or second is None:
            return None
        origin=(self.data.geom_xpos[g1]+self.data.geom_xpos[g2])/2
        def support(direction):
            tangent=direction-float(direction@normal)*normal
            return first(tangent)-origin,second(-tangent)-origin-gap*normal
        seed=np.eye(3)[int(np.argmin(np.abs(normal)))]
        solved=closest_witness(support,np.cross(normal,seed),tolerance=1e-10)
        if solved is None or solved['upper']>1e-9:
            return None
        segment=np.r_[solved['a']+origin,solved['b']+origin+gap*normal]
        return self.validate(g1,g2,gap,segment,limit,normal_hint=normal)[0]

    def separating_faces(self, g):
        """候选法向由实际几何产生；是否为有效支撑面仍以全体顶点投影验证。"""
        m,d=self.model,self.data
        rotation=d.geom_xmat[g].reshape(3,3); kind=m.geom_type[g]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:
            return [sign*rotation[:,axis] for axis in range(3) for sign in (1,-1)]
        if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_PLANE):
            return [rotation[:,2],-rotation[:,2]] if kind==mujoco.mjtGeom.mjGEOM_CYLINDER else [rotation[:,2]]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(m.geom_dataid[g])
            if mesh not in self.face_normals:
                triangles=self.vertices(g)[1]
                normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                lengths=np.linalg.norm(normals,axis=1)
                normals=normals[lengths>1e-15]/lengths[lengths>1e-15,None]
                unique={tuple(np.round(n,12)):n for n in np.vstack((normals,-normals))}
                self.face_normals[mesh]=list(unique.values())
            return [rotation@n for n in self.face_normals[mesh]]
        return []

    def vertices(self, g):
        m = self.model
        mesh = int(m.geom_dataid[g])
        if mesh not in self.meshes:
            va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            v = m.mesh_vert[va:va+vn].astype(float)
            self.meshes[mesh] = (v, v[m.mesh_face[fa:fa+fn]])
        return self.meshes[mesh]

    def interval(self, g, normal):
        m,d = self.model,self.data
        axis = d.geom_xmat[g].reshape(3,3).T @ normal
        center = float(d.geom_xpos[g] @ normal)
        kind,size = m.geom_type[g],m.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            values = self.vertices(g)[0] @ axis
            return center+float(values.min()),center+float(values.max())
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            radius = float(np.abs(axis) @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = size[0]*np.linalg.norm(axis[:2])+size[1]*abs(axis[2])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0]+size[1]*abs(axis[2])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radius = np.linalg.norm(size*axis)
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE and np.linalg.norm(axis[:2]) < 1e-7:
            radius = 0
        else:
            return -math.inf,math.inf
        return center-radius,center+radius

    def mesh_support_error(self, g, point, normal):
        """在凸包支撑面内核对见证点；不能误用原始凹网格三角面。"""
        rotation=self.data.geom_xmat[g].reshape(3,3)
        axis=rotation.T@normal
        p=(point-self.data.geom_xpos[g])@rotation
        vertices=self.vertices(g)[0]
        projection=vertices@axis
        height=float(projection.max())
        plane_error=abs(float(p@axis)-height)
        if plane_error>2e-6:
            return plane_error
        points=vertices[projection>=height-2e-6]
        # 支撑面投影为二维凸多边形；单调链只依赖顶点，不依赖凸包三角化。
        seed=np.eye(3)[int(np.argmin(np.abs(axis)))]
        u=np.cross(axis,seed);u/=np.linalg.norm(u);v=np.cross(axis,u)
        points=sorted(set(map(tuple,np.column_stack((points@u,points@v)))))
        target=np.array([p@u,p@v])
        def cross(a,b,c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        halves=[]
        for seq in (points,points[::-1]):
            half=[]
            for x in seq:
                while len(half)>=2 and cross(half[-2],half[-1],x)<=0:
                    half.pop()
                half.append(x)
            halves.append(half)
        hull=halves[0][:-1]+halves[1][:-1] if len(points)>1 else points
        if len(hull)>=3 and all(cross(hull[i],hull[(i+1)%len(hull)],target)>=0 for i in range(len(hull))):
            return plane_error
        nearest=math.inf
        for i,a in enumerate(hull):
            a=np.array(a);b=np.array(hull[(i+1)%len(hull)]);edge=b-a
            t=np.clip(float((target-a)@edge)/max(float(edge@edge),1e-30),0,1)
            nearest=min(nearest,float(np.linalg.norm(target-a-t*edge)))
        return math.hypot(plane_error,nearest)

    def surface_error(self, g, point, normal=None):
        m,d = self.model,self.data
        p = (point-d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3,3)
        kind,size = m.geom_type[g],m.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            q = np.abs(p)-size
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            q = np.array([np.linalg.norm(p[:2])-size[0],abs(p[2])-size[1]])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return abs(float(np.linalg.norm(p)-size[0]))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return abs(float(np.linalg.norm(p-[0,0,np.clip(p[2],-size[1],size[1])])-size[0]))
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            return abs(float(p[2]))
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return abs(float(np.linalg.norm(p/size)-1))*float(size.min())
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            if normal is not None:
                return self.mesh_support_error(g,point,normal)
            triangles = self.vertices(g)[1]
            if len(triangles) > 200000:return math.inf
            return triangle_surface_distance(triangles,p)
        else:
            return math.inf
        return abs(float(np.linalg.norm(np.maximum(q,0))+min(q.max(),0)))

    def validate(self, g1, g2, distance, segment, limit, normal_hint=None):
        eps = 2e-6  # 数值一致性阈值，不是允许穿透或悬空的公差。
        if not math.isfinite(distance) or not np.isfinite(segment).all():
            return None,"距离或接触点不是有限数"
        if distance >= limit-1e-10 and np.linalg.norm(segment) < 1e-12:
            return {"distance_m":None,"lower_bound_m":limit,"points":None,"normal":None},None
        if abs(float(np.linalg.norm(segment[3:]-segment[:3]))-abs(distance)) > eps:
            return None,"接触点间距与有符号距离不一致"
        segment_length=float(np.linalg.norm(segment[3:]-segment[:3]))
        if abs(distance) > 1e-8 and segment_length>1e-12:
            candidates = [(segment[3:]-segment[:3])/segment_length*(1 if distance>0 else -1)]
        else:
            candidates = [np.array([0.,0.,1.]),np.array([0.,0.,-1.])]
            if normal_hint is not None:
                candidates.insert(0,normal_hint)
            for g in (g1,g2):
                candidates.extend(self.data.geom_xmat[g].reshape(3,3).T)
                candidates.extend(-self.data.geom_xmat[g].reshape(3,3).T)
            delta = self.data.geom_xpos[g2]-self.data.geom_xpos[g1]
            if np.linalg.norm(delta) > eps:
                candidates.append(delta/np.linalg.norm(delta))
        normal = None
        errors = None
        for axis in candidates:
            a,b = self.interval(g1,axis)[1],self.interval(g2,axis)[0]
            if (math.isfinite(a) and math.isfinite(b) and abs((b-a)-distance) <= eps
                    and abs(float(segment[:3]@axis)-a) <= eps and abs(float(segment[3:]@axis)-b) <= eps):
                surface_errors=[self.surface_error(g1,segment[:3],axis),self.surface_error(g2,segment[3:],-axis)]
                if max(surface_errors)<=eps:
                    normal = axis.tolist()
                    errors = surface_errors
                    break
        if normal is None:
            return None,"无法核对最近点、接触法向与实际碰撞形状支撑面"
        return {"distance_m":distance,"lower_bound_m":None,"points":segment.tolist(),
                "normal":normal,"surface_errors_m":errors},None

    def analytic_face(self, g1, g2, limit):
        """支撑平面下界加支撑区域求交见证；适用于网格面、箱体面及圆柱端面。"""
        m,d=self.model,self.data
        for face,other,reverse in ((g1,g2,False),(g2,g1,True)):
            for normal in self.separating_faces(face):
                normal=normal/np.linalg.norm(normal)
                far=self.interval(face,normal)[1]; near=self.interval(other,normal)[0]
                gap=near-far
                # 刚体变换和点积会把精确接触变成机器舍入量级的微小负值。
                # 仅容纳浮点运算误差，不使用碰撞/承托验收公差放宽求解范围。
                roundoff=64*np.finfo(float).eps*max(1.,abs(far),abs(near),
                    float(np.linalg.norm(d.geom_xpos[face])),float(np.linalg.norm(d.geom_xpos[other])))
                if not math.isfinite(gap) or gap < -roundoff or roundoff>1e-9:
                    continue
                if gap>=limit:
                    return {'distance_m':None,'lower_bound_m':limit,'points':None,'normal':None}
                if m.geom_type[face]==mujoco.mjtGeom.mjGEOM_PLANE:
                    p=self.support_point(other,-normal)
                    if p is None:
                        continue
                    q=p-gap*normal
                    segment=np.r_[p,q] if reverse else np.r_[q,p]
                    result=self.validate(g1,g2,gap,segment,limit,normal_hint=-normal if reverse else normal)[0]
                else:
                    result=self.support_witness(g1,g2,-normal if reverse else normal,gap,limit)
                if result is not None:
                    if gap<0:
                        # 原样保留有符号投影及其数值不确定区间，不伪造精确零。
                        result['distance_bounds_m']=[gap,0.]
                        result['roundoff_bound_m']=roundoff
                    return result
        return None

    def box_overlap(self, g1, g2, limit):
        """两个箱体的分离轴定理；仅在还能构造表面见证点时输出穿透深度。"""
        m,d=self.model,self.data
        if any(m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX for g in (g1,g2)):
            return None
        a,b=[d.geom_xmat[g].reshape(3,3) for g in (g1,g2)]
        axes=[*a.T,*b.T,*[np.cross(x,y) for x in a.T for y in b.T]]
        depth,normal=math.inf,None
        for axis in axes:
            length=np.linalg.norm(axis)
            if length<1e-10:
                continue
            axis=axis/length
            al,ah=self.interval(g1,axis);bl,bh=self.interval(g2,axis)
            if ah<bl or bh<al:
                return None
            for penetration,n in ((ah-bl,axis),(bh-al,-axis)):
                if penetration<depth:
                    depth,normal=penetration,n
        if normal is None:
            return None
        return self.support_witness(g1,g2,normal,-depth,limit)

    def native_candidates(self, g1, g2, limit):
        """Native CCD answers in both argument orders, each with both witness orders.

        MuJoCo 3.3.7 reorders a pair by geom type internally and, with native CCD, can
        return the witness points without swapping them back. The candidates are only
        re-labelled measurements; validate() still certifies every one independently.
        """
        m = self.model
        for first,second,flip in ((g1,g2,False),(g2,g1,True)):
            segment = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(m,self.data,first,second,limit,segment))
            if flip:
                segment = np.r_[segment[3:],segment[:3]]
            yield distance,segment
            yield distance,np.r_[segment[3:],segment[:3]]

    def measure(self, g1, g2, limit=.05):
        m = self.model
        original = int(m.opt.disableflags)
        solver = (int(m.opt.ccd_iterations),float(m.opt.ccd_tolerance))
        errors = []
        try:
            # Measurement only: converge mesh-mesh overlap depth instead of the step-time budget.
            # MuJoCo 3.3.7 sizes its native polytope buffers for about 100 iterations; more crashes.
            m.opt.ccd_iterations,m.opt.ccd_tolerance = max(solver[0],100),min(solver[1],1e-9)
            for method,flags in (("native",original & ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
                                 ("legacy_checked",original | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))):
                m.opt.disableflags = flags
                if method == "native":
                    for distance,segment in self.native_candidates(g1,g2,limit):
                        result,error = self.validate(g1,g2,distance,segment,limit)
                        if result is not None:
                            break
                else:
                    segment = np.zeros(6)
                    distance = float(mujoco.mj_geomDistance(m,self.data,g1,g2,limit,segment))
                    result,error = self.validate(g1,g2,distance,segment,limit)
                if result is not None:
                    result.update(geom1=m.geom(g1).name,geom2=m.geom(g2).name,method=method,
                                  consistency_checked=True)
                    if errors:
                        result["recovered_from"] = errors
                        if len(self.recoveries) < 32:
                            self.recoveries.append({"geom1":m.geom(g1).name,"geom2":m.geom(g2).name,"errors":errors})
                    return result
                errors.append({"method":method,"error":error})
                if method=='native':
                    analytic=self.analytic_face(g1,g2,limit)
                    analytic_method='analytic_face'
                    if analytic is None:
                        analytic=self.box_overlap(g1,g2,limit)
                        analytic_method='analytic_box_overlap'
                    if analytic is not None:
                        analytic.update(geom1=m.geom(g1).name,geom2=m.geom(g2).name,method=analytic_method,
                                        consistency_checked=True,recovered_from=errors)
                        if len(self.recoveries)<32:
                            self.recoveries.append({'geom1':m.geom(g1).name,'geom2':m.geom(g2).name,'errors':errors,'method':analytic_method})
                        return analytic
        finally:
            m.opt.disableflags = original
            m.opt.ccd_iterations,m.opt.ccd_tolerance = solver
        result=self.convex_distance(g1,g2,limit)
        if result is not None:
            result.update(geom1=m.geom(g1).name,geom2=m.geom(g2).name,method='convex_distance',
                          consistency_checked=True,recovered_from=errors)
            if len(self.recoveries)<32:
                self.recoveries.append({'geom1':m.geom(g1).name,'geom2':m.geom(g2).name,'errors':errors,'method':'convex_distance'})
            return result
        if self.recovery:
            result = self.overlap_certificate(g1,g2)
            if result is not None:
                return {**result,"recovered_from":errors}
        raise MeasurementError(f"距离证据不可靠：{m.geom(g1).name} / {m.geom(g2).name}；{errors}")
