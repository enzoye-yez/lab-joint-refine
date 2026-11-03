"""Fixed numerical tools: search sets candidate poses only in a data copy and never moves the execution world."""
import heapq
import math
import mujoco
from .common import copy_data
import numpy as np
from .common import OperationError, need, vector


def interpolate(a, b, resolution):
    a, b = np.asarray(a), np.asarray(b)
    count = max(1, int(np.ceil(np.max(np.abs(b-a))/resolution)))
    return [a+(b-a)*t/count for t in range(1, count+1)]


def edge_valid(a, b, valid, resolution):
    return all(valid(q) for q in interpolate(a, b, resolution))


def rrt_connect(start, goal, limits, valid, rng, iterations=2000, resolution=.04, start_tolerance=0.):
    start, goal, limits = np.asarray(start), np.asarray(goal), np.asarray(limits)
    need(start.shape == goal.shape and limits.shape == (len(start), 2), "Search dimension mismatch")
    def legal(q):
        return bool(np.all(q >= limits[:,0]) and np.all(q <= limits[:,1]) and valid(q))
    need(np.isfinite(start_tolerance) and start_tolerance>=0,"Invalid start-state measurement tolerance")
    projected=np.clip(start,limits[:,0],limits[:,1])
    if not np.array_equal(projected,start):
        # Soft constraint observations may cross a limit slightly. Validate the recovery edge;
        # only bounded setpoints are commanded, never teleport the measured state.
        if np.max(np.abs(projected-start))>start_tolerance or not valid(start) or not edge_valid(start,projected,valid,resolution):
            raise OperationError("path_not_found","Measured start state is beyond recoverable limits or the recovery edge collides")
        return [start]+rrt_connect(projected,goal,limits,valid,rng,iterations,resolution)
    if not legal(start) or not legal(goal):
        raise OperationError("path_not_found", "Start or goal violates collision or limit constraints")
    if edge_valid(start, goal, legal, resolution): return [start, goal]
    trees = [([start], [-1]), ([goal], [-1])]
    step = max(resolution*4, .1)
    def extend(tree, target):
        points, parents = tree
        idx = int(np.argmin([np.linalg.norm(q-target) for q in points]))
        delta = target-points[idx]
        new = points[idx]+delta*min(1., step/max(np.linalg.norm(delta), 1e-12))
        if not edge_valid(points[idx], new, legal, resolution): return None, False
        points.append(new); parents.append(idx)
        return len(points)-1, np.linalg.norm(new-target) < 1e-9
    def chain(tree, idx):
        out = []
        while idx != -1:
            out.append(tree[0][idx]); idx = tree[1][idx]
        return out[::-1]
    for iteration in range(iterations):
        side = iteration % 2
        first, second = trees[side], trees[1-side]
        target = second[0][0] if rng.random() < .15 else rng.uniform(limits[:,0], limits[:,1])
        a, _ = extend(first, target)
        if a is None: continue
        for _ in range(100):
            b, reached = extend(second, first[0][a])
            if b is None: break
            if reached:
                left, right = chain(first, a), chain(second, b)
                result = left+right[-2::-1]
                if side: result.reverse()
                # Bounded shortcut smoothing still checks the whole edge.
                for _ in range(40):
                    if len(result) < 3: break
                    i, j = sorted(rng.choice(len(result), 2, replace=False))
                    if j > i+1 and edge_valid(result[i], result[j], legal, resolution):
                        result[i+1:j] = []
                return result
    raise OperationError("path_not_found", "This bounded search found no path; this is not a proof of unreachability")


def astar(start, goal, bounds, valid, grid=.05, max_nodes=100000):
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    start, goal = np.asarray(start), np.asarray(goal)
    need(np.all(hi > lo), "Invalid navigation bounds")
    key = lambda p: tuple(np.round((p-lo)/grid).astype(int))
    position = lambda k: lo+np.array(k)*grid
    s, g = key(start), key(goal)
    def legal(p): return bool(np.all(p >= lo) and np.all(p <= hi) and valid(p))
    if not legal(start) or not legal(goal): raise OperationError("path_not_found", "Invalid navigation endpoint")
    queue, cost, parent = [(0., s)], {s: 0.}, {}
    directions = [(a,b) for a in (-1,0,1) for b in (-1,0,1) if a or b]
    for _ in range(max_nodes):
        if not queue: break
        _, current = heapq.heappop(queue)
        if current == g:
            route = [goal]
            while current in parent:
                route.append(position(current)); current = parent[current]
            route.append(start)
            route.reverse()
            if all(edge_valid(a,b,legal,grid/3) for a,b in zip(route,route[1:])): return route
            break
        for dx,dy in directions:
            nxt = (current[0]+dx, current[1]+dy)
            p, q = position(current), position(nxt)
            if not edge_valid(p,q,legal,grid/3): continue
            value = cost[current]+math.hypot(dx,dy)*grid
            if value < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = value, current
                heapq.heappush(queue, (value+float(np.linalg.norm(q-goal)), nxt))
    raise OperationError("path_not_found", "This navigation search found no path")


def rotation_error(target, actual):
    # MuJoCo quaternion order is wxyz.
    qa, qb, neg, delta, velocity = (np.zeros(n) for n in (4,4,4,4,3))
    mujoco.mju_mat2Quat(qa, np.asarray(target).reshape(9))
    mujoco.mju_mat2Quat(qb, np.asarray(actual).reshape(9))
    mujoco.mju_negQuat(neg, qb)
    mujoco.mju_mulQuat(delta, qa, neg)
    if delta[0] < 0: delta *= -1
    mujoco.mju_quat2Vel(velocity, delta, 1.)
    return velocity


class Kinematics:
    def __init__(self, model, data, joints, valid, settings, rng):
        self.model, self.settings, self.valid, self.rng = model, settings, valid, rng
        self.data = mujoco.MjData(model)
        copy_data(self.data, model, data)
        ids = [model.joint(name).id for name in joints]
        need(all(model.jnt_type[j] in (2,3) for j in ids), "Planning joints must be slide or hinge joints")
        self.qids, self.vids = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        need(all(model.jnt_limited[j] for j in ids), "Operation joints must have explicit limits")
        self.limits = model.jnt_range[ids].copy()

    def solve(self, targets):
        need(bool(targets),"Inverse kinematics targets must not be empty")
        start = np.clip(self.data.qpos[self.qids],self.limits[:,0],self.limits[:,1])
        best = {"position_error_m":float("inf"),"rotation_error_rad":float("inf")}
        for restart in range(self.settings.ik_restarts):
            q = start.copy() if restart == 0 else self.rng.uniform(self.limits[:,0], self.limits[:,1])
            for _ in range(self.settings.ik_iterations):
                self.data.qpos[self.qids] = q
                mujoco.mj_forward(self.model, self.data)
                errors, jacobians, satisfied = [], [], True
                for target in targets:
                    sid = self.model.site(target["site"]).id
                    dp = vector(target["position"],3)-self.data.site_xpos[sid]
                    dr = rotation_error(np.asarray(target["rotation"]).reshape(3,3), self.data.site_xmat[sid].reshape(3,3))
                    jp, jr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
                    mujoco.mj_jacSite(self.model,self.data,jp,jr,sid)
                    errors.extend(dp); errors.extend(dr)
                    jacobians.extend(jp[:,self.vids]); jacobians.extend(jr[:,self.vids])
                    satisfied &= np.linalg.norm(dp) <= self.settings.position_tolerance_m and np.linalg.norm(dr) <= self.settings.rotation_tolerance_rad
                position_error=max(float(np.linalg.norm(errors[i:i+3])) for i in range(0,len(errors),6))
                rotation_residual=max(float(np.linalg.norm(errors[i+3:i+6])) for i in range(0,len(errors),6))
                if position_error+rotation_residual < best["position_error_m"]+best["rotation_error_rad"]:
                    best={"position_error_m":position_error,"rotation_error_rad":rotation_residual,"joint_candidate":q.tolist()}
                if satisfied and self.valid(q): return q
                jac, err = np.asarray(jacobians), np.asarray(errors)
                dq = jac.T @ np.linalg.solve(jac@jac.T+np.eye(len(err))*.0025, err)
                q = np.clip(q+np.clip(dq,-.12,.12),self.limits[:,0],self.limits[:,1])
        raise OperationError("ik_not_found", "This inverse kinematics run found no valid solution; this is not a proof of unreachability",
                             {**best,"restarts":self.settings.ik_restarts,"iterations_per_restart":self.settings.ik_iterations})


def wheel_commands(vx, vy, omega, wheels):
    result = []
    for wheel in wheels:
        x,y = wheel["xy"]
        wx,wy = vx-omega*y, vy+omega*x
        angle, speed = math.atan2(wy,wx), math.hypot(wx,wy)/wheel["radius"]
        if angle > math.pi/2: angle -= math.pi; speed *= -1
        if angle < -math.pi/2: angle += math.pi; speed *= -1
        result.append((angle, speed*wheel.get("sign",1)))
    return result
