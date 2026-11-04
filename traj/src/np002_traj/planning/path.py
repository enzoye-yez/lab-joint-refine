"""Joint-space paths: RRT-Connect with shortcutting, then smooth time scaling under velocity and acceleration limits."""
import numpy as np
from scipy.interpolate import CubicSpline


class NoPath(Exception):
    pass


def rrt_connect(start, goal, limits, edge_valid, rng, iterations=3000, step=0.25):
    start, goal, limits = np.asarray(start, float), np.asarray(goal, float), np.asarray(limits, float)
    if edge_valid(start, goal):
        return [start, goal]
    trees = [[start], [goal]]
    parents = [[-1], [-1]]

    def extend(tree, parent, target):
        points = np.array(tree)
        i = int(np.argmin(np.linalg.norm(points - target, axis=1)))
        delta = target - tree[i]
        distance = np.linalg.norm(delta)
        new = target if distance <= step else tree[i] + delta * (step / distance)
        if not edge_valid(tree[i], new):
            return None
        tree.append(new)
        parent.append(i)
        return len(tree) - 1

    def chain(tree, parent, i):
        out = []
        while i != -1:
            out.append(tree[i])
            i = parent[i]
        return out[::-1]

    for iteration in range(iterations):
        side = iteration % 2
        target = rng.uniform(limits[:, 0], limits[:, 1])
        a = extend(trees[side], parents[side], target)
        if a is None:
            continue
        other = 1 - side
        while True:                                  # connect greedily
            b = extend(trees[other], parents[other], trees[side][a])
            if b is None:
                break
            if np.linalg.norm(trees[other][b] - trees[side][a]) < 1e-9:
                first, second = chain(trees[side], parents[side], a), chain(trees[other], parents[other], b)
                path = first + second[::-1][1:]
                return shortcut(path[::-1] if side else path, edge_valid, rng)
    raise NoPath(f"RRT-Connect found no path in {iterations} iterations")


def shortcut(path, edge_valid, rng, attempts=60):
    path = list(path)
    for _ in range(attempts):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and edge_valid(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def time_scale(path, max_velocity, max_acceleration, rate=500.0):
    """Cubic spline through the waypoints with zero end velocities, slowed until within limits.

    Returns (times, positions, velocities) sampled at `rate` Hz.
    """
    path = np.asarray(path, float)
    if len(path) == 1:
        return np.zeros(1), path, np.zeros_like(path)
    lengths = np.max(np.abs(np.diff(path, axis=0)) / max_velocity, axis=1)
    knots = np.concatenate([[0], np.cumsum(np.maximum(lengths, 1e-3) * 1.5)])
    for _ in range(30):
        spline = CubicSpline(knots, path, bc_type="clamped")
        t = np.linspace(0, knots[-1], max(2, int(knots[-1] * rate) + 1))
        v, a = spline(t, 1), spline(t, 2)
        scale = max(np.max(np.abs(v) / max_velocity), np.sqrt(np.max(np.abs(a) / max_acceleration)), 1e-9)
        if scale <= 1.0:
            return t, spline(t), v
        knots = knots * scale * 1.02
    raise RuntimeError("time scaling did not converge")
