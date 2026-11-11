"""Base navigation: a 2-D occupancy map from the scene geometry and A* with a closed set.

The map is the floor-plane projection of every non-robot collision geometry between the floor and
the robot's height. Search uses a disc that circumscribes the base footprint, so any heading is
safe along the path; the final docking pose is checked with the exact footprint by the caller.
"""
import heapq
import math

import mujoco
import numpy as np

PLANES = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))


def world_aabbs(model, data, geoms):
    rot = data.geom_xmat[geoms].reshape(-1, 3, 3)
    local = model.geom_aabb[geoms]
    centre = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rot, local[:, :3])
    half = np.einsum("nij,nj->ni", np.abs(rot), local[:, 3:])
    return centre - half, centre + half


class OccupancyMap:
    def __init__(self, model, data, robot, resolution=0.04, height=(0.03, 2.0), bounds=None):
        robot_geoms = set(robot.geoms.tolist())
        geoms = np.array([g for g in range(model.ngeom) if g not in robot_geoms and model.geom_type[g] not in PLANES
                          and (model.geom_contype[g] or model.geom_conaffinity[g])], int)
        low, high = world_aabbs(model, data, geoms)
        keep = (high[:, 2] > height[0]) & (low[:, 2] < height[1])
        low, high, self.geoms = low[keep], high[keep], geoms[keep]
        if bounds is None:
            planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
            size = model.geom_size[planes[0]][:2] if planes else np.array([5.0, 5.0])
            centre = data.geom_xpos[planes[0]][:2] if planes else np.zeros(2)
            bounds = (centre - size, centre + size)
        self.origin, self.resolution = np.asarray(bounds[0], float), resolution
        shape = np.ceil((np.asarray(bounds[1]) - self.origin) / resolution).astype(int)
        self.occupied = np.zeros(shape, bool)
        # Walls beyond the floor's edge
        self.occupied[[0, -1], :] = True
        self.occupied[:, [0, -1]] = True
        for lo, hi in zip(low[:, :2], high[:, :2]):
            i0, j0 = np.floor((lo - self.origin) / resolution).astype(int)
            i1, j1 = np.ceil((hi - self.origin) / resolution).astype(int)
            self.occupied[max(i0, 0):max(i1, 0), max(j0, 0):max(j1, 0)] = True
        self._clearance = None

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int))

    def point(self, cell):
        return self.origin + (np.asarray(cell) + 0.5) * self.resolution

    @property
    def clearance(self):
        """Distance (m) from each cell centre to the nearest occupied cell."""
        if self._clearance is None:
            from scipy.ndimage import distance_transform_edt
            self._clearance = distance_transform_edt(~self.occupied) * self.resolution
        return self._clearance

    def free(self, xy, radius):
        i, j = self.cell(xy)
        if not (0 <= i < self.occupied.shape[0] and 0 <= j < self.occupied.shape[1]):
            return False
        return self.clearance[i, j] >= radius

    def plan(self, start, goal, radius, goal_radius=None, max_expansions=400000):
        """Shortest 8-connected path from start to goal (xy) keeping `radius` clearance.

        Cells within `goal_radius` of the goal may have less clearance (the dock pose itself is
        validated separately with the exact footprint).
        """
        start_cell, goal_cell = self.cell(start), self.cell(goal)
        goal_radius = goal_radius if goal_radius is not None else 0.0
        clear = self.clearance >= radius
        near_goal = lambda c: math.dist(c, goal_cell) * self.resolution <= goal_radius
        frontier = [(0.0, start_cell)]
        cost, parent, closed = {start_cell: 0.0}, {}, set()
        steps = [(a, b, math.hypot(a, b)) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
        expansions = 0
        while frontier and expansions < max_expansions:
            _, current = heapq.heappop(frontier)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1
            if current == goal_cell:
                cells = [current]
                while current in parent:
                    current = parent[current]
                    cells.append(current)
                return self.simplify([self.point(c) for c in cells[::-1]], radius, start, goal)
            for di, dj, length in steps:
                nxt = (current[0] + di, current[1] + dj)
                if nxt in closed or not (0 <= nxt[0] < clear.shape[0] and 0 <= nxt[1] < clear.shape[1]):
                    continue
                if not (clear[nxt] or near_goal(nxt) or math.dist(nxt, start_cell) * self.resolution <= radius):
                    continue
                value = cost[current] + length * self.resolution
                if value < cost.get(nxt, math.inf):
                    cost[nxt], parent[nxt] = value, current
                    heapq.heappush(frontier, (value + math.dist(nxt, goal_cell) * self.resolution, nxt))
        return None

    def simplify(self, points, radius, start, goal):
        """Keep only the waypoints needed for straight segments that stay clear."""
        points = [np.asarray(start, float)] + points[1:-1] + [np.asarray(goal, float)]
        out, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_clear(points[i], points[j], radius):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def segment_clear(self, a, b, radius):
        count = max(2, int(np.linalg.norm(b - a) / (self.resolution / 2)))
        return all(self.free(a + (b - a) * t, radius) for t in np.linspace(0, 1, count)[1:-1])
