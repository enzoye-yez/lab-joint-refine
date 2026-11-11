"""Closest points between convex sets and paired witness points; used both for solid distance and for intersecting support regions."""

from itertools import combinations
import math
import numpy as np


def closest_witness(support, direction, *, tolerance=1e-9, limit=math.inf, iterations=96):
    """GJK (convex-set distance algorithm): keeps convex-combination witnesses; zero distance is not discarded at the solver layer.

    support(direction) returns the extreme points of the two sets in opposite directions. Zero distance only proves
    that the sets intersect or are very close; the caller must separately prove a supporting plane before claiming solid contact.
    """
    direction = np.asarray(direction,dtype=float)
    if np.linalg.norm(direction)<1e-12:
        direction = np.array([1.,0.,0.])
    simplex = []
    for _ in range(iterations):
        pair = support(direction)
        if pair is None:
            return None
        a,b = pair
        simplex.append((a-b,a,b))
        best = None
        for size in range(1,min(4,len(simplex))+1):
            for indices in combinations(range(len(simplex)),size):
                points = np.array([simplex[i][0] for i in indices])
                if size==1:
                    weights = np.array([1.])
                else:
                    tail = np.linalg.lstsq((points[1:]-points[0]).T,-points[0],rcond=None)[0]
                    weights = np.r_[1-tail.sum(),tail]
                if weights.min() < -1e-10:
                    continue
                weights = np.maximum(weights,0); weights /= weights.sum()
                closest = weights@points
                squared = float(closest@closest)
                if best is None or squared<best[0]:
                    best = squared,indices,weights,closest
        if best is None:
            return None
        squared,indices,weights,closest = best
        simplex = [simplex[i] for i in indices]
        upper = math.sqrt(squared)
        a = sum(w*p[1] for w,p in zip(weights,simplex))
        b = sum(w*p[2] for w,p in zip(weights,simplex))
        if upper <= tolerance:
            return {'upper':upper,'lower':0.,'a':a,'b':b}
        direction = -closest/upper
        pair = support(direction)
        if pair is None:
            return None
        lower = max(0.,float((pair[0]-pair[1])@(-direction)))
        if lower>=limit or upper-lower<=tolerance:
            return {'upper':upper,'lower':lower,'a':a,'b':b}
    return None
