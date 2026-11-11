"""Conservative spatial candidates; exact box filtering and input ordering retained."""
import numpy as np
from scipy.spatial import cKDTree

class BoxIndex:
    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.lower = np.array([b[0] for b in boxes], dtype=float).reshape(-1,3)
        self.upper = np.array([b[1] for b in boxes], dtype=float).reshape(-1,3)
        centers = (self.lower+self.upper)/2
        radii=np.linalg.norm((self.upper-self.lower)/2,axis=1)
        self.groups=[]
        # Large trays must not inflate every tiny feature's search radius.
        # Disjoint size bands retain every candidate without a global worst-case radius.
        bands=np.frexp(radii)[1]
        for band in np.unique(bands):
            ids=np.flatnonzero(bands==band)
            tree=cKDTree(centers[ids]) if len(ids)>32 else None
            self.groups.append((ids,tree,float(radii[ids].max())))

    def overlap(self, lo, hi, tolerance=0.):
        lo,hi = np.asarray(lo),np.asarray(hi)
        center=(lo+hi)/2
        own_radius=float(np.linalg.norm((hi-lo)/2))+np.sqrt(3)*tolerance
        candidates=[]
        for ids,tree,group_radius in self.groups:
            radius=own_radius+group_radius
            selected=ids if tree is None else ids[tree.query_ball_point(center,radius+1e-10*(1+radius))]
            candidates.extend(selected.tolist())
        ids=np.asarray(sorted(candidates),dtype=int)
        return ids[np.all(self.upper[ids]+tolerance>=lo,axis=1) & np.all(hi+tolerance>=self.lower[ids],axis=1)].tolist()
