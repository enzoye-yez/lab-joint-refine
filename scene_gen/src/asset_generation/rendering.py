"""Allocate visualization capacity from the loaded model, not a fixed 10k limit."""
import mujoco


def renderer(model, *, width, height):
    # Geometry plus sites/body decorations/tendon segments; static views add no trajectories.
    capacity=max(10000,int(model.ngeom+model.nsite+model.nbody+2*model.ntendon+1024))
    return mujoco.Renderer(model,width=width,height=height,max_geom=capacity)
