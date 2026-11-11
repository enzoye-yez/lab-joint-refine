"""Static geometry state without the global dynamics solver; never advances time."""
import mujoco

def geometry_state(model, data, *, contacts=False):
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    if model.nflex:
        mujoco.mj_flex(model, data)
    if contacts:
        mujoco.mj_collision(model, data)
