"""独立碰撞分离证书和球体接触解析量测；不更改引擎选项或验收公差。"""

from pathlib import Path
import numpy as np
import mujoco

from asset_generation.checks import load_scene_model
from asset_generation.contracts import read_json
from asset_generation.distance import DistanceQuery, MeasurementError


def load_scene(root, limits=None, contract=None, checkpoint_dir=None):
    if checkpoint_dir is None:
        model = load_scene_model(root)
    else:
        from .checkpoints import Checkpoints
        from asset_generation.checks import measure_model
        cache = Checkpoints(checkpoint_dir, root, limits, contract, True)
        model, _ = cache.load_model(lambda: measure_model(root, scope='scene',
                                                        static=limits.validation_mode == 'static'))
    meta = read_json(Path(root)/'scene.json',8000000)
    data = mujoco.MjData(model)
    data.qpos[:] = meta['home_qpos']
    data.ctrl[:] = meta['home_ctrl']
    from asset_generation.static_state import geometry_state
    geometry_state(model,data,contacts=True)
    return model,data,meta


class CertificateQuery(DistanceQuery):
    def __init__(self, model, data, support=False):
        super().__init__(model,data)
        self.for_support = support

    def measure(self, g1, g2, limit=.05):
        m,d = self.model,self.data
        if all(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE for g in (g1,g2)):
            delta = d.geom_xpos[g2]-d.geom_xpos[g1]
            length = float(np.linalg.norm(delta))
            if length > 1e-12:
                normal = delta/length
                a = d.geom_xpos[g1]+normal*m.geom_size[g1,0]
                b = d.geom_xpos[g2]-normal*m.geom_size[g2,0]
                result,error = self.validate(g1,g2,length-m.geom_size[g1,0]-m.geom_size[g2,0],np.r_[a,b],limit)
                if result is not None:
                    return {**result,'method':'analytic_spheres','consistency_checked':True}
        # 投影间隙只是有符号距离的下界；不得伪造成接触法向或承托证据。
        axes = [*np.eye(3),*d.geom_xmat[g1].reshape(3,3).T,*d.geom_xmat[g2].reshape(3,3).T,
                d.geom_xpos[g2]-d.geom_xpos[g1]]
        for raw in axes:
            length = np.linalg.norm(raw)
            if length < 1e-12:
                continue
            axis = raw/length
            a,b = self.interval(g1,axis),self.interval(g2,axis)
            lower = max(b[0]-a[1],a[0]-b[1])-2e-6
            if np.isfinite(lower) and lower > (limit if self.for_support else 0):
                return {'distance_m':None,'lower_bound_m':float(lower),'points':None,'normal':None,
                        'method':'separating_projection','axis':axis.tolist(),'consistency_checked':True}
        # 不重复失败的引擎组合；对本能力不能回答的范围明确返回未知。
        raise MeasurementError('独立解析或分离证书不覆盖此几何对')


class RecoveryQuery(DistanceQuery):
    """保留已可量测范围，用独立证书补足原方法失败的几何对。"""
    def __init__(self, model, data, support=False):
        super().__init__(model,data,recovery=True)
        self.certificates = CertificateQuery(model,data,support)

    def measure(self, g1, g2, limit=.05):
        try:
            return self.certificates.measure(g1,g2,limit)
        except MeasurementError:
            return super().measure(g1,g2,limit)
