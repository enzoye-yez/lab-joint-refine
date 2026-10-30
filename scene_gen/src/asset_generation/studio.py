"""Presentation renders: each asset alone on a small pedestal under studio lighting, a
turntable video that also plays every mechanism, and a lit version of the full scene.

These images are for people (papers, reports) and for the visual reviewer. They never
replace check evidence: geometry is loaded unchanged and only lights, a pedestal, a floor
and a backdrop are added to a copy of the model.
"""
import math
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

WIDTH, HEIGHT = 1600, 1200


def _bounds(model, data, skip=()):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if g in skip or model.geom_rgba[g, 3] == 0 or model.geom_group[g] >= 3:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        r = model.geom_rbound[g]
        if not np.isfinite(r) or r <= 0:
            continue
        c = data.geom_xpos[g]
        # Oriented AABB of the geom in world frame.
        local = model.geom_aabb[g]
        center, half = local[:3], local[3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        world_c = c+rot@center
        world_h = np.abs(rot)@half
        lo = np.minimum(lo, world_c-world_h); hi = np.maximum(hi, world_c+world_h)
    return lo, hi


def _studio_assets(spec, floor_rgb=(.93, .935, .94)):
    spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[.96, .965, .97], rgb2=[.90, .905, .91],
                     width=512, height=3072)
    spec.add_texture(name="studio_floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=list(floor_rgb),
                     rgb2=[c-.015 for c in floor_rgb], width=512, height=512)
    # Emission keeps the seamless floor evenly bright beyond the spot cones; shadows stay soft.
    spec.add_material(name="studio_floor", textures=["", "studio_floor_tex"], texrepeat=[40, 40],
                      reflectance=.0, specular=0, shininess=0, emission=.62)
    spec.add_material(name="studio_pedestal", rgba=[.95, .95, .96, 1], specular=.1, shininess=.2, emission=.25)


def _lights(spec, center, radius, scale=1.):
    def add(name, offset, diffuse, shadow, specular=.1):
        pos = np.asarray(center)+np.asarray(offset)*radius*scale
        direction = (np.asarray(center)-pos); direction = direction/np.linalg.norm(direction)
        spec.worldbody.add_light(name=name, pos=pos.tolist(), dir=direction.tolist(), type=mujoco.mjtLightType.mjLIGHT_SPOT,
                                 castshadow=shadow, diffuse=[diffuse]*3, specular=[specular]*3,
                                 cutoff=60, exponent=1, attenuation=[1, 0, 0])
    add("studio_key", (2.2, -2.6, 3.4), .48, True, .2)
    add("studio_fill", (-3.0, -1.2, 1.6), .22, False)
    add("studio_rim", (-.6, 3.2, 2.6), .30, False, .15)
    add("studio_top", (0, 0, 4.0), .10, False)


def _visual(model, width, height):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.quality.shadowsize = 4096
    model.vis.quality.offsamples = 8
    model.vis.quality.numslices = 96
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .34
    model.vis.headlight.diffuse[:] = .04
    model.vis.headlight.specular[:] = 0
    model.vis.map.shadowclip = 1.5
    model.vis.map.znear = .005
    model.vis.rgba.haze[:] = [.93, .935, .94, 1]


def asset_studio(xml_path, qpos0=None):
    """Compiled copy of the asset on a pedestal with studio lights; returns (model, data, frame)."""
    base = mujoco.MjModel.from_xml_path(str(xml_path))
    d0 = mujoco.MjData(base)
    if qpos0 is not None and len(qpos0) == base.nq:
        d0.qpos[:] = qpos0
    mujoco.mj_forward(base, d0)
    lo, hi = _bounds(base, d0)
    center = (lo+hi)/2
    size = float(np.max(hi-lo))
    footprint = float(np.linalg.norm((hi-lo)[:2]))/2
    spec = mujoco.MjSpec.from_file(str(xml_path))
    _studio_assets(spec)
    pedestal_h = max(.02, .06*size)
    r = footprint*1.12+.01
    spec.worldbody.add_geom(name="studio_pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[r, pedestal_h/2, 0], pos=[center[0], center[1], -pedestal_h/2],
                            material="studio_pedestal", contype=0, conaffinity=0, group=2)
    spec.worldbody.add_geom(name="studio_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[60*r+10*size, 60*r+10*size, .05],
                            pos=[center[0], center[1], -pedestal_h], material="studio_floor",
                            contype=0, conaffinity=0, group=2)
    radius = max(size, r*2)/2
    _lights(spec, [center[0], center[1], lo[2]+.5*(hi[2]-lo[2])], radius)
    model = spec.compile()
    _visual(model, WIDTH, HEIGHT)
    data = mujoco.MjData(model)
    if qpos0 is not None and len(qpos0) == model.nq:
        data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    frame = {"center": ((lo+hi)/2).tolist(), "radius": float(np.linalg.norm(hi-lo)/2), "pedestal_m": pedestal_h}
    return model, data, frame


def camera(frame, azimuth, elevation=-22., fovy=45., margin=.9):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = frame["center"]
    cam.distance = frame["radius"]*margin/math.sin(math.radians(fovy)/2)
    cam.azimuth, cam.elevation = azimuth, elevation
    return cam


def _options():
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.sitegroup[g] = 0
    return opt


class Shooter:
    def __init__(self, model, width=WIDTH, height=HEIGHT):
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width, max_geom=max(10000, 2*model.ngeom+64))
        self.option = _options()

    def __call__(self, data, cam):
        self.renderer.update_scene(data, cam, self.option)
        flags = self.renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def still_views(xml_path, output, qpos0=None, azimuths=(35, 125, 215, 305), elevation=-22):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, data, frame = asset_studio(xml_path, qpos0)
    shoot = Shooter(model)
    paths = []
    try:
        for az in azimuths:
            path = output/f"studio_{az:03d}.png"
            Image.fromarray(shoot(data, camera(frame, az, elevation))).save(path)
            paths.append(path)
    finally:
        shoot.close()
    return paths


def _ease(t):
    return .5-.5*math.cos(2*math.pi*t)


def mechanism_schedule(model, mechanisms, max_slots=12):
    """Group mechanisms into at most `max_slots` animation slots, each a list of (joint, lo, hi)."""
    from asset_generation.motion import _ranges
    items = [(m["id"], _ranges(model, m)) for m in mechanisms]
    items = [(i, r) for i, r in items if r]
    if not items:
        return []
    per = math.ceil(len(items)/max_slots)
    slots = []
    for k in range(0, len(items), per):
        group = items[k:k+per]
        slots.append(([i for i, _ in group], [x for _, r in group for x in r]))
    return slots


def turntable(xml_path, mechanisms, output, qpos0=None, seconds=12., fps=30, elevation=-20,
              width=1280, height=960):
    """One full turn of the camera while each mechanism plays low -> high -> low in sequence."""
    import imageio.v2 as imageio
    from asset_generation.motion import _welded_followers, _follow
    model, data, frame = asset_studio(xml_path, qpos0)
    home = data.qpos.copy()
    # Frame must contain the mechanisms' swept poses too (an opened lid reaches higher).
    slots = mechanism_schedule(model, mechanisms)
    followers = _welded_followers(model, data)
    lo, hi = np.array(frame["center"])-frame["radius"], np.array(frame["center"])+frame["radius"]
    for _, ranges in slots:
        for t in (0., 1.):
            data.qpos[:] = home
            for j, a, b in ranges:
                data.qpos[model.jnt_qposadr[j]] = a+t*(b-a)
            _follow(model, data, followers)
            skip = {model.geom("studio_pedestal").id, model.geom("studio_floor").id}
            blo, bhi = _bounds(model, data, skip)
            lo, hi = np.minimum(lo, blo), np.maximum(hi, bhi)
    frame = {**frame, "center": ((lo+hi)/2).tolist(), "radius": float(np.linalg.norm(hi-lo)/2)}
    shoot = Shooter(model, width, height)
    frames = int(seconds*fps)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=8,
                                pixelformat="yuv420p")
    labels = []
    try:
        for f in range(frames):
            t = f/frames
            data.qpos[:] = home
            active = None
            # Mechanisms open one after another and stay open, so inner parts revealed by an
            # earlier one (a lid) remain visible; the last fifth closes everything together.
            if slots:
                opening = .8
                for k, (ids, ranges) in enumerate(slots):
                    if t < opening:
                        local = t/opening*len(slots)-k
                        amount = 0. if local <= 0 else 1. if local >= 1 else .5-.5*math.cos(math.pi*local)
                        if 0 < local < 1:
                            active = ids
                    else:
                        local = (t-opening)/(1-opening)
                        amount = .5+.5*math.cos(math.pi*local)
                    for j, a, b in ranges:
                        start = float(home[model.jnt_qposadr[j]])
                        target = b if abs(b-start) >= abs(a-start) else a
                        data.qpos[model.jnt_qposadr[j]] = start+amount*(target-start)
            _follow(model, data, followers)
            writer.append_data(shoot(data, camera(frame, 30+360*t, elevation)))
            labels.append(active)
    finally:
        writer.close()
        shoot.close()
    return {"video": str(output), "frames": frames, "fps": fps,
            "slots": [ids for ids, _ in slots]}


def scene_lit(xml_path, home_qpos, output, views, width=1920, height=1080):
    """The assembled scene with an overhead light grid, a key light and a pale floor."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for light in list(spec.lights):
        light.diffuse = [.25]*3
    _studio_assets(spec, floor_rgb=(.80, .81, .82))
    floor = spec.geom("environment__floor")
    if floor is not None:
        floor.material = "studio_floor"; floor.rgba = [1, 1, 1, 1]
    base = spec.compile()
    d0 = mujoco.MjData(base); d0.qpos[:] = home_qpos; mujoco.mj_forward(base, d0)
    skip = {g for g in range(base.ngeom) if (base.geom(g).name or "").startswith("environment__")}
    lo, hi = _bounds(base, d0, skip)
    center = (lo+hi)/2; span = hi-lo
    for i, x in enumerate(np.linspace(lo[0]+.15*span[0], hi[0]-.15*span[0], 3)):
        for k, y in enumerate(np.linspace(lo[1]+.2*span[1], hi[1]-.2*span[1], 2)):
            spec.worldbody.add_light(name=f"studio_ceiling_{i}_{k}", pos=[x, y, hi[2]+1.6], dir=[0, 0, -1],
                                     castshadow=(i == 1 and k == 0), diffuse=[.28]*3, specular=[.05]*3,
                                     cutoff=75, exponent=2)
    spec.worldbody.add_light(name="studio_sun", pos=[center[0]+span[0], center[1]-span[1], hi[2]+4],
                             dir=(-np.array([span[0], -span[1], 4])/np.linalg.norm([span[0], -span[1], 4])).tolist(),
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True, diffuse=[.45]*3, specular=[.1]*3)
    model = spec.compile()
    _visual(model, width, height)
    model.vis.map.shadowclip = 2.
    model.vis.quality.shadowsize = 8192
    data = mujoco.MjData(model); data.qpos[:] = home_qpos; mujoco.mj_forward(model, data)
    shoot = Shooter(model, width, height)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    frame = {"center": center.tolist(), "radius": float(np.linalg.norm(span)/2)}
    paths = []
    try:
        for name, (az, el, zoom) in views.items():
            cam = camera(frame, az, el, margin=zoom)
            path = out/f"{name}.png"
            Image.fromarray(shoot(data, cam)).save(path)
            paths.append(path)
    finally:
        shoot.close()
    return paths


def mechanism_stills(xml_path, mechanisms, output, qpos0=None, azimuth=35, elevation=-24, limit=16):
    """Each mechanism at its home pose and at its far limit, same camera, for side-by-side review."""
    from asset_generation.motion import _welded_followers, _follow
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, data, frame = asset_studio(xml_path, qpos0)
    home = data.qpos.copy()
    followers = _welded_followers(model, data)
    slots = mechanism_schedule(model, mechanisms, max_slots=limit)
    shoot = Shooter(model, 960, 720)
    out = {}
    try:
        for ids, ranges in slots:
            name = "+".join(ids)[:60]
            paths = []
            for label, amount in (("home", 0.), ("far", 1.)):
                data.qpos[:] = home
                for j, a, b in ranges:
                    start = float(home[model.jnt_qposadr[j]])
                    target = b if abs(b-start) >= abs(a-start) else a
                    data.qpos[model.jnt_qposadr[j]] = start+amount*(target-start)
                _follow(model, data, followers)
                path = output/f"mech_{name}_{label}.png"
                Image.fromarray(shoot(data, camera(frame, azimuth, elevation, margin=.9))).save(path)
                paths.append(path)
            out[name] = paths
    finally:
        shoot.close()
    return out
