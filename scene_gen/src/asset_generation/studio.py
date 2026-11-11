"""Presentation renders for people (papers, reports): each asset alone on a seamless backdrop
under three-point lighting, a turntable video that also plays every mechanism, and a lit view
of the full scene.

They never replace check evidence: geometry is loaded unchanged; only lights, a matte floor
that catches shadows and presentation settings are added to a copy of the model.

Image formation, all on the CPU rasterizer:
  * supersampling: each frame is rendered at SUPERSAMPLE x the output size and downsampled;
  * soft shadows: the key light is moved over a small disc and the passes are averaged;
  * seamless backdrop: a segmentation pass finds floor and empty pixels; they are replaced by
    a flat backdrop colour, keeping only the floor's shadow (shadowed / unshadowed brightness)
    as a soft contact shadow, so no horizon, floor texture or pedestal shows.
"""
import math
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

WIDTH, HEIGHT = 1600, 1200
SUPERSAMPLE = 2
BACKDROP = np.array([.965, .968, .972])  # near-white, faintly cool
SHADOW_STRENGTH = .55                     # fraction of the floor's darkening kept as shadow
KEY = "studio_key"


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
        # Oriented AABB of the geom in world frame.
        center, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        world_c = data.geom_xpos[g]+rot@center
        world_h = np.abs(rot)@half
        lo = np.minimum(lo, world_c-world_h); hi = np.maximum(hi, world_c+world_h)
    return lo, hi


def _floor(spec, center, z, size):
    spec.add_material(name="studio_floor", rgba=[1, 1, 1, 1], specular=0, shininess=0, reflectance=0)
    spec.worldbody.add_geom(name="studio_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[size, size, .05],
                            pos=[center[0], center[1], z], material="studio_floor", contype=0, conaffinity=0, group=2)


def _lights(spec, center, radius):
    """Key (shadow-casting, warm), fill (cool, soft), rim (separates the silhouette), top."""
    def add(name, offset, diffuse, shadow, specular=.08):
        pos = np.asarray(center)+np.asarray(offset)*radius
        direction = np.asarray(center)-pos; direction = direction/np.linalg.norm(direction)
        spec.worldbody.add_light(name=name, pos=pos.tolist(), dir=direction.tolist(), type=mujoco.mjtLightType.mjLIGHT_SPOT,
                                 castshadow=shadow, diffuse=list(diffuse), specular=[specular]*3,
                                 cutoff=80, exponent=1, attenuation=[1, 0, 0])
    add(KEY, (2.4, -2.2, 3.6), (.56, .54, .51), True, .12)
    add("studio_fill", (-3.2, -1.6, 1.4), (.20, .21, .23), False)
    add("studio_rim", (-.8, 3.4, 2.4), (.26, .26, .27), False, .1)
    add("studio_top", (0, 0, 4.5), (.08, .08, .08), False, 0)


def _visual(model, width, height):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width*SUPERSAMPLE)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height*SUPERSAMPLE)
    model.vis.quality.shadowsize = 8192
    model.vis.quality.offsamples = 4
    model.vis.quality.numslices = 96
    model.vis.quality.numstacks = 48
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .30
    model.vis.headlight.diffuse[:] = .06
    model.vis.headlight.specular[:] = 0
    model.vis.map.shadowclip = 3.
    model.vis.map.znear = .005
    model.vis.rgba.haze[:] = [*BACKDROP, 1]
    # Glossy plastic highlights read as noise in print; keep materials satin.
    model.mat_specular[:] = np.minimum(model.mat_specular, .25)
    model.mat_shininess[:] = np.minimum(model.mat_shininess, .35)
    model.mat_reflectance[:] = 0


def asset_studio(xml_path, qpos0=None):
    """Compiled copy of the asset on a shadow-catching floor with studio lights; returns (model, data, frame)."""
    base = mujoco.MjModel.from_xml_path(str(xml_path))
    d0 = mujoco.MjData(base)
    if qpos0 is not None and len(qpos0) == base.nq:
        d0.qpos[:] = qpos0
    mujoco.mj_forward(base, d0)
    lo, hi = _bounds(base, d0)
    center = (lo+hi)/2
    size = float(np.max(hi-lo))
    spec = mujoco.MjSpec.from_file(str(xml_path))
    _floor(spec, center, lo[2], 50*size+5)
    radius = float(np.linalg.norm(hi-lo))/2
    _lights(spec, [center[0], center[1], lo[2]+.5*(hi[2]-lo[2])], max(radius, .05))
    model = spec.compile()
    _visual(model, WIDTH, HEIGHT)
    data = mujoco.MjData(model)
    if qpos0 is not None and len(qpos0) == model.nq:
        data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    frame = {"center": center.tolist(), "radius": radius}
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
    """Renders presentation frames: supersampled, soft-shadowed, composited on the backdrop."""

    def __init__(self, model, width=WIDTH, height=HEIGHT, passes=6, backdrop=True, cone=2.):
        self.model, self.width, self.height, self.passes, self.backdrop = model, width, height, passes, backdrop
        self.cone = cone  # degrees; half-angle of the directional key's jitter
        self.renderer = mujoco.Renderer(model, height*SUPERSAMPLE, width*SUPERSAMPLE, max_geom=max(10000, 2*model.ngeom+64))
        self.option = _options()
        self.key = model.light(KEY).id if KEY in [model.light(i).name for i in range(model.nlight)] else None
        floor = [g for g in range(model.ngeom) if model.geom(g).name == "studio_floor"]
        self.floor = floor[0] if floor else None
        # Key-light offsets on a disc (golden-angle spiral) for soft shadow edges.
        self.jitter = [(math.sqrt((k+.5)/passes)*math.cos(2.39996*k), math.sqrt((k+.5)/passes)*math.sin(2.39996*k))
                       for k in range(passes)]

    def _render(self, data, cam, offset=None, shadow=True):
        self.renderer.update_scene(data, cam, self.option)
        if offset is not None and self.key is not None:
            # The scene's light list may start with the headlight; find the key by position.
            scene = self.renderer.scene
            light = min((scene.lights[i] for i in range(scene.nlight)),
                        key=lambda l: np.linalg.norm(np.array(l.pos)-data.light_xpos[self.key]))
            pos = np.array(light.pos)
            if light.type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                # Tilt the direction inside a small cone, pivoting about the look-at depth so the
                # shadow map stays centred on the scene.
                d = np.array(light.dir)
                u = np.cross(d, [0, 0, 1.]); u = u/np.linalg.norm(u); v = np.cross(d, u)
                tilted = d+math.tan(math.radians(self.cone))*(offset[0]*u+offset[1]*v)
                tilted = tilted/np.linalg.norm(tilted)
                depth = float(np.dot(np.array(cam.lookat)-pos, d))
                light.dir[:] = tilted
                light.pos[:] = pos+depth*(d-tilted)
            else:
                span = .08*np.linalg.norm(pos-np.array(cam.lookat))
                light.pos[:] = pos+span*np.array([offset[0], offset[1], 0.])
        flags = self.renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        return self.renderer.render()

    def __call__(self, data, cam):
        image = np.mean([self._render(data, cam, o).astype(np.float32) for o in self.jitter], axis=0)/255
        if self.backdrop:
            self.renderer.enable_segmentation_rendering()
            try:
                self.renderer.update_scene(data, cam, self.option)
                seg = self.renderer.render()
            finally:
                self.renderer.disable_segmentation_rendering()
            empty = seg[..., 0] < 0
            floor = (seg[..., 0] == self.floor) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) if self.floor is not None else np.zeros_like(empty)
            # Shadow alone = shadowed / unshadowed floor brightness; this removes the light's
            # own falloff across the floor, which would otherwise read as a grey gradient.
            shade = np.ones(image.shape[:2])
            if floor.any():
                weights = np.array([.2126, .7152, .0722])
                unshadowed = (self._render(data, cam, shadow=False).astype(np.float32)/255)@weights
                shade = np.where(unshadowed > .08, np.clip((image@weights)/np.maximum(unshadowed, 1e-3), 0, 1), 1)
            ground = BACKDROP*(1-SHADOW_STRENGTH*(1-shade))[..., None]
            image = np.where((floor | empty)[..., None], np.where(floor[..., None], ground, BACKDROP), image)
        frame = Image.fromarray((np.clip(image, 0, 1)*255+.5).astype(np.uint8))
        return np.asarray(frame.resize((self.width, self.height), Image.LANCZOS))

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


PLAN_SAMPLES = 8
PLAN_TOL = .001


def _pose(model, home, opened):
    qpos = home.copy()
    for ranges in opened:
        qpos = far_pose(model, qpos, ranges)
    return qpos


def mechanism_plan(model, data, items, tol=PLAN_TOL):
    """Penetration-free animation plan for [(id, ranges)]: moves (id, +1 out to the far limit or
    -1 back home) that move every mechanism once and end with all of them home, plus, per
    mechanism, the ones held at their far limit while it moves.

    Mechanisms go in manifest order, but each waits until its sweep from the current pose adds no
    penetration deeper than `tol`; every visible geom counts, so a door drawn without collision
    geometry still blocks the drawer behind it. A lower lid thus waits for the upper one. If
    nothing can move, mechanisms already out go back home, last first, until one can (racks
    slide back in before their door swings shut). One that collides whatever the pose moves in
    its manifest place; its own checks report it. Everything out returns home last-first."""
    from asset_generation.motion import _subtree, _welded_followers, _follow, _contacts
    home = data.qpos.copy()
    followers = _welded_followers(model, data)
    # Collision filters are also kept per body (the union of its geoms'), so both are opened.
    filters = (model.geom_contype, model.geom_conaffinity, model.body_contype, model.body_conaffinity)
    saved = [f.copy() for f in filters]
    visible = ((model.geom_rgba[:, 3] > 0) & (model.geom_group < 3)
               & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)).astype(model.geom_contype.dtype)
    model.geom_contype[:] = visible; model.geom_conaffinity[:] = visible
    model.body_contype[:] = 0
    model.body_contype[model.geom_bodyid[visible > 0]] = 1
    model.body_conaffinity[:] = model.body_contype
    ranges = dict(items)
    moving = {i: _subtree(model, {int(model.jnt_bodyid[j]) for j, _, _ in r}) for i, r in items}

    def clean(i, out):
        start = _pose(model, home, [ranges[k] for k in out])
        data.qpos[:] = start; _follow(model, data, followers)
        base = _contacts(model, data, tol)
        for t in np.linspace(0, 1, PLAN_SAMPLES+1)[1:]:
            data.qpos[:] = far_pose(model, start, ranges[i], t); _follow(model, data, followers)
            for (a, b), depth in _contacts(model, data, tol).items():
                if (int(model.geom_bodyid[a]) in moving[i]) != (int(model.geom_bodyid[b]) in moving[i]) \
                        and depth < base.get((a, b), 0.)-tol:
                    return False
        return True

    moves, held, out = [], {}, []
    pending = [i for i, _ in items]
    try:
        while pending:
            back, trial = [], list(out)
            i = next((i for i in pending if clean(i, trial)), None)
            while i is None and trial:
                back.append(trial.pop())
                i = next((i for i in pending if clean(i, trial)), None)
            if i is None:
                back, trial, i = [], list(out), pending[0]
            moves += [(k, -1) for k in back]
            out = trial
            held[i] = list(out)
            moves.append((i, 1)); out.append(i); pending.remove(i)
        moves += [(k, -1) for k in reversed(out)]
    finally:
        for f, old in zip(filters, saved):
            f[:] = old
        data.qpos[:] = home; mujoco.mj_forward(model, data)
    return moves, held


def mechanism_schedule(model, mechanisms, max_slots=12, data=None):
    """The animation plan of `mechanism_plan` in at most `max_slots` slots of consecutive moves in
    the same direction. Each slot is (ids, [(joint, lo, hi)], direction, held ranges): direction
    +1 moves the slot's mechanisms out, -1 back home; the held mechanisms stay at their far limit
    meanwhile (for an outward slot, what had to open first)."""
    from asset_generation.motion import _ranges
    items = [(m["id"], _ranges(model, m)) for m in mechanisms]
    items = [(i, r) for i, r in items if r]
    if not items:
        return []
    if data is None:
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    moves, held = mechanism_plan(model, data, items)
    ranges = dict(items)
    per = math.ceil(len(moves)/max_slots)
    groups = []
    for i, direction in moves:
        if groups and groups[-1][1] == direction and len(groups[-1][0]) < per:
            groups[-1][0].append(i)
        else:
            groups.append(([i], direction))
    slots = []
    for ids, direction in groups:
        kept = [k for k in held[ids[0]] if k not in ids] if direction > 0 else []
        slots.append((ids, [x for i in ids for x in ranges[i]], direction, [ranges[k] for k in kept]))
    return slots


def slot_pose(model, home, slots, upto, amount=1.):
    """qpos after the first `upto` slots, with slot `upto` (if any) `amount` of the way through."""
    level = {}
    for k, (_, ranges, direction, _) in enumerate(slots[:upto+1]):
        a = 1. if k < upto else amount
        for r in ranges:
            level[r] = a if direction > 0 else 1.-a
    qpos = home.copy()
    for r, a in level.items():
        if a > 0:
            qpos = far_pose(model, qpos, [r], a)
    return qpos


def far_pose(model, home, ranges, amount=1.):
    """qpos with each joint moved `amount` of the way from home to its farther limit."""
    qpos = home.copy()
    for j, a, b in ranges:
        start = float(home[model.jnt_qposadr[j]])
        target = b if abs(b-start) >= abs(a-start) else a
        qpos[model.jnt_qposadr[j]] = start+amount*(target-start)
    return qpos


def turntable(xml_path, mechanisms, output, qpos0=None, seconds=12., fps=30, elevation=-20,
              width=1280, height=960, passes=3):
    """One full turn of the camera while the mechanisms play their planned moves (see
    `mechanism_plan`): out one after another, held out where another needs them, back home."""
    import imageio.v2 as imageio
    from asset_generation.motion import _welded_followers, _follow
    model, data, frame = asset_studio(xml_path, qpos0)
    home = data.qpos.copy()
    # Frame must contain the mechanisms' swept poses too (an opened lid reaches higher).
    slots = mechanism_schedule(model, mechanisms, data=data)
    followers = _welded_followers(model, data)
    lo, hi = np.array(frame["center"])-frame["radius"], np.array(frame["center"])+frame["radius"]
    skip = {model.geom("studio_floor").id}
    for k in range(len(slots)):
        data.qpos[:] = slot_pose(model, home, slots, k)
        _follow(model, data, followers)
        blo, bhi = _bounds(model, data, skip)
        lo, hi = np.minimum(lo, blo), np.maximum(hi, bhi)
    frame = {**frame, "center": ((lo+hi)/2).tolist(), "radius": float(np.linalg.norm(hi-lo)/2)}
    shoot = Shooter(model, width, height, passes=passes)
    frames = int(seconds*fps)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=9, macro_block_size=8,
                                pixelformat="yuv420p")
    try:
        for f in range(frames):
            t = f/frames
            # Every slot gets an equal share of the clip, eased in and out.
            k, local = divmod(t*len(slots), 1.) if slots else (0, 0.)
            data.qpos[:] = slot_pose(model, home, slots, int(k), .5-.5*math.cos(math.pi*local))
            _follow(model, data, followers)
            writer.append_data(shoot(data, camera(frame, 30+360*t, elevation)))
    finally:
        writer.close()
        shoot.close()
    return {"video": str(output), "frames": frames, "fps": fps, "slots": [ids if d > 0 else [f"{i} (back)" for i in ids] for ids, _, d, _ in slots]}


def scene_lit(xml_path, home_qpos, output, views, width=1920, height=1080, passes=8, elevation=55.):
    """The assembled scene: a shadow-casting directional key (jittered for soft shadows), a soft
    opposite fill, raised ambient light, and the empty surroundings replaced by the backdrop colour."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for light in list(spec.lights):
        light.diffuse = [.12]*3
        light.castshadow = False
    spec.add_material(name="studio_floor", rgba=[.88, .885, .89, 1], specular=0, shininess=0, reflectance=0)
    floor = spec.geom("environment__floor")
    if floor is not None:
        floor.material = "studio_floor"; floor.rgba = [1, 1, 1, 1]
    e = math.radians(elevation)
    heading = np.array([-.45, .35])/np.linalg.norm([-.45, .35])
    lights = ((KEY, (*(math.cos(e)*heading), -math.sin(e)), (.52, .50, .47), True), ("studio_fill", (.5, -.3, -.6), (.16, .17, .19), False))
    for name, _, diffuse, shadow in lights:
        spec.worldbody.add_light(name=name, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=shadow,
                                 diffuse=list(diffuse), specular=[.05]*3)
    model = spec.compile()
    _visual(model, width, height)
    model.vis.headlight.ambient[:] = .36
    # Coplanar faces of touching primitives z-fight in the additive shadowed-light pass and read
    # as speckled acne; a per-geom inflation of .03-.3 mm separates them invisibly.
    prim = ~np.isin(model.geom_type, [mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD])
    model.geom_size[prim] += 3e-5*(1+9*(np.arange(model.ngeom)*.618034 % 1))[prim, None]
    data = mujoco.MjData(model); data.qpos[:] = home_qpos; mujoco.mj_forward(model, data)
    room_lo, room_hi = _bounds(model, data)
    centre, room = (room_lo+room_hi)/2, float(np.linalg.norm(room_hi-room_lo))/2
    reach = 1.5*room
    for name, direction, _, _ in lights:
        d = np.array(direction)/np.linalg.norm(direction)
        model.light(name).dir[:] = d; model.light(name).pos[:] = centre-reach*d
    mujoco.mj_forward(model, data)
    # Orthographic shadow box just covering the room. Its depth range is the camera's: a far
    # plane close behind the room (~2*reach) leaves too little depth bias on distant floor and
    # shows moire acne in this GL path, so keep it far out.
    extent = model.stat.extent
    model.vis.map.shadowclip = 1.05*room/extent
    model.vis.map.znear = .05/extent
    model.vis.map.zfar = 1000./extent
    skip = {g for g in range(model.ngeom) if (model.geom(g).name or "").startswith("environment__")}
    lo, hi = _bounds(model, data, skip)
    frame = {"center": ((lo+hi)/2).tolist(), "radius": float(np.linalg.norm(hi-lo)/2)}
    shoot = Shooter(model, width, height, passes=passes)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for name, (az, el, zoom) in views.items():
            path = out/f"{name}.png"
            Image.fromarray(shoot(data, camera(frame, az, el, margin=zoom))).save(path)
            paths.append(path)
    finally:
        shoot.close()
    return paths


def assembled_views(xml_path, home_qpos, instances, output, azimuths=(35, 215), elevation=-22):
    """One job's instances as assembled in the scene, everything else hidden, under the studio
    lights of the stills. Parts that meet another configuration of the same asset (a cooler's
    hoses on its host's coolant ports) show connected here, while the per-configuration stills
    show them loose."""
    prefixes = tuple(f"{i}__" for i in instances)

    def hide(model):
        for g in range(model.ngeom):
            if not model.body(int(model.geom_bodyid[g])).name.startswith(prefixes):
                model.geom_group[g] = 4  # not drawn (MjvOption shows groups 0-2), skipped by _bounds
    base = mujoco.MjModel.from_xml_path(str(xml_path)); hide(base)
    d0 = mujoco.MjData(base); d0.qpos[:] = home_qpos; mujoco.mj_forward(base, d0)
    lo, hi = _bounds(base, d0)
    center, radius = (lo+hi)/2, float(np.linalg.norm(hi-lo))/2
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for light in list(spec.lights):
        light.active = False
    _floor(spec, center, lo[2], 50*radius+5)
    _lights(spec, [center[0], center[1], lo[2]+.5*(hi[2]-lo[2])], max(radius, .05))
    model = spec.compile(); hide(model)
    model.geom_group[model.geom("studio_floor").id] = 2
    _visual(model, WIDTH, HEIGHT)
    # The scene's extent is the room; scale the shadow box and near plane to this asset.
    model.vis.map.shadowclip = 3*radius/model.stat.extent
    model.vis.map.znear = .005*radius/model.stat.extent
    data = mujoco.MjData(model); data.qpos[:] = home_qpos; mujoco.mj_forward(model, data)
    frame = {"center": center.tolist(), "radius": radius}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    shoot = Shooter(model)
    paths = []
    try:
        for az in azimuths:
            path = output/f"assembled_{az:03d}.png"
            Image.fromarray(shoot(data, camera(frame, az, elevation))).save(path)
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
    slots = mechanism_schedule(model, mechanisms, max_slots=2*limit, data=data)
    shoot = Shooter(model, 960, 720, passes=4)
    out = {}
    try:
        for ids, ranges, direction, held in slots:
            if direction < 0:
                continue
            name = "+".join(ids)[:60]
            paths = []
            for label, amount in (("home", 0.), ("far", 1.)):
                # Prerequisites stay open in both stills (the drawer is shown behind its open door).
                data.qpos[:] = far_pose(model, _pose(model, home, held), ranges, amount)
                _follow(model, data, followers)
                path = output/f"mech_{name}_{label}.png"
                Image.fromarray(shoot(data, camera(frame, azimuth, elevation, margin=.9))).save(path)
                paths.append(path)
            out[name] = paths
    finally:
        shoot.close()
    return out
