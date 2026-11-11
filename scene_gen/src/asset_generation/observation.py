"""Targeted read-only observation with measured visibility, not a geometry verdict."""
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from .contracts import require
from scene_generation.geometry import bounds


def _segmentation_mask(renderer, data, camera, option, geoms):
    """Segmentation is exact but not always available; returns None when unusable."""
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera, scene_option=option)
    segmentation = renderer.render()
    return np.isin(segmentation[:, :, 0], geoms) & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))


def _depth_mask(renderer, data, camera, option, model, geoms, hidden_groups):
    """Fallback without segmentation ids: a target pixel is visible when the full render
    reaches it at the same depth as a target-only render.

    Segmentation maps scene-local geom slots back to global ids, so hiding geometry
    (which shrinks the scene) can overflow that table. Depth comparison depends only on
    the rendered distance and stays valid for any number of hidden parts.
    """
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera, scene_option=option)
    full = renderer.render().copy()
    isolated_option = mujoco.MjvOption()
    isolated_option.geomgroup[:] = 0
    isolated_option.sitegroup[:] = 0
    for g in geoms:
        isolated_option.geomgroup[0] = 1
    for g in range(model.ngeom):
        if g not in geoms:
            model.geom_group[g] = 5
    try:
        renderer.update_scene(data, camera, scene_option=option)
        isolated = renderer.render().copy()
    finally:
        model.geom_group[:] = hidden_groups
    renderer.disable_depth_rendering()
    finite = np.isfinite(isolated) & (isolated > 0)
    visible_mask = finite & np.isfinite(full) & (np.abs(full - isolated) <= 1e-4)
    return visible_mask, int(finite.sum())

def _target_visibility(renderer, data, camera, option, model, geoms, groups):
    """Return (mask, visible, expected) without letting a renderer id bug void evidence.

    Segmentation is exact but maps scene-local geom slots to global ids; hiding geometry
    shrinks the scene and can make MuJoCo index past the end of that table. Depth
    agreement needs no id table, so it is used whenever segmentation cannot complete.
    """
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data,camera,scene_option=option)
        segmentation=renderer.render()
        mask=np.isin(segmentation[:,:,0],geoms) & (segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        visible=int(mask.sum())
        try:
            for g in range(model.ngeom):
                if g not in geoms: model.geom_group[g]=5
            renderer.update_scene(data,camera,scene_option=option)
            isolated=renderer.render()
            expected=int((np.isin(isolated[:,:,0],geoms) & (isolated[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))).sum())
        finally:
            model.geom_group[:]=groups
        renderer.disable_segmentation_rendering()
        return mask,visible,expected
    except IndexError:
        model.geom_group[:]=groups
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        try:
            renderer.update_scene(data,camera,scene_option=option)
            full=renderer.render().copy()
            for g in range(model.ngeom):
                if g not in geoms: model.geom_group[g]=5
            renderer.update_scene(data,camera,scene_option=option)
            isolated=renderer.render().copy()
        finally:
            model.geom_group[:]=groups
            renderer.disable_depth_rendering()
        # A target pixel is unoccluded when the full render meets it at the depth the
        # target-only render reaches. Background returns the far plane, never the target.
        far=float(model.vis.map.zfar*model.stat.extent)
        reaches=np.isfinite(isolated)&(isolated>0)&(isolated<far*(1-1e-4))
        mask=reaches&np.isfinite(full)&(np.abs(full-isolated)<=1e-4)
        return mask,int(mask.sum()),int(reaches.sum())

def render_observation(model, data, request, output, number, descendants):
    hidden=set()
    for name in request["hide_bodies"]:
        hidden |= descendants(model,name)
    focus=descendants(model,request["focus_body"]) if request["focus_body"] else set(range(1,model.nbody))
    if request.get("focus_geoms"):
        geoms=[model.geom(name).id for name in request["focus_geoms"]]
        require(not any(int(model.geom_bodyid[g]) in hidden for g in geoms),"target geometry is hidden")
    else:
        geoms=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in focus-hidden
               and model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE and model.geom_rgba[g,3]>0 and model.geom_group[g]<3]
    require(geoms,"view has no visible target")
    boxes=[bounds(model,data,g) for g in geoms]
    lo,hi=np.min([b[0] for b in boxes],axis=0),np.max([b[1] for b in boxes],axis=0)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    camera.lookat[:]=(lo+hi)/2
    camera.distance=max(.003,float(np.linalg.norm(hi-lo))*1.8)/request.get("zoom",1)
    camera.azimuth,camera.elevation=request["azimuth"],request["elevation"]
    groups=model.geom_group.copy();rgba=model.geom_rgba.copy()
    light=(model.vis.headlight.active,model.vis.headlight.ambient.copy(),model.vis.headlight.diffuse.copy())
    try:
        for g in geoms:model.geom_group[g]=0
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in hidden: model.geom_group[g]=5
        option=mujoco.MjvOption();option.geomgroup[5]=0
        if request.get("hide_sites",True): option.sitegroup[:]=0
        model.vis.headlight.active=1
        model.vis.headlight.ambient[:]=[.5]*3;model.vis.headlight.diffuse[:]=[.6]*3
        if request.get("diagnostic_light",True):
            model.geom_rgba[:,:3]=np.maximum(model.geom_rgba[:,:3],.25)
        model.vis.global_.offwidth=max(640,model.vis.global_.offwidth)
        model.vis.global_.offheight=max(480,model.vis.global_.offheight)
        from .rendering import renderer as make_renderer
        with make_renderer(model,width=640,height=480) as renderer:
            renderer.update_scene(data,camera,scene_option=option)
            pixels=renderer.render().copy()
            camera_position=np.array(renderer.scene.camera[0].pos,dtype=float)
            from .distance import DistanceQuery
            query=DistanceQuery(model,data)
            inside=[]
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) in hidden or rgba[g,3]<=0 or groups[g]>=3:continue
                radius=query.interior_radius(g,camera_position)
                if radius is not None and radius>1e-6:inside.append(model.geom(g).name)
            working=model.geom_group.copy()
            mask,visible,expected=_target_visibility(renderer,data,camera,option,model,geoms,working)
        brightness=float(np.percentile(pixels[mask],90)) if visible else 0.
        fraction=visible/max(expected,1)
        observable=visible>=16 and fraction>=.15 and brightness>=15 and not inside
        name=f"inspection_{number}.png"
        (Path(output)/"renders").mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(Path(output)/"renders"/name)
        quality={"target_pixels":visible,"target_only_pixels":expected,"visible_fraction":fraction,
                 "brightness_p90":brightness,"camera_inside_geoms":inside,"observable":observable}
        result={"kind":"view","request":request,"image":name,"quality":quality,
                "targets":[model.geom(g).name for g in geoms],
                "camera":{"lookat":camera.lookat.tolist(),"distance":float(camera.distance),
                          "azimuth":float(camera.azimuth),"elevation":float(camera.elevation)},
                "meaning":"Diagnostic static view; hidden parts are not removed. Lighting/material brightness may be normalized. Visibility does not prove physical fit."}
        if not observable: result.update(error_kind="observation_incomplete",error="target is obscured, too small or too dark")
        return result
    finally:
        model.geom_group[:]=groups;model.geom_rgba[:]=rgba
        model.vis.headlight.active=light[0];model.vis.headlight.ambient[:]=light[1];model.vis.headlight.diffuse[:]=light[2]
