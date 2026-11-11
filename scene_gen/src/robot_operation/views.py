"""Bounded read-only scene views, kept outside the immutable scene tree."""
from pathlib import Path
import mujoco
from PIL import Image
from asset_prep.storage import file_hash,atomic_json,safe_path
from .common import read,need
from .scene import load


def render_current(scene,sha,directory):
    directory=Path(directory);root=directory.parent.parent
    if (directory/"receipt.json").exists():
        receipt=read(directory/"receipt.json")
        need(receipt["scene_sha256"]==sha,"Preview belongs to a different scene version")
        for row in receipt["images"]:need(file_hash(safe_path(root,row["path"]))==row["sha256"],"Scene preview evidence was modified")
        return receipt["images"],receipt["error"]
    directory.mkdir(parents=True,exist_ok=True);rows=[];error=None;renderer=None
    try:
        model,data,meta,manifest=load(scene,sha)
        renderer=mujoco.Renderer(model,height=240,width=320)
        for i,(azimuth,elevation) in enumerate(((120,-35),(30,-35),(90,-85))):
            camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
            camera.lookat[:]=model.stat.center;camera.distance=max(1,float(model.stat.extent)*1.3)
            camera.azimuth=azimuth;camera.elevation=elevation
            renderer.update_scene(data,camera=camera)
            path=directory/(str(i)+".png");Image.fromarray(renderer.render()).save(path)
            rows.append({"path":path.relative_to(root).as_posix(),"sha256":file_hash(path)})
    except Exception as exc:error="Scene preview was not fully generated: "+type(exc).__name__+": "+str(exc)[:500]
    finally:
        if renderer is not None:renderer.close()
    atomic_json(directory/"receipt.json",{"scene_sha256":sha,"images":rows,"error":error})
    return rows,error


def show_joints(scene,sha):
    """Diagnostic joint sliders only: no physics step, no execution or success receipt."""
    import time
    import mujoco.viewer
    model,data,_,_=load(scene,sha)
    with mujoco.viewer.launch_passive(model,data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                data.qvel[:]=0;data.time=0
                mujoco.mj_forward(model,data)
            viewer.sync();time.sleep(1/30)
