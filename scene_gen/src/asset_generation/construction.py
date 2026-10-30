"""Versioned geometric helpers. Construction is not certification; all sizes are metres."""
import math
import re
from xml.etree.ElementTree import SubElement
import numpy as np

VERSION = 1
TOOLS = ("hinge", "slide", "container", "bracket")


def vector(value, n=3):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError("构建参数须为有限向量")
    return a


def positive(value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("构建尺寸必须为正数")
    return float(value)


def text(value):
    return " ".join(format(float(x), ".12g") for x in value)


def axis_rotation(axis):
    z = vector(axis)
    if abs(np.linalg.norm(z)-1) > 1e-6:
        raise ValueError("构建轴须为单位向量")
    helper = np.array([1., 0, 0]) if abs(z[0]) < .9 else np.array([0., 1, 0])
    x = np.cross(helper, z); x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def attach(source_position, source_rotation, target_position, target_rotation):
    """Return the rigid transform aligning two local frames, without world-coordinate guesses."""
    s, t = np.asarray(source_rotation, float), np.asarray(target_rotation, float)
    for r in (s, t):
        if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T@r, np.eye(3)) or not np.isclose(np.linalg.det(r), 1):
            raise ValueError("安装基准必须是右手正交坐标系")
    rotation = t@s.T
    return vector(target_position)-rotation@vector(source_position), rotation


class Builder:
    """Small composable MJCF helpers, usable inside the existing build worker."""
    def __init__(self):
        self.records = []
        self.names = set()

    def record(self, kind, name, parameters):
        if kind not in TOOLS or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name) or name in self.names:
            raise ValueError("构建组件名称无效或重复")
        self.names.add(name)
        self.records.append({"tool":kind, "name":name, "version":VERSION,
                             "parameters":parameters, "provenance":"engineering_constructor", "certified":False})

    @staticmethod
    def box(parent, name, position, half_size):
        size = vector(half_size)
        if np.any(size <= 0): raise ValueError("盒体半尺寸必须为正")
        return SubElement(parent, "geom", name=name, type="box", pos=text(vector(position)),
                          size=text(size), density="1000")

    def container(self, parent, name, size, wall):
        size = vector(size); wall = positive(wall)
        if np.any(size <= 2*wall): raise ValueError("壁厚封闭了容器内腔")
        self.record("container", name, {"size":size.tolist(), "wall":wall})
        body = SubElement(parent, "body", name=name)
        x,y,z = size
        self.box(body,name+"_bottom",[0,0,wall/2],[x/2,y/2,wall/2])
        for sign in (-1,1):
            suffix = "p" if sign>0 else "n"
            self.box(body,name+"_x"+suffix,[sign*(x-wall)/2,0,(z+wall)/2],[wall/2,y/2,(z-wall)/2])
            self.box(body,name+"_y"+suffix,[0,sign*(y-wall)/2,(z+wall)/2],[(x-2*wall)/2,wall/2,(z-wall)/2])
        SubElement(body,"site",name=name+"_opening",pos=text([0,0,z]))
        return body

    def bracket(self, parent, name, size, opening):
        """A plate with a genuine rectangular through-hole, normal along local Z."""
        size, opening = vector(size), vector(opening,2)
        if np.any(size <= 0) or np.any(opening <= 0) or np.any(opening >= size[:2]):
            raise ValueError("支架开孔须严格位于板内")
        self.record("bracket",name,{"size":size.tolist(),"opening":opening.tolist()})
        body = SubElement(parent,"body",name=name)
        x,y,z=size; a,b=opening
        for sign in (-1,1):
            suffix="p" if sign>0 else "n"
            self.box(body,name+"_x"+suffix,[sign*(x+a)/4,0,0],[(x-a)/4,y/2,z/2])
            self.box(body,name+"_y"+suffix,[0,sign*(y+b)/4,0],[a/2,(y-b)/4,z/2])
        SubElement(body,"site",name=name+"_mount",pos=text([0,0,z/2]))
        return body

    def hinge(self, parent, name, origin, axis, limits, radius=.004, length=.04):
        origin, axis, limits = vector(origin),vector(axis),vector(limits,2)
        radius, length = positive(radius),positive(length)
        if limits[0] >= limits[1] or not limits[0] <= 0 <= limits[1]: raise ValueError("铰链范围须含初始零位且非空")
        rotation=axis_rotation(axis)
        self.record("hinge",name,{"origin":origin.tolist(),"axis":axis.tolist(),"limits":limits.tolist(),"radius":radius,"length":length})
        SubElement(parent,"geom",name=name+"_pin",type="cylinder",size=text([radius,length/2]),
                   fromto=text([*(origin-axis*length/2),*(origin+axis*length/2)]),density="1000")
        body=SubElement(parent,"body",name=name,pos=text(origin))
        SubElement(body,"joint",name=name+"_joint",type="hinge",axis=text(axis),range=text(limits),damping=".1")
        # Twelve finite-volume sectors leave a real bore around the pin.
        for i in range(12):
            angle=2*math.pi*i/12; radial=np.array([math.cos(angle),math.sin(angle),0])
            tangent=np.array([-math.sin(angle),math.cos(angle),0])
            node=self.box(body,name+"_sleeve_"+str(i),rotation@(radial*radius*1.65),[radius*.25,radius*.5,length*.3])
            node.set("xyaxes",text([*(rotation@radial),*(rotation@tangent)]))
        SubElement(body,"site",name=name+"_axis",pos="0 0 0",zaxis=text(axis))
        return body

    def slide(self, parent, name, origin, axis, limits, rail_length=.2, width=.02, clearance=.001):
        origin, axis, limits=vector(origin),vector(axis),vector(limits,2)
        length,width,clearance=positive(rail_length),positive(width),positive(clearance)
        if limits[0]>=limits[1] or not limits[0]<=0<=limits[1] or max(abs(limits)) + width/2 >= length/2:
            raise ValueError("滑块行程必须在导轨有效长度内")
        rotation=axis_rotation(axis)
        self.record("slide",name,{"origin":origin.tolist(),"axis":axis.tolist(),"limits":limits.tolist(),"rail_length":length,"width":width,"clearance":clearance})
        frame=SubElement(parent,"body",name=name+"_frame",pos=text(origin),xyaxes=text([*rotation[:,0],*rotation[:,1]]))
        for sign in (-1,1):
            self.box(frame,name+"_rail_"+str(sign).replace('-','n'),[sign*(width+clearance),0,0],[width/2,width/2,length/2])
        for sign in (-1,1):
            self.box(frame,name+"_stop_"+str(sign).replace('-','n'),[0,0,sign*(length/2+width/4)],
                     [width*1.5+clearance,width/2,width/4])
        body=SubElement(frame,"body",name=name)
        SubElement(body,"joint",name=name+"_joint",type="slide",axis="0 0 1",range=text(limits),damping="1")
        self.box(body,name+"_carriage",[0,0,0],[width/2,width/2,width/2])
        SubElement(body,"site",name=name+"_mount",pos=text([0,width/2,0]))
        return body
