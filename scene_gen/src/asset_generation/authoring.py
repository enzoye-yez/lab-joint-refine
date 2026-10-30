"""Compact, backend-derived authoring limits; no credential inspection."""
def capabilities(generation, backend):
    return {"packages":backend.get("packages",{}),"backend":backend.get("backend"),
            "max_artifact_files":generation.max_artifact_files,"max_artifact_bytes":generation.max_artifact_bytes,
            "max_script_bytes":generation.max_script_bytes,
            "construction":{"module":"asset_generation.construction", "version":1,
                "helpers":["Builder.hinge","Builder.slide","Builder.container","Builder.bracket","attach"],
                "provenance":"engineering_constructor", "certification":False,
                "signatures":["container(parent,name,size,wall)","bracket(parent,name,size,opening)",
                    "hinge(parent,name,origin,axis,limits,radius=.004,length=.04)",
                    "slide(parent,name,origin,axis,limits,rail_length=.2,width=.02,clearance=.001)",
                    "attach(source_position,source_rotation,target_position,target_rotation) -> position,rotation"],
                "units":"metres and radians; compiler angle=radian; emit records as manifest.construction_records",
                "slide_rule":"max(abs(limits))+width/2 < rail_length/2; rails and end stops derive from same dimensions"},
            "guidance":["同一网格和纹理复用资源文件；不要为每个重复零件输出一个副本。",
                        "碰撞网格须具有有效体积；薄面可用有真实厚度的几何表达，不用零体积网格。",
                        "活动部件须有有效质量和惯量；保留真实孔槽、接口和承托结构。",
                        "缺少可选依赖时改用已声明能力，不联网安装；完整原图仍用于视觉核查。"]}
