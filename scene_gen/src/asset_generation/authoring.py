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
            "guidance":["Reuse resource files for identical meshes and textures; do not emit a copy for every repeated part.",
                        "Collision meshes must have valid volume; express thin faces with geometry of real thickness, never zero-volume meshes.",
                        "Moving parts must have valid mass and inertia; keep real holes, slots, interfaces and supporting structures.",
                        "When an optional dependency is missing, use a declared capability instead and do not install from the network; the full original images are still used for visual verification."]}
