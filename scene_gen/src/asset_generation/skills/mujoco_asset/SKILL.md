---
name: mujoco-asset
description: Recipes for MuJoCo lab assets that pass the framework's mechanism, button, connection, material and stability checks on the first draft.
---

# MuJoCo asset recipes

Distilled from the defects that most often sent drafts back for repair. The contract in
`instructions.md` stays authoritative; this file only shows ways to meet it.

## Work loop
1. Read `context.json` (`job`, `fixed_contract`, `instance_contract`, `feedback`, `repair_tasks`) and the
   reference images. List every motion/interaction requirement before writing geometry.
2. Write the static body first, run `./check`, then add one mechanism family at a time
   (hinges, slides, buttons, connections) and run `./check` after each.
3. Fix `fail` rows by id, reading their numbers (`peak_position`, `rest_position`, `depth_mm`,
   `blocking_contacts`, `repair_hint`). `partial` rows are measurement gaps, not defects.
4. Stop when `./check` has no `fail`; a slow check (see `seconds` against
   `framework_budget_seconds`) is rejected by the framework too.

## Frame and initial state
- Origin at the centre of the footprint on the support plane (z=0 is the bottom).
- The operator works from local -Y: controls, display, door and loading slot on the -Y face.
- The default pose (all joints 0) is the protocol's starting state (closed, folded, loaded).
  Scene assembly never moves anything.

## Numerical stability (checked first; a diverging model skips every dynamic check)
- No moving body under 5 g; give tiny parts an explicit `inertial` (mass and diaginertia
  >= 1e-7 kg m^2) instead of relying on tiny geoms.
- Add `armature="0.0005"` to the joints of light links and `damping` to every hinge/slide
  (0.01-0.05 for lids and drawers, 0.5-2 for buttons).
- Never chain welds between free bodies to fake rigidity; build one body with several geoms.
- Free parts start resting on their support with 0.2-0.5 mm clearance, never interpenetrating.

## Material integrity
- Every visible geom of a rigid body overlaps or touches another geom of that body
  (0.5-1 mm overlap for primitives); no floating decoration.
- Collision geoms stay inside the visible material (`collision_envelope` fails when a
  collider sticks out); name every collision geom.
- Jointed parts touch or are enclosed by their guide/pin: a hinge leaf overlaps its knuckle,
  a drawer sits inside its rails.

## Push buttons (most frequent failure: "did not return" and "event not triggered")
```xml
<body name="btn_start" pos="0 -0.0995 0.06">          <!-- in a hole of the -Y panel -->
  <joint name="btn_start_slide" type="slide" axis="0 1 0" range="0 0.003"
         stiffness="400" springref="-0.002" damping="1" armature="0.0005"/>
  <geom name="btn_start_cap" type="cylinder" size="0.006 0.003" euler="1.5708 0 0"
        mass="0.004" friction="0.3"/>
</body>
```
- Press direction = +joint axis, pointing into the panel; the cap's front face is the first
  thing a presser meets (nothing of the housing in front of it).
- Clearance: the hole in the panel is 0.5 mm wider than the cap in radius, so the cap never
  rubs the housing (contact there holds the button pressed).
- Preload: `springref` outside the travel, opposite the press, with
  `stiffness * |springref| >= 2 * button mass * 9.81`; the range lower limit 0 stops it.
- `pressed_at` <= 0.8 * travel, `released_at` around 0.15 * travel, both inside the range.
- Travel ends before the cap's back face reaches the panel interior (no pressing into solid).

## Detachable connections (caps, lids, tubes in racks)
- A weld or connect equality between the two bodies, active at home; the removable part is a
  free body. Declare it under `connections` with the event that releases it.
- Under a 3x-weight pull the active connection holds within 2 mm; after release the part must
  move freely >= 5 mm: leave 0.3-0.5 mm radial clearance between a cap and its bottle neck
  and do not let threads or rims interpenetrate (model threads as a smooth sleeve).
- The release event must be reachable: a button that really fires, a declared software
  interface, or a mechanical event on a real interface with its trigger joint.

## Hinged lids and doors
- Hinge axis on the real hinge line, `range` from closed (0) to fully open, a stop geom or the
  range limit at the end; the lid's closed pose rests on its seat with 0.2 mm clearance.
- Sweep check: nothing of the lid passes through the body anywhere in its travel; keep 2 mm
  gap at the hinge side.
- Locked lids: declare `lock` and a `release_event` that drives a real catch joint past `at`.

## Drawers, trays and pull-out parts
- Rails first: rail length along the axis >= travel / 0.85; the drawer keeps >= 30% of its
  length inside the rails at full travel; declare `extends=true`.
- Use `Builder.slide(...)` from `asset_generation.construction` when it fits.

## Hoses, tubes and cables between instances
- Compute the mating port in this model's frame from `scene_layout` (world = position +
  Rz(yaw) @ local). Run the last segment coaxially with the port axis and slide the hose bore
  over a barb whose outer radius is smaller than the bore, or stop 0.5 mm short of a solid face.

## Repeated parts
- One model is one asset unit unless `instance_contract.expected_model_units` says otherwise;
  the scene copies instances. Reuse mesh and texture files for identical parts.
