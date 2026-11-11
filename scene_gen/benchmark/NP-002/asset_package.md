# NP-002 | Microbial nucleic acid extraction and sequencing from air particles: Asset Generation Input

## Task scope and initial conditions

Choose the PM10 (particulate matter with an aerodynamic diameter not exceeding 10 µm) membrane filter path, magnetic funnel refiltration, magnetic bead purification, original protocol step-by-step library construction and high-output sequencing path. Only fresh samples are processed, without entering frozen storage, column purification substitution, or failed retry branches. Sequencing is the core endpoint of this task and is retained; additional purity and library quality checks are not included in the action tasks, and their qualified results are used as input status.

**Initial state:** Single-sample path. The filter membrane and contacting equipment are sterilized, reagents are prepared, and equipment calibration and fixed installation are complete. Start with the initial weighing of the wrapped filter membrane. Retain loading the membrane into the sampler, starting the sampler, and removing the membrane, then proceed to the processing station inside the biosafety cabinet.

**Final state:** The library has been loaded into the sequencing instrument, which has completed a normal run; no subsequent calculation and analysis scenarios are generated.

All reagents are prepared in advance. Do not generate chemical recipes or simulate biochemical reactions; retain the liquid additions, transfers, washes, mixing, loading, startup, and unloading required by this path. Represent internal device processes as operating states. One ROBOTIS AI Worker FFW-SG2 integrated mobile dual-arm robot with two stock grippers serves this task. Design a bounded room with a justified number of variable-size worktables and independent floor instruments. Choose table dimensions, heights, positions and orientations; retain real instrument dimensions, mechanisms, sample quantities and required operations. The robot may translate, turn, lift and use either arm, with different docking poses for different operations on the same instrument. Reserve whole-robot/tool transit space, door/lid opening, loading and necessary maintenance clearance. Prefer shared work surfaces and justified compact layouts, without a universal table-count cap or shrinking equipment. Record room revisions and additional-table reasons. Reuse stock grippers when sufficient; custom adapters, replacement tools and fixtures require a task reason, arm and mounting interface. Check payload per arm; do not sum arm ratings or assume unverified cooperative payload sharing. Do not add another robot or unapproved powered assistance. Static scene delivery requires modeled movable mechanisms and evidence-backed coarse approach suggestions, not a claim of completed navigation or workflow execution. The complete mechanical workflow below remains the later execution target. Public robot identity, nominal parameters and approximations: ../_evaluation/generation_environment.json.

## Asset inventory

| ID | Object | Quantity and allocation |
| --- | --- | --- |
| A01 | Air sampler, filter holder and quartz filter | Single sample path: 1 sampler and 1 filter membrane |
| A02 | Precision balances and weighing trays | 1 set; the same filter membrane and corresponding aluminum foil are weighed twice before and after completion |
| A03 | Biological safety cabinets and operating surfaces inside the cabinets | 1 unit and fixed bracket |
| A04 | Filter membrane supporting culture dish, aluminum foil and cutting interface | 1 piece of original filter membrane corresponds to 1 piece of aluminum foil, 1 petri dish with lid, and 1 stable cutting work area |
| A05 | 50ml cone bottom screw cap tube and support stand | 1 sample tube, plus a balance tube of the same type and at least 2 support positions |
| A06 | Large-capacity refrigerated centrifuge and buckets | 1 unit, 1 rotor, 4 bucket; opposite identical adapters and mass matching tubes |
| A07 | Magnetic filter funnel, filter membrane, suction filter bottle, pipeline and vacuum pump | 1 set |
| A08 | Screw-cap tubes preloaded with grinding beads | 1 sample grinding tube, and the same type of balancing tube |
| A09 | Constant temperature water bath and immersed tube rack | 1 unit and 1 immersion stand matching screw grinding tube |
| A10 | Vortex mixer and tube holder | 1 mixer, 1 cup head, 1 24-position horizontal tube holder |
| A11 | Refrigerated microcentrifuge and rotor | 1 set, 1 24-hole rotor and its cover; balancing tube matching each tube type are additional |
| A12 | 4°C short-term refrigerator | 1 set, built-in stable stand matching 2 ml tube |
| A13 | 1.5 ml and 2 ml microtubes | At least 3 new 2 ml collection tubes and 9 1.5 ml process tubes for a single sample; additional matching tubes required for each centrifugation stage |
| A14 | Single-channel pipettes, tips and tip boxes | 5 single-channel pipettes with different volume ranges, matching tips and tip boxes; the tips are not reused after use |
| A15 | Microtube magnetic stand | 1x 16-position magnetic stand |
| A16 | Focused ultrasonicator, dedicated tubes, and tube holders | 1 host, 1 cooler, 1 tube clamp, 1 dedicated sample tube |
| A17 | Thermal cycler, reaction tubes and loading bracket | 1 set, 1 compatible 96-well bracket, 1 countertop splash-proof base, 4 independent 0.2 ml reaction tubes |
| A18 | Cluster-generation instrument and loading components | 1 set, 1 high-output flow cell and supporting manifold/reagent plate/template tube strip |
| A19 | Sequencer, flow cell, reagent rack | 1 unit, using the flow cell area on one side and retaining the complete shell and hatch on the other side |
| A20 | Tube racks, reagent containers and waste receiving containers | At least one 15-position microtube rack and one 50ml tube rack; independently pre-installed liquid source, waste receiver and waste tip box |
| A21 | Sequencer loading containers, priming collection tubes, and staging rack | 8 15 ml priming collection tubes; 10 first-load 15 ml tubes and 9 second-end replacement tubes; 8 first-load 250 ml bottles and 1 second-end replacement bottle; additional staging rack |

## Asset structure and interactions

Numerical specifications and drawings take precedence over visual proportions. Equipment not selected for this task, people, and backgrounds in reference images are excluded from generation. When a reference model differs from the model specified in the original protocol, its image does not establish exact mating dimensions. The generation model may choose unpublished hole coordinates, hinge axes, masses, inertias, contact friction, joint damping, travel limits, and opening/closing forces, but must explicitly label these as modeling values rather than manufacturer measurements. These modeling degrees of freedom must satisfy the structural, capacity, opening/closing, and mating constraints below. Determine hole/container, lid/opening, and rotor/adapter dimensions together to allow insertion, removal, and the necessary motion clearance; do not guess them independently and ignore conflicts.

### A01 | Air sampler, filter holder and quartz filter

Model and selection: VFC (constant flow large volume air sampler), using PM10 size-selective inlet; RFPS-1287-063 is the reference method certification number for this type of sampler; 2500 QAT-UP (quartz membrane) 7204

![A01: This task uses the series shells of a, the rectangular filter membrane of d and the subsequent processing relationship of e–h; b and c do not belong to the selected branch](images/np002_filter.png)

**Specifications and mating constraints:** The filter membrane is 203.2×254 mm; the sampling volume flow rate is 1.13 cubic meters/minute, which is used as the equipment status parameter. The sampler outline, filter clamp locking position and working height are not given.

**Required interactions:** After the initial weighing, unfold the wrapper, open the sampler, place the filter membrane into the filter membrane clamp and fix it, close it, and start it; after the end, unfasten the filter membrane clamp, move the entire filter membrane clamp to the safety cabinet, open it in the cabinet, and take out the membrane. Sampling airflow and particle capture only indicate the status of the device; the filter membrane must support folding and cutting.

**Task-specific details:** The sampler, filter membrane clamp, and filter membrane are independent objects. The combination diagram a is only used as a reference for the sampler series shell, d is the rectangular quartz filter membrane clamping; b and c are PM2.5 (another sampling particle size branch) inlet gasket, which will not be generated this time. The internal pneumatic structure of the PM10 inlet can be abstracted into states, retaining the opening, closing, pressing and removal of the real filter membrane clip; the unknown outer contour is modeled according to the picture and the 203.2×254 mm membrane interface, and the inference is marked. Do not shrink the filter membrane to fit the layout.

### A02 | Precision balances and weighing trays

Model and selection: PL303 (precision balance) series reference; original suffix PL303-IC not independently confirmed by the existing manual

![A02: Precision balances and weighing trays — appearance or assembly reference](images/pl303_drawing.png)

**Specifications and mating constraints:** The reference host is 238 mm wide × 335.2 mm deep × 287 mm high, with a net weight of 5.1 kg; a round weighing pan diameter of 100 mm, and a usable height of 150 mm inside the glass cover; a measuring range of 310 g and a reading resolution of 1 mg. The glass side door slides along the depth direction of the instrument body, and the scale pan, door, handle and feet are retained respectively. The drawing gives 335.2mm and the spec sheet rounds the depth to 335mm.

**Required interactions:** Open the side glass sliding door, put in the wrapped filter membrane, close the door and read the stable mass; open the door and take it out, the same filter membrane maintains its identity, and the two weighings correspond to the same object.

**Task-specific details:** The PL303-IC suffix does not appear in this manual, and the drawings are only used as a reference for the PL303 series. The folded, wrapped filter must fit within the 100 mm diameter weighing pan, with clearance from the glass enclosure walls and a total mass no greater than 310 g. Use the same wrapper for both weighings. Fold the membrane with its processing side facing inward.

### A03 | Biological safety cabinets and operating surfaces inside the cabinets

Model and selection: MSC-Advantage (biological safety cabinet series) 1.2-meter model, as a reference for the same function as the original equipment example

![A03: Biological safety cabinets and operating surfaces inside the cabinets — appearance or assembly reference](images/mscadvantage.png)

![A03: Biological safety cabinets and operating surfaces inside the cabinets — appearance or assembly reference](images/mscadvantage_dimensions.png)

**Specifications and mating constraints:** The outer width of the cabinet is 1300×depth 802×height 1522 mm, excluding the 680 mm fixed bracket; the inner width of the cabinet is 1200×depth 630×height 780 mm. The normal opening of the operation window is 200 mm, and the maximum opening is 533 mm; the working surface height of the fixed bracket configuration is 750 mm. The front window, working surface, front edge and rear grille should be retained. Accurate rails, handles, window masses and opening and closing forces are modeling degrees of freedom.

**Required interactions:** The working window is preset in the normal operating position; after receiving the entire filter holder, remove the membrane, unfold the wrapper, and perform membrane cutting, transfer, and refiltration. The movable front window retains the movement structure, but does not treat the maximum opening as a normal working condition. Maintain a collision-free path for the gripper and equipment in and out of the window.

**Task-specific details:** The manufacturer does not require the refrigerated centrifuge to be placed inside the cabinet for this task. Transfer sealed tubes to the equipment outside the cabinet, then return them for further processing. The layout inside the cabinet needs to leave space for the inlet and exhaust grilles, filter membrane clips and suction filtration devices; the air cleanliness only indicates the status of the equipment and does not claim to have verified true biological protection.

### A04 | Filter membrane supporting culture dish, aluminum foil and cutting interface

Model and selection: Falcon (consumables series) 353003 standard petri dish, corresponding to the 08-772-E series; aluminum foil is the original pre-sterilized covering

![A04: Filter membrane supporting culture dish, aluminum foil and cutting interface — appearance or assembly reference](images/np002_filter.png)

![A04: Standard petri dish series appearance and removable lid; generated according to the size of 353003, not according to the stacking quantity in the photo](images/falcon_standard_dishes.png)

**Specifications and mating constraints:** The entire quartz membrane is 203.2×254 mm, and this path takes a quarter of the effective filter area; the 47 mm collected membrane is cut into approximately 20 mm² pieces. The nominal dimensions of the petri dish is 100 × 20 mm, and the actual outer diameter × height of the dish body given in the manufacturer's catalog is 89.43 x 19.18 mm; the petri dish cover is separately matched, and the nominal 100 mm must not replace the actual dish-body dimensions. The inner cavity needs to support the 47 mm film and allow cutting; the aluminum foil supports unfolding and folding inward. Precise lid clearance, foil thickness, and cutting and clamping interfaces are modeling degrees of freedom.

**Required interactions:** Unfold the aluminum foil and filter membrane, secure the membrane for cutting, clamp the rolled membrane into a 50 ml tube, clamp the 47 mm membrane from the funnel, move it to the petri dish, cut into pieces and transfer to the grinding tube. Cutting and clamping are realized by the subsequent fixture co-design, retaining the reaction-force constraints needed to hold the membrane pieces and containers in place.

**Task-specific details:** Sterilization and material preparation have been completed manually, and cutting is retained. The handheld tool in the figure only illustrates the clamping/shearing function; subsequent clamp co-design provides these capabilities and does not force the generation of scissors and long tweezers.

### A05 | 50ml cone bottom screw cap tube and support stand

Model and selection: Falcon (consumable series) 352070; original supplier catalog number 14-432-22

![A05: Figure f shows the rolled membrane loaded into a 50 ml tube in the original protocol.](images/np002_filter.png)

![A05: The manufacturer's drawing lists both 352070 and 352098; this task uses 352070 tube body and matching 358206 screw cap](images/falcon50.png)

**Specifications and mating constraints:** The capacity is 50 ml; the manufacturer's catalog gives the tube's outer diameter x length of approximately 30 x 115 mm, and the blue flat-top threaded cap has an additional edge. The original protocol adds 50 ml at a time and inserts the rolled membrane; when generating the inner cavity, it must accommodate the liquid state, the membrane and the clearance needed to close the cap at the same time. The nominal volume cannot be simply regarded as the full mouth volume. The aperture, depth and anti-rotation structure of the support frame are modeled values. The mm value of the manufacturer's drawing: the tube body height is 114.55, the main outer diameter is 29.72, the top inner diameter is 27.81, the thread outer diameter is 31.50; the total height with cover is 115.80. The 358206 blue screw cap has a height of 13.21 and an outer diameter of 34.75. There is only a nominal difference of 0.20 mm between the 116 mm height limit of the selected bucket cover and the total height of the tube. Tolerances and loading and unloading clearance inspections must be retained, and the actual fit cannot be claimed to have been verified.

**Required interactions:** Support screwing/unscrewing caps, pick-and-place, pouring, and pipetting. The rack must provide reaction support during cap rotation, and a separate cap placement area must be available.

**Task-specific details:** Only 50ml size reserved. The tube body can be inserted into an adapter with an upper limit of 30 mm, and the cover remains above the opening. The opening cannot cut off the edge of the screw cap; the total height of the capped tube must meet the envelope of the selected bucket cover.

### A06 | Large-capacity refrigerated centrifuge and buckets

Model and selection: 5810 R (Eppendorf refrigerated centrifuge)

![A06: Large-capacity refrigerated centrifuge and buckets — appearance or assembly reference](images/centrifuge5810.png)

![A06: Rotor, rectangular bucket and removable bucket cover; the upper cover of the main unit is not included in this picture](images/a462_rotor.png)

![A06: The 50 ml tube and 3-hole adapter row selected in this version is not the 25 ml configuration with the top layer removed.](images/a462_adapter50.png)

**Specifications and mating constraints:** The instrument body is 700mm wide x 608mm deep x 345mm high, and weighs 99kg excluding the rotor. A-4-62 (swing-bucket rotor) is equipped with four 250 ml rectangular bucket, all bucket should be installed; the total load limit of each basket's adapter, tube and contents is 620 g. Use the 5810758009 adapter to accommodate 50 ml conical bottom tubes, 3 tubes per basket, 12 tubes in a full rotor, and the tube diameter does not exceed 30 mm. The upper limit of total tube height is 116 mm when the anti-aerosol bucket cover is installed, and 122 mm when the cover is not installed; the upper limit of the ordinary conical-bottom tube configuration is 3094 times the acceleration of gravity. The low-speed stage of this process requires 200 times the acceleration of gravity and 4°C, which is separate from the high-speed stage of the microtube.

**Required interactions:** Open the upper cover of the centrifuge, disassemble the selected bucket cover, load and balance the tubes, reinstall the bucket cover, close the main cover, and start; after stopping and unlocking, open the main cover and bucket cover in sequence, and remove the tubes. The bucket, adapter and tube body must have an independent cooperation relationship. The microtube high-speed stage uses another listed microcentrifuge.

**Task-specific details:** The balance tube does not carry the sample product and its identity is recorded separately. The bucket droops when stationary and swings out when rotating, and a fixed upright hole cannot be built; the main cover is locked during operation. Do not use unselected 15 ml adapter.

### A07 | Magnetic filter funnel, filter membrane, suction filter bottle, pipeline and vacuum pump

Model and selection: PALL (filtration equipment manufacturer) 47 mm magnetic funnel 4238, Supor 200 (polyethersulfone membrane) 66234; 1 liter suction filter bottle; MaximaDry (oil-free diaphragm vacuum pump) 13-880-15

![A07: Figures g and h are used as reference for the 500ml funnel and suction filtration connection in this process.](images/np002_filter.png)

![A07: The 150/300 ml products of the same series are only for reference of the magnetic pressing and separation base structure; this task generates 500 ml 4238, and the capacity in the picture cannot be copied.](images/pall_magnetic_family.png)

![A07: The outline drawing of the 115-volt pump set of the same series included in the university equipment catalog is only for reference of the instrument body, pump head and connecting pipe. The dimensions of the 230-volt version are restricted by this text. The resolution is low and no fine dimensions of the interface can be inferred from this.](images/maximadry_family.png)

**Specifications and mating constraints:** The 4238 funnel has a capacity of 500 ml. The manufacturer's catalog gives an overall height of 229 mm, a main body diameter of 89 mm, a 47 mm membrane adapter surface, and a No. 8 plug interface; the effective filtration area is 13.1 cm². The nominal pore size of the filter membrane is 0.2 µm; the suction filter bottle is 1 liter. The middle hole of the bottle stopper is sealed with the funnel rod, and the exhaust side pipe is connected to the vacuum pump through a hose; the upper cup of the funnel is separated from the membrane support base. The hose inner diameter, barbed fittings, pump housing and undisclosed seal geometry are modeling degrees of freedom and must maintain continuous filtration and extraction paths. The selected pump is 330 mm long x 165 mm wide x 220 mm high, with a rated gas flow rate of 20 liters per minute and an ultimate vacuum of 75 Torr; a hose with an inner diameter of approximately 9.5 mm is required. This gas flow rate is not the liquid flow rate on the membrane surface, and it cannot be used directly to set the suspension membrane penetration speed.

**Required interactions:** Open the pressing part of the funnel, lay the film, press it, pour the suspension, start the suction filtration, stop the pump to relieve the pressure difference, disassemble and take out the film; the flow rate is indicated by status, and the film must not penetrate the supporting surface.

**Task-specific details:** Mount the funnel directly on the 1-liter vacuum filtration bottle; the bottom suction filtration waste liquid remains in the bottle and is not poured into the sample receiving container. The pre-installed hose still retains a visible path, and the start/stop control and differential pressure release interfaces are accessible. The switch position of the pump and the fine structure of the port are for modeling freedom, and the series of pictures are not claimed to be the exact same photos of 230 volts.

### A08 | Screw-cap tubes preloaded with grinding beads

Model and selection: PowerBead (pre-installed grinding bead tube), original PowerSoil (extraction set) matching tube; the picture is a reference for the same series from the manufacturer

![A08: The manufacturer has the same series of screw-top bead tubes, but the bead types and contents in the picture are different; use only the tube-body and screw-cap structure as a reference, without claiming that this is the exact model from the original legacy kit.](images/qiagen_powerbead.png)

**Specifications and mating constraints:** Use 1.5–2 ml tube holder-compatible screw-top grinding tubes. Preloaded grinding beads and liquid are used as internal states and do not collide particle by particle; real tube caps, thread seals, graspable tube bodies, lumens and ends must still be generated. The undisclosed precise tube diameter, length and thread are modeling degrees of freedom and must jointly meet the constraints of the horizontal tube holder, water bath bracket and micro rotor. Do not use reaction formulas from different sets or use flip-top tubes as screw-top tubes.

**Required interactions:** Unscrew the cap, add filter membrane fragments and pre-mixed solution, seal, water bath, lock in a special shaking adapter, centrifuge, and take the supernatant.

**Task-specific details:** Three new 2 ml collection tubes are required for extraction, which are listed in the microtube entry; grinding tubes and collection tubes are not merged into the same object.

### A09 | Constant temperature water bath and immersed tube rack

Model and selection: Precision GP 05 (5 liter constant temperature water bath), as a reference model when the equipment is not specified

![A09: GP 05 independent product picture on page 5 of the manufacturer's brochure; does not use the GP 2S shallow groove type next to it](images/waterbath.png)

**Specifications and mating constraints:** The cavity is 154 mm long x 300 mm wide x 150 mm high; without the cover, the outer shape is 246 mm long x 355 mm wide x 232 mm high; the capacity is 5 liters. The manufacturer's rated temperature range is from ambient temperature to 100°C, and the actual setting is in accordance with the requirements of this task. This task is set at 65°C. The removable transparent cover and handle are retained; the bracket and the tube body are combined to determine the immersion depth, and the tube opening must not be submerged.

**Required interactions:** Opening/removing lids, loading tube racks, placing and taking out closed containers, temperature adjustment and starting; the water bath does not need to be fluid simulated, but the container immersion position and anti-floating support must exist.

**Task-specific details:** The original protocol lists examples of water baths with the same function; this version uses a 5-liter reference equipment that has obtained the manufacturer's outline drawings. The tube rack needs to match the screw-cap grinding tube.

### A10 | Vortex mixer and tube holder

Model and selection: Vortex-Genie 2 (adjustable speed vortex mixer); SI-HP524 (original 24-position plastic clip horizontal microtube holder) is used as the same functional structure reference of the original 13000-V1 accessory

![A10: Vortex mixer and tube holder — appearance or assembly reference](images/vortex.png)

![A10: The selected manufacturer horizontal tube holder is shown loaded; the flip-cap tubes only illustrate loading orientation. This task uses screw-cap grinding tubes.](images/si_hp524.png)

![A10: The independent clamping position of the same empty tube holder must remain insertable and releasable](images/si_hp524_empty.png)

**Specifications and mating constraints:** The base is 165 mm deep x 122 mm wide x 165 mm high, with a mass of 4 kg; it has contact start and continuous operation modes, and the round cup/platform is replaceable. The maximum rotation speed of different power frequency versions is different and does not apply across versions.

**Required interactions:** For short-term mixing, use cup head contact to start and keep holding; for long-term oscillation, switch to the horizontal adapter, buckle the closed grinding tube into the tube holder, with the tube cover facing the center, load at least two tubes in a balanced manner, run continuously, and unfix the tube after shutdown. The adapter and cup head are interchangeable accessories and should not be used together.

**Task-specific details:** The selected reference tube holder is suitable for 1.5-2.0 ml standard microtubes. The tube diameter and clamp size must be determined together with the screw-top grinding tube. The tube caps should face inward and do not interfere with each other. This is a clear reference to accessory selection and does not claim to be the same catalog number as the original 13000-V1; the original box image is not used. The short-term mixing cup head and long-term tube holder should be stored separately, leaving space for disassembly and assembly of the replacement interface.

### A11 | Refrigerated microcentrifuge and rotor

Model and selection: 5424 R (Eppendorf refrigerated microcentrifuge), only used if equivalent equipment is permitted by the original protocol

![A11: Refrigerated microcentrifuge and rotor — appearance or assembly reference](images/centrifuge5424r.png)

**Specifications and mating constraints:** The overall shape is 290mm wide x 480mm deep x 260mm high; the mass excluding the rotor is 21kg. FA-45-24-11 (fixed angle rotor) accommodates 24 1.5/2 ml tubes, with a maximum acceleration of 21130 times gravity and 15000 rpm; the rotor mass is 800 g. The upper limit of the total load of the single-hole tube and its contents is 3.75 g; this task operates according to the original maximum acceleration of 10,000 times gravity, and the maximum value of the equipment is not used to replace the task conditions.

**Required interactions:** Unlock and open the upper cover, disassemble and install the rotor cover, load the tubes symmetrically and balance them, close the cover, start, wait for the rotor to stop, open the cover and take out the tube. It is prohibited to open the lid during operation; a balancing tube is required when the sample is insufficient. The rotor aperture/coordinates, cover hinge axis and locking force are modeling degrees of freedom and need to be coordinated according to the selected tube body.

**Task-specific details:** The original example is 5424 room temperature type; this version clearly selects 5424 R as the reference equipment to meet the microtube centrifugation requirements, and its size cannot be said to be the original 5424 size.

### A12 | 4°C short-term refrigerator

Model and selection: TSX505GA (glass door undercounter refrigeration cabinet), as a reference when the original protocol does not limit the exact model.

![A12: 4°C short-term refrigerator — appearance or assembly reference](images/tsx505_glass.png)

**Specifications and mating constraints:** The outer width is 600×depth 661×height 813 mm, and the depth including the handle is 686 mm; the inner width is 496×depth 521×height 585 mm, and the mass is 61.5 kg. The front door hinges, door seals, handles and movable shelves are expressed independently, leaving 25 mm of clearance around the sensor; this task is placed at 4°C for a short period of time.

**Required interactions:** Open the door, put the capped collection tube together with the stable tube rack, and close the door; when the time condition is met, open the door, take it out, and close the door.

**Task-specific details:** The immediate processing path does not require a minus 80°C freezing equipment; no rotators are installed in the cabinets, and the ice boxes are not treated as the same object.

### A13 | 1.5 ml and 2 ml microtubes

Model and selection: Axygen (consumables manufacturer) MCT-150-L-C (1.5 ml low-adsorption flip-top tube); 2 ml collection tube uses the same series MCT-200-C as a clear structural reference

![A13: Factory catalog 1.5 ml series one-piece clamshell picture, the original protocol MCT-150-L-C is in the same series](images/axygen_mct15.png)

![A13: The manufacturer's diagram of the 2 ml one-piece flip cap is used as a structural reference for the collection tube of the original set, and does not claim to be the same catalog number.](images/axygen_mct2.png)

**Specifications and mating constraints:** The 1.5 ml and 2 ml models are modeled separately. Both are flip caps that are integrally connected to the tube body. You cannot just change the label for different capacities. The manufacturer's transparent series has a rated relative centrifugal force of 14,000 times gravity acceleration. This task does not exceed the original requirement of 10,000 times gravity acceleration. The undisclosed tube length, diameter, cone bottom and cap axis are modeling degrees of freedom; they must be coordinated with the micro rotor and magnetic stand. The 2 ml collection tube is not a screw-cap grinding tube.

**Required interactions:** Support opening/closing lids, grasping, pipetting, pouring, and insertion into a rotor or tube rack. Create a new instance whenever material is transferred to a fresh tube. Maintain distinct identities for the original tube, supernatant receiver, and product tube.

**Task-specific details:** The nine 1.5 ml process tubes include 2 extraction magnetic separation tubes, 1 combined product tube, 1 tube each of three library construction and purification tubes, 2 fragment screening tubes, and 1 final elution tube. Preloaded reagent source tubes and balancing tubes are counted separately and must not use any of these fresh tubes.

### A14 | Single-channel pipettes, tips and tip boxes

Model and selection: Research plus (mechanical pipette series), as a reference when the model is not specified; see the supplement of this task for the original specified model.

![A14: Only the single-channel model in the picture is generated, multi-channel models are not selected.](images/pipette.png)

![A14: Original box and refill tray; the matching tips are already installed in the box in the initial state.](images/eptips_box_photo.png)

![A14: The relationship between the hinged lid, box body, replaceable well plate and independent pipette tip; the human hands and gloves in the picture are not generated](images/eptips_box_loading.png)

**Specifications and mating constraints:** Configure different measuring ranges according to the transfer volume in this task. You cannot use a large-volume pipette to complete small-volume dispensing beyond the measuring range. The suction tip is matched with the interface; the hole position of the suction tip box should allow vertical installation of the tip. Select the epT.I.P.S. (original tip series) corresponding range tip and Box 2.0 (refillable tip box) reference structure; a single box has 96 positions and is initially filled. The aperture, hole spacing, box outline and lid hinge are modeling degrees of freedom; it must be ensured that the tip of the tip is suspended, the tip can be inserted vertically, and the lid is opened without blocking the tip pickup area.

**Required interactions:** Support grasping, tip attachment, pressing to the first and second stops, releasing to aspirate, dispensing, and tip ejection; the volume adjustment knob must rotate. Tips are detachable objects, and used and unused tips are stored separately. Represent liquids only through container identity and volume state.

**Task-specific details:** According to the original protocol, the volume ranges of 0.1–2.5, 0.5–10, 2–20, 20–200, and 100–1000 µl are provided, which are matched with the tip specifications of 0.1–10, 2–200, and 50–1000 µl respectively; 2–20 µl pipettes use compatible range tips. Select single channel only. 50 ml of pre-prepared solution is quantitatively poured from an independent pre-filled bottle, and the micropipette is not responsible for transferring the entire bottle.

### A15 | Microtube magnetic stand

Model and selection: DynaMag-2 (16-position microtube magnetic stand), Cat. No. 12321D

![A15: Microtube magnetic stand — appearance or assembly reference](images/dynamag2.png)

**Specifications and mating constraints:** 16 numbered holes, 8 on each side; suitable for 1.5–2 ml microtubes; upper tube rack and magnet base can be separated. Working volume 10–2000 µl. Overall dimensions, hole spacing and disassembly force are not given.

**Required interactions:** Insert/remove tubes, detach or replace the upper rack, aspirate liquid and resuspend the contents within the tubes. Magnetic separation is only represented by status, and the tube cannot be mistakenly adsorbed to the magnet, nor can discarding liquid and washing transfer be omitted.

**Task-specific details:** The washing action includes the rotation of the tube body; the clamp needs to rotate the tube near the magnetic separation working position and must not immobilize the tube against the required rotation.

### A16 | Focused ultrasonicator, dedicated tubes, and tube holders

Model and selection: Covaris S220 (focused ultrasonic processor), special buckle cap tube Cat. No. 520045

![A16: Focused ultrasonicator, dedicated tubes, and tube holders — appearance or assembly reference](images/s220.png)

![A16: Manufacturer 520045 tube body, diaphragm cover and internal fiber structure diagram](images/covaris520045.png)

![A16: Manufacturer's 500114 single tube clamp structure diagram, not a tube rack or ordinary centrifugal adapter](images/covaris500114.png)

![A16: Manufacturer's manual diagram of the same CH03 cooler, this task connects S220](images/e220_ch03.png)

**Specifications and mating constraints:** The main unit is about 200mm wide × 530mm deep × 330mm high, and the mass is about 12kg. 520045 is a 6 x 16 mm glass specialty tube with a pre-slit diaphragm cover and internal acoustic fiber; this is the tube specification, not the total envelope including the cover and clamp. Choose 500114 single tube clamp and CH03 (external circulation cooler). This procedure contains 50 µl of liquid. The old version of the manufacturer's guide lists that the tube is compatible with 50/130 µl; 50 µl is not written as the nominal capacity of the tube. The undisclosed dimensions of the cover, tube clamp clamping groove and positioning structure are modeling degrees of freedom. The glass tube must not penetrate the clamp or collide with the bottom of the sink. The cooling circuit uses approximately 9.525 mm inner diameter hoses and quick connectors; the undisclosed parts of the cooler's outer contour and interface coordinates are modeling degrees of freedom, and the space occupied by the entire machine and hoses must be retained.

**Required interactions:** Insert the tip through the pre-cut slit in the cover to add the sample, keeping the internal fiber and cover in place; open the instrument safety cover, fix the special tube to the 500114 tube clamp and install it into the processing position, close the cover and receive the software startup event. At the end of the operation, remove the tube clamp and tube, and suck out the sample through the pre-cut slit; it is not required to remove the entire buckle cover each time. The water tank and pre-connected cooling circuit preserve form and space.

**Task-specific details:** When filling liquid on the table, a special tube needs to be stably supported; the support seat can be designed for subsequent fixtures, and ordinary tube rack holes cannot be used to pretend to be compatible. The E220 energizer listed in the original protocol does not belong to this version of the S220 configuration. Software startup is a control event and does not involve imaginary body buttons.

### A17 | Thermal cycler, reaction tubes and loading bracket

Model and selection: Veriti (96-well 0.2 ml thermal cycler), Cat. No. 4375786

![A17: Thermal cycler, reaction tubes and loading bracket — appearance or assembly reference](images/veriti.png)

![A17: Original text PCR-02-C corresponds to the manufacturer's 0.2 ml flat-top attached lid single tube picture](images/axygen_pcr02_flat.png)

![A17: 96-position bracket and countertop splash-proof base structure; for this task, select a 0.2 ml single tube with attached lid, and do not select the 0.1 ml model shown in the picture](images/veriti_single_tray.png)

![A17: Orientation of the bracket/plate into the heating zone, does not require the creation of a plate reaction vessel](images/veriti_loading.png)

**Specifications and mating constraints:** The instrument body is 237 mm wide × 485 mm deep × 245 mm high; the height with the lid open is 400 mm. It is recommended that the total space above is 555 mm and the rear clearance is 155 mm. 96 wells, 0.2 ml per well; do not mix 0.1 ml rapid version. The tube, bracket and countertop base are separate objects and the base must not be loaded into the heating block.

**Required interactions:** The tube cover is closed, the hot cover is opened, the bracket is loaded according to the direction, the hot cover is closed, the touch screen is started, and after the operation is completed, open the heated lid and remove the loaded tray. Reaction only records running status. Single tubes must have compatible brackets; opening angle, hinge position and clamping force are not given.

**Task-specific details:** Select the original PCR-02-C (thin-walled flat-cap reaction tube) for four rounds of step-by-step reactions, with independent new tubes for each round. The manufacturer's catalog states that it fits the standard 96-hole block; the specific fit of the bracket hole and the tube is determined by the generated model and the thermal cover closure is checked. The manufacturer's MicroAmp (reaction consumable series) pictures cannot be regarded as proven to be exactly the same type. Only the individual-tube path with attached caps is generated, and no hold-down frame for capless tubes or 8-tube strips is added. Instrument mass is listed as modeling degrees of freedom due to inconsistencies within manual versions.

### A18 | Cluster-generation instrument and loading components

Model and selection: cBot (pre-sequencing clusterer, not second generation model)

![A18: Cluster-generation instrument and loading components — appearance or assembly reference](images/cbot.png)

![A18: Cluster-generation instrument and loading components — appearance or assembly reference](images/cbot_stage.png)

![A18: Refer to the position of the reagent table and template tube strip. The specific carrier board uses the v3 configuration specified in this task.](images/cbot_reagent.png)

![A18: Manifold outlet and outlet clamp](images/cbot_outlet.png)

![A18: The buckles and positioning pins on both sides of the sipper comb](images/cbot_sipper.png)

**Specifications and mating constraints:** Width 380 x depth 620 x height 390 mm, height with lid open 700 mm, mass 31 kg. Includes hot stage, flow cell clamp, distribution manifold, reagent plate, template tube strip, sipper comb and waste bottle. This version of the v3 reagent carrier plate has 11 rows of pre-installed 8-tube strips, and the 12th row is empty; the white sealing film of the 10th row needs to be peeled off, and other sealing films are pierced by a sipper comb. The sample template uses an additional 8-tube strip. The reagent plate plastic protective cover, peelable sealing film, carrier plate and tube strip are detachable objects.

**Required interactions:** The prepared library is transferred to the template tube strip with the pre-prepared loading solution; open the instrument cover, remove the reagent plate protective cover and peel off the white film of the 10th row, pull the plate clamp lever, install the plate in the front right diagonal direction and fix it. Install the flow cell into the hot stage with the hole facing up, align the manifold with the positioning pin and press the flow cell clamp, connect the manifold outlet and fasten the outlet clamp, and buckle the two sides of the sipper comb in position. Put in the template tube strip, close the cover and touch the screen to start. After completion, release the outlet clamp, outlet, sipper comb, and flow cell clamps in sequence, remove the manifold, and then remove the same flow cell.

**Task-specific details:** Use the original third edition high-output clustered consumables and do not mix manifolds and flow cells from other sequencing platforms. A single sample occupies one template channel, and other channels are processed as configured blank status; the tube strip is still fully loaded according to the equipment requirements. Template tubes, reagent plates, manifolds, and flow cells cannot be combined into one decorative model. Only use the original protocol-compatible single indexing scheme and do not add additional primer rows specific to another library. The flow cell cleaning contact surface is provided by the subsequent fixture; the pre-installed wet wipe surface and rinse solution are considered prepared materials, and human gloves or tweezers are not produced. Reserve necessary flushing, wiping and transfer space.

### A19 | Sequencer, flow cell, reagent rack

Model and selection: HiSeq 2500 (high-throughput sequencer), select the high-output mode that matches the original reagents

![A19: Sequencer, flow cell, reagent rack — appearance or assembly reference](images/hiseq2500.png)

![A19: Sequencer, flow cell, reagent rack — appearance or assembly reference](images/hiseq_flowcell.png)

![A19: Independent structural relationship of reagent rack, conical bottom flask, 15 ml tube and vertically movable aspiration tubes](images/hiseq_reagent_cabinet.png)

![A19: 8 flow cell waste pipes and independent condensation drain pipes; only the first 8 pipes are transferred to the priming collection tube](images/hiseq_waste_lines.png)

**Specifications and mating constraints:** The entire machine is 1186mm wide x 760mm deep x 940mm high. The height can be increased by up to 12.7 mm; the weight is 226kg. The waste bottle on the right side extends an additional 100 mm, and the keyboard tray on the front extends an additional 190 mm. Recommended clearance is 610mm on each side, 102mm on the rear and 610mm on the top. Contains dual flow cell areas, vacuum fixtures and independent reagent refrigeration compartments, while only using one side while retaining the complete body. Use eight 250 ml conical bottom flask positions on one side and a total of 10 15 ml tube positions No. 10-19. A funnel cap is used to guide the aspiration tube at the mouth of the bottle, and the conical bottom flask cannot be replaced with an ordinary wide-mouth bottle. In addition to the sample flow cell, a used and clean priming flow cell is required; the two are always of different identities. The precise bottle outline, cap hole and rack hole are modeled as degrees of freedom, and are coordinated with the equipment icon slots and lifting aspiration tube.

**Required interactions:** Open the cabin, lift up the aspiration tube handle and lock it in the upper position, pull out the frame, replace the funnel cover and insert the pre-mixed bottle and 15ml tube, push back and lower the aspiration tube. First install the priming flow cell, with the hole facing down and the barcode facing right, push to the positioning pin, and the control lever moves from 0 to 1 to establish vacuum, and then to 2 to connect the manifold. Put the 8 waste lines into 8 empty 15 ml receiving tubes respectively, start priming and check the collection volume status, and return the waste lines to their original connections. Release the control levers 2 to 1 to 0 and remove the priming flow cell, perform contact surface cleaning, and then load the clustered sample flow cell in the same orientation. After normal fluid-path check, the cabin is closed and the software is started. At the end of the first-end reading section, follow the prompts to install the pre-prepared second-end reagent tube, replace the No. 1 bottle and transfer the funnel cover, lower the aspiration tube and close the chamber and continue until completion.

**Task-specific details:** Select high-output v3, single-index, paired-end 101-base reads. The internal chemistry and optical imaging of the run are in status, but the normal required replacement of the priming flow cell, rerouting of the waste lines, and mid-stream second-end reagent replacement are still retained. The holes in the clustering instrument are facing up, and the holes in the sequencer are facing down. The direction of the flow cell must be changed when transferring. The normal fluid-path/collected-volume check is the starting condition and is not a product quality assessment that is excluded.

### A20 | Tube racks, reagent containers and waste receiving containers

Model and selection: Functional auxiliary assets; geometry adapted to selected equipment

![A20: Tube racks, reagent containers and waste receiving containers — appearance or assembly reference](images/coolrack_m15.png)

![A20: The large red tube rack in Figure f is used as a reference for supporting 50 ml tubes; the bottles and receivers in Figure G only illustrate the layout of independent containers.](images/np002_filter.png)

![A20: A series of capped glass bottles from the original factory supply catalog, as a reference for pre-packed dispensing sources and independent waste bottles](images/pyrex_media_bottles.png)

![A20: Wide-mouth lidded polypropylene container series, select 500 ml as the waste tip receiving container; not all sizes shown are generated](images/fisher_wide_jars.png)

**Specifications and mating constraints:** The slot location is determined based on the number of samples, the number of new tubes, the number of balancing tubes and the turnover of each step; the reagent container is pre-installed and still serves as the source of liquid extraction. The waste liquid container and the waste tip container are expressed separately; the capacity and number of slots are determined according to the task configuration. For large-volume liquid sources, choose Pyrex (borosilicate glass bottle series) 1395-0100 100 ml capped bottle, pre-filled with the required 50 ml of liquid; small-volume source liquids can be pre-filled in listed microtubes. The independent waste liquid bottle refers to the 250 ml bottle of 1395-0250; the waste tip container refers to the 500 ml polypropylene wide-mouth jar of W014372, with a cap size of 89 mm-400. The unknown outer contour and wall thickness of each container are modeling degrees of freedom, which must meet capacity and grasping stability. All source liquid containers are separated from product tubes and discard bottles.

**Required interactions:** Support containers, preserve lid-opening and pipetting paths, and provide graspable regions. Containers must fit their positions without interpenetration or unsupported suspension; decorative geometry must not block their cavities.

**Task-specific details:** At least distinguish the positions of sample/supernatant/elution/library tubes; replace new tubes repeatedly in multiple stages, and the number of instances cannot be estimated based on photos. This version of the microtube support uses CoolRack M15 (15-position metal tube rack) 432037: length 102 × width 64 × height 38 mm, 3 rows and 5 columns of round holes, hole diameter 11.1 mm, hole depth 32.7 mm, center-to-center spacing of 17.8 mm in both rows and columns. The CoolRack diagram only covers the micro-tube rack, and other auxiliary parts are supplemented by the corresponding pictures in this task; the round hole rack does not automatically provide capping reaction force, and the anti-rotation clamp still needs to be designed later.

### A21 | Sequencer loading containers, priming collection tubes, and staging rack

Model and selection: Comes with 250 ml conical bottom flask and funnel cap; refer to Falcon (consumables series) 352096 for 15 ml tube

![A21: Sequencer loading containers, priming collection tubes, and staging rack — appearance or assembly reference](images/falcon15.png)

![A21: Sequencer loading containers, priming collection tubes, and staging rack — appearance or assembly reference](images/hiseq_reagent_cabinet.png)

![A21: Sequencer loading containers, priming collection tubes, and staging rack — appearance or assembly reference](images/hiseq_waste_lines.png)

![A21: Sequencer loading containers, priming collection tubes, and staging rack — appearance or assembly reference](images/hiseq_flowcell.png)

**Specifications and mating constraints:** The 15 ml tube has an approximate outer diameter × length of 17 × 120 mm and must fit the sequencer rack. Place 8 empty priming collection tubes on a separate stable staging rack; do not reuse the loaded source tubes for this purpose. Initially, the 10 reagent-rack positions hold 3 prepared index tubes and 7 prefilled water tubes. For the second read, replace the specified 9 positions and retain the original index tube in position 17. The 250 ml conical-bottom bottles use the instrument-compatible funnel caps; do not substitute flat-bottom source bottles. Also prepare bottle 1 for the second read. Model unpublished staging-rack dimensions to match the corresponding tubes. Manufacturer drawing dimensions for the 15 ml tube, in mm: body height 118.8, main outer diameter 17.37, opening inner diameter 14.66, thread outer diameter 19.25, and capped total height 120.75. The matching cap is 10.41 high with an outer diameter of 22.98. Rack holes support the tube body and leave clearance for the cap rim.

**Required interactions:** Open each container and load it into its corresponding rack position according to its identity; make sure the access is open before lifting and lowering the suction tubes. The source tube and receiving tube cannot be interchanged. The removed bottle cap and funnel cap have independent places. Place the flow cell on a stable support during transfer and waiting to prevent edges and interfaces from contacting the table.

**Task-specific details:** The 15 ml specification is used for sequencing loading and priming, without adding pre-processing centrifuge rotor configuration. The refill reagents have been pre-prepared; mixing and thawing do not open another scene, but opening, changing and loading and unloading are reserved.

## Operation sequence and object relationships

1. **Initial weighing, sampling, and reweighing** (A02, A01, A04, A03): Weigh the pre-sterilized and wrapped filter membrane, unfold it and install it into the sampler clamp and start it; after completion, remove the filter membrane clamp, open it in the cabinet and take out the membrane, fold the sampling surface inward and weigh it again in the same aluminum foil. mass status is saved twice.
2. **Cutting, rolling the membrane, and low-speed centrifugation** (A03, A04, A05, A06, A20): Unfold the fresh filter membrane, take a quarter of the effective filter area, roll it into a 50 ml tube, add the pre-mixed solution and tighten the lid; balance the load and place it in the refrigerated centrifuge, close the lid and start, stop and unlock before taking it out.
3. **Refiltration and transfer of membrane fragments** (A03, A10, A05, A07, A04, A08): Mix the contents of the closed large tube; put the 47 mm membrane into the magnetic funnel and press it together, connect the filter bottle and the pump, and when pouring the suspension, the original quartz membrane remains in the large tube; start and stop the pump after completion to release the pressure difference, open the funnel to take out the membrane, put it in a petri dish, cut it into pieces, and put it into the grinding tube.
4. **Grinding and stepwise supernatant transfer** (A08, A09, A10, A11, A12, A14, A13, A20): Add pre-mixed liquid to the grinding tube and tighten the lid. After the water bath, fix it in a horizontal tube holder for operation, take it out and centrifuge. The first supernatant is transferred into a new 2 ml tube, and two rounds of liquid addition, short mixing, temporary storage at 4°C, centrifugation, and transfer to a new 2 ml tube are completed in sequence; finally, the supernatant is transferred to two 1.5 ml tubes.
5. **Magnetic bead washing and merging of extracts** (A15, A13, A14, A11, A20): Add the pre-installed magnetic bead suspension to the two tubes and mix well. Discard the supernatant after magnetic separation. Transfer the tubes and wash them for two rounds according to the original protocol. After a short centrifugation, magnetically separate again and remove the residual liquid. Open the lid and wait. After elution, magnetic separation was performed and the two products were combined into a new tube. Waste liquid and sample identity are separated.
6. **Sonication in a dedicated tube and three sequential reaction rounds** (A16, A17, A14, A13, A15, A20): Transfer the sample to the special tube through the diaphragm port, clamp, load, close the cover and start. After completion, unload and take the liquid into the first 0.2 ml tube. Perform three rounds of reactions in sequence, sealing tubes, installing racks, running, and unloading in each round; transfer each round of purification into independent 1.5 ml tubes and use a magnetic stand, and then transfer the eluent into the next round of new reaction tubes. You cannot directly insert the 0.2 ml tube into the magnetic stand that only supports 1.5/2 ml.
7. **Amplification and two-stage size selection** (A17, A13, A14, A15, A20): The fourth reaction tube completes amplification loading, running and removal. For the first screening, transfer to a new 1.5 ml tube and perform magnetic separation. Keep the supernatant and discard the beads. The supernatant enters another new tube for the second screening. This time, keep the beads and discard the supernatant. Wash and elute to the final new product tube. Do not reverse which fraction is retained and which is discarded in either selection step.
8. **Library loading and cluster generation** (A18, A14, A13, A20): Transfer the final library and pre-mixed loading solution into the template tube strip; remove the reagent plate cover, peel off the designated sealing film, install the reagent plate, flow cell, manifold and sipper comb, connect the outlet and buckle the clamp, install the template tube strip, close the cover and start. After finishing, release the pipeline buckle and clamp, remove the manifold and take out the sample flow cell; do not pull out the piece directly with the manifold.
9. **Sequencing priming and sample flow cell loading** (A19, A21, A20): Install the pre-prepared reagent bottles and source tubes, and install the independent priming flow cell; route the 8 waste lines into 8 separate empty 15 ml tubes, run and confirm the normal volume status and restore the waste-line connections. Take out the priming flow cell and clean the contact surface, replace it with the same clustered sample flow cell, with the hole facing downwards, complete the two-level locking and normal fluid-path check, and close the chamber and start the run.
10. **Reagent replacement for the second read and completion** (A19, A21, A20): When the first-end reading and indexing are completed and the equipment prompts to allow loading, raise the aspiration tube and pull out the rack, insert the pre-configured second-end tube and replace the No. 1 bottle, transfer the funnel cover; push the rack back, lower the aspiration tube, close the chamber, and continue running until the paired-end sequencing is completed. No additional product measurements, data analysis, or maintenance washes are performed for the next run.

These descriptions define the operations that the equipment must support; they do not prescribe robot grasp points, fixture shapes, or trajectories. Different samples, aliquot tubes, waste liquids, and products must retain distinct identities. A successful operating state indicates completion of the operation sequence only; it does not establish biological validity.
