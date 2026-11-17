<!-- Read-only code audit of cf2851d by an audit agent on 17 Nov 2026; spot-checked by the task owner against the import graph. Nothing was run. -->

# Architecture audit body: sim, robot, perception, navigation, skills, semantics
Repo C:\Users\24adi\DREAM-cleanup, cf2851d. Read-only. Importer facts come from import_graph.json
(static imports; lazy/importlib uses spot-checked with grep). "FUNC" marks a change that would alter
behaviour or results; everything else is behaviour-neutral (removal of unreferenced code, renames, moves).
Result counts were read from results/*/summary.json unless marked "README only".

## 0. Headline findings
1. The G1 grasp/posture stack (g1_posture 1077, wholebody 445, g1_supervisor 216, g1_grasp_geometry 266,
   sim/g1_grasp_fixtures 87) is reachable from exactly ONE caller: experiments/g1_grasp_bringup.py (1148 lines),
   plus g1_wbc_envelope.py for the envelope. Nothing in dream/tools or dream/runtime can command arms, posture or
   fingers. tests/test_isolation.py:17 (FORBIDDEN_FOR_DECISIONS includes dream.robot) makes that structural: the R1
   grasp is a development diagnostic, not a capability the agent can call.
2. dream/robot/g1_rollout.py (341) and g1_correction.py (202) are a dead-end pair. g1_rollout is imported by
   nothing in dream/ or experiments/, only tests/test_g1_posture.py:185 (verify_top_k). g1_correction is imported
   only by g1_rollout and tests/test_g1_posture.py:146,226. They are the "PLAN C5 contact-aware feasibility gate"
   from the W6 whole-body work, superseded in R1 by the rehearsal path (g1_grasp_geometry.rehearse_pinches,
   sim/g1_mobile.build_grasp_rehearsal:252, trial backend at g1_grasp_bringup.py:614-623).
3. Two G1 embodiments: supported-pelvis (sim/g1.py + robot/g1.py, torque PD, fixed pelvis) and free-standing WBC
   (sim/g1_mobile.py + robot/g1_wbc.py). g1_mobile._add_hands (g1_mobile.py:111) re-adds the Unitree finger hinges
   to the NVIDIA MJCF, a second hand-model path alongside sim/g1.py.
4. Two procedural-scene generators (sim/room.py RoomSpec, sim/home.py HomeSpec) and two navigators in
   MobileToolbox ("classic" vs "rpp"), each tied to different frozen results.
5. Oracle inputs are labelled in docstrings and README; the cleanup risk is dead variants, not mislabelling.

## 1. Embodiments

### 1.1 Gantry + mechanisms (Active Physical Discovery)
- Scene dream/sim/mechanisms.py (292; MechanismSpec = truth; gripper is a ball-joint connect constraint, stated in its
  docstring "Simplifications"); env dream/sim/env.py (205); controller is the abstract force-limited 3-axis gantry
  inside env.py (move_ee_to, grasp, execute_probe). Callers: dream.eval.episode, experiments run_benchmark, demo,
  calibrate_adequacy, validate_adequacy, 5 tests.
- Validated per AGENTS section 2 item 1 (about 60% fewer probes than random). Narrow: attaches by constraint, not
  contact. Not a cleanup target.

### 1.2 Panda tabletop
- Scene dream/sim/tabletop.py (242, TabletopSpec truth) + tabletop_env.py (351). Controller dream/robot/panda.py
  (315): damped-LS differential IK on the TCP site, workspace clamp, speed cap, force latch. Embodiment adapter
  SimPanda at tabletop_env.py:~295. Commands reach it via skills/manipulation.py -> Embodiment.move_tcp /
  set_gripper (tabletop_env.py:303-310).
- Evidence: results/pick_place/summary.json n=50 success=50 (oracle tracker, seeds 1000-1049, 0 protective stops).
  Camera arm is results/vision/summary.json (README "39/40"; I did not re-derive the cell mapping: README only).
  results/tasks/summary.json: agent 40/40 in every task/condition cell incl. disturbed and cluttered; fixed_sequence
  0/40 when disturbed. Gates pass.
- Narrow/oracle: cubes only; oracle tracker (6.1); camera path is a colour+depth baseline for box-like objects
  (perception/detect.py docstring states its assumptions).
- Duplicate: dream/robot/joint.py (192), "separate experimental Panda absolute-joint controller". Callers: 3
  molmo/joint experiments + 1 test; no dream/ importer. README: MolmoAct2 verified negative; results/
  joint_hold_failure_dev kept. One-off for foundation-model experiments; leave, label as experimental.

### 1.3 Compartment (Panda + hidden drawer/lid/fixed)
- dream/sim/compartment.py (294) built into the tabletop scene; skills/articulation.py (292); discovery through
  dream/tools/discovery.py and dream/agent.
- Evidence: results/compartment/summary.json openable 54/60 (0.9, gate 0.8), fixed correct rejection 30/30, false
  claims 0: gates pass. Camera: results/compartment_vision_closeout/closeout_summary.json openable 0.55 (11/20;
  gate 0.8 FAIL), fixed declined 10/10, false claims 0. Oracle reference arm: results/
  compartment_oracle_reference_closeout.
- Narrow: knob handle, shallow cavity, counterbalanced lid (AGENTS section 2 item 3). Oracle: handle pose,
  compartment AABB, visible cavity (6.1).

### 1.4 Supported-pelvis G1 (fixed pelvis, Dex3 hands)
- Scene dream/sim/g1.py (130; MjSpec removes the free joint); controller dream/robot/g1.py (268), "proprioceptive
  torque PD only, not a locomotion or grasp policy". Callers: dream/tools/dexterous.py:88 and object_exploration.py
  via robot.set_hand(side, closure) (types.py:215); experiments g1_dexterous, g1_contact_resolution,
  capture_groot_input, audit_groot_actions. Results: g1_dex3_dev(_v2,_v3), g1_dex3_challenge, g1_dex3_checkpoint,
  g1_contact_resolution_dev (commissioning; no README status row).
- Role now: commissioning testbed and GR00T input capture. Keep (AGENTS section 14); say "commissioning only" in
  the docstring.

### 1.5 Free-standing mobile G1 under NVIDIA Decoupled WBC (main G1 embodiment)
- Model dream/sim/g1_mobile.py (481): NVIDIA g1_gear_wbc.xml unmodified plus optional hands (_add_hands:111), touch
  sensors (_add_touch_sensors:171) and head camera. Controller dream/robot/g1_wbc.py (246): two ONNX actors; Balance
  when command norm < 0.05, Walk otherwise (g1_wbc.py:225-229).
- Command interface and rates:
  * velocity: WBCLocomotion.set_command(vx, vy, wz) (g1_wbc.py:169) rejects, never clips, beyond MAX_COMMAND
    [0.5, 0.3, 0.6] (DREAM commissioning envelope, not the policy's trained range).
  * posture: set_posture(height, rpy, rates) (g1_wbc.py:175); HEIGHT_RANGE 0.20-0.74 m, RPY_RANGE +-0.52/+-0.52/+-1.0;
    out of range is rejected. Arms: callers write controller.arm_target (14 values), PD-held at kp 100.
  * rates: policy 50 Hz (tick, g1_wbc.py:219); physics 200 Hz, DECIMATION 4; G1MobileEnv.step (g1_mobile.py:412) does
    one policy tick then 4 PD-torque physics steps; fingers are position servos (set_hand:397).
  * callers, navigation: MobileToolbox (dream/tools/mobile.py) -> MobileBase.walk(forward, lateral, turn, seconds)
    (G1MobileBase.walk, g1_mobile.py:473), velocity updated every STEP_S=0.1 s (mobile.py:44), replan every
    REPLAN_S=1.0 s (mobile.py:43).
  * callers, whole body: WBCBackend.execute / follow (wholebody.py:357, 405) set zero velocity, set_posture and
    arm_target on each 50 Hz tick in _run (wholebody.py:415). Only g1_grasp_bringup.py and g1_wbc_envelope.py use it.
- Safety: fall latch at tilt > 0.6 rad (g1_wbc.py FALL_TILT, damping only afterwards); g1_supervisor 50 Hz monitor
  (joint range, torque, tilt, contact force, foot support) exists only in the WBCBackend path. The walking path has
  the fall latch and MobileToolbox._blocked (mobile.py:522).
- Validated (README): object search on G1 (H4) 14/15 vs 13/15, 0 falls, PASS G8; room search frozen seeds FAIL
  (0 falls); posture envelope results/g1_wbc_envelope, g1_wbc_envelope_v2. R1 development results, all
  oracle-assisted pose and odometry: g1_r1_tabletop_dev_20261110 verified 3/6; rehearsal dev 5/6 cubes + box; depth
  dev 8/9; compound dev (mugs pass, bottles and sphere fail); bottle dev 13/15 (heavy bottles, sphere fail);
  g1_r1_walk_dev_20261111 three cubes pass the full gate, cylinder slipped. R1 gate itself still "pending".
- Failed / not integrated: W6 whole-body reach (branch claude/wholebody-autonomy at 8f47475, 35 commits not in this
  branch; off-band 0.667 vs 0.70, refusal 0.864 vs 0.90, held-out not run: README table, Reports/2026-11-10/
  001-strategic-audit-and-plan/REPORT.md line 19). That branch holds dream/tools/reach.py and dream/sim/
  reach_scenes.py, which do not exist here. R0 ported model, planner, supervisor selectively (Reports/2026-11-10/
  003-r0-r1-start/REPORT.md section C). Grasp failures: heavy bottles, sphere, cylinder slip, shifted-cube cases
  (README G1 roadmap).
- Narrow/oracle: object pose, table geometry, odometry, localisation (6.3).
- Duplicates / dead code inside the stack:
  * g1_rollout.py + g1_correction.py: see headline 2 (543 lines, test-only).
  * g1_assisted_demo.py (95): RightArm IK + a MuJoCo `connect` equality constraint "ASSISTED grasp" (docstring
    lines 8-13). Only experiments/isaac_task.py:148 imports it; add_grasp_constraint is called nowhere. That is an
    attachment constraint, which Invariant 2 forbids for claims.
  * wrap() defined at navigation/follow.py:18, navigation/control.py:15, robot/g1_posture.py:114,
    sim/g1_mobile.py:203, plus eval/task_scoring.py:34 (_wrap). yaw from quaternion at g1_mobile.py:207 and
    tabletop_env.py:289. yaw_matrix (wholebody.py:49) and kinematic_base._rz (54) are the same rotation.
- Isaac adapter dream/sim/isaac_g1.py (548): "drop-in for G1MobileEnv" in Isaac Sim 6.2: same WBCLocomotion,
  PD torques computed at 200 Hz and clipped at MJCF limits, PhysX drives zeroed; oracle odometry by calling
  G1MobileEnv._integrate_odometry(self) (isaac_g1.py:495). Callers: experiments isaac_g1_kitchen, isaac_g1_live,
  isaac_home_runs, isaac_replay_video + one test. Evidence: README Isaac rows are FAIL or INCOMPLETE (frozen matrix
  3 false claims; semantic re-test gates INCOMPLETE, `l_island` never loaded). AGENTS section 2: no Isaac embodiment
  validated. It copies constants from g1_mobile (STAND_HEIGHT 0.79 at isaac_g1.py:27 and g1_mobile.py;
  JOINT constants): drift risk. Cleanup: none required (g1_mobile imports mujoco, isaac_g1 must stay CPU-light);
  add a test that the two constants agree.

### 1.6 Kinematic surrogate base
- dream/sim/kinematic_base.py (277): 0.25 m capsule on a HomeSpec floor, first-order velocity lag, same oracle odometry
  model, same head camera pose (head_camera_pose:67), optional oracle semantic camera. Callers: only
  experiments/semantic_search_eval.py:248 and tests. Docstring says headline results use the real G1, yet the
  validated search results H1/H2/H3 (README 83/87, 31/32, 14/16) ran on this base; the G1 H4 14/15 is the secondary
  check. It is the validated embodiment for semantic search. Keep.

### 1.7 Procedural scenes: room vs home
- sim/room.py (266) RoomSpec (cabinet with hidden door/drawer/fixed, Milestone B): used by 12 experiments
  (g1_explore_room, g1_command_demo, g1_interruption_demo, g1_memory_demo, g1_navigation_eval, ...) and
  tests/test_mobile_exploration.py; imported by no dream module. sim/home.py (1559): HomeSpec used by
  kinematic_base.py:30 and semantic_sensor.py:33 plus experiments. Different object models; both stay (seeds tie
  results to the generators).
- home.py also holds evaluation oracles: GroundTruthView:799, g_visibility:899, verify_scenario:906, viewpoints:876,
  TargetInfo:322. Scoring code inside the scene module. Cleanup (neutral): move them to dream/eval.

### 1.8 T1 (Isaac Lab, experimental)
- dream/robot/t1.py (100), dream/sim/isaac_t1.py (309), dream/navigation/control.py (161: NavigationRequest,
  LocalWaypoint, WaypointTracker). Importers: experiments/isaac_t1_walk.py and inspect_t1_stand.py only (control.py is
  imported by dream only via sim/isaac_t1). results/t1_* dirs are cloud failures (preallocation, IAM, init, stand
  target/state) and local stand follow-ups; README has no T1 capability row; isaac_t1.py docstring: "no claim of
  walking validation". Candidate to archive with the two experiments; keep result dirs (failed evidence, Invariant 7).

## 2. Perception
Rendered sensor arrays: sim/camera.py (152) RGBDCamera returns colour + depth with depth noise
sigma(z) = 1 mm + 1.5 mm/m^2 z^2 and a robot self-mask from the robot's own geometry; "No segmentation, geom ids or
body poses leave this module". Used by tabletop_env, g1_mobile, isaac_g1, kinematic_base. Oracle channels: section 6.

| Component | Input | Used by | Validated |
|---|---|---|---|
| detect.py (538) classical RGB-D box/container detector | rgbd frame | vision.py, spatial.py, sphere.py, eval.vision | camera Panda 39/40 (README); compartment camera 11/20 FAIL |
| world.py (184) WorldModel tracks | detections | only vision.py | via VisionPerception |
| vision.py (208) VisionPerception | 1-2 fixed cameras | session.py, run_vision, run_compartment, run_vlm | as above |
| sphere.py (118) red sphere | frame | tools/object_exploration (supported G1) | commissioning only |
| occupancy.py (89) log-odds grid | head RGB-D in odometry frame | spatial, planning, semantics.*, tools/mobile, memory.relocalise | room search, navigation results |
| spatial.py (194) occupancy + colour-keyed furniture Candidates | frames, detect.py | tools/mobile.py:29, 16 experiments | Milestone B, inspection, g1_navigation_* |
| semantic.py (285) SemanticMap Tracks + coverage | frame.detections | semantics.world/belief/rooms/verify, runtime.semantic_search | H1-H4, V1 (README) |
| vlm_detect.py (116) VLMDetector (Qwen3-VL boxes -> masks) | rgb | box_detectors, isaac_home_runs, isaac_semantic(_dev), detector_compare | Isaac semantic: gates INCOMPLETE/FAIL |
| box_detectors.py (434) Qwen/OWLv2/GroundingDINO Grounders + ModelDetector | rgb | only experiments/detector_compare.py + 1 test | dev comparison (results/detector_compare_stub_dev) |
| sim/semantic_sensor.py (210) SemanticCamera | TRUE instance segmentation + assumed noise | kinematic_base, semantic_search_eval | assumed noise, labelled oracle |
Findings:
- P1 Three object-state layers: SpatialMemory.Candidate (spatial.py:45), SemanticMap.Track (semantic.py) and, on the
  Panda, WorldModel.Track (world.py). `Track` is defined in both semantic.py and world.py with different fields.
  MobileToolbox holds Candidates and, when semantic is enabled, a SemanticWorld (mobile.py:162). Keys differ
  (colour, label, entity class), so no merge; rename one Track (neutral).
- P2 Two Qwen box paths: vlm_detect.VLMDetector vs box_detectors.QwenGrounder + ModelDetector (box_detectors.py:180),
  same backend, same vocabulary, same output type. Only detector_compare uses box_detectors; the OWLv2 and DINO
  adapters have no other caller. Move box_detectors.py to experiments/ (neutral). Re-pointing the Isaac runs at it
  would be FUNC.
- P3 spatial.py's "saturated colour = furniture, grey/white = structure" heuristic is scene-specific (its docstring
  says so); keep it labelled.
- P4 geometry.py and occupancy.py are genuinely shared; no cleanup.

## 3. Navigation
- planning.py (128): traversable (dilated free), distances (8-connected Dijkstra; one call prices all goals), path_to,
  frontiers (information-gain proxy ignoring occlusion). No A*. Callers: tools/mobile.py (228, 300, 559, 618),
  runtime/semantic_search.py:266, experiments audit_mobile_inspection and isaac_task.
- follow.py PathFollower (66): "classic" pure-pursuit, the MobileToolbox default (mobile.py:155, navigator="classic").
  rpp.py (211): Nav2 Regulated Pure Pursuit plus line_of_sight, shortcut, densify, smooth; mobile._go_to_rpp:652.
- Who uses which: 15+ experiments and tests pass navigator="rpp" (g1_command_demo, interruption, memory demo,
  isaac_g1_live, isaac_home_runs, phase1_interface, closeout tests). Every frozen semantic-search config uses
  "navigator": "classic" (semantic_exploration_dev.json:12, semantic_search_heldout.json:15 and others) and so does
  the g1_navigation_eval A/B. README: "RPP vs classic: v1 25/30 vs 29/30; v2 28/30 each, PARTIAL, no superiority".
  Both are live; removing "classic" is FUNC (invalidates frozen search results and the A/B). Minimal action: keep,
  document in mobile.py why classic remains.
- Duplicates: _go_to (classic, mobile.py:535) and _go_to_rpp (652) repeat the timeout/blocked/stop prologue; extract
  shared helper only if closeout tests stay byte-identical (neutral if so). line_of_sight in sim/home.py:732 (truth
  grid) vs navigation/rpp.py:33 (perceived grid): different data, rename for clarity.
- navigation/control.py: T1-only (1.8); navigation/__init__.py docstring describes it, so it also goes with 1.8.

## 4. Skills
- skills/manipulation.py (291): pick, place, move_via_travel_height, close_on, etc. over Embodiment + Perception
  protocols. Callers: tools/toolbox.py, tools/discovery.py, skills/articulation.py; experiments run_pick_place,
  run_tasks, stress_shoulder_singularity. skills/articulation.py (292): grasp_handle, probe, pull_along,
  release_handle; callers tools/discovery, tools/toolbox, memory/experience, compartment experiments.
- Validated: pick-place oracle 50/50, camera 39/40 (README), tasks 40/40 agent cells, results/
  shoulder_singularity_stress; compartment oracle 54/60 + 30/30 fixed declined; camera 11/20 FAIL.
- No G1 skill exists in dream/skills. G1 reach/grasp/hold phases are inline in experiments/g1_grasp_bringup.py
  (around lines 194-221, 614-623, 795, 1077). The stack behind the R1 claims is not reusable by the runtime. No
  cleanup now; extracting a typed G1 tool for R2 is new work and needs the isolation rule to allow an adapter.
- No pure-forwarding wrappers in skills. Embodiment/Perception Protocols (types.py) have 2-3 implementations.

## 5. Wrappers, single-implementation abstractions, redundant state, dead or one-off code
| # | Item | Evidence | Cleanup | FUNC? |
|---|---|---|---|---|
| W1 | g1_rollout.py + g1_correction.py | importer table | archive or delete with their test cases | no |
| W2 | g1_assisted_demo.py | only isaac_task.py:148; constraint never added | remove | no |
| W3 | box_detectors.py OWL/DINO adapters | detector_compare only | move to experiments | no |
| W4 | T1 chain: robot/t1, sim/isaac_t1, navigation/control | T1 experiments only | archive | no |
| W5 | sim/semantic_sensor.SemanticG1Base (185-210) | forwards time, stop_reason, odometry, walk to base, adds detections | fold into G1MobileBase option or keep | no if frames identical |
| W6 | G1MobileBase (g1_mobile.py:450) | forwards few members but owns the walk loop; MobileBase Protocol impl | keep | n/a |
| W7 | wholebody.ControllerCapabilities.from_envelope_json vs from_envelope_v2 (wholebody.py:151, 171) | two envelope formats | not verified which callers remain; drop v1 only if unused | no |
| W8 | semantics/graph.py (107) vs memory/scene_graph.py (Spark-DSG) | two scene graphs, distinct homes | document, no merge | no |
| W9 | home.py truth readers | 1.7 | move to eval | no |
| W10 | sim/room.py vs sim/home.py | 1.7 | keep both | n/a |
Modules with no dream/ importer: robot/g1_rollout, g1_correction (via rollout only), g1_assisted_demo,
g1_grasp_geometry (experiment only), joint, t1; sim/room; perception/box_detectors. Of these only rollout,
correction, assisted_demo and the T1 chain lack an evidence-backed caller.

## 6. Oracle vs estimated: what decision code receives from an oracle

### 6.1 Panda tabletop and compartment (oracle arm)
- observe_objects (tabletop_env.py:155): TRUE body position + Gaussian noise, TRUE yaw + noise, size = exact
  2*half_size, color = obj.name (identity given). No occlusion, misses or false positives; every spec object visible.
- tray_region / fixtures (172-190): exact tray geometry (centre, inner half width), noise-free. Compartment cavity
  (_visible_cavity:232) is ray-cast against TRUE geometry, robot geometry ignored.
- observe_articulated (192-215): TRUE handle-site pose + noise, TRUE compartment AABB, exact handle_half.
- robot_state: proprioception (legitimate). The camera arm (VisionPerception) produces the same types from rendered
  RGB-D (vision.py docstring).
- Not oracle: mass, friction, joint type (the discovery targets); gripper-object contact is real physics.

### 6.2 Mobile search and inspection (G1, kinematic base, Isaac)
- Odometry: ORACLE-ASSISTED true increments with multiplicative noise and gyro bias (g1_mobile.py:421-437;
  kinematic_base.py:265; isaac_g1.py:495). No drift correction in the map (occupancy.py docstring). Camera
  calibration is composed from estimated pelvis pose (g1_mobile.py:372-396); true_pose (440) is evaluation only.
- Occupancy, frontiers, path following: estimated from rendered depth.
- Furniture candidates: classical detect.py on rendered RGB-D plus the colour heuristic (spatial.py).
- Semantic search H1-H3 (kinematic base): frame.detections are SemanticCamera output: TRUE instance segmentation of
  scene geoms (robot excluded) with ASSUMED miss / confusion / spurious noise (semantic_sensor.py:36,
  SOURCE="oracle_segmentation_noisy"). Room/object knowledge: ConceptNet extract, not the generator (knowledge.py).
- G1 H4 and Isaac Qwen runs: detections from VLM (vlm_detect.VLMDetector) on rendered RGB, synonyms to classes.
- Evaluator-only: true_pose, contact_log (kinematic_base.py:227), GroundTruthView. Import isolation enforced by tests.

### 6.3 G1 grasp (R1), experiments/g1_grasp_bringup.py
- Object pose, table geometry, starting pose and localisation: oracle-assisted (header line 1; Reports/2026-11-10/
  003 section C; README R1 text). Odometry: oracle with noise.
- Hand position: FK from the robot's own sensors (g1_correction docstring). Obstacles are PerceivedBox objects the
  runner supplies (g1_posture.PerceivedBox), not rendered.
- Sensed hold: tactile opposition + encoder travel + oracle-assisted object observation; the independent gate is
  true lowest-collider clearance (dream/eval/g1.py:9). Contact is real; no equality constraints.
- Assumed priors: shared 0.10 kg mass for rehearsal; configured object friction override in material tests.

### 6.4 Gantry
- Noisy sensor observations only; handle position sampled independently of the joint; truth only for build/scoring.

## 7. Minimal cleanup plan (neutral unless marked FUNC)
1. Archive or delete g1_rollout.py, g1_correction.py and their test cases (543 lines); update AGENTS section 14.
2. Delete g1_assisted_demo.py and the RightArm path in isaac_task.py (attachment constraint vs Invariant 2).
3. Move box_detectors.py (and its test) to experiments/; keep vlm_detect.
4. Archive the T1 chain (robot/t1, sim/isaac_t1, navigation/control, 2 experiments, tests); keep results/t1_*.
5. Move truth-reading parts of sim/home.py to dream/eval.
6. One wrap()/yaw helper; rename the duplicate `Track` class; test that Isaac and MuJoCo G1 constants agree.
7. Document rather than change: classic vs RPP split (removal is FUNC), room vs home generators, the kinematic
   base as the validated search embodiment, the experiment-only G1 grasp stack.
Gate for 1-6: pytest tests/ -q plus the isolation test; no result numbers should move.
