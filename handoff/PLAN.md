# DREAM whole-body autonomy: audit, decisions, migration plan (2026-10-26)

Branch `claude/wholebody-autonomy`, cut from `claude/isaac-g1-kitchen@d582159`, in worktree
`C:\Users\24adi\DREAM-wholebody` (outside OneDrive). The original checkout
`C:\Users\24adi\OneDrive\DREAM` is in use by another live session (it committed d582159 during
this audit and owns the running VM `dream-qwen-isaac-20261026-131049-a164c9`): never switch its
branch, edit it, or touch that VM/bucket.

## 1. Audit summary (evidence from five read-only audits)

Branches: `claude/isaac-g1-kitchen` contains every other branch (admiring-carson, foundation-models,
codex/g1-dex3-physics, research/embodied-vla-isaac-demo, fresh-adequacy-validation). Nothing must be
cherry-picked. All branches are preserved untouched.

Validated, reusable engineering (KEEP):
- NVIDIA Decoupled WBC Balance/Walk ONNX adapter `dream/robot/g1_wbc.py` (contract copied from
  upstream, tests/test_g1_wbc.py; results/g1_locomotion: 0 falls, 30/30 go-to, 120 N pushes).
- Mobile G1 env `dream/sim/g1_mobile.py` (IMU+encoder proprioception, head RGB-D in odometry frame).
- Isaac Sim 6.2 G1 bring-up (results/isaac_g1_kitchen_bringup), cloud tooling with deletion deadlines,
  ledger and caps (cloud/isaac_session.py, cloud/launch.py).
- Navigation (classic + Nav2-RPP), occupancy, SpatialMemory, HomeMemory (SQLite), relocalise,
  Spark-DSG scene graph, TaskScheduler/phone API, experience memory.
- Active Physical Discovery (dream/agent belief/EIG/operate, dream/tools/discovery.py) - Panda-validated.
- Typed interfaces dream/types.py (Embodiment, DexterousEmbodiment, Perception, MobileBase,
  ObjectObservation), Toolbox/ToolResult, VLMAgent safety wrappers.

Blocking generalisation (REPLACE or ISOLATE as labelled baselines):
- `experiments/isaac_task.py::run_task`: per-task branches (`open_cabinet`/`fetch_banana`), per-object
  STAND_OFF table, target-height gate 0.55-1.15 m ("no crouch"), fixed 8 cm pre-grasp, walk-back-for-N-s
  opening, +0.15 m lift, carry to origin, success read from USD inside the control flow.
- Oracle perception in the task path: USD semantic labels + instance segmentation -> target (invariant 1).
- Assisted grasps: Isaac UsdPhysics.FixedJoint (isaac_task.py:114) and MuJoCo mjEQ_CONNECT
  (g1_reach.py) (invariant 2).
- Upright-only body: g1_wbc.py HEIGHT_CMD=0.74 constant and torso rpy command = 0; right-arm-only,
  position-only IK with no balance, collision or torque check (g1_reach.py).
- Oracle-assisted odometry and true start pose in odom->world conversion (labelled; replace later).
- VLM runtime is Panda-only; G1 tasks never go through reasoning.

Key enabling facts:
- The pretrained Decoupled WBC already takes base-height and torso roll/pitch/yaw command inputs
  (upstream g1_gear_wbc_policy.py obs[3], obs[4:7]); upstream teleop lowers height 0.74 -> 0.2 m clip;
  torso rpy command = FK of torso_link relative to the yaw-only pelvis frame. Trained range UNVERIFIED.
- g1_gear_wbc.xml IS Unitree g1_29dof_with_hand_rev_1_0 with the 14 Dex3 finger joints commented
  out; hand links (0.781 kg/wrist subtree) are already present and welded; masses/inertias identical.
- Mink 1.3.0 (Apache-2.0, py>=3.10, mujoco>=3.10) has a G1 whole-body example (frame tasks on
  pelvis/torso/feet/palms, CoM task, posture, configuration and collision limits). Not installed.
- GEAR-SONIC: 50 Hz motion-token controller + 10 Hz kinematic planner (height for squat/kneel),
  C++/TensorRT deployment, MuJoCo sim env exists. GR00T N1.7+SONIC: 64-d token + 2x7 hand joints,
  no released NVIDIA G1 checkpoint -> needs fine-tuning. Isaac Lab Arena G1 loco-manip: GR00T N1.6
  checkpoint `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`, Isaac Lab + Docker + GPU.
- Cloud: cap US$100 (config), ~US$20 reconciled + one live session (~US$5.7 est). L4 quota 8.
  One orphan bucket `dream-qwen-20261024-161406-11ebe6` (41 MB code tarball). Laptop C: 14 GB free.

## 2. Engineering decisions

D1 Branch/worktree: as above. Subagents commit only the paths they own (`git add <paths>`), chain
   `ruff check && pytest <focused> && git commit`. No pushes until a milestone checkpoint.
D2 Whole-body controller: KEEP NVIDIA Decoupled WBC as the balance/locomotion/posture layer (it is the
   only validated pretrained controller we run). Promote its height and torso-rpy observations from
   constants to commanded inputs, exactly per the upstream contract. Commands outside the safety bound
   are REJECTED, never clipped. The usable envelope is MEASURED in physics (D3), not assumed.
   GEAR-SONIC is the planned upgrade for kneeling/deep squats and VLA coupling; adopted only after its
   official MuJoCo path is reproduced (Milestone A2).
D3 Envelope characterisation (Milestone A, extension of the reproduced baseline): sweep height command
   0.74..0.30 m and torso pitch/roll/yaw commands, standing and with arms extended, 5 seeds; measure
   achieved pelvis height, torso tracking, falls, torque saturation ratio, CoM margin over the support
   polygon, settle time. Output: a frozen calibration artifact (`results/g1_wbc_envelope`) that the
   posture planner reads as bounds. The 0.2-0.74 m upstream teleop clip is the hard safety bound.
D4 Unified embodiment (Milestone C prerequisite): one G1 = g1_gear_wbc.xml + the 14 Dex3 finger
   hinges re-enabled at build time via MjSpec (values from third_party/unitree_g1
   g1_29dof_with_hand_rev_1_0.xml), 14 position-servo finger actuators (kp 2, kd 0.06, Unitree
   forceranges; as commissioned in dream/sim/g1.py). Body inertias untouched (test asserts identity).
   The WBC still owns 29 body motors. Gate: locomotion regression matches results/g1_locomotion within
   noise; fingers stable over full range at 0.005 s. Any integrator/armature change is escalated.
D5 Whole-body posture planner (Milestone B): `dream/robot/g1_posture.py` using Mink on the ROBOT-ONLY
   MJCF (robot self-knowledge, not scene truth). Decision variables: stance (x, y, yaw) sampled over
   perceived free space around the target, then full-configuration IK: hand frame task (position,
   optional approach axis), both feet fixed at the stance, CoM task over the support polygon, posture
   regulariser, configuration limits, self-collision limit, and collision limits against PERCEIVED
   obstacle boxes only (from RGB-D), never scene geoms. Candidates scored by IK residual, CoM margin,
   joint-limit margin, gravity-torque ratio (mj_rnea on the IK pose vs actuator limits), distance from
   nominal, and walking cost; either arm is a candidate. No height rules anywhere: crouch/lean/arm
   choice emerge from the optimisation. The chosen configuration is projected onto controller commands:
   pelvis height, torso rpy (upstream FK definition), 14 arm joint targets; bounded by the D3 envelope.
D6 Physical feasibility and safety supervisor `dream/robot/g1_supervisor.py`: rate-limited command
   trajectories; joint position/velocity/torque limits; tilt and tilt-rate abort; contact-force limit;
   protective stop returns to the nominal posture. Every executed posture is verified in physics; IK
   success alone never counts. Reports reach error, support contacts, CoM trace, torques, tracking,
   stops. If execution fails, the planner tries the next candidate (re-stance, other arm), then
   declares the target unreachable with the measured reason.
D7 Reach skill & first vertical slice: `reach(target_observation)` typed tool on a new G1 toolbox:
   perceive target with head RGB-D (existing classical detector for this slice; labelled), walk with
   the existing navigator to the chosen stance, execute the posture, re-observe, verify by
   proprioception + vision. The language instruction is parsed into a typed goal (structured parser now,
   VLM tool choice later through the same Toolbox).
D8 Unfamiliar-height acceptance test (owner's non-negotiable first test): MuJoCo scenes with a handle on
   a fixture at heights the system never saw. Splits: development heights {0.35, 0.75, 1.10} m;
   held-out heights drawn from [0.10, 1.45] m plus lateral/depth offsets, disjoint seeds, frozen gates
   before running. Independent evaluator (truth only in dream/eval): fingertip-handle distance, falls,
   protective stops, torque peaks, unintended contacts, correct refusals for infeasible targets.
   Baseline: the current upright right-arm reach (labelled scripted baseline) on identical scenes.
D9 Retirement: `experiments/isaac_task.py` becomes a labelled `scripted_baseline`; assisted grasp only
   behind an explicit `--assisted-grasp` flag that stamps every result. Removed from the new path.
   Isolation test extended to every new decision module.
D10 Learned VLA (Milestone D): no learned output reaches actuators before (a) the upstream baseline is
   reproduced in its own environment and (b) the action contract is verified. Order: GEAR-SONIC MuJoCo
   reproduction -> Arena G1 checkpoint reproduction (cloud, Isaac Lab) -> GR00T N1.7+SONIC fine-tune on
   correctly mapped DREAM demonstrations (collected with the D5/D6 stack + Dex3).
D11 Perception (Milestone E): open-vocabulary detector/segmenter (SAM 3 or OWLv2/GroundingDINO) as a
   VisionPerception backend producing ObjectObservation with depth-derived 3D + covariance; USD labels
   stay in dream/eval for scoring only.
D12 Grasping (Milestone C): contact-only Dex3 grasps (no attachment). Classical baseline first:
   geometry-derived candidates from the perceived point cloud, finger-closure until contact/torque,
   verify by lift + slip + contact persistence. GraspGen-class learned candidates evaluated after.
D13 Discovery (Milestone F): port dream/tools/discovery.py probes to the G1 hand via the D5/D6 stack
   with force limits, so "open the cupboard" investigates slide/swing/fixed instead of walking back.
D14 Cloud: within the standing authorisation and the US$100 config cap, US$6/session ceiling; only one
   DREAM session at a time while the other session's VM is live; any new spend >US$6 or cap change is
   proposed to the owner first.

D15 W6 continuation decision (overseer, 27 Oct): First perform an independent read-only audit of
   W6 and its partial outputs. Verify and fix the metadata/scorer semantic and provenance mismatch
   before trusting or resuming protocol v2; the handoff reports a v1 success definition in v2 metadata,
   but its cause and scoring implications remain unverified. Preserve existing partial outputs and
   never silently overwrite them. Resolve the reported reference (i) low-height infeasibility before
   trusting its labels. Complete the development agent, scripted baseline and truth-input diagnostic
   labels before freeze review. Do not run fresh held-out seeds 6000-6023 without the overseer's explicit
   freeze and go. These are decision gates, not findings from the pending audit.

## 2b. Owner review corrections (binding; supersede anything above that conflicts)

C1 Input bound vs execution envelope. [0.20, 0.74] m is only the controller's INPUT bound (reject
   outside). The planner may request only postures inside the physically MEASURED stable region from
   W1 (results/g1_wbc_envelope/envelope.json), reported with trial counts and uncertainty. Unmeasured
   regions are not executable autonomously.
C2 Explicit 29+14 actuator bridge. The WBC 29-joint observation/output contract stays byte-exact.
   G1MobileEnv maps body joints and the 14 finger actuators BY JOINT NAME into index arrays; body torques
   are written only to body actuator indices and finger commands only to finger indices (no
   `data.ctrl[:] = ...` full-array writes; the `model.nu != 29` assertion becomes an explicit name check).
C3 No pelvis-height fall criterion for posture work. Collapse vs intended crouch is decided from:
   tilt latch, undesired ground contact (any non-foot robot geom touching the floor), loss of foot support,
   tracking error vs command, joint-limit contact, CoM margin. experiments/g1_locomotion.py stays unchanged
   for its own validated walking results; new evaluators use the new criterion.
C4 Protective stop = HOLD, not stand-up. On violation: freeze posture/arm references at the current
   measured state, zero velocity commands, keep the balance policy running (damping-only only after a
   real fall), and report. Recovery to nominal is a separate, explicit action, permitted only when the
   support state is validated (both feet in contact, CoM margin > 0, tilt and tilt rate small) and executed
   by reversing the recorded trajectory at the safe rate.
C5 IK proposes, physics disposes. Mink IK and mj_rnea gravity torques only RANK candidates. The
   feasibility gate is a contact-aware rollout: the controller executing the plan on a planning copy of
   the robot model (robot + floor + PERCEIVED obstacle boxes only; no scene truth), measuring torque,
   foot force/slip, CoM margin, tilt and hand error; then the real execution, monitored by the supervisor.
   Only measured real execution counts as success.
C6 One frozen protocol. Envelope: 5 seeds per cell, plus repeat trials, transitions between postures,
   and push disturbances at low postures; report counts and Wilson 95% intervals, never "safe" from a few
   zero-fall trials. Reach test reports feasible-reach success and correct-refusal results separately.
C7 Controller-independent capability interface (`dream/robot/wholebody.py`): planners emit a
   `WholeBodyTarget` (full configuration q + task-space goals: hand poses, pelvis height/torso as
   derived quantities). Backends (`WBCBackend` now; `SonicBackend` later, consuming joint-position
   reference frames) expose `capabilities()` (supported request kinds, measured limits, rates) and
   `execute(target) -> ExecutionReport` (achieved vs requested, stops, measurements). Posture/task code
   never references WBC height/rpy constants directly.
C8 Naming. W5 is a GEOMETRIC AND CONTROL milestone with a structured target; it is not language-
   conditioned autonomy. Full autonomy requires the VLM choosing goals/capabilities, a learned policy
   generating movement and physical recovery without task routines (Milestones D-F). No collection of
   classical skills is presented as a language-conditioned humanoid.
C9 Integration gating. Parallel work builds interfaces; integration (W5) starts only after W1's envelope
   and W2's 43-actuator contract pass their gates.

## 3. Execution order (vertical slices)

W1-W4: initial command-channel/envelope, unified embodiment, planner/supervisor and SONIC work
are recorded in `handoff/CODEX_HANDOFF.md`; its claims remain subject to independent verification.
W5: the v1 unfamiliar-height reach evaluation ran; preserve its frozen record and do not rescore it.
W6: step 5 development runs are complete; development review and diagnostics continue under D19.
Protocol v2 is not frozen. Decisions D15-D18 and their historical evidence remain recorded below.
W7: contact-only Dex3 grasp on the unified hand (D12), after the required reach work.
W8: retire `experiments/isaac_task.py` to a labelled scripted baseline (D9); update README status
with owner approval as specified in the handoff.

Operational constraints for continuation: local free space was observed at about 14 GB; check before
each batch and pause below 2 GB. Run at most two MuJoCo processes concurrently. Redirect background
output to bounded logs, print progress lines only and never print arrays in loops. No cloud allocation
has been made for this continuation; apply D14 and the repository's cloud safeguards before any launch.

## W6 continuation decisions

**D16 — scoring and provenance.** The evaluator actually uses contact mode, with 0.3 s
unbroken hand-handle contact and a 25 N peak-force limit. The old distance description in
`success_definition` is stale descriptive metadata, not the scoring path. Preserve the original
14 agent episodes, 14 reference records and original metadata unchanged. Continue in
`results/g1_reach_heights_v2_dev_continuation`, reusing those episodes only through a manifest
recording source and destination SHA-256 checksums, original metadata and configuration hash,
and why scoring remains unchanged. The runner generates new launch metadata normally. Reference
(i) and (ii) are finite-search operational labels, not proof of impossibility; report their
disagreements and incomplete coverage. The scripted baseline uses a different success-claim
rule from the agent despite the shared evaluator and sensors; disclose this in comparisons.
For label (ii), truth-input diagnostics on every agent physical failure plus agent physical
successes cover all 60 scenes by the implemented union; assert coverage before reporting.

**D17 — execution.** Run agent, scripted baseline, then truth-input diagnostics on agent
physical failures, as sequential batches, with at most two workers and no simultaneous runners
sharing the reference cache. Do not change imported code during any batch. Check free disk before
each batch and stop below 2 GB; use bounded progress logs without looped array output. Complete
all 60 development scenes and review evidence before any freeze decision. Do not run fresh
held-out seeds 6000–6023 before a separate freeze and explicit go.


**D18 - strict height capability validation.** A development incident reached the WBC setter with
height `0.7400000107504702`, outside its strict 0.74 m upper bound. Capability validation had
a 1e-6 m tolerance, admitting that value before the setter rejected it. The minimal correction
uses strict measured height bounds in capability validation; the setter and physical limits stay
unchanged, and no planner interior margin is introduced without evidence it is needed. Class:
General algorithmic correction. Candidate selection can change, so the earlier continuation remains
development diagnostic evidence. Rerun all 60 development agent and baseline episodes and the
truth-input diagnostics for agent physical failures in `results/g1_reach_heights_v2_dev_strict_height`,
without copying pre-fix episode or reference caches. Review the results before freeze; D17's
sequential execution and held-out gate remain binding.


**D19 - development review decision.** W6 step 5 completed at `5f9add2` with 60 agent,
60 baseline and 29 truth-input diagnostic episodes, and full truth-input union coverage. Preserve
`results/g1_reach_heights_v2_dev_strict_height/summary.json`, `attribution.json`, and
`development_review.json` with the underlying results. The off-band success and correct-refusal
gates failed, and the slow correction test remains unresolved; there is no freeze or held-out go.

First diagnose the measured rollback/error contract: the slow test reports 24.2 mm before and
38.1 mm after an undo reported complete, which does not establish restoration of measured state.
Its pre-existing status is not proven. Then investigate finite-reference disagreements for
`dev_v1ho_s5012`, `dev_h1.10_L1_s101`, and `dev_v1ho_s5023`, and refusal failures; then investigate
the tactile/evaluator claim mismatch. Treat these as diagnostics until evidence supports a testable
hypothesis and regression before any correction. Do not advance to W7 before this review. Source
changes require a later rerun policy decided case by case. W5 and W6 use different success definitions;
do not claim same-metric improvement from their comparison. Preserve fresh held-out seeds 6000-6023
untouched until a separate freeze and explicit go.
