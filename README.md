# DREAM

**Long-term mission.** DREAM is a research project toward general-purpose embodied artificial
intelligence: a system that receives an unfamiliar physical goal, works out from what it observes
and what it can physically do what needs to happen, acquires missing knowledge by observation and
experiment, adapts to failure, and completes the task or recognises that it cannot, without a
task-specific program per goal. This is the direction, **not a current capability**, and
generality is something we measure, not assume. The behavioural contract, anti-overfitting
rules, architecture and engineering standards are in [`AGENTS.md`](AGENTS.md).

**What exists today: two foundations, now connected in one narrow task.**

1. **Active Physical Discovery** (abstract gantry robot). A simulated robot meets a mechanism it
   has never seen, a panel with a handle that might be fixed, slide, or swing on a hinge. It
   keeps competing explanations of how the panel moves, picks the physical experiment expected
   to best tell them apart, predicts what that experiment will do, runs it, updates its beliefs
   from what it measures, and then opens the mechanism.
2. **A Panda manipulation foundation.**
   - A simulated Franka Panda (MuJoCo Menagerie model) grasps cubes through real finger contact.
   - Closed-loop skills, a typed tool interface and a rule-based task runtime carry out
     *structured* goals: "in red tray", or "on red blue" (stack red on blue).
   - It recovers from dropped objects, clears obstructions, and runs as a live session that can
     be paused or cancelled.
   - There is no natural-language understanding and no memory yet.
3. **Discovery as a tool on the Panda (milestone 11).** For "in red compartment", the Panda
   probes a closed box whose mechanism is hidden (drawer, lid or fixed) through its own compliant
   controller, using the same belief and EIG code as the gantry. It then opens the box along the
   learned constraint or declines, places the cube and verifies. This runs in one continuous
   simulation, on a deliberately narrowed scene (knob handle, 5 cm cavity, counterbalanced lid):
   see the [audit of simplifications](#audit-of-milestone-11-changes).
4. **Camera perception (milestone 10, first slice).** A calibrated RGB-D camera and a classical
   RGB-D pipeline replace the object tracker for tabletop pick-and-place: the Panda finds cubes
   and the tray in images, grasps and places from those estimates, and verifies from new frames
   (held-out 39/40 against the oracle tracker's 40/40). It is a colour-and-depth baseline for
   box-like objects, not open-world vision. Camera-driven *mechanism discovery* remains
   unreliable: the original development run was 0/12; the continuation diagnostic completed
   both lids but neither drawer on two new development seeds. Held-out compartment results
   still use the oracle tracker.
5. **Foundation-model integration: interface only, no model run yet.** A model-driven agent
   (`dream/runtime/vlm.py`) lets a vision-language model (selected: Qwen3-VL-4B-Instruct) choose
   DREAM's typed tools from a natural-language instruction and the camera image. Its contracts
   are tested with a scripted test double. **No pretrained model has been executed in this
   repository**: the current Windows host can reach model hosts, but has insufficient free
   RAM for the selected full planner checkpoint and no CUDA GPU (see
   [Foundation models](#foundation-models-dreamruntimevlmpy-resultsfoundation_model_audit)).

![Panda: red cube into the tray, then blue stacked on green](results/demo/panda_session/replay.gif)

The discovery loop:

```
observe -> hypothesise -> predict -> act (probe) -> measure -> update -> ... -> operate
```

Discovery is explicit Bayesian system identification with information-seeking probes in MuJoCo,
with no learned components (Stage Zero and One of the Physics Apprentice research plan). In the
long-term architecture it becomes a tool that the task runtime calls when a skill meets
unfamiliar physical behaviour.

## Status

Every row marked "implemented and tested" has passing tests and/or saved results in `results/`.

Last full verification (23 Oct 2026, code `21969bf`, branch
`codex/camera-discovery-continuation`): **163 passed** in 411 s, including slow closed-loop,
compartment, vision and live-session tests; Ruff lint/format passed. The fast suite passed
142 tests in 71 s on this Windows host, exceeding the 60 s target. Python 3.13.14,
MuJoCo 3.16.0, NumPy 2.5.4, GLFW; CPU control with Intel graphics for rendering. The full
suite ran alongside the development diagnostic. XML and environment metadata are saved in
`results/continuation_validation`. Earlier Linux verification was 132 tests in about 6 min.

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | MuJoCo mechanism scenes plus force-limited gantry probing | Implemented and tested | `tests/test_sim.py` |
| 2 | Sensor-only observations, ground-truth isolation, episode logging with git SHA and config | Implemented and tested | `tests/test_isolation.py`, `results/*/meta.json`. Raw observations and MJCF are logged since `results/main_geometric_prior`; `main` and `generalisation` predate this. |
| 3 | Bayesian belief over fixed / prismatic / revolute mechanisms | Implemented and tested | `tests/test_belief.py` |
| 4 | Predictive model validated against MuJoCo; held-out prediction error | Implemented and tested | `tests/test_model.py`, `results/main` |
| 5 | EIG probe selection vs random, goal-directed and scripted baselines | Implemented and tested | `results/main`, `results/main_geometric_prior` |
| 6 | Generalisation (unseen parameters and families), out-of-class detection | Implemented and tested | `results/generalisation`, `results/adequacy_calibration` |
| 6a | Minimum-evidence (3-probe) gate for adequacy false alarms | Implemented and tested on fresh held-out seed; adopted in `Belief.fit_check` | `results/fresh_adequacy_validation`, `tests/test_adequacy.py` |
| 6b | Demo GIF and narration / live viewer | Tested / implemented, unverified (no display) | `results/demo`, `experiments/demo.py --viewer` |
| 7 | Simulated Panda arm, safety-limited controller, contact grasping, reusable skills | Implemented and tested | `tests/test_panda.py`, `results/pick_place` |
| 8 | Typed tool interface and task runtime (structured goals, rule-based planner) | Implemented and tested | `tests/test_tools_runtime.py`, `results/tasks` |
| 9 | Two goal types (tray, stack) in shared scenes from the same tools, without per-task scripts | Implemented and tested | `results/tasks` |
| 9a | Live session: background physics, pause, resume, cancel, logs, replay | Implemented and tested (headless); live viewer implemented, unverified (no display) | `tests/test_session.py` |
| 10 | Visual perception replacing oracle poses | **Tabletop pick-and-place: implemented and tested** (classical RGB-D baseline). Camera discovery: partial development success, **not held-out validated** | `tests/test_vision.py`, `results/vision`, `results/compartment_vision_dev`, `results/compartment_vision_continuation_dev` |
| 11 | **Discovery invoked on demand by the runtime (unfamiliar compartment on the Panda)** | **In progress**: implemented and tested on the narrowed scene; held-out evaluation below; challenge variants open | `tests/test_compartment.py`, `results/compartment`, audit table |
| 12 | Experience memory, failure-driven replanning, skill adaptation | Planned | n/a |
| 12a | Vision-language planner (Qwen3-VL-4B-Instruct) choosing typed tools | Interface implemented and contract-tested with test doubles. **Model inference not run**; caller-bound success contract, atomic call rejection and cancellation checks | `tests/test_vlm.py`; hardware audit below |
| 12b | Learned motor policy (GR00T N1.7-LIBERO) | **Feasibility inspected; adapter not implemented**. Checkpoint/configs reachable; missing wrist camera, rotation/control mapping and CUDA inference hardware | Hardware and source-contract audit below |
| 13 | Generalisation evaluation across goals, arrangements and task families | Planned | n/a |
| 14 | Learned world/motor models, mobile or whole-body embodiments | Planned, conditional on evidence and compute | n/a |

## Install

Python >= 3.10 (developed on 3.13). CPU only.

```bash
pip install -e ".[dev]"            # mujoco==3.16.0, numpy, matplotlib, pytest, ruff
sudo apt-get install libosmesa6    # only for headless rendering of the demo GIF
```

On Windows, use GLFW instead of Linux OSMesa. In PowerShell, set
`$env:MUJOCO_GL='glfw'; $env:PYOPENGL_PLATFORM='win32'` before running tests or experiments.
The continuation uses an isolated `.venv`; its interpreter is `.\.venv\Scripts\python`.

## Run

```bash
pytest -m "not slow"                                   # fast suite (~55 s)
pytest                                                 # + slow closed-loop, vision, session tests (~6 min)

python experiments/demo.py --seed 3                    # hidden mechanism drawn from the seed
python experiments/demo.py --family left_hinge         # or pick the hidden family
python experiments/demo.py --family slider --viewer    # live MuJoCo viewer (needs a display)

python experiments/run_benchmark.py experiments/configs/main.json --workers 4            # ~15 min
python experiments/calibrate_adequacy.py --workers 4                                    # ~5 min
python experiments/run_benchmark.py experiments/configs/generalisation.json --workers 4  # ~20 min
python experiments/run_benchmark.py experiments/configs/main_geometric_prior.json --workers 4
python experiments/validate_adequacy.py --workers 4   # fresh held-out adequacy validation (~12 min)
python experiments/summarise.py results/main          # re-summarise saved episodes

# Panda manipulation
python experiments/run_pick_place.py --n 50 --seed0 1000              # acceptance test (~1 min)
python experiments/run_tasks.py experiments/configs/tasks.json         # task evaluation (~10 min)
python experiments/session.py --seed 0 [--viewer] [--rate 1]           # interactive live session
python experiments/session.py --perception vision                      # ... seeing through the camera
python experiments/run_vision.py experiments/configs/vision.json       # vision evaluation (~40 min)
python experiments/demo_vision.py                                      # recorded camera-based demo
python experiments/run_compartment.py experiments/configs/compartment.json  # discovery on the Panda
python experiments/replay.py results/sessions/<time> --width 320       # session replay GIF
python experiments/calibrate_handle_measurement.py                     # camera handle-measurement calibration
python experiments/audit_foundation_models.py --name foundation_model_audit_local  # new output directory
python experiments/run_vlm.py experiments/configs/vlm.json --model-path <dir>  # needs the checkpoint
```

In the live session, type structured goals such as `in red tray` or `on blue green`, then
`status`, `scene`, `pause`, `resume`, `cancel` or `quit`. Natural language is rejected with an
explicit message.

The demo prints, for every probe, the chosen probe and its expected information gain, the
predicted motion distribution, the observed motion and the updated belief. It writes
`results/demo/<family>_<seed>/` containing `demo.gif` (sampled hypotheses drawn as translucent
hinge axes in blue and slide arrows in orange), `summary.png` and `log.json`.
Benchmarks write `results/<name>/`: every step of every episode (`episodes.jsonl`),
`summary.json`, plots, and `meta.json` with the git SHA, versions and config.

## How Active Physical Discovery works

| Piece | File | What it does |
|---|---|---|
| Mechanisms | `dream/sim/mechanisms.py` | Visually identical panels: fixed, drawer, slider, 4 hinge families. Handle placement is independent of the joint. Held out: sash, pivot. Out of model class: rising hinge (helical). |
| Robot and sensing | `dream/sim/env.py` | 3-axis gantry with force-limited position servos. It grips the handle (ball-joint constraint) and drags it. The agent receives only noisy sensors: handle pose (1 mm, 0.5°), end-effector position, actuator forces. |
| Hypotheses and prediction | `dream/agent/model.py` | Fixed / prismatic (axis) / revolute (axis, pivot line), with travel limit and friction. A quasi-static, friction-limited model of the compliant probe predicts the settled handle pose. |
| Belief | `dream/agent/belief.py` | One SMC particle set per joint type (adaptive tempering, MH rejuvenation). Type posterior from each set's evidence. Gaussian likelihood whose width grows with predicted motion. χ² adequacy check. |
| Experiment choice | `dream/agent/exploration.py` | EIG via a pairwise Gaussian-mixture estimate over 128 posterior samples, across 104 candidate probes (26 directions × 2 distances × 2 force caps). Baselines: random, goal-directed pull, axis sweep, all on the same candidate set. |
| Operation | `dream/agent/operate.py` | Follows the believed constraint manifold to open the part (0.10 m or 0.5 rad), or declares it fixed. |
| Scoring | `dream/eval/` | Identification, held-out prediction (on forked simulators), task success, paired statistics. |

**Integrity guarantees.** `dream/agent` never imports MuJoCo, the simulator or the evaluation
code (`tests/test_isolation.py`). Ground truth is used only to build scenes and score results.
The agent is given the visible panel geometry and the closed-state handle pose (the report's
"oracle geometry, unknown joint" rung), and nothing about the joint.

**Model validity.** With the *true* parameters, the quasi-static predictor matches MuJoCo with
about 0.2–2 mm median error (worst case ~9 mm, from hinge inertia during the probe ramp;
`tests/test_model.py`). The likelihood noise is sized from this measurement.

## How the Panda foundation works

| Layer | File | What it does |
|---|---|---|
| Robot model | `third_party/mujoco_menagerie/franka_emika_panda` | Canonical Menagerie Panda, vendored unmodified (Apache-2.0, pinned commit in `PROVENANCE.txt`) |
| Scene | `dream/sim/tabletop.py`, `tabletop_env.py` | Table, free cubes, tray; physics stepping; object tracker with 2 mm / 2° noise |
| Control and safety | `dream/robot/panda.py` | Damped-least-squares differential IK with null-space posture; workspace clamp; speed caps; protective stop on link (80 N) or finger (40 N) contact force |
| Skills | `dream/skills/manipulation.py` | `move_to`, `pick` (yaw chosen for finger clearance, miss and slip detection), `place`; success judged from finger width and observations |
| Tools | `dream/tools/` | Seven typed tools with schemas, validation, preconditions and status reporting; structured goals and observation-based verification |
| Runtime | `dream/runtime/agent.py` | One-action-at-a-time loop; rule-based planner that reduces both goal types to grasp → place and clears obstructions; bounded retries |
| Session | `dream/session.py` | Physics and agent threads, pause, resume and cancel into a safe hold, logs for replay |

`tests/test_isolation.py` keeps skills, tools and runtime away from MuJoCo, the scene code and
ground truth. Decisions see only `Embodiment` and `Perception`.

## Results

### Identification on visually identical mechanisms (`results/main`)

7 families × 12 random mechanisms, the same mechanisms and an 8-probe budget for every strategy.
"Identified" means the most probable type is correct with P >= 0.9, the axis error is below 10°
and, for hinges, the hinge-line error is below 2 cm, and all of this holds to the end of the
budget. These thresholds are project criteria, not published standards.

| Strategy | Mean probes to identify [95% CI] | Median | Identified within 8 |
|---|---|---|---|
| **EIG (active)** | **2.14 [1.96, 2.33]** | 2 | **100%** |
| Random | 5.18 [4.57, 5.79] | 5 | 80% |
| Goal-directed pull | 3.38 | 1 | 71% |
| Axis sweep (script) | 2.15 | 1 | 98% |

![main benchmark curves](results/main/curves.png)

- **EIG vs random: -60% median probes** (2 vs 5), better on 65 of 84 mechanisms and worse on 8,
  paired permutation p < 1e-4. This passes the report's proposed gate (>= 25–30%).
  Held-out prediction error falls faster too (3.8 mm vs 7.8 mm after 2 probes), and the
  opening task succeeds in 99% of episodes after 2 EIG probes, against 67% after 2 random ones.
- **EIG vs a hand-written axis sweep: no difference** (2.14 vs 2.15, p = 1.0). The sweep's first
  probe, pulling straight out, happens to suit drawers and hinges. EIG is better on fixed panels
  (3.3 vs 5.0) and sliders (1.75 vs 2.75), and worse on drawers and hinges, where it typically
  needs 2 probes against 1. EIG's agent prior is symmetric, so it has no reason to expect parts
  to open *toward* it, and its first probe often tests the inward direction. The script carries
  that assumption implicitly. The ablation below tests this explanation.
- **Goal-directed pulling** never identifies sliders or fixed panels (0/24).

### Ablation: a soft geometric prior (`results/main_geometric_prior`)

These are the same 84 mechanisms as the main benchmark, but the agent's prior gives weight 0.1 (not 0)
to hypotheses that open *into* the cabinet. The agent infers the front face from the side the
handle protrudes from. Caveat: this assumption matches how the generator builds every mechanism
in this set, so it is favourable.

| Strategy | Mean probes [95% CI] | Identified after 1 probe |
|---|---|---|
| **EIG + prior** | **1.63 [1.45, 1.82]** (uniform prior: 2.14) | **64%** (uniform: 24%) |
| Axis sweep + prior | 1.96 [1.65, 2.31] (uniform: 2.15) | 65% |
| Random + prior | 5.27 | 23% |

With the prior, EIG beats the axis sweep (p = 0.008; 24 better, 11 worse, 49 ties). This supports
the explanation that the main benchmark's tie came from the prior, not from the planner. Sliders get slightly
worse (1.75 → 2.08), because sliding parallel to the face sits at the edge of the "outward" region.
The default prior stays uniform.

### Generalisation (`results/generalisation`)

Mechanisms were drawn from wider distributions than anything used during development: axes tilted
up to 15°, hinge lines up to 4 cm off the panel edge, friction up to 4.5 N. Two families were
never used during development: a vertical **sash** and an interior **pivot**. Each family has 8
mechanisms × 4 strategies. Nothing was retuned.

| In-class (72 mechanisms) | Mean probes to identify [95% CI] | Identified within 8 | Opening succeeds after 2 probes |
|---|---|---|---|
| **EIG** | **1.92 [1.71, 2.11]** | **100%** | **97%** |
| Random | 5.44 [4.81, 6.08] | 78% | 49% |
| Goal-directed pull | 3.74 | 71% | 78% |
| Axis sweep | 2.53 [2.10, 3.00] | 96% | 85% |

- EIG vs random: -65% mean probes (-67% median; p < 1e-4). EIG vs axis sweep: -24% mean probes (30 better, 23 worse, p = 0.01).
  The script's assumptions break on tilted axes and the vertical sash, while EIG's do not.
- On the two unseen families alone, EIG needs 1.69 probes, against 2.94 for the axis sweep and
  4.31 for random.

**Out-of-class mechanism: the rising hinge.** This hinge lifts 3–6 cm per radian as it turns.
That helical motion is outside the agent's {fixed, prismatic, revolute} model class.

- Every strategy ends up confidently wrong: P(revolute) ≈ 1.0. Every strategy still opens it,
  because the compliant controller absorbs the lift.
- The χ² model-adequacy check (`Belief.fit_check`) was calibrated on development families only
  (`results/adequacy_calibration`, threshold frozen at 0.28 before this run). It flags the
  rising hinge as an unknown mechanism class in **100% of EIG episodes by probe 6** (88% by
  probe 4). For comparison it flags 12% of random episodes, 50% of axis-sweep episodes and 100%
  of goal-directed episodes. What exposes the misfit is probes that move the mechanism far.
  EIG and repeated goal-directed pulls do; random probes rarely do.
- False alarms on in-class mechanisms: 0% at the end of the budget, but **3% of individual EIG
  steps**, almost all after only 1–2 probes, when χ² over few records is noisy. These are the
  results under the original rule, which alerts from probe 1. See the fresh validation below.

### Fresh held-out validation of a minimum-evidence rule (`results/fresh_adequacy_validation`)

A rule was preregistered in PR #1 before any of these data existed: alert only from probe 3
onwards, with the threshold frozen at 0.28. It was then evaluated once on a new seed (14819):
108 in-class mechanisms (the 7 development families plus sash and pivot, 12 each) and 12
rising hinges, using EIG with a budget of 8. Before running, a seeding bug was found and fixed
in review: families with equal-length names shared random streams.

| Rule | In-class episodes with any false alarm [95% CI] | Rising hinges detected [95% CI] | Median first alert |
|---|---|---|---|
| Original (alert from probe 1) | 6/108 = 5.6% [2.6, 11.6] | 12/12 = 100% [76, 100] | probe 3 |
| **3-probe minimum evidence** | **0/108 = 0% [0, 3.4]** | **12/12 = 100% [76, 100]** | probe 3 |

All 6 original-rule false alarms happened at probe 1. The gate prevents no detection, but it
delays 3 of 12 (from probe 1–2 to probe 3), so mean latency rises from 3.33 to 3.67 probes.
Only one out-of-class family was tested, so the detection interval is wide. The rule is now used
by `Belief.fit_check`. Results saved before this change keep their original flags.

### Demo outputs (`results/demo`)

`left_hinge_4`, `fixed_6` and `rising_hinge_5` each contain a GIF, a summary plot and a log.
In the hinge GIF, the sampled hinge lines start scattered and collapse onto the true hinge
(final axis error 1.5°, hinge-line error 4.9 mm). The rising-hinge run reports
`DOES NOT FIT: unknown mechanism class` from probe 3 onwards, while P(revolute) = 1.0.

### Panda pick-and-place acceptance (`results/pick_place`)

These are 50 held-out scenes (seeds 1000–1049, never used for development, which used seeds
0–39). In each, a random cube among three is picked and placed in the tray, with grasps made by
finger contact only: the scene has no attachment constraint (`tests/test_panda.py` checks this).
Success is judged from true poses by `dream/eval/manipulation.py`, independently of the skills.

| Success [95% CI] | Grasps | Placement error, median (max) | Protective stops | Link collisions |
|---|---|---|---|---|
| 50/50 [92.9, 100] | 50/50 | 3.2 mm (7.3 mm) | 0 | 0 |

### Task-level evaluation (`results/tasks`)

This is a paired design on 40 held-out scenes per cell (seeds 2000–2039): the same scene,
goal, controller, skills and sensing for each method.

- **Methods:**
  - `agent` is the runtime;
  - `agent_no_retry` is an ablation with one attempt per action;
  - `fixed_sequence` is an open-loop script that plans every waypoint once from the first
    observation and runs it through the same controller.
- **Tasks:**
  - tray, `in X tray`;
  - stack, `on X Y`.
- **Conditions:**
  - nominal (3 cubes);
  - disturbed: a physical downward tug knocks the held cube out of the hand during its first
    lift, and it lands within about 1 cm;
  - cluttered: 4 cubes, never used in development.

| Condition (tray + stack) | agent | agent_no_retry | fixed_sequence |
|---|---|---|---|
| Nominal | **80/80** | 80/80 | 78/80 (2 protective stops) |
| Cluttered | **80/80** | 80/80 | 76/80 (4 protective stops) |
| Disturbed | **80/80** (recovered 80/80) | 0/80 | 0/80 |

- Per cell (n = 40) every agent result is 40/40, Wilson 95% [91, 100]. On the disturbed
  condition, agent vs each alternative gives exact McNemar p = 1.8e-12 per task. The nominal and
  cluttered differences are not significant (p ≥ 0.5).
- **Interpretation.** In undisturbed scenes, a simple open-loop script is nearly as good. The
  agent's measured advantage comes from re-observing, verifying and retrying. Its skills' finger
  clearance also avoids the clutter collisions that stopped the script. The disturbance is one
  designed failure mode (a drop during the lift), so this shows recovery from that mode, not
  general robustness.
- **Other measurements:**
  - zero false success claims and zero arm-link collisions for any method;
  - mean 2 tool calls per goal nominally and 3 when disturbed;
  - about 15 s of simulated time per goal (19 s when disturbed), and 3–4.5 s of wall time.
- **Placement error:**
  - for stacking, the top-to-bottom centre offset has a median of 3.5 mm (agent) and 3.2 mm
    (script);
  - for the tray, the agent deliberately places into a free corner spot to leave room for more
    objects, so distance to the tray centre is not an error measure; success (inside, at rest)
    is.
- The same scene supports sequences of goals in one session. `tests/test_tools_runtime.py`
  runs tray, then stack, then a goal whose object is buried under another, which the agent
  clears first.

### Live session (`dream/session.py`, `experiments/session.py`)

- A physics thread keeps the controller running while goals are planned and executed. Pause
  and cancel interrupt the current tool, and the robot holds still (tested).
- Pause then resume completed the goal in 20/20 headless runs at 4× real time, with the result
  checked against ground truth.
- In the starved maximum-speed mode (`--rate 0`), one of 20 runs stalled during a grasp lift,
  and the cause is not yet understood.

### Discovery as a tool on the Panda (`results/compartment`)

Goal `in red compartment`: a closed box whose mechanism is hidden (drawer, lid or fixed). The
runtime finds the compartment inaccessible, calls `investigate_physics` (the gantry's belief and
EIG, executed as compliant Panda probes), opens along the learned constraint or declines, then
picks, places and verifies, in one continuous simulation.

- 30 held-out scenes per kind (seeds 5000–5029, disjoint from development seeds 0–9 and
  100–109), paired across methods.
- All methods share the robot, skills, sensing (oracle-assisted trackers), the 104-probe
  candidate set, a 12-probe budget and the 15 N force cap.
- Success is the cube inside the cavity by ground truth.
- An interaction is one compliant motion of the gripped handle: a probe, an opening segment or
  a straight pull.

| | DREAM (EIG) | random probing | fixed pull (no discovery) |
|---|---|---|---|
| Drawers | 27/30 | 27/30 | 28/30 |
| Lids | 27/30 | 26/30 | 0/30 |
| Openable, Wilson 95% | **54/60 [80, 95]** | 53/60 [78, 94] | 28/60 [35, 59] |
| Fixed boxes declined | **30/30** [89, 100] | 30/30 (28 as "not identified" after 12 probes) | 0/30 (pulls 3 times) |
| Mean interactions, openable / fixed | 5.7 / 5.7 | 8.4 / 11.9 | 2.0 / 3.0 |
| Successes per 10 interactions (openable) | 1.58 [1.42, 1.74] | 1.05 [0.91, 1.19] | **2.33** [1.54, 3.37] |
| Correct outcomes per 10 interactions (all scenes) | **1.64** | 0.96 | 1.33 |
| False success claims | 0 | 0 | 0 |

- **Regression rerun after the camera-discovery fixes** (`results/compartment_regression_vision_fixes`,
  run at `723b68e`, the same seeds and gates; these seeds were already seen, so this checks for
  regressions and is not a fresh evaluation): DREAM 55/60 openable (was 54), fixed 30/30, random
  probing 54/60 (was 53), fixed pull 28/60, zero false claims. The only per-episode changes are
  two drawers (one DREAM, one random) that ended in a protective stop before and now succeed. The
  table below is the original run.
- **Gates** (pre-registered in `experiments/configs/compartment.json`), all passed:
  - openable success ≥ 80% (90%);
  - fixed-box rejection ≥ 90% (100%);
  - zero false claims.
- **DREAM vs random:** the same success (McNemar p = 1.0) with 2.7 fewer interactions per
  openable box (paired 95% CI [2.1, 3.4]).
- **DREAM vs fixed pull:** 28 vs 2 discordant successes (p = 9e-7).
- **The primary metric does not favour DREAM.** On openable boxes, the fixed pull has the most
  successes per interaction, because a drawer opens with one pull. DREAM leads only when lids and
  fixed boxes count: its "correct outcomes per interaction" includes correct refusals.
- **Identification:**
  - drawers 29/30 (one investigation stopped unsafe), lids 30/30, fixed 30/30;
  - median axis error 1.0° (drawers) and 2.1° (lids);
  - about 3 probes per openable box and 5.7 per fixed box.
- **DREAM's six failures are all protective stops:**
  - four hand or arm contacts near the open container while picking or placing (the motions
    are not collision-aware);
  - one during probing;
  - one stop latched by the handle release after an opening that the tool still reported as
    successful. This was a reporting defect, since fixed with a regression test.
- The run was launched at `029bd63`; see `sha_note` in `summary.json`.

**Shift and challenge runs.** Pre-registered, report-only, single method (DREAM), at `7e7c178`:

| Condition | Result | Reading |
|---|---|---|
| Parameter shift: friction 3–5 N (dev 0.5–3), front yaw ±35° (dev ±15°) | drawers 10/10, lids 10/10, fixed declined 10/10, 0 false claims | the frozen thresholds transferred to this shift |
| Out of model class: return spring (60 N/m) closes the part | 0/16 success, 0 false claims, **0/16 flagged inadequate** (all "identified") | the Panda adequacy check (0.59) showed no detection power here; the agent fails after 3 openings without knowing why |
| Original bar handle | drawers 8/8, lids 5/8 (two release stops, one 40 N stop) | the knob simplification matters for lids, though it does not decide them |
| Original 7 cm cavity | drawers 8/8, lids 6/8 true but only 3 claimed (stops after placing) | placement is not collision-aware; the 5 cm cavity hides this |
| Gravity-loaded lid (no counterbalance) | 8/8 | lids opened to about 1.3–1.6 rad stay open under joint friction |
| Menagerie shoulder armature 0.1 | drawers 8/10 (one grasp failure, one 40 N stop) | no stall observed |

**Shoulder-singularity stress test** (`results/shoulder_singularity_stress`). 40 motions from
folded poses near the base through the travel height, at armatures 0.1, 0.2, 0.3 and 0.5 kg m²:
**0 stalls at every value, including Menagerie's 0.1**. The stall seen once in development has
not been reproduced beyond that one trajectory, so the armature change is not shown to be
necessary (see the audit).

### Audit of milestone-11 changes

Each change that affects task success, classified per `AGENTS.md` Rule 3. The original
conditions remain available as parameters (`CompartmentSpec.handle`, `cavity_height`,
`counterbalanced`, `return_stiffness`; `TabletopEnv(shoulder_armature=...)`).

| Change | Class | Evidence / open |
|---|---|---|
| Gripper servo ×10 stiffer (about 2 N → about 20 N on a cube) | Physically justified calibration | The Franka Hand grips up to 70 N; tests. Still a position servo, not force control. |
| Net wrist-force estimate; two-stage finger close | General correction | `tests/test_panda.py` |
| Spherical knob instead of a bar handle | **Environment simplification** | Bar challenge: lids 5/8, drawers 8/8. Open: grasps that follow a rotating handle. |
| Cavity 7 → 5 cm | **Environment simplification** | Deep challenge: 3/8 lids claimed. Open: collision-aware placement. |
| Counterbalanced lid | **Environment simplification** (from the start) | Gravity challenge 8/8 at the opening angles reached |
| Shoulder armature 0.1 → 0.3 kg m² | **Unverified workaround** | Stall not reproduced at 0.1 in 40 stress motions; 8/10 full drawer tasks at 0.1. Neither necessity nor harm shown. |
| Panda adequacy threshold 0.59 | Calibration by the frozen development procedure | No false alarms in 90 held-out and 30 shifted scenes; **no detection shown** on the out-of-class spring (0/16) |
| P(fixed) ≥ 0.99 to declare fixed | Task-specific decision heuristic (asymmetric cost) | Held-out 30/30 declined, 0/60 openable declined; shift 10/10 and 0/20 |
| Opening deviation tolerance 2 cm + 10% of travel | Heuristic tolerance | No false stops held-out or shifted. Its ability to catch a wrong model is untested. |
| Place yaw aligned with the container's front face | Task-specific heuristic | Deep-cavity lids still fail. Open: general collision-aware placement. |
| Release retreat toward 0.5 m radius | Heuristic | Development failures only |
| Open only until the opening is wide enough | General (task-directed stopping from perception) | n/a |
| Retain occluded opening evidence, accepting wider measurements and clear closure | General algorithmic correction | Regression contracts; continuation development: lids 2/2, drawers 0/2, no causal attribution to this change alone |
| Five-second lifetime for old openings and released-handle estimates | Task-specific heuristic (uncalibrated freshness policy) | Expiry tests; does not model how fast a released mechanism can close |
| Recent release position as a fallback for an unchanged, uncertain handle track | Task-specific heuristic (rigid knob grasp) | Scoped to a successful opening and confirmed finger width, same track, one immediate retry; clear newer vision wins; rotating/slipping grasps remain unsupported |
| Failed release fails the opening | General correction | Regression test |
| Cubes kept out of the drawer's path and in the validated workspace | **Environment simplification** | The agent does not reason about objects blocking an opening |
| Drawer- or lid-specific code | None in planning or discovery: segment sizes depend on the *learned* joint class. The fixed-pull baseline is a drawer strategy by definition. | `dream/runtime`, `dream/tools/discovery.py` |

**Stale results.** `results/pick_place` and `results/tasks` predate the gripper, armature and
floor-height changes and have not been rerun at the current code.

### Camera perception (`results/vision`, `dream/perception/`)

**Pipeline** (NumPy only, isolation-guarded: it never imports MuJoCo or the scene code).
- **Camera.** A fixed calibrated RGB-D camera, rendered by MuJoCo at 480×360, about 1 m from
  the objects. Depth noise is 1 mm + 1.5 mm/m²·z², with RGB noise.
- **Self mask.** The robot's own body is masked by rendering its own geometry, the standard
  self-filter.
- **Detection.**
  - back-projection, then a RANSAC table plane;
  - segmentation by 3D and hue continuity;
  - a box fit per component (minimum-area rectangle and height);
  - colour names from a fixed hue vocabulary;
  - open-container detection;
  - occlusion flags and a 1σ position estimate.
- **World model.** Tracks with visibility and last-seen time. An object the camera loses stays
  at its *last-known* place and is flagged as not visible; it is never moved to where it really
  is.
- **Placement in the hand.** Placement uses the grasp point when the hand hides a held cube.
  Pick confirms a lift by finger width when the cube is hidden.
- **Two backends, same interface.**
  - `TabletopEnv` is the oracle backend (noisy ground-truth trackers).
  - `VisionPerception` sees only frames.
  - Logs record which backend ran.

**Held-out evaluation** (seeds 9000+; development used 0–11; run at `11b5cd8`; gates
pre-registered in `experiments/configs/vision.json`, all passed):

| Condition (tray + stack) | oracle | vision | vision, no retries | vision, single frame |
|---|---|---|---|---|
| Nominal (20 + 20) | 40/40 | **39/40** | 38/40 | 33/40 |
| Small cubes (3.2 cm) | 24/24 | 24/24 | | |
| Large cubes (5 cm) | 24/24 | 24/24 | | |
| Colours never used in development (orange, purple, yellow) | 24/24 | 22/24 | | |
| Dim light (×0.45) | 24/24 | 24/24 | | |
| Bright light (×1.6) | 24/24 | 24/24 | | |
| Calibration error (1 cm, 1°) | 24/24 | 24/24 | | |
| Sensor noise ×3 | 24/24 | 18/24 | | |
| Clutter (5 cubes) | 24/24 | 24/24 | | |

- **Zero false success claims** in all 544 episodes.
- **Nominal perception** (first frame, 40 scenes):
  - recall 1.0, precision 1.0;
  - position error median 2.2 mm, p90 2.8 mm, max 3.4 mm;
  - yaw error median 0.9°;
  - colour names 100% correct;
  - tray centre error median 1.5 mm.
- **Under shift:**
  - *Calibration error* raises the position error to 12 mm. The tasks still succeed, but the
    reported uncertainty is wrong: **0% of errors within 2σ**. The pipeline cannot know its
    calibration is off.
  - *Sensor noise ×3* gives a median error of 5.3 mm and precision 0.86. On the tray task the
    tray estimate shrinks, and the agent declines with "tray too small" (5 of 12). These are
    correct refusals, not false claims.
  - *New colours* gave one colour misnamed per task ("orange is not visible").
- **Uncertainty calibration:** 88% of nominal errors lie within 2σ.
- **The world model matters:** without memory (single-frame ablation), a cube hidden in the
  hand is forgotten, and success drops to 33/40.
- The **oracle arm** also re-measures tray and stack at the current code: 40/40 nominal and
  24/24 under every shift.

**Supported distribution:**
- distinctly coloured, box-shaped objects (3–5 cm) on one plane;
- objects that contrast in colour with the tray and with each other;
- one fixed camera in front of the workspace.

Two touching objects of the same colour merge. Colour names come from a fixed vocabulary, not
learning. This is a classical baseline, not open-world perception. The backend can be replaced
behind `Perception` without touching skills or the runtime.

- **Open-vocabulary and unseen-object segmenters** (for example Grounding-DINO-like detectors,
  SAM-like segmenters) were not added. The original Linux environment had no GPU or model-host
  access, and that slice did not need them. Their licences and versions remain unverified.
- **Compute:** one camera frame takes about 0.35 s to render on CPU (OSMesa), and detection
  about 0.25 s.
- **Live demo** (`results/demo_vision/`): "in red tray" then "on blue green" in one live vision
  session. The demo paused and resumed with 6e-15 rad of joint drift; both goals were true in
  the world after 137 frames. The replay shows the camera view with the logged detections and
  uncertainties.

**Camera-driven discovery: negative result** (`results/compartment_vision_dev`, development
seeds 0–5, oracle-calibrated thresholds, nothing tuned on it).

What exists:
- box, opening and handle detection from depth;
- compartment identity across the closed and open states;
- a second camera beside the robot, added after measuring that a lid swung open hides its
  cavity from the default camera;
- a position-only likelihood, since a round knob's orientation is not observable;
- probes with an unseen handle give no measurement;
- a `retract` tool to look with the arm out of view.

Measured: **0/12 drawers and lids, 0 false claims**. The fixed boxes were declined 6/6, but as
"model inadequate" rather than identified as fixed. Causes:
- the knob estimate shifts when the fingers cover it, which is read as slip;
- handle grasps miss on vision estimates;
- the oracle-calibrated noise model (3 mm) and adequacy threshold (0.59) raise false alarms on
  camera measurements;
- position-only evidence misidentified lids as prismatic.

**Since then (development work; continuation smoke below).**
- *Handle measurement calibrated* (`experiments/calibrate_handle_measurement.py`,
  `results/handle_measurement_calibration`, development seeds 0–9, 30 scenes, with the true
  handle pose used only for scoring):
  - camera error with the knob in clear view: median 4.3 mm, p90 6.7 mm;
  - with the gripper on the knob: median 5.4 mm, but p90 24.3 mm;
  - the rigid-grasp offset (TCP to handle) changes by at most 2.2 mm during probing.
- So observations now carry a 1σ (7 mm clear, 25 mm occluded). When perception is worse than
  the grasp, probe displacements come from the gripped TCP, and the camera only checks for gross
  slip.
- A bug made position-only probes use the full 6D likelihood (the averaging step dropped the
  orientation flag). This, not the evidence, was why lids came out prismatic. Fixed, with a
  regression test.
- Runtime fixes found on these scenes: no re-grasp of a knob already held, bounded set-aside
  attempts, one compartment track across its closed and open states.
- Continuation branch: occluded opening measurements can widen retained space; clear views
  can shrink or remove it. Retained space expires for placement after five simulated seconds
  without supporting evidence. Handle associations stay attached to their original perception
  track rather than switching to whichever handle moves nearest. A successful opening can
  supply a recent release position for one re-grasp when vision still shows an old, uncertain
  estimate. A lost grasp or failed release supplies no such estimate. The time limits and
  contact fallback are uncalibrated heuristics, not demonstrated transfer or learning.
  Colour/nearest-neighbour detection association can still confuse indistinguishable handles
  crossing in the image; persistent ownership does not solve that perception ambiguity.
- Saved continuation smoke (`results/compartment_vision_continuation_smoke`, inspected
  development seed 1): drawer and lid both completed and verified; fixed box identified and
  declined, zero false claims and protective stops. This is a pipeline smoke, not a success
  rate estimate or a controlled attribution to these changes. It ran at `7ef4d8a` with the
  uncommitted code snapshot preserved in the `patch` field of `working_tree.json`; the summary
  records `dirty=true`. The original negative result remains above.
- Frozen continuation diagnostic (`results/compartment_vision_continuation_dev`, development
  seeds 12–13, clean code at `21969bf`): lids 2/2, drawers 0/2, fixed identified and declined
  2/2; zero false claims (the pre-registered gate passed), one protective stop. Drawer 12 opened
  physically but the release triggered the 40 N hand-force protective stop. Drawer 13 opened and grasped
  the cube, then placement was rejected three times with "no free spot in compartment".
  The log establishes the rejection, not its underlying perception/geometry cause. No safety
  threshold or scene geometry was changed in response. These are within-generator development
  outcomes, not a paired causal comparison or a generalisation result.
- Still open: grasping the knob again from a camera estimate, the grasp being lost while
  opening, verification right after placing (the cube estimate is occluded), and the
  position-only adequacy threshold (still the 0.59 placeholder).

### Foundation models (`dream/runtime/vlm.py`, `results/foundation_model_audit`)

**Selected models.**
- Planner: **Qwen/Qwen3-VL-4B-Instruct**, Apache-2.0 per the reachable
  [official model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
  The checkpoint index lists 8.88 GB of weights before inference buffers.
- Learned motor policy: **NVIDIA GR00T N1.7-LIBERO**. The
  [model card](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) specifies NVIDIA Open Model
  License Agreement; upstream inference code is Apache-2.0. No upstream assets or model
  dependencies have been added. OpenVLA remains an untested fallback.

**Original Linux audit** (`experiments/audit_foundation_models.py`, saved run at `8366bfe`):
- 4 CPUs, 16.5 GB RAM, 31.5 GB free disk, **no GPU**, no torch or transformers installed.
- huggingface.co, github.com, modelscope.cn and download.pytorch.org are refused by the
  environment's network policy (403 at the proxy); pypi.org is reachable.
- Qwen3-VL could not be downloaded in that environment, nor could GR00T run with its checkpoint:
  it needs a CUDA GPU.

**Windows continuation audit** (`results/foundation_model_audit_windows_final/audit.json`,
clean code at `21969bf`): all checked hosts returned HTTP 200. The host has 16.8 GB total RAM,
1.91 GB available at the audit, no `nvidia-smi`, and no inference packages installed in `.venv`. Qwen's full
checkpoint exceeds available RAM before buffers; no hosted endpoint was provided. Model
inference was not attempted, and no weights were downloaded. The audit records the script's
hash and code revision. The earlier Linux audit and initial dirty Windows diagnostic
(`results/foundation_model_audit_windows_sources`) are preserved.

The audit also records pinned official source URLs and hashes. GR00T's `libero_10` configuration
uses an external image plus wrist image, eight state values (xyz, three rotation-vector
components despite the names roll/pitch/yaw, two finger positions), and a 16-step chunk of
seven action values. The official wrapper uses 256-square RGB views flipped in both axes,
and transforms gripper commands from [0,1] to a binary command with inverted sign. The
processor config labels actions `ABSOLUTE/NON_EEF`; that label alone does **not** establish
world-frame position commands because the wrapper passes them into LIBERO's controller.
OSC scaling, reference frames and frequency remain unverified for DREAM. DREAM lacks the
wrist camera, per-finger state interface and validated arbitrary-orientation action mapping for a
faithful mapping. The [upstream hardware requirements](https://github.com/NVIDIA/Isaac-GR00T#hardware-requirements)
specify a GPU with at least 16 GB VRAM. There is no validated motor adapter or learned action
execution here; passing these action values directly to DREAM would be incorrect.

**What exists** (`dream/runtime/vlm.py`, `dream/runtime/qwen.py`):
- `VLMAgent` gives the model the instruction, the camera image (at most two images kept in
  context), the structured scene, tool results and investigation findings. Named physical tools
  are offered as JSON-schema functions, plus `finish`.
- Output is parsed only as JSON inside the model's own `<tool_call>` format. Nothing it
  writes is executed as code. Tool arguments go through the same toolbox validation,
  preconditions, budgets and safety limits as the rule-based planner.
- A success claim must match a caller-bound structured objective, which is not shown to the
  model, and `verify_goal` must confirm it with nothing in the hand. Without a trusted objective,
  the runtime can use tools or decline but cannot certify natural-language completion.
  Multiple calls are rejected together; hosted responses preserve all calls for this check.
  Invalid or rejected calls are
  bounded (`max_invalid`), and the run stops.
- The model chooses tools; it never supplies coordinates from images, motor commands or
  mechanism types. Coordinate-level moves and the fixed-pull benchmark strategy are rejected
  at the model interface. Discovery stays the only way a mechanism is identified.
- Tool observations expose visibility, timestamps and uncertainty. Cancellation is checked
  after inference before dispatch. Local generation checks cancellation/time between tokens;
  it cannot interrupt a blocked kernel or loading. Hosted requests can remain blocked until
  their socket timeout, but a cancelled result cannot trigger an action. Inference errors
  stop the episode and hold the robot.
- Backends: local transformers inference (`QwenTransformersBackend`, the model's chat template
  with `tools`), or an OpenAI-compatible server such as vLLM (`OpenAICompatibleBackend`).
- `experiments/run_vlm.py` compares condition A (rule-based planner given the structured goal)
  with B (the model given the paraphrased instruction), including an impossible instruction
  scored as a correct refusal. It refuses to run B without a real backend.

**What is tested:** schemas, parsing, bounded invalid outputs, rejected false success claims,
image budget, and one scripted episode through real physics (`tests/test_vlm.py`). The test
double replays fixed outputs. These tests show the plumbing works. **They are not evidence of
any model's ability.**

**To run the model:** on a host with enough available memory, install
`pip install -e ".[vlm]"`, download the checkpoint, then
`python experiments/run_vlm.py experiments/configs/vlm.json --model-path <dir>`.
The CPU backend remains unvalidated with model weights. GR00T needs a CUDA inference host
and a validated observation/action mapping before any DREAM execution.

## Limitations (read before citing any number)

- **Perception.** Tabletop pick-and-place runs from camera images (classical RGB-D baseline,
  above). Held-out compartment results use the oracle-assisted handle tracker; camera-driven
  discovery has mixed development results only. The import order matters for headless
  rendering: MuJoCo picks its GL backend when first imported, so scripts set `MUJOCO_GL` first
  (`dream/__init__.py`, runners, `tests/conftest.py`).
- **No language understanding has been demonstrated.** The rule-based planner takes structured
  goals (`in X Y`, `on X Y`). The model-driven agent accepts language, but no model has been run
  through it, so there is no evidence of any model's ability on these tasks.
- **Panda scope.**
  - Cubes of one size, one tray fixture, top-down grasps only.
  - The rule-based planner covers two goal types plus obstruction clearing.
  - The gripper is a position servo stiffened ×10 (about 20 N on a cube), not force-controlled.
  - Compartment opening covers one drawer and lid family on a narrowed scene (see the audit).
    Out-of-class mechanisms are not detected on the Panda.
- **Simplified contact (discovery gantry only).** That grasp is an ideal ball-joint
  constraint, the mechanism has no collision geometry (joint limits act as stops), and panels
  are gravity-compensated. The Panda's grasps use real finger contact.
- **Quasi-static predictor.** It predicts settled outcomes only. Inertial overshoot is absorbed
  by the noise model rather than modelled.
- **Travel limits** are only learned if a probe reaches them, so predictions near an unvisited
  limit are poor. This is visible as the held-out error rising late in some episodes.
- **Uninformative prior.** EIG allows that a panel might open into the cabinet, which costs it a
  probe against scripts that assume otherwise. See the geometric-prior ablation.
- **Model-adequacy check:** cannot alert before probe 3, by design. Out-of-class detection
  was tested on only one out-of-class family (the rising hinge).
- **Live viewers** (`--viewer` in both demos and the session) were not exercised: development
  ran in a headless container. Headless replay GIFs are tested.
- Compute: an EIG step takes about 0.6 s to select and 0.5–2 s to update, on one CPU core.

## Next milestone: camera-driven discovery

Milestone 11 (discovery as a tool on the Panda) is implemented on a narrowed scene with the
oracle tracker; see its audit. Milestone 10's first slice (camera pick-and-place) is done. The
next technical dependency joins them: **mechanism discovery from camera measurements**.
1. A measurement model for a tracked handle: noise and bias calibrated on development frames,
   including partial occlusion by the fingers.
2. Slip detection that does not mistake measurement shifts for slip.
3. The adequacy threshold re-calibrated for the position-only likelihood.
4. Grasping a handle from a vision estimate, with re-observation.
Gate: fresh held-out compartment scenes in vision mode against the oracle-tracker results above,
with no false claims.

Open follow-ups:
- out-of-class detection on the Panda (0/16 on the return spring);
- collision-aware placement (the 7 cm cavity);
- grasps that follow a rotating handle (the bar challenge);
- rerun `results/pick_place` and `results/tasks` at the current code.

## Roadmap

The remaining milestones in the status table (10–14), one vertical slice at a time (see
`AGENTS.md` → Development priorities). Each needs its own baselines and fresh held-out
evaluation before it is marked done.
