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
   - There is no natural-language understanding, no camera perception (object poses come from a
     simulated tracker) and no memory yet.
3. **Discovery as a tool on the Panda (milestone 11).** For "in red compartment", the Panda
   probes a closed box whose mechanism is hidden (drawer, lid or fixed) through its own compliant
   controller, using the same belief and EIG code as the gantry. It then opens the box along the
   learned constraint or declines, places the cube and verifies. This runs in one continuous
   simulation, on a deliberately narrowed scene (knob handle, 5 cm cavity, counterbalanced lid):
   see the [audit of simplifications](#audit-of-milestone-11-changes).

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

Last full verification (22 Oct 2026, branch `claude/panda-manipulation`): `pytest` (98 passed,
including slow closed-loop, persistent-scene and live-session tests; the fast suite is 92 tests
in about 50 s) and `ruff check`/`format`. The discovery statistics were independently recomputed
from the saved `episodes.jsonl`. Environment: Python 3.13.16, MuJoCo 3.16.0, NumPy 2.5.3, Linux,
CPU only.

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
| 10 | Visual perception replacing oracle poses | Planned | n/a |
| 11 | **Discovery invoked on demand by the runtime (unfamiliar compartment on the Panda)** | **In progress**: implemented and tested on the narrowed scene; held-out evaluation below; challenge variants open | `tests/test_compartment.py`, `results/compartment`, audit table |
| 12 | Experience memory, failure-driven replanning, skill adaptation | Planned | n/a |
| 13 | Generalisation evaluation across goals, arrangements and task families | Planned | n/a |
| 14 | Learned world/motor models, mobile or whole-body embodiments | Planned, conditional on evidence and compute | n/a |

## Install

Python >= 3.10 (developed on 3.13). CPU only.

```bash
pip install -e ".[dev]"            # mujoco==3.16.0, numpy, matplotlib, pytest, ruff
sudo apt-get install libosmesa6    # only for headless rendering of the demo GIF
```

## Run

```bash
pytest -m "not slow"                                   # fast suite (~40 s)
pytest                                                 # + closed-loop episode tests (~1 min)

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
python experiments/replay.py results/sessions/<time> --width 320       # session replay GIF
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

## Limitations (read before citing any number)

- **No camera perception, anywhere.** Mechanism handle poses and tabletop object poses come
  from simulated trackers with Gaussian noise. The report's MOSART warning applies: perception
  error may dominate on a real system.
- **No language understanding.** Goals are structured (`in X Y`, `on X Y`). No language model
  is integrated, and none is available in this environment. The `Planner` protocol is where one
  would plug in.
- **Panda scope.**
  - Cubes of one size, one tray fixture, top-down grasps only.
  - The rule-based planner covers two goal types plus obstruction clearing.
  - The gripper's grip force is low (about 2 N, Menagerie's actuator range), which is fine for
    50 g cubes but untested for heavier objects.
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

## Next milestone: 11, discovery as a tool on the Panda

The goal is "open the unfamiliar compartment and put the cube inside". This connects DREAM's two
foundations and is the first task that needs physical knowledge the agent lacks.

- **Scene.** A compartment (drawer or hinged lid, hidden joint) in the tabletop world. Reuse the
  mechanism generator, now with collision geometry, so it physically blocks access until opened.
- **Skills.** `grasp_handle`, `probe(direction, distance, max_force)` and `pull_along(path)` on
  the Panda, all force-limited. These were in the earlier milestone 7 plan and have not been
  built.
- **Tool.** `investigate_physics(object)` runs the existing EIG belief loop through the Panda's
  probe skill and returns the identified joint, its uncertainty and the adequacy check.
  The planner calls it when `place` into a closed compartment is impossible or a pull fails.
- **Gates.**
  1. The discovery benchmark reruns through the Panda's probe skill. The change in
     identification performance relative to the gantry is reported, whatever its size.
  2. On held-out scenes, the open-then-place task beats two baselines: a script that pulls
     straight out, and the runtime without `investigate_physics`.

Visual perception (milestone 10) stays planned. Every result above uses tracked poses.

Open follow-ups: test out-of-class detection on more mechanism families; diagnose the
max-speed live-session stall; learn friction and travel limits; try receding-horizon EIG.

## Roadmap

The remaining milestones in the status table (10–14), one vertical slice at a time (see
`AGENTS.md` → Development priorities). Each needs its own baselines and fresh held-out
evaluation before it is marked done.
