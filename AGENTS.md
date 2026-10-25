# DREAM: mission, behavioural contract and engineering guide

This is the authoritative specification for everyone who changes this repository, human or
coding agent. `CLAUDE.md` is a symlink to this file, so there is exactly one set of instructions.

- **`AGENTS.md`** (this file) defines the mission and the behaviour DREAM must be developed
  toward. It also holds the architecture and invariants, the anti-overfitting rules, the
  engineering and research standards, and how to decide what to build. It is not a progress
  diary.
- **`README.md`** is the human-facing record. It holds the status table, install and run
  commands, measured results, the audit of environment simplifications and known failures,
  limitations and the next milestone. It is the source of truth for what is done.

Read both before changing code. Then read the code you are about to change and the commit
history of the files involved.

Contents:
1. Mission and ambition
2. What exists today
3. Expected embodied intelligence behaviour
4. Anti-overfitting and research integrity
5. Architecture
6. Invariants
7. How to decide what to build
8. Development priorities
9. Adding tools, models and dependencies
10. Code style
11. Testing and evaluation standards
12. Experiments
13. Source control, documentation and scope
14. Layout

---

## 1. Mission and ambition

DREAM is a research project toward **general-purpose embodied artificial intelligence**. The
objective is not a robot that opens drawers, moves a few blocks, or completes a fixed list of
tasks.

The objective is a system that can:

- receive an unfamiliar natural-language physical goal;
- understand its environment and work out what needs to happen;
- use the physical capabilities it has;
- acquire missing knowledge through observation and experimentation;
- adapt to failures;
- complete the task, or recognise that it cannot.

The long-term question DREAM should be able to answer is:

> Given what I can observe, what I can physically do, and what I have learned so far, how can I
> accomplish this new goal?

It must not need a complete task-specific program for every new objective. The long-term
target is a humanoid in ordinary homes ("go to the bedroom cabinet and get the charger",
"figure out how this container opens", "explore this room and remember where things are"):
it explores unfamiliar spaces, finds out how unfamiliar things work through observation,
external knowledge and bounded physical experiments, composes physical skills and remembers
what it learned. When it does not know how to do something it works out how to acquire the
missing knowledge, safely and with purpose, or explains what remains unresolved. None of this
is established by architectural intent; each part is claimed only when measured.

**Target behaviours.** These are directions, not claims:

- Organise unfamiliar objects according to instructions.
- Manipulate different objects without a separate grasp script for each.
- Open a previously unseen mechanism by discovering its physical constraints.
- Use available objects as tools when appropriate.
- Combine existing capabilities to accomplish a new multistep task.
- Recognise an unfamiliar physical interaction and investigate it.
- Recover when an object slips, a prediction is wrong or an action fails.
- Reuse previously learned knowledge in a new environment.
- Identify that a task is impossible or unsafe with the current embodiment.

**The scientific question.**

> Does an embodied agent solve unfamiliar tasks more effectively when it combines general skills
> with active physical discovery, predictive reasoning and experience-driven adaptation?

General-purpose intelligence is the direction. **Measured transfer across unfamiliar tasks is
how we determine whether we are approaching it.** Never claim DREAM performs arbitrary tasks,
reasons generally or learns autonomously until that behaviour has been demonstrated and evaluated
(sections 4 and 11).

**Governing principle.** DREAM should learn to handle unfamiliar situations. It must not require
developers to keep altering unfamiliar situations until DREAM can handle them. Every
contributor must preserve that distinction.

## 2. What exists today (do not overstate it)

The README status table is the source of truth. Anything not marked "implemented and tested"
there must not be described as working. In summary:

1. **Active Physical Discovery** on an abstract gantry (`dream/agent/`).
   - A force-limited 3-axis gantry probes single-joint mechanisms in MuJoCo.
   - A per-type SMC belief, a quasi-static predictive model and expected-information-gain (EIG)
     probe selection identify fixed, prismatic, revolute and held-out families.
   - Against random probing on the same candidates, EIG needs about 60% fewer probes (median),
     with paired statistics.
   - A χ² adequacy check flags an out-of-class (helical) mechanism.
2. **Panda manipulation foundation** (`dream/robot`, `skills`, `tools`, `runtime`, `session`).
   - A simulated Franka Panda (Menagerie model) with a deterministic safety-limited controller.
   - Contact-based grasping and closed-loop skills.
   - Typed tools, and a rule-based runtime for structured goals.
   - A live session with pause and cancel.
3. **Discovery integrated with the Panda** (milestone 11, in progress).
   - The Panda probes a closed compartment with a hidden mechanism (drawer, lid or fixed)
     through its own compliant controller, using the same belief and EIG code.
   - It opens the compartment along the learned constraint, or declines.
   - It then places a cube inside and verifies the result, all in one continuous simulation.
   - The task scope was narrowed on the way (knob handle, shallow cavity, counterbalanced
     lid), and the audit table in the README lists these simplifications. Held-out results
     are in the README.

4. **Camera perception, first slice** (`dream/perception/`): a classical RGB-D pipeline replaces
   the object tracker for tabletop pick-and-place. It is a colour-and-depth baseline for box-like
   objects, not open-world vision. Camera-driven mechanism discovery has mixed development
   outcomes and is not held-out validated; validated compartment results still use the
   oracle-assisted handle tracker. Keep successes and failures visible in the README.

5. **Mobile G1 under a pretrained whole-body controller** (`dream/robot/g1_wbc.py`,
   `dream/sim/g1_mobile.py`): NVIDIA's released Decoupled WBC walk/balance policies drive a
   free-standing G1 in MuJoCo (velocity commands; arms held). On top, a classical
   exploration agent (`dream/perception/spatial.py`, `dream/navigation/`,
   `dream/tools/mobile.py`, `dream/runtime/search.py`) maps procedural rooms from the head
   camera and finds and inspects a cabinet. Odometry is oracle-assisted. No manipulation
   while walking, no discovery on this robot yet; results and limits are in the README.

A pretrained multimodal runtime now selects typed tools from language and camera observations;
its measured scope belongs in the README. Its motor execution is classical. The only learned
motor policy in use is NVIDIA's pretrained G1 locomotion (item 5); no learned manipulation
policy, persistent experience memory or Isaac Sim embodiment is validated.

---

## 3. Expected embodied intelligence behaviour

This section is DREAM's behavioural contract and one of the most important parts of the
repository. Every capability added must move the system toward these behaviours without
faking them. Each subsection states the requirement and what it means in code.

### 3.1 Goal-directed, not script-directed

DREAM reasons about desired outcomes, not about which known command sequence matches a goal
string. For "put the red object inside the closed container", it must work out what needs to
happen from current observations:

- the container is not accessible;
- something must change that;
- how it moves is unknown;
- so investigate, operate it according to what was learned, then place and verify.

It must not assume the container is a drawer, that it opens in a particular direction, or run a
hardcoded trajectory because the task resembles a known one. Existing skills are reused and
composed, but task-level behaviour must depend on the actual environment.

In code:

- Planners act on goal preconditions and observed state.
- An `if <scene type>: <motion>` branch in planning code is a defect.
- A motion chosen for a container must come from a model of that container, built from
  evidence.

### 3.2 Act on observations, not hidden answers

The robot uses only what is plausibly available through its sensors, its proprioception, its
own earlier interactions and permitted external knowledge.

- It never uses privileged simulator state to infer an unknown physical property or to plan.
- Ground truth is for scene construction, debugging and independent evaluation only.
- Oracle-assisted perception (a tracker that reads true poses and adds noise) is allowed during
  development. It must be labelled as oracle-assisted wherever it is used or reported, and
  eventually replaced by estimation from real sensor data.

Section 6 (invariant 1) makes this enforceable.

### 3.3 Recognise uncertainty

Represent uncertainty where it changes decisions. Examples:

- object pose;
- mechanism type, friction and stiffness;
- grasp success;
- unknown obstacles or constraints;
- competing task strategies;
- confidence in a physical prediction.

The system must distinguish four states:

- **known with sufficient evidence**;
- **suspected but uncertain**;
- **not yet observed**;
- **outside the existing model class**, where none of the agent's models explain the data.

A confidently wrong model is not intelligent behaviour. Decision thresholds must reflect the
cost of being wrong. For example, declaring a mechanism immovable abandons the task, so it
needs stronger evidence than identifying a mechanism that can still be tested by operating it.

### 3.4 Actively seek useful information

When missing knowledge prevents reliable action, DREAM investigates. Possible ways:

- observe from another viewpoint;
- make a safe physical probe;
- test whether something moves;
- compare candidate physical models;
- simulate possible outcomes;
- consult permitted external information;
- retrieve earlier experience.

Information gathering must have a measurable purpose:

- Choose experiments by their expected value to the task, weighed against effort and safety
  cost, as EIG with an effort penalty does.
- Stop when the task-relevant uncertainty is resolved, when more evidence cannot help, or when
  the budget is spent.
- Never experiment repeatedly without considering the task.

### 3.5 Predict, act, observe, update

DREAM compares expected physical outcomes with observed ones, continuously. When a prediction
fails, it does not blindly repeat the action. It works out whether the cause is:

- incorrect perception;
- an incorrect physical model;
- a missing motor capability;
- a poor grasp or contact;
- an unreachable or unsafe target;
- an unexpected change in the environment;
- an incorrect high-level plan.

It then revises its behaviour accordingly. Responses to failure must be based on evidence (a
measured deviation, a failed verification, a safety signal), not on an ever-growing collection
of hardcoded exceptions.

In code:

- Skills report measured outcomes, not success flags.
- Operations compare the observed state to the model's prediction while they run.
- The runtime re-observes after every action.

### 3.6 Reuse and transfer capabilities

Design each capability around a reusable physical or reasoning operation, not a benchmark
example:

- Grasping works across a supported distribution of objects and poses, and that distribution
  is stated.
- Opening follows learned constraints, not a known cabinet type.
- Placement uses measured geometry and collision constraints.
- Planning composes reusable skills.
- Recovery responds to observed failures, never to known seeds.

Wherever possible, demonstrate a new capability on situations different from those used to
develop it (sections 4 and 11).

### 3.7 Maintain and reuse experience

The architecture must support meaningful experience memory: physical knowledge, successful
strategies and earlier failures that **change future decisions**.

- Saving a JSON log is not learning.
- Per-session state is not memory either. For example, the toolbox holds investigation results
  only for the current session, and the code must say so.
- When memory is implemented, measure whether it improves performance on new instances or
  reduces repeated exploration, against the same system without it.

### 3.8 Know the limits of its capabilities

DREAM must be able to:

- report insufficient information;
- detect model inadequacy;
- reject unsafe actions;
- recognise unreachable targets;
- stop after repeated unsuccessful attempts;
- distinguish an *unsupported* task from a *failed execution*.

It never forces a mechanism to satisfy a benchmark. A failure with a correct explanation can be
more informative than an accidental success. Evaluations must score correct refusals and false
success claims explicitly.

### 3.9 Real-time closed-loop embodiment

DREAM should support continuous sensing, action and interaction with a user.

- High-level reasoning and tool calling stay separate from real-time deterministic control and
  safety.
- The system supports interruption, cancellation, a safe hold and recovery where the embodiment
  permits.
- No language model or planner may bypass force, velocity, joint or workspace limits
  (invariant 6).

### 3.10 Generality is measured, not assumed

DREAM must not be called general-purpose because it can choose between a few hardcoded tasks.
Generality is evaluated through:

- new task compositions;
- unseen object configurations;
- different physical parameters;
- different geometries;
- new environments;
- changed observations and disturbances;
- transfer of previously acquired knowledge.

Always state whether a result shows **interpolation within a familiar distribution** (new seeds
from the development generator) or **generalisation to genuinely unfamiliar conditions**
(shifted parameters, new families, new compositions). Report them separately.

---

## 4. Anti-overfitting and research integrity

These rules are non-negotiable.

**Rule 1: Never optimise for a specific demonstration.**
- Do not adjust the system until one GIF, scenario or seed succeeds and then present it as
  general.
- A demonstration shows a capability exists under particular conditions. It does not show that
  it generalises.

**Rule 2: Understand failures before fixing them.** For every significant failure:
1. Reproduce it.
2. Record the original conditions (seed, config, commit).
3. Identify the failure category, using the list in 3.5.
4. Find the root cause, not a correlate.
5. State a testable hypothesis for the correction.
6. Make the smallest justified change.
7. Verify the original case.
8. Verify on independent cases the change was not developed on.
9. Check for regressions.
10. Document the remaining limitations.

Trial-and-error parameter tweaking does not replace this process. Every bug fix gets a
regression test first (section 11).

**Rule 3: Separate algorithmic corrections from environment simplifications.** A better
controller and an easier environment are different kinds of change:
- A different handle shape may make grasping easier without the robot learning to grasp
  unfamiliar handles.
- A shallower cavity may make placement easier without better collision-aware manipulation.
- Changing simulated motor inertia may suppress an instability without the controller becoming
  generally stable.

Classify every change that affects task success as one of:

| Class | Meaning | Required evidence |
|---|---|---|
| General algorithmic correction | A defect in a general mechanism was fixed | The original failure, plus independent cases from outside the development set |
| Physically justified calibration | A simulated parameter was moved toward documented physical reality | A cited value or measurement, and the effect on validated results |
| Environment simplification | The task or world was made easier | The harder variant kept as a reproducible challenge case |
| Task-specific heuristic | A rule that encodes knowledge of the task family | Its scope stated in the code, and a test of where it breaks |
| Unverified workaround | It works, and why is not established | Kept on the open-issues list until validated or replaced |

State the class in the commit message. Keep the current classification in the README audit
table.

**Rule 4: Preserve difficult cases.**
- Never permanently remove challenging scenes, object types or failure conditions because they
  lower success rates.
- Keep original failures reproducible, behind a parameter or a challenge test.
- If development needs a simplified environment, keep the harder variant as a separate
  challenge benchmark.

**Rule 5: Never hardcode benchmark answers.** Forbidden:
- seed-specific branches;
- looking up true joint types;
- privileged action selection per mechanism family;
- hardcoded solutions for particular task examples;
- returning success without observing the physical result;
- altering simulator state to bypass manipulation;
- training on hidden evaluation labels;
- changing thresholds after seeing held-out outcomes while claiming the evaluation is untouched.

**Rule 6: Evaluate beyond the development distribution.**
- New seeds from one procedural generator are useful but insufficient evidence of broad
  generalisation.
- When justified, add the following:
  - within-distribution held-out seeds;
  - parameter, geometry and dynamics shifts;
  - new mechanism or object families;
  - sensor-noise and calibration perturbations;
  - unseen task compositions.
- Use the smallest set of tests that distinguishes the competing explanations of observed
  performance. Do not invent dozens of benchmarks.

**Rule 7: Freeze evaluation before testing.**
- Keep development, calibration and final evaluation data separate.
- Fix metrics and acceptance gates before the final run.
- A held-out result used to guide a fix becomes development evidence. An unbiased evaluation
  after that needs fresh data.
- Never silently overwrite earlier benchmark results. A rerun goes to a new directory, or the
  README says what changed and why.

**Rule 8: Do not chase 100%.**
- A perfect score on a narrow benchmark is not necessarily meaningful.
- Do not sacrifice generality or physical validity to maximise one metric.
- Measure success, safety, efficiency, uncertainty, failure recognition and transfer together.

**Rule 9: Reject artificial-intelligence theatre.**
- Do not add a neural network or language model to make the system look more intelligent.
- Do not rename heuristics as learning.
- Do not treat generated reasoning text as evidence of physical reasoning.
- Do not claim a reusable skill was learned unless behaviour measurably improves or transfers.

**Rule 10: Allow legitimate engineering.**
- Calibration, better control, better grasp geometry and deliberately narrow initial task
  scopes are all legitimate.
- The requirement is to understand what improved and what changed, and to test whether it
  transfers.
- Do not preserve broken code for the appearance of objectivity. Preserve honest evidence.

---

## 5. Architecture

DREAM is a set of cooperating components, each separately testable, forming one system rather
than separate demonstrations. Build each component only when a vertical slice needs it, and
never create placeholder packages. The MuJoCo Panda is the first manipulation embodiment, not a
limit of the architecture. Mechanism discovery is one tool, not the project's purpose.

The intended hierarchy combines multimodal goal understanding; bounded programmatic skill
composition; persistent object-centric and relational world representation; reusable learned
manipulation policies over robot-specific controllers; active discovery; learned predictive
models only when validated; experience and acquired-skill memory; deterministic safety with
independent evaluation; and embodiment adapters. Planned components are not created until a
vertical slice uses and tests them. Generated task programs must use typed operations, explicit
preconditions and predicates, bounded loops/retries and cancellation, never arbitrary Python.

| | Component | Responsibility | Status / home |
|---|---|---|---|
| A | Embodied reasoning and task planning | Interpret goals, choose skills and tools, track multi-step progress; may use pretrained (V)LMs behind an interface | structured-goal planner and multimodal typed-tool runtime: `dream/runtime/`; bounded task-program composition planned |
| B | Perception and scene understanding | Object poses, relations, affordances *with uncertainty*, from robot sensors | camera backend `dream/perception/` (RGB-D detection, world model; camera driver `dream/sim/camera.py`); oracle-assisted trackers in `dream/sim/tabletop_env.py` as the baseline |
| C | Reusable motor skills and control | Reach, grasp, place, probe, pull along a constraint, release; reuse existing controllers | `dream/skills/` (manipulation, articulation); typed tools in `dream/tools/` |
| D | Physical prediction and world modelling | Interpretable models first; learned dynamics only when they measurably win | quasi-static mechanism model, `dream/agent/model.py` |
| E | Active physical discovery | Choose experiments that reduce task-relevant uncertainty | `dream/agent/` (belief, EIG); on the Panda via `dream/tools/discovery.py` |
| F | Experience and skill memory | Stored outcomes and learned properties that *change future decisions* (logs do not count) | planned, `dream/memory/` |
| G | Verification, recovery and safety | Check expected effects, detect failure or model inadequacy, replan or declare a task unsupported; deterministic low-level safety | partial: `verify_goal`, `Belief.fit_check`, constraint-deviation checks while opening, Panda safety layer |
| H | Robot-independent capability interfaces | Robot-specific actuation behind explicit capability interfaces | `Embodiment`, `Perception`, `CompliantMode` in `dream/types.py`; Panda adapter `dream/robot/panda.py` + `dream/sim/tabletop_env.py`; gantry in `dream/sim/env.py` |

Intended flow, for example:
1. The goal arrives, and the runtime finds that a precondition fails (the container is not
   accessible).
2. Discovery (E) runs experiments through the robot (C, H), using the predictive model (D).
3. The skill (C) operates the container along the learned constraint, while verification (G)
   monitors deviation from the model.
4. Placement and verification complete the task. Memory (F) will later record what was learned.

When the runtime arrives in full, `dream/agent/` will be renamed `dream/discovery/` in a
dedicated refactor commit that keeps the isolation test and all results reproducible.

## 6. Invariants (non-negotiable)

1. **Ground truth stays isolated.**
   - Simulator state about objects and mechanisms (joint types, axes, poses, friction) is used
     only to build scenes, debug and score results in `dream/eval/`.
   - Decision-making code (`dream/agent`, `skills`, `tools`, `runtime`, and later perception and
     memory) must not import `mujoco`, `dream.sim`, `dream.eval` or `dream.robot`.
   - `tests/test_isolation.py` enforces this. Extend it to every new decision-making package and
     every new ground-truth type, and never weaken it.
   - Robot adapters may read the robot's own proprioception, but never pass object ground truth
     upward. Diagnostics that read scene state (for example, which body a hand touched) stay out
     of the `Embodiment` interface and are used only by evaluation.
2. **Real execution only.**
   - Tasks succeed through simulated physics and measured feedback.
   - No scripted animations, teleported objects, direct state writes, attachment or equality
     constraints, or LLM-generated "explanations" in place of execution.
   - Tests may set up scene state before an agent acts. Agents never write it.
3. **No faked or inflated results.**
   - Every number in the README or a commit message comes from a script in this repository that
     was actually run, with config, seeds and git SHA saved under `results/`.
   - Report failed gates and negative results.
   - Never tune on held-out data. If you have seen it, evaluate on fresh data.
4. **Fair baselines.** Compared methods share environments, seeds, budgets, action sets,
   controllers, models and noise. Only the component under test differs.
5. **Name things for what they are.**
   - A fixed sequence is not "planning", and a heuristic is not "learning".
   - Session state is not memory.
   - An oracle-assisted tracker is not perception.
   - Architectural portability is not demonstrated cross-embodiment generalisation.
6. **Reasoning is separate from control.**
   - Language and planning layers never produce raw motor commands and are never the source of
     precise physical state.
   - Safety limits (force, velocity, workspace, joint limits) are deterministic, live below the
     reasoning layer and cannot be overridden by it.
7. **Preserve validated work and honest evidence.**
   - Do not delete or silently change validated components, their tests or their results.
   - If a change alters validated results, rerun the affected evaluations and update the README
     in the same commit or the next one. Say which results are stale until then.
   - Keep original failure cases reproducible (Rule 4).

## 7. How to decide what to build

Before a substantial feature, answer these questions (in the commit message or the PR when
they are not obvious):

- What DREAM capability does this improve (section 3)?
- Why is the existing implementation insufficient? What evidence shows it?
- Does a mature open-source implementation already exist?
- What is the simplest defensible approach?
- Does it generalise beyond one scripted case? On what distribution will that be checked?
- What observable evidence would show improvement?
- What test could prove the approach wrong?
- What does it cost in complexity, compute and dependencies?
- Does it preserve existing verified functionality?

Then:

- Prefer solutions that improve transferable capabilities over ones that improve one demo.
- Use pretrained models and external tools when they meaningfully expand capability. Do not
  train foundation models from scratch when open ones can be reused.
- Stay simulation-first and budget-conscious.
- Do not claim generalisation because an architecture could, in theory, support it.

When a fix works, ask which class in Rule 3 it belongs to before writing the commit message.

## 8. Development priorities

Work on **one coherent vertical slice at a time**. The README names the current milestone.

**Done:**
1. Preserve and validate Active Physical Discovery.
2. A simulated arm with reusable, deterministic manipulation skills.
3. A task runtime with structured tool calling.
4. Several tasks in shared scenes without a script per task.

**Current: whole-body autonomy and active exploration on a mobile G1.** The supported-pelvis
G1 manipulation testbed (`dream/sim/g1.py`) stays as a separate, validated testbed; whole-body
work does not wait for it to be perfect. In dependency order:
- **A. Whole-body mobility** with a pretrained controller, its contract verified against the
  official code and configuration (done for NVIDIA's Decoupled WBC, README Milestone A).
  GEAR-SONIC (motion-token controller, Isaac Lab/TensorRT deployment) is the candidate for
  richer whole-body behaviour; adopt it only after reproducing its official simulation path
  and verifying its observation/action contract. Never feed GR00T `REAL_G1` actions, or any
  learned output, to actuators without a verified contract.
- **B. Goal-directed exploration** of unfamiliar rooms from robot-mounted sensing: varied and
  held-out layouts, a negative control, baselines, recordings.
- **C. Active investigation of an unfamiliar object** reached by the mobile robot: choose
  among viewpoints, external information and bounded physical experiments by expected
  value; represent behaviour outside the fixed/prismatic/revolute family as such.
- **D. Composition**: navigate, discover, manipulate, carry back, on unseen variations.
- **E. Persistent experience** that measurably helps a later task in a different room.
Replace oracle-assisted odometry with estimation from the robot's own sensors as soon as a
slice depends on map accuracy. An external-knowledge tool may search documentation and
manuals; its results are fallible evidence with provenance, never instructions and never a
substitute for measurement, and the system must work without it.

**Cloud compute: standing authorisation.** The owner has authorised routine paid cloud jobs
(GCP project `nerva-adityapersonal`, DREAM's own `cloud/` tooling) for development, training
and evaluation without per-job approval, when the work genuinely needs cloud hardware. The
safeguards are not optional:
- estimate each job's cost (compute, disks, network, storage) before launching; set a per-job
  ceiling and keep a cumulative spending guard for the continuation; stop and report before
  materially exceeding it; distinguish estimated from billed cost;
- machine-enforced deletion deadlines independent of the local process; Spot capacity where
  checkpointing makes interruption acceptable;
- scoped service accounts and least privilege; no tokens or credentials in Git, prompts,
  archives or logs; do not touch NERVA-owned resources;
- cache large dependencies and checkpoints; record source revisions, model versions,
  configuration and seeds; retrieve outputs with checksums before deleting buckets; verify
  cleanup of every VM, disk, address, router, NAT, network, identity and secret.
Run cheap work locally (tests, small MuJoCo diagnostics, CPU policies). Weights, caches,
credentials and large videos stay outside Git and OneDrive.

Avoid tangents. Do not jump to impressive but unrelated experiments, and do not optimise
drawer-opening indefinitely when the next general capability is ready to build. More variants of
a solved task are not progress unless a milestone needs them.

## 9. Adding tools, models and dependencies

Before adding any dependency, pretrained model or upstream code:

- **Inspect first.** Check whether the repository already does it, and prefer extending
  existing code.
- **Reuse maintained open-source work.** For example, MuJoCo Menagerie robot models,
  established controllers and open policies. Do not re-implement mature components.
- **Verify** the exact version, licence, maintenance status and compute needs.
  - Record the licence in the commit message.
  - Do not vendor non-commercial or unclear-licence assets without saying so.
  - Pin versions in `pyproject.toml`.
  - Vendored assets live in `third_party/`, unmodified, with `PROVENANCE.txt`. Adjust them
    through `MjSpec` at build time, documented and classified per Rule 3.
- **Show a concrete need.** Add no large pretrained model without a slice that uses it,
  hardware that can run it, and a baseline it must beat.
  - The core test suite runs on CPU, without network access or paid APIs.
  - LLM and VLM calls sit behind an interface that a deterministic stub replaces in tests.
- **Keep it thin.** Wrap external components behind a small adapter in the package that uses
  them. Add no frameworks, plugin registries or speculative abstraction layers.
- **Stay simulation-first.** Nothing may require physical hardware to validate.

## 10. Code style

- Python >= 3.10, NumPy, MuJoCo 3.16.
- Small flat modules, pure functions, and frozen dataclasses where possible. No speculative
  abstractions or redundant code.
- Pass seeded `numpy.random.Generator` objects explicitly. No global RNG state.
- SI units everywhere. Name the unit when it is ambiguous. Vectorise over particles and
  candidates.
- Type hints on public functions. Docstrings say what and why. Comments are rare and explain
  physics or decisions.
- Constants that encode a decision (thresholds, tolerances, margins) carry a comment saying
  where their value came from: a measurement, a calibration result or a cited value.
- Make small, focused changes.
- `ruff check . && ruff format --check .` must pass.

## 11. Testing and evaluation standards

**Test suite.**
- `pytest -m "not slow"` must pass before every commit. It takes about 40–60 s now; keep it
  under 60 s.
- Mark long simulation tests `@pytest.mark.slow`. The full suite must pass before pushing a
  milestone.
- Every bug fix gets a regression test first.
- After an edit made by a script, verify that it applied (an assertion or a test). Silent
  no-op replacements have already caused bugs here.

**Standards for every capability:**
- **Physical validity.** Actions run through real simulation dynamics and the robot controller.
- **Observation isolation.** Decision code cannot read privileged simulator information
  (invariant 1).
- **Reproducibility.** Record seeds, parameters, versions, code revision and configuration.
- **Contracts, not smoke.** Test physical and statistical contracts. For example: "a revolute
  hypothesis predicts no motion for a pull along its axis", "a skill reports failure when the
  grasped object slips", "a fixed box is declined without an opening attempt".
- **Capability-based tests.** Prefer tests of transferable behaviour, across seeds and
  parameter ranges, over tests of one predefined trajectory.
- **Adversarial evaluation.** Test reasonable conditions under which a proposed improvement
  might fail: parameter shifts, geometry shifts, noise, disturbances.
- **Regression testing.** Validated capabilities stay functional. Rerun their evaluations when a
  shared component changes.
- **Independent verification.** Compare the agent's success claims with ground truth, without
  letting that truth reach the agent. Count false claims.
- **Fair comparison** (invariant 4).
- **Negative results** are legitimate results. Preserve and report them.
- **Calibration discipline.** Thresholds and hyperparameters are calibrated on development data
  by a saved procedure, then frozen. Never adjust them against final results.
- **Falsifiable acceptance.** Every major feature has an acceptance test or gate written before
  its final evaluation, which a wrong approach would fail.
- **Tolerances** use fixed seeds and are justified by measured noise.

## 12. Experiments

- Configs go in `experiments/configs/*.json`, and runners in `experiments/*.py`.
- Outputs go in `results/<name>/`: per-episode logs, a summary, plots, and metadata (git SHA
  at the start of the run, versions, config, seeds).
- Commit results only when they back a README claim.
- Keep seed ranges disjoint and record them in configs: development, calibration, held-out
  evaluation, and shifted or challenge sets.
- Calibrate on development data, freeze, then evaluate on held-out data.
- Pre-register gates in the config before running them, and report every gate, pass or fail.
- Start long runs only after a smoke config has run the full pipeline.
- Do not change code an evaluation imports while it runs. Commit after it finishes, so the
  recorded SHA is the code that ran.

## 13. Source control, documentation and scope

- Develop on a focused branch per milestone, created from the latest validated branch. Never
  overwrite or discard unmerged work.
- Commit in small, meaningful checkpoints that pass the fast suite. Commit messages say what
  changed, why, how it was verified, and the Rule 3 class of any change that affects results.
  No model identifiers in commits.
- Do not open pull requests unless asked.
- There are exactly two Markdown files: `README.md` and this file. Do not add more. Design notes
  belong in docstrings, and history belongs in commit messages.
- Update the README status table, results and audit table in the same commit as the change that
  affects them. Delete stale text rather than appending to it.
- Never mark a milestone "implemented and tested" without execution evidence in `results/` or
  the test suite.
- New top-level packages from the architecture table are created only together with their first
  working, tested code.

## 14. Layout

```
dream/types.py        Shared data and protocols (observations, RobotState, CompliantMode, Embodiment, Perception)
dream/sim/            MuJoCo scenes (ground truth): mechanisms + gantry, tabletop + Panda, compartment,
                      supported G1, mobile G1 + procedural rooms;
                      the RGB-D camera driver (renders sensor arrays only)
dream/robot/          Panda controller; supported G1 controller; pretrained G1 WBC locomotion adapter (g1_wbc.py)
dream/agent/          Active Physical Discovery: belief, predictive model, probe selection, operation
dream/perception/     Camera perception: RGB-D geometry, detection, world model, VisionPerception;
                      occupancy map and spatial memory for the mobile robot
dream/skills/         Closed-loop skills over Embodiment/Perception (manipulation, articulation)
dream/tools/          Typed tools, structured goals, investigate_physics/opening (discovery.py)
dream/navigation/     Path following, planning, frontiers (odometry frame; no simulator access)
dream/runtime/        Task agent loop and planners (structured, multimodal, room search)
dream/session.py      Live session: physics thread + agent thread (wiring, like dream/eval)
dream/eval/           Episode runners and scoring against ground truth
experiments/          Reproducible runners, configs, calibration, demos, interactive session CLI
third_party/          Vendored external assets (pinned, with licence and PROVENANCE.txt)
tests/                pytest suite, including the isolation guard
results/              Saved outputs that back README claims
```
