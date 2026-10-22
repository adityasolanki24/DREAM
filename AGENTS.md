# DREAM: engineering and architecture guide

This is the authoritative guide for coding agents and contributors. `CLAUDE.md` is a symlink to
this file, so there is exactly one version of these instructions. `README.md` holds the
milestone status table and the measured results. Read both before changing code.

## Mission

DREAM aims to become a general-purpose embodied agent. Given a natural-language goal such as "put
the red cube in that cabinet", "figure out how this mechanism works" or "retrieve the object you
cannot reach", it should be able to:

1. interpret the goal and the scene;
2. plan with reusable motor skills;
3. execute through real (simulated) physics;
4. notice when physical knowledge is missing, and run experiments to acquire it;
5. verify outcomes and recover from failure;
6. improve from experience.

It should do this without a hand-written script for every task.

**This is a research direction, not a capability claim.** The scientific question is:

> Does an embodied agent solve unfamiliar tasks more effectively when it combines general skills
> with active physical discovery, predictive reasoning and experience-driven adaptation?

Every capability must be measured against relevant baselines in simulated physics with
observable feedback.

## What exists today (do not overstate it)

Two validated foundations exist, and they are **not yet connected**.

1. **Active Physical Discovery** (gantry robot, `dream/agent/`):
   - A force-limited 3-axis gantry probes single-joint mechanisms in MuJoCo: fixed, prismatic
     and revolute panels, plus a held-out helical mechanism.
   - A per-type SMC belief, a quasi-static predictive model and expected-information-gain probe
     selection identify those mechanisms. They beat random probing by about 60% in median
     probes, with paired statistics.
   - A χ² check, validated on fresh data, flags an out-of-class mechanism.
2. **Panda manipulation foundation** (`dream/robot`, `skills`, `tools`, `runtime`):
   - A simulated Franka Panda (Menagerie model) with a deterministic safety-limited controller.
   - Contact-based grasping and closed-loop skills.
   - A typed tool interface and a rule-based task runtime for *structured* goals ("in red
     tray", "on red blue").
   - A live session with pause and cancel.

Object poses come from a simulated tracker (oracle plus noise), not cameras. There is no
natural-language understanding (structured goals only; no LLM is integrated), no memory, and
discovery is not yet callable as a tool from the runtime.

The README status table is the source of truth for what is done. Anything not marked
"implemented and tested" there must not be described as working.

## Target architecture

These are cooperating components, each separately testable. Build each one only when a vertical
slice needs it, and never create placeholder packages.

| | Component | Responsibility | Status / home |
|---|---|---|---|
| A | Embodied reasoning | Interpret goals, choose skills and tools, track multi-step progress. May use pretrained (V)LMs. | rule-based planner over structured goals: `dream/runtime/` (LLM planner planned) |
| B | Perception | Object poses, relations, affordances *with uncertainty*, from robot sensors | planned, `dream/perception/` |
| C | Motor skills | reach, grasp, place, push, pull, follow a constrained path. Reuse existing controllers and policies. | reach/grasp/place: `dream/skills/`; typed tools: `dream/tools/` |
| D | Predictive physics | Interpretable models first; learned dynamics only when they measurably win | exists for mechanisms, `dream/agent/model.py` |
| E | Active discovery | Choose experiments that reduce uncertainty when knowledge is missing | **exists**, `dream/agent/` |
| F | Experience memory | Stored outcomes and learned properties that *change future decisions*. Logs alone do not count. | planned, `dream/memory/` |
| G | Verification and safety | Check expected effects, detect failure or model inadequacy, replan or declare a task unsupported. Deterministic low-level safety limits. | partial: `verify_goal`, `Belief.fit_check`, Panda safety layer |
| H | Embodiment adapters | Robot-specific actuation behind explicit capability interfaces | Panda: `dream/robot/panda.py` + `Embodiment` protocol (`dream/types.py`); the gantry lives in `dream/sim/env.py` |

Intended flow, for example: goal → runtime plans "open cabinet, then place cube" → the open
skill fails or its model is inadequate → the runtime invokes discovery (E) → discovery learns
the constraint → skill (C) operates it → verification (G) checks the goal → memory (F) records
what was learned.

When the runtime arrives, `dream/agent/` will be renamed to `dream/discovery/` in a dedicated
refactor commit that keeps the isolation test and all results reproducible.

## Invariants (non-negotiable)

1. **Ground truth stays isolated.** Simulator state about objects and mechanisms (joint types,
   axes, poses, friction) is used only to build scenes and to score results in `dream/eval/`.
   Decision-making code (`dream/agent/`, and later runtime, skills, perception and memory) must
   not import `mujoco`, `dream.sim` or `dream.eval`. `tests/test_isolation.py` enforces this.
   Extend it to every new decision-making package, and never weaken it. Robot adapters may
   read the robot's *own* proprioception but must never pass object ground truth upward.
2. **Real execution only.** Tasks succeed through simulated physics and measured feedback.
   No scripted animations, teleported objects, direct state writes, or LLM-generated
   "explanations" in place of execution.
3. **No faked or inflated results.** Every number in the README or a commit message comes from a
   script in this repository that was actually run, with its config, seed and git SHA saved under
   `results/`. Report failed gates and negative results. Never tune on held-out data, and if
   you have already seen it, evaluate on a fresh seed.
4. **Fair baselines.** Compared methods share the same environments, seeds, budgets, action
   sets, models and noise.
5. **Name things for what they are.** A fixed sequence is not "planning", a heuristic is not
   "learning", and architectural portability is not demonstrated cross-embodiment
   generalisation.
6. **Reasoning is separate from control.** Language and planning layers never produce raw motor
   commands and are never the source of precise physical state. Safety limits (force,
   velocity, workspace, joint limits) are deterministic, live below the reasoning layer, and
   cannot be overridden by it.
7. **Preserve validated work.** Do not delete or silently change the discovery component, its
   tests or its results. If a change alters those results, rerun the benchmark and update the
   README in the same commit.

## Development priorities

Work on **one coherent vertical slice at a time**. The README names the current next milestone.

1. Preserve and validate Active Physical Discovery. *(done; follow-ups are tracked in the README)*
2. A simulated robot arm with reusable, deterministic manipulation skills.
3. A task runtime with structured tool calling over those skills.
4. Several tasks in shared scenes without a full script per task.
5. Visual perception that replaces oracle poses.
6. Discovery invoked on demand when a skill fails or a model is inadequate.
7. Memory, failure-driven replanning and skill adaptation.
8. Evaluation of generalisation across goals, arrangements and task families.
9. Learned world models, mobile manipulation or whole-body embodiments, only when experiments and
   compute justify them.

Avoid tangents. More mechanism variants are not progress unless a milestone needs them.

## Adding tools, models and dependencies

Before adding any dependency, pretrained model or upstream code:

- **Inspect first.** Check whether the repository already does it, and prefer extending existing
  code.
- **Reuse maintained open-source work.** Examples: MuJoCo Menagerie robot models, established
  controllers, open policies. Do not re-implement mature components.
- **Verify** the exact version, licence (record it in the commit message, and do not vendor
  non-commercial or unclear-licence assets without saying so), maintenance status and compute
  needs. Pin versions in `pyproject.toml`.
- **Show a concrete need.** Add no large pretrained model without a slice that uses it,
  hardware that can run it, and a baseline it must beat. The core test suite must run on CPU
  without network access or paid APIs. LLM and VLM calls sit behind an interface that can be
  replaced by a deterministic stub in tests.
- **Keep it thin.** Wrap external components behind a small adapter in the package that uses them.
  Add no frameworks, plugin registries or speculative abstraction layers.
- Stay simulation-first. Nothing may require physical hardware to validate.

## Code style

- Python >= 3.10, NumPy, MuJoCo 3.16. Use small flat modules, pure functions and frozen
  dataclasses where possible.
- Pass seeded `numpy.random.Generator` objects in explicitly. No global RNG state.
- Use SI units everywhere, and name the unit when it is ambiguous. Vectorise over particles
  and candidates.
- Type hints on public functions. Docstrings say what and why. Comments are rare and
  explain physics or decisions.
- `ruff check . && ruff format --check .` must pass.

## Testing

- `pytest -m "not slow"` must pass before every commit (currently about 40 s; keep it under 60 s).
  Mark long simulation tests `@pytest.mark.slow`. The full suite must pass before pushing a
  milestone.
- Test physical and statistical *contracts*, not just smoke. For example: "a revolute
  hypothesis predicts no motion for a pull along its axis", or "a skill reports failure when
  the grasped object slips".
- Use fixed seeds and tolerances justified by measured noise.
- Every bug fix gets a regression test first.
- After any edit made by a script, verify that the edit actually applied, for example with an
  assertion or a test. A silent no-op replace has already caused one bug here.

## Experiments

- Configs go in `experiments/configs/*.json`, and runners in `experiments/*.py`.
- Outputs go in `results/<name>/`: per-episode logs, summary, plots and `meta.json` (git SHA,
  versions, config, seeds).
- Commit results only when they back a README claim.
- Calibrate on development data, freeze, then evaluate on held-out data. Pre-register gates
  before running them.

## Documentation and scope control

- There are exactly two Markdown files: `README.md` (mission summary, status table, install and
  run, measured results, limitations, next milestone) and this file. Do not add more. Design
  notes belong in docstrings, and history belongs in commit messages.
- Update the README status table and results in the same commit as the change that affects them.
  Delete stale text rather than appending to it.
- Never mark a milestone "implemented and tested" without execution evidence in `results/` or
  in the test suite.

## Layout

```
dream/types.py      Shared data and protocols (observations, RobotState, Embodiment, Perception)
dream/sim/          MuJoCo scenes and environments: mechanisms + gantry, tabletop + Panda (ground truth)
dream/robot/        Panda controller and safety layer (robot kinematics only, no scene access)
dream/agent/        Active Physical Discovery: belief, predictive model, probe selection, operation
dream/skills/       Closed-loop manipulation skills over Embodiment/Perception
dream/tools/        Typed tool interface and structured goals
dream/runtime/      Task agent loop and planners
dream/session.py    Live session: physics thread + agent thread (wiring, like dream/eval)
dream/eval/         Episode runners and scoring against ground truth
experiments/        Reproducible runners, configs, demo, interactive session CLI
third_party/        Vendored external assets (pinned, with licence and PROVENANCE.txt)
tests/              pytest suite, including the isolation guard
results/            Saved outputs that back README claims
```

New top-level packages from the architecture table are created only together with their first
working, tested code.
