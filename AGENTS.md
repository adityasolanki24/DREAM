# Agent and contributor guide

Read this before changing code. It applies to humans and coding agents alike.
`CLAUDE.md` is a symlink to this file.

## What this repository is

DREAM is a research codebase. A simulated robot meets an articulated mechanism it does not
recognise, keeps competing hypotheses about how the mechanism moves, chooses experiments to tell
them apart, updates its beliefs from what it measures, and then operates the mechanism.
The scientific claim under test is that **choosing experiments for their information gain
identifies mechanisms with fewer interactions than random probing**. The code has to be able to
show that claim is false as easily as it shows it is true.

## Non-negotiable integrity rules

1. **Privileged information never reaches the agent.** `dream/agent/` must not import `mujoco`,
   `dream.sim`, or `dream.eval`. The agent receives only `dream.types` objects that the
   environment builds from sensor readings, plus the disclosed part geometry (panel extent and
   initial handle pose). `tests/test_isolation.py` enforces this. Do not weaken that test.
2. **Ground truth is for scoring only.** `MechanismSpec` (joint type, axis, pivot, friction) may
   be used by `dream/sim` to build scenes and by `dream/eval` to score results. It is never
   passed to a probe selector, belief, or controller.
3. **No faked results.** Every number, plot, or claim in the README or a commit message must come
   from a script in this repository that you actually ran, with its seed and config saved under
   `results/`. If an experiment fails or a gate is not met, report that.
4. **Baselines run under identical conditions:** same mechanisms, seeds, probe budget, candidate
   probe set, belief model, and noise. Otherwise the comparison is invalid.
5. **Name things for what they are.** Do not call a heuristic "learning" or a fixed sequence
   "planning". Add a learned model only when an experiment shows it does measurably better than
   the explicit model it replaces.

## Code style

- Python >= 3.10, NumPy, MuJoCo. Add no dependency without a measured need.
- Small, flat modules. No speculative abstraction layers, plugin registries, or placeholder files.
  Write code when a feature needs it.
- Pure functions and frozen dataclasses wherever possible. Seeded `numpy.random.Generator`
  objects are passed in explicitly; no global RNG state.
- SI units everywhere (metres, radians, newtons, seconds). Name the unit in the variable or
  docstring when it is ambiguous.
- Vectorise across particles and probes with NumPy. Do not loop over particles in Python.
- Type hints on public functions. Docstrings say *what* and *why*, and the code says *how*.
  Comments are rare and explain non-obvious physics or decisions.
- `ruff check . && ruff format --check .` must pass.

## Testing

- `pytest -m "not slow"` must pass before every commit. It should stay under about 60 s.
- Every module gets tests for its physical or statistical contract, not just smoke tests.
  Examples: "a revolute hypothesis predicts zero displacement for a pull parallel to its axis",
  "the posterior concentrates on the true type after diagnostic probes".
- Simulation tests use fixed seeds and tolerances justified by the noise model.
- Bug fix means a regression test first.
- Mark long simulation or benchmark tests `@pytest.mark.slow`.

## Workflow

1. Make one focused change. Run it, then test it.
2. Fix failures before moving on. Never commit red tests.
3. Commit small working checkpoints with messages that state what changed and what was verified.
4. Experiments: put the config in `experiments/configs/*.json` and the runner in
   `experiments/*.py`. Outputs (metrics JSON, plots, generated MJCF) go to `results/<name>/`
   together with the git SHA and seeds.

## Documentation

- There are exactly two Markdown files: `README.md` (what it is, how to install and run, current
  measured results) and this file. Do not add more. Put design notes in docstrings, and put
  history in commit messages.
- When behaviour or results change, update the README section involved in the same commit.
  Delete stale text rather than adding to it.

## Layout

```
dream/types.py      Data passed between sim and agent (observations, probes, geometry)
dream/sim/          MuJoCo scenes, mechanism generator, environment (holds ground truth)
dream/agent/        Belief, predictive model, probe selection, operation (no simulator access)
dream/eval/         Metrics that compare agent beliefs with ground truth
experiments/        Reproducible experiment runners and configs
tests/              pytest suite
results/            Saved experiment outputs (committed when they back a README claim)
```
