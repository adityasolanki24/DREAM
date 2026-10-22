# DREAM

A simulated robot meets a mechanism it has never seen, a panel with a handle that might be fixed,
slide, or swing on a hinge. It keeps competing explanations of how the panel moves, picks
the physical experiment expected to best tell them apart, predicts what that experiment will do,
runs it, updates its beliefs from what it measures, and then opens the mechanism.

```
observe -> hypothesise -> predict -> act (probe) -> measure -> update -> ... -> operate
```

This is the "Stage Zero / Stage One" core of the Physics Apprentice plan (see the research
report): explicit Bayesian system identification with information-seeking probes, in MuJoCo, with
no learned components. It tests whether choosing experiments for their information gain
identifies mechanisms faster than other probing strategies.

## Install

Python >= 3.10 (developed on 3.13). CPU only.

```bash
pip install -e ".[dev]"            # mujoco==3.16.0, numpy, matplotlib, pytest, ruff
sudo apt-get install libosmesa6    # only for headless rendering of the demo GIF
```

## Run

```bash
pytest -m "not slow"                                   # fast suite (~20 s)
pytest                                                 # + closed-loop episode tests (~1 min)

python experiments/demo.py --seed 3                    # hidden mechanism drawn from the seed
python experiments/demo.py --family left_hinge         # or pick the hidden family
python experiments/demo.py --family slider --viewer    # live MuJoCo viewer (needs a display)

python experiments/run_benchmark.py experiments/configs/main.json --workers 4            # ~15 min
python experiments/calibrate_adequacy.py --workers 4                                    # ~5 min
python experiments/run_benchmark.py experiments/configs/generalisation.json --workers 4  # ~20 min
```

The demo prints, for every probe, the chosen probe and its expected information gain, the
predicted motion distribution, the observed motion and the updated belief. It writes
`results/demo/<family>_<seed>/` containing `demo.gif` (sampled hypotheses drawn as translucent
hinge axes in blue and slide arrows in orange), `summary.png` and `log.json`.
Benchmarks write `results/<name>/`: every step of every episode (`episodes.jsonl`),
`summary.json`, plots, and `meta.json` with the git SHA, versions and config.

## How it works

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
  that assumption implicitly.
- **Goal-directed pulling** never identifies sliders or fixed panels (0/24).

### Generalisation (`results/generalisation`)

_Pending: the run is in progress._

## Limitations (read before citing any number)

- **No camera perception.** The handle pose comes from a simulated tracker with Gaussian noise.
  The report's MOSART warning applies: perception error may dominate on a real system.
- **Simplified contact.** The grasp is an ideal ball-joint constraint, the mechanism has no
  collision geometry (joint limits act as stops), and panels are gravity-compensated.
- **Quasi-static predictor.** It predicts settled outcomes only. Inertial overshoot is absorbed
  by the noise model rather than modelled.
- **Travel limits** are only learned if a probe reaches them, so predictions near an unvisited
  limit are poor.
- **Single-joint model class.** Mechanisms outside it are detected only through the χ²
  adequacy check (see Generalisation).
- The live viewer (`--viewer`) was not exercised here: development ran in a headless container.
