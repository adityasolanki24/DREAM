# DREAM Research / Development Report: Permanent Reports system; Isaac semantic exploration with learned perception, knowledge sources and larger homes

## Task identification
- Task title: Permanent Reports system; Isaac semantic exploration with learned perception, knowledge sources and larger homes
- Task ID: 2026-11-09-001
- Date and time: 2026-11-09T13:58:11+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: claude/isaac-semantic-research-20261109
- Starting commit: df00ae9e30cb6c32ade4d3155e4c83cac489b6a2
- Final commit: efd6872f4c375f948244e04c5beb71eade626925
- Task status: partially completed

## A. Objective

This task has two parts.

1. **A permanent reporting system.** Build a hierarchical `Reports/` record: one dated task per development prompt and one record per experimental run, with indexes and lightweight tooling. Make it mandatory in AGENTS.md for every future prompt.
2. **Continue the semantic object-search research.** The question is whether semantic world knowledge improves how an embodied robot searches unfamiliar environments when it relies on genuine visual observation and physical navigation. Isaac Sim is the primary realistic environment, using:
   - the Unitree G1 with the pretrained whole-body controller;
   - rendered RGB-D;
   - learned or VLM perception instead of the segmentation oracle;
   - DREAM's existing navigation, memory, tools and verifier.

   The research requires:
   - an audit of hardcoding and oracle use;
   - validated assets and larger multi-room environments;
   - a comparison of ConceptNet, Qwen, combined and no knowledge;
   - better active exploration;
   - frozen held-out comparisons against frontier baselines, with ablations;
   - cloud GPU runs with reliable self-deletion and a report for every run.

## B. Starting state

Branch `claude/semantic-goal-search-20261108` at `9ed947c`, pushed on 9 November 2026 and not merged into `main`. The README's status, results and Git references describe this state, with evidence in `results/`.

- **Original-goal consistency.** Request-to-goal substitution is fixed in the tested paths.
  - The fault-injection gate holds: 7/40 false claims without the original-goal record, 0/40 with it.
  - Isaac held-out re-test (frozen at `c4100e2`): the old path claimed success on the wrong goal in 4 of 6 cases, and the new path in 0 of 9. Those gates are incomplete: `l_island` never loaded, and 6 cases were not run.
- **Object search in MuJoCo** used an oracle-assisted segmentation detector and a kinematic base, except H4, which ran on the G1 controller.
  - H1: the proposed planner found 83 of 87 targets with 0 of 120 false claims.
  - H2: 1 of 48 false claims, a shower claimed as a bathtub. Gates G0 and G7 fail.
  - G3, the speed gate: the planner was no faster than the gain-frontier explorer (median paired difference 0 s, p = 0.73).
  - H2's larger homes showed a non-significant hint: 5 discordant pairs to 0 against the no-semantics ablation, p = 0.063.
- **Verifier fix.** The Wilson lower-bound check was confirmed on fresh seeds (V1: 0 of 192 false claims, 94 of 96 found, against 93) and is the default since `6d680b4`.
- **Knowledge.** ConceptNet never recognises laundry or storage rooms. Qwen's text priors agreed with the benchmark's furnishing tables in 23 of 27 classes, against 17 for ConceptNet. The agent has never used them.
- **Isaac.** The five NVIDIA kitchens are single rooms. Their refrigerator and oven assets return 404. Counter-mounted objects never verify: the sink track's purity was 0.39, because Qwen's countertop and cabinet boxes fuse with the sink.
- **Cloud.** About US$31.20 estimated for 8-9 November. The project was audited clean, and only the runtime image is kept.
- **Reports.** No `Reports/` directory existed. AGENTS.md allowed only two Markdown files.

## C. Work performed

### Implementation

Added the Reports workflow and observation-isolation hardening. Built the Isaac asset checks and procedural home composer, comparison tooling for offline detectors, and the optional semantic evidence from unexplored rooms.

### Evaluation

Completed K1 on larger MuJoCo homes. Ran Isaac asset, capture and standing diagnostics, and a local development check of the unexplored-room option. The 10 November note records a separate offline detector comparison.

### Runs

The 23 registered runs are indexed under runs/. The session 5 offline detector comparison is described in the 10 November note. No dedicated run record for that comparison was found in this checkout. Its numbers remain development notes.

## D. What was achieved

K1 completed 576 episodes. Its false-claim and target-present gates passed, and all arms refused the absent-target cases. The Isaac work produced asset diagnostics, a fix for extra PhysicsScene prims and development captures from composed homes, as recorded in the run reports and final progress note.

## E. What was not achieved

No learned detector was evaluated in a closed-loop search on the composed Isaac homes. K2 was not frozen or run. experiments/configs/semantic_exploration_k2.json and experiments/configs/isaac_home_dev.json preserve the drafts in this checkout.

## F. Passed tests and experiments

K1 passed K-G1 and K-G3. The detector pipeline completed its CPU stub check, which is a plumbing result. Isaac validation and capture smoke results are in their respective run records.

## G. Failed tests and experiments

The primary K-G2 speed gate failed for all knowledge sources. The unexplored-room option did not improve the development comparison consistently. Missing assets, failed standing checks and counter-mounted object recognition failures remain in the run records.

## H. Improvements

The 10 November note reports that removing the extra PhysicsScene prims allowed four development homes to stand, cross rooms and capture frames. This is development evidence. No search-efficiency improvement was demonstrated.

## I. Regressions and dis-improvements

The K1 utility arms walked farther on average than the frontier arms. In the exploratory room check, ConceptNet and Qwen each found five fewer targets with the option, while combined found five more. Other effects were not measured.

## J. Comparison against baseline

K1 knowledge arms had 0 s median paired time difference against the frontier baseline and Holm-adjusted p = 1.0 for all three. The same utility planner without knowledge found 62/64 targets and was indistinguishable on timing from the knowledge arms.

## K. Research integrity and assumptions

K1 used the existing generator, oracle-assisted detections and odometry, and a kinematic base. Isaac and detector results in the 10 November progress note are development observations. The closeout preserves the recorded evidence and adds no new run.

## L. Cloud resource usage

The 10 November note gives US$30.50 estimated for the Isaac work in this task and records cleanup verification per session. Billed costs were not read. See individual run records for evidence and the retained runtime image.

## M. Remaining work

The R0 adoption defers the K2, detector integration and wider Isaac work. Future work can resume from the saved draft configurations and notes when those capabilities become dependencies.

## N. Final conclusion

The reporting system and development tooling were delivered. The primary semantic search claim failed its frozen speed gate, and the Isaac learned-perception loop remained incomplete.
