# DREAM Research / Development Report: Permanent Reports system; Isaac semantic exploration with learned perception, knowledge sources and larger homes

## Task identification
- Task title: Permanent Reports system; Isaac semantic exploration with learned perception, knowledge sources and larger homes
- Task ID: 2026-11-09-001
- Date and time: 2026-11-09T13:58:11+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: claude/isaac-semantic-research-20261109
- Starting commit: df00ae9e30cb6c32ade4d3155e4c83cac489b6a2
- Final commit: _TO COMPLETE_
- Task status: in progress

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
_TO COMPLETE_

### Evaluation
_TO COMPLETE_

### Runs
Run records: see runs/ and the daily index
_TO COMPLETE_

## D. What was achieved

_TO COMPLETE_

## E. What was not achieved

_TO COMPLETE_

## F. Passed tests and experiments

_TO COMPLETE_

## G. Failed tests and experiments

_TO COMPLETE_

## H. Improvements

_TO COMPLETE_

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
_TO COMPLETE_

## J. Comparison against baseline

_TO COMPLETE_

## K. Research integrity and assumptions

_TO COMPLETE_

## L. Cloud resource usage

_TO COMPLETE_

## M. Remaining work

_TO COMPLETE_

## N. Final conclusion

_TO COMPLETE_
