# DREAM Research / Development Report: Locate and test the causes of R1 grasp failures

## Task identification
- Task title: Locate and test the causes of R1 grasp failures
- Task ID: 2026-11-27-002
- Date and time: 2026-11-27T04:33:03+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: 96964b8b013cf4f28b570520d491d9c9702dd20d
- Final commit: 3425e62
- Task status: partially completed

## A. Objective

Explain the failures of the complete R1 tabletop grasp distribution and test a correction for unsafe or unavailable nominal approaches.

## B. Starting state

The latest controller had verified targeted walking holds but lacked a complete 50-case development result. Cylinder and mug pregrasp refusals, a sphere without sustained opposing contact and loss of load during lifting had been observed.

## C. Work performed

### Implementation
Added a bounded arm correction from encoders after nominal reach refusal. Each increment must reduce hand pose error in a guarded WBC forecast and the live supervisor monitors execution.

### Evaluation
Ran the unmodified controller on all 50 cases from a clean detached worktree. Separately replayed two refused approaches and a walking control with measured-posture refinement.

### Runs
Run records: see runs/ and the daily index
The baseline-50 and measured-reach records give configurations, commands and file checksums.

## D. What was achieved

Measured the first complete development rate and located the dominant failure stage. Recovered one previously refused cylinder approach.

## E. What was not achieved

R1 missed the 35/50 target. The revised approach has no broad rate, held-out evaluation or final full-suite validation.

## F. Passed tests and experiments

Forty focused cohort, pose interpolation and isolation cases passed. Changed Python files passed Ruff check and format. The baseline bottle 1003 and mug 1024 held. The matched walking control retained the verified hold.

## G. Failed tests and experiments

Baseline first failure stages: walk 2, raise 2, pregrasp 29, reach 2, preclose 3, close 2, lift or hold 8. The cylinder and mug still failed eventual holding in the replay. The pose-step test initially failed for the missing helper. Its SciPy angle construction was corrected before it passed.

## H. Improvements

The matched cylinder advanced from a refused approach to opposing touch. Independent scoring rejected the subsequent loss. The sphere trace showed physical contact without sustained bilateral preload.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
The walking control retained its hold. Regressions outside these matched cases remain unmeasured. Historical tests apply to their original source.

## J. Comparison against baseline

Two of 50 baseline episodes passed. The three matched physical replays tested previously executed choices, separately from the broad run.

## K. Research integrity and assumptions

Development cases only. Pose, geometry and odometry are oracle-assisted. Friction is assumed. Object material is private to execution. All motion is physical and the correction uses measured pose and forecast progress.

## L. Cloud resource usage

Local CPU and cached ONNX controllers only. No cloud jobs or costs.

## M. Remaining work

At the below-40% review the owner retained the Dex3 and all five shapes and prioritised stance and approach. Broad current-tree verification, reliability of contact and the original held-out gate remain open.

## N. Final conclusion

Pregrasp was the most frequent loss. Measured-arm corrections recovered one approach while preserving safety checks. The full R1 goal remains open.
