# DREAM Research / Development Report: Implement the G1 roadmap one validated rung at a time

## Task identification
- Task title: Implement the G1 roadmap one validated rung at a time
- Task ID: 2026-11-10-004
- Date and time: 2026-11-10T10:36:01+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: efd6872f4c375f948244e04c5beb71eade626925
- Final commit: efd6872f4c375f948244e04c5beb71eade626925
- Task status: partially completed

## A. Objective

Advance R0 only, clearing lint and the monitored reach regression before promoting the integration branch or resuming R1.

## B. Starting state

The branch had an uncommitted WBC/Dex3 port. The source and integration branches both reproduced the slow reach rollback failure. Project-wide lint and formatting were red on the base.

## C. Work performed

### Implementation

Formatted and repaired lint in the existing active experiment scripts. Excluded the immutable bisect evidence copy using the same principle as the existing source snapshot exclusion. Added a bounded, supervised hand-stability check before calculating reach corrections and assessing rollback. Kept the original independent hand-error assertion and reset the simulator between measured and widened capability checks.

### Evaluation

Compared the original failure, arm-only correction, unchanged-command control, the stability check, and a shifted target with a separate development seed. The shifted test completed after the conversation was interrupted.

### Runs

Five executions, with reports under runs/ and checked traces stored outside the repository.

## D. What was achieved

The original and shifted physical checks pass with final independent hand error of 4.0 and 4.1 mm. Repo-wide Ruff lint and formatting pass.

## E. What was not achieved

The turn ended before full integration verification. No new R1 tests or capability claims were made. Integration edits remained uncommitted.

## F. Passed tests and experiments

The original and shifted simulated reach checks passed. Repository-wide Ruff lint and formatting passed.

## G. Failed tests and experiments

The original regression and arm-only change both failed the same hand-error assertion. An unchanged-command control exposed post-motion base drift. These are retained as separate run records.

## H. Improvements

The monitor now observes a stationary window rather than assuming the hand is stable after the nominal posture hold. On the original development regression, final independent hand error decreased from 38.08 mm after rollback to 4.0 mm after correction.

## I. Regressions and dis-improvements

No regression was detected in the two simulated reach checks. Full suite evidence was not available at the end of the turn.

## J. Comparison against baseline

At unchanged command the simulated base moved about 26 mm in x during the first extra hold; the independent hand trace and the encoder/odometry estimate agreed on the drift. Holding torso command constant alone did not fix the failure.

## K. Research integrity and assumptions

Class: general algorithmic correction for evaluating a moving hand as though settled. The bounded window uses the existing 1 mm improvement resolution. Cases are development evidence. No thresholds were relaxed.

## L. Cloud resource usage

None.

## M. Remaining work

Run the complete fast and full suites and isolation checks before proceeding to R1. Task 2026-11-10-005 continues that work.

## N. Final conclusion

R0 lint and the known physical regression were repaired. Full integration verification remains the gate for grasp development.
