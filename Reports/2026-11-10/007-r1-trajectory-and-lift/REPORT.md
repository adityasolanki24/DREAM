# DREAM Research / Development Report: Resolve the R1 trajectory and grasp without fixed answers

## Task identification
- Task title: Resolve the R1 trajectory and grasp without fixed answers
- Task ID: 2026-11-10-007
- Date and time: 2026-11-10T12:12:35+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: 52d3c4aa200cd5f08b3201559015315024ae994b
- Final commit: 52d3c4aa200cd5f08b3201559015315024ae994b
- Task status: partially completed

## A. Objective

Investigate the R1 hand-clearance stop without storing a scene-specific motion.

## B. Starting state

At 52d3c4a the guard stopped the right index finger near the table before closure.

## C. Work performed

Implemented sensor-initialised physics preview and tested joint ordering, geometrically varied targets and sampled detours. See the four run records.

## D. What was achieved

The preview rejected unsafe moves before live actuation. It ruled out the tested ordering and midpoint families.

## E. What was not achieved

No path or lift was verified in this session. The final run finished after the user interrupted the turn.

## F. Passed tests and experiments

Ruff passed for the experimental code. The preview reset crash was corrected and subsequent runs completed.

## G. Failed tests and experiments

The first preview crashed during odometry update. The next three runs refused every candidate due to IK, clearance, force or torque.

## H. Improvements

Failure can now be detected in a separate physics preview before arm motion.

## I. Regressions and dis-improvements

No new live reach was executed. Broader regressions were not measured.

## J. Comparison against baseline

The direct, reordered, endpoint-varied and geometric waypoint approaches each refused the original case.

## K. Research integrity and assumptions

The preview used observed robot state and labelled fixture geometry. It did not copy live contact partners or object dynamics. Results are development evidence.

## L. Cloud resource usage

None.

## M. Remaining work

Continue in task 2026-11-10-008 using the measured initial wrist and stability evidence.

## N. Final conclusion

The interrupted session produced useful negative trajectory results, with R1 still open.
