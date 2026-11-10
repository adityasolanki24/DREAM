# DREAM Research / Development Report: Resolve the R1 tabletop grasp failure

## Task identification
- Task title: Resolve the R1 tabletop grasp failure
- Task ID: 2026-11-10-006
- Date and time: 2026-11-10T11:50:09+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: 6299f156b07785d65615fba43a642e1c6a2be237
- Final commit: aae289e0164920338c628057f2571629efe331a6
- Task status: partially completed

## A. Objective

Continue R1 in the fixed-stance tabletop fixture. Reproduce and prevent the table-directed lift request, then test whether a guarded approach reaches a real box grasp.

## B. Starting state

R0 passed at 33e23c0. At 6299f15 the both-arms-clear development trial reached and closed on the tabletop. The evaluator showed no box contact and no lift.

## C. Work performed

### Implementation

Added an optional grip position and orientation to the posture planner, retaining the default reach tool frame and axis-only behavior. The tabletop runner uses a horizontal Dex3 pinch frame derived from the overlap of the opposing pads. It checks the articulated meshes against the labelled tabletop dimensions at the planned endpoint, during motion, and across the closing sweep. After opposed touch it requires an observed box rise in a bounded probe before the full lift. The original fixed stance and a closer matched development fixture remain available.

### Evaluation

Reproduced the old false positive, inspected the hand geometry, and compared the inflated fixed-hand collision proxy with a planning-only control. Tested several distinct changes to isolate the remaining motion failure. Kept the existing force, torque and joint limits.

### Runs

13 registered development executions. The unsuccessful ready, Cartesian feedback, and compact-hand controls have evidence outside Git. The evidence directory for each run is listed in its report.

## D. What was achieved

The original test reproduced a lift request after tabletop-only contact. The guarded version prevents that request. The closer fixture now stops during the raise, before making physical table contact, even when its planned endpoint clears the surface.

## E. What was not achieved

The G1 has not grasped or lifted the box in these runs. The preclose sweep and probe are implemented, but the current physical trials stop earlier. R1 has no capability claim or held-out evaluation. R2 remains pending.

## F. Passed tests and experiments

The targeted suite passed 85 tests, including the new original-case and closer-stance regressions, posture, hands, supervisor and isolation checks. Repository Ruff lint and format passed.

## G. Failed tests and experiments

The original regression failed before the change. The corrected pinch point was initially infeasible with the fixed-hand proxy. Direct and staged posture transitions failed the hand clearance guard. Full static ready reset failed tilt rate, arm-only ready reset failed clearance, Cartesian feedback failed the elbow position guard, and the compact-finger variant still failed clearance.

## H. Improvements

Observed pose and geometry now distinguish a reach endpoint from a safe trajectory. At the closer stance the planning-only control reduced endpoint residual from 22.6 mm with the inflated proxy to 3.2 mm when the hand used the separate geometry check. The near-edge trace identified the distal right index link.

## I. Regressions and dis-improvements

The targeted posture and safety tests passed. Full suite rerun and general grasp performance were not measured for this R1 checkpoint.

## J. Comparison against baseline

The geometry control preserved table, box, stance and limits while changing the representation used for hand clearance. The body remains checked against the conservative table obstacle. With both the original hand path and its shorter-finger variant, the arm still approaches the near edge before it is high enough.

## K. Research integrity and assumptions

General algorithmic correction: check the task-specific hand envelope and measured object response before lift. The horizontal pinch is a scoped morphology heuristic. The closer start and ready-pose tests are development fixture changes. Box height and tabletop geometry are oracle-assisted. Contact identity remains evaluator-only. The clearance test samples the sweep and does not claim continuous collision guarantees.

## L. Cloud resource usage

None. All work used the local cached WBC on CPU.

## M. Remaining work

Build a raise trajectory with continuous clearance from the near tabletop edge. Use a planning copy with the articulated hand to reject unsafe transitions before execution, and keep monitoring the live reach. Then exercise closure, probe and hold on a real box grasp before widening R1 to varied objects or walking.

## N. Final conclusion

The tabletop false lift request is covered by a regression. R1 is still at approach bring-up, with the remaining blocker localized to the right index trajectory near the edge.
