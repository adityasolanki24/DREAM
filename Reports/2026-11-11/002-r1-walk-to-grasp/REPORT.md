# DREAM Research / Development Report: R1 walk to a reachable grasp stance

## Task identification
- Task title: R1 walk to a reachable grasp stance
- Task ID: 2026-11-11-002
- Date and time: 2026-11-11T12:20:09+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6
- Final commit: b7559fa0f3ef7d35e91a0568cd9697c468324908
- Task status: partially completed

## A. Objective

Walk from a distant initial pose to a stance selected from observed geometry and arm reachability. Verify the same physical contact lift and hold after the gait handover.

## B. Starting state

The earlier fixed-stance selector passed 13 of 15 matched cases and one of two fresh bottles. Walking and posture planning existed separately. A grasp episode began at a supplied stance.

## C. Work performed

### Implementation
Sample candidate stances around the observed object and check direct access from an exterior table face, robot mesh clearance, and arm reachability to multiple approach points. Prefer a stance that aligns the object with the working shoulder. Walk with bounded forward, lateral and yaw commands using odometry, then settle and reobserve before reaching. Derive the grasp orientation and the raise and retreat directions from the chosen table face. The robot and object remain in one MuJoCo execution scene. Record table contacts independently and supervise robot clearance and unintended object displacement.

### Evaluation
Replay two starting poses through the forward follower, planar feedback, mesh clearance and shoulder alignment, and the full grasp selector. Check changed scene orientation and object geometry as separate development cases.

### Runs
Run records: see runs/ and the daily index
The run records link exact source snapshots, fixture configurations, and complete physical traces.

## D. What was achieved

All four final episodes walked from 1.24 to 1.42 m from the table into an automatically selected stance. Three held a cube with at least 0.10 m of complete-object clearance for three seconds. The rotated and translated cube recovered after stance ties and grasp depths were computed in the relevant approach frame. The sensed hold and independent evaluation agreed in all four cases.

## E. What was not achieved

The cylinder arrived and lifted but slipped. The typed humanoid interface and toolbox, the 50-episode development evaluation and the 60-episode held-out gate remain open.

## F. Passed tests and experiments

The 109 focused fixed-stance, hand, posture, safety and isolation checks passed before the final frame correction. Four targeted tests passed on the final implementation, covering two walking and two fixed-stance holds. Repository-wide Ruff lint and formatting passed. The final physical walking cohort passed three of four hold checks.

## G. Failed tests and experiments

The forward follower drifted during the final heading adjustment. Planar feedback reached the stance but the arm failed the approach until shoulder alignment and motion prediction were improved. The rotated case first selected a diagonal stance due to numerical ties, then hit the force limit. A softer preload changed the stopping condition to slip and was removed. Grouping the grasp candidates by world X was the remaining frame error. The cylinder slipped from both attempted stances.

## H. Improvements

The actual episode now joins a reachability-selected walk, balance handover, grasp and hold in one physics state. The planner derives the approach normal from the observed table, aligns the object with the working shoulder and groups candidate depths along that normal. The tested walking trajectory forecasts use their own material prior.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
Both matched positive walking controls passed in the final replay. Fixed-stance regression checks passed in the focused suite and the final contact tests. The cylinder's changed-height fixed control held, while both walking attempts slipped. The complete R1 suite and planning throughput across the full distribution remain unmeasured.

## J. Comparison against baseline

The forward follower and early planar variants had zero verified holds across the two matched starts. Both matched starts passed with full grasp selection. The first four-case comparison passed two holds. Correcting the approach-frame depth grouping recovered the rotated cube, yielding three verified holds in four final episodes.

## K. Research integrity and assumptions

Frame handling is a general algorithmic correction. Stance sampling and feedback gains are task-family commissioning choices inside the WBC envelope. Table geometry, object pose and odometry are oracle-assisted. The live episode uses the declared material and independent scoring. The fixed-stance cylinder control also uses the original fixed-stance motion forecast, so it does not isolate posture history from every other difference. These four cases are development evidence.

## L. Cloud resource usage

None. Local CPU only, with at most three concurrent simulation and test processes.

## M. Remaining work

Connect the walking and grasp controllers through one humanoid interface and toolbox. Condition grasp forecasts on the measured arrival state and evaluate shape and payload transfer, including the failed walking cylinder. Freeze and run the 50 development and 60 held-out episodes and complete repository-wide acceptance at that milestone.

## N. Final conclusion

A continuously simulated G1 can choose a stance, walk to it and hold a cube under the physical gate in the tested scenes. R1 remains active pending the full interface and evaluation.
