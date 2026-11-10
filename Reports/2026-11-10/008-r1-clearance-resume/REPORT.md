# DREAM Research / Development Report: Continue R1 from the trajectory search evidence

## Task identification
- Task title: Continue R1 from the trajectory search evidence
- Task ID: 2026-11-10-008
- Date and time: 2026-11-10T21:53:02+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: 52d3c4aa200cd5f08b3201559015315024ae994b
- Final commit: d4fab226df0bcc3b9a0c35271257cdadcb644fee
- Task status: partially completed

## A. Objective

Resolve the fixed-stance tabletop approach and demonstrate a measured physical grasp on more than one box position. Retain the broader R1 distribution and the three-second gate.

## B. Starting state

R0 was green. The prior R1 fixture stopped at the table edge. Sensor-initialised previews and sampled joint waypoints were in progress from the interrupted task 2026-11-10-007.

## C. Work performed

### Implementation

The fixed-stance runner settles using measured base motion, then selects a trajectory from forward simulations initialised from the robot observation and labelled tabletop fixture. The successful first stage raises in the measured wrist orientation. Horizontal alignment is measured above the box. The final descent uses small pose corrections and executes only a previewed safe improvement. Before closure it checks table clearance, grasp depth and opposing geometry. Opposing tactile force must persist before probing. The lift keeps a fixed body reference, uses small relative vertical commands, and adjusts the pinch from measured opposing force and box motion relative to the wrist. Independent simulation contact identity is used only to score and diagnose.

### Evaluation

Preserved the direct reach as a comparison path. Evaluated the same final controller on six fixed-stance box configurations. Two fixtures changed position and mass after the controller was frozen. The runner summary, exact fixture configs and source hashes are in results/g1_r1_tabletop_dev_20261110/summary.json.

### Runs

27 executions are recorded under runs/. The report keeps the interrupted summary, unsuccessful control variants, and the partial lifts distinct from verified holds.

## D. What was achieved

The nominal, lateral-shift and fresh-near cases passed the 10 cm and 3 s independent gate. The two other lifted boxes lost grasp during hold. The smallest tested cube was refused before closure. The table-edge approach failure is resolved for several fixed-stance cases.

## E. What was not achieved

The full R1 gate is pending. These trials do not test varied shape families, the full planned width and mass range, automatic stance selection or walking. The oracle-assisted pose setup remains in use.

## F. Passed tests and experiments

Three independent physical hold passes with zero protective stops in those successful episodes. Ruff lint and format passed. All 87 focused posture, hands, supervisor, isolation and grasp tests passed.

## G. Failed tests and experiments

Negative results included unsuccessful direct or sampled transitions, early closure without depth, slip during probing and slip during hold. On the final controller, shift_in refused closure, and shift_out and fresh_far slipped during hold. These are retained as challenge cases.

## H. Improvements

The nominal failure progressed from a preclose table stop to a verified hold. The unchanged controller also passed a lateral shift and the fresh-near fixture with a different centre and mass.

## I. Regressions and dis-improvements

An early-acceptance adaptive descent regressed the nominal and lateral cases. A measured pinch depth restored both. The narrower development suite still contains three physical failures. The full repository suite was not rerun for this development checkpoint.

## J. Comparison against baseline

The unchanged-command control advanced about 39 mm during balance settling. Raising in the sensed wrist orientation cleared the table. Above-box alignment enabled a real pinch. In the final controller, relative-pose and force feedback kept three cases held above the required height for the full interval. The original direct comparator still stops before closure.

## K. Research integrity and assumptions

General algorithmic corrections: observation-initialised trajectory preview, measured alignment and slip regulation. The horizontal pinch and search bounds are task-family heuristics, and fixture start poses are scoped development conditions. There are no seed or case branches in the motion selection. Box pose and odometry are oracle-assisted. The six episodes are development evidence and do not constitute the frozen R1 acceptance distribution.

## L. Cloud resource usage

None. All execution used local CPU and the cached WBC weights.

## M. Remaining work

Investigate loss of contact during hold across box widths and offsets using measured relative rotation, contact distribution and preload. Preserve the current six fixtures and evaluate on fresh cases before claiming a wider grasp envelope. Only then continue to automatic stance and the broader R1 gate.

## N. Final conclusion

The tabletop blocker is resolved for the initial fixed-stance box slice on multiple fixtures. R1 remains open because retention and width transfer are incomplete.
