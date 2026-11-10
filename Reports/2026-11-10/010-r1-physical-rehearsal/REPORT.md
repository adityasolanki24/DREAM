# DREAM Research / Development Report: R1 grasp retention rehearsal

## Task identification
- Task title: R1 grasp retention rehearsal
- Task ID: 2026-11-10-010
- Date and time: 2026-11-10T22:30:11+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4
- Final commit: a841927e2c90bf93d985e3fb360eb9e89fb1c63f
- Task status: partially completed

## A. Objective

Improve R1 grasp retention without per-fixture rules. Test whether a separate simulation can identify which contact poses will retain a load, and use the unchanged independent live hold gate to judge each selection.

## B. Starting state

Static articulated contact ranking verified four of six matched cubes and none of the three added geometries. A smaller cube lifted and then slipped. The larger outward-shifted cube slipped in the probe. A previous successful hold had additional support from the proximal middle link.

## C. Work performed

### Implementation
Reuse the exact articulated Dex3 right hand on a force-limited virtual vertical stage. Screen up to 12 contact-ranked poses under gravity and use the same slip-to-preload regulator as the whole-body controller. The optional whole-body check replays the stage proposal in a fresh G1 simulation with the same deterministic limits, a common mass prior, and an independent planning odometry stream. If the replay does not retain the object, the actual attempt uses the original static ranking. Live success still requires 150 physics-verified hold samples. The projected-wrist and fixed-loaded-orientation variants are preserved in the run evidence.

### Evaluation
Compare all variants on the same six development cubes and three added geometries. Keep material, pose and seeds fixed in the live scene across variants. Verify the independent planning stream after finding the initial rollout reused the episode noise seed. Run the focused grasp, posture, hand, supervisor and isolation regressions.

### Runs
Run records: see runs/ and the daily index
See the run records for commands, per-run parameters and evidence hashes. The local evidence folders retain the exact source snapshots, full traces and scenario configurations. The compact comparison in results/ records each outcome, trace hash and source snapshot hash.

## D. What was achieved

The independent-noise preview recovered the outward-shifted cube and the rectangular box. All four earlier live passes were retained. The whole-body check rejected predicted grips that the isolated stage had incorrectly rated as stable, including the original nominal and farther cubes.

## E. What was not achieved

The inward-shifted cube, cylinder and sphere still failed the live hold. The pilot does not cover the full object and mass distributions, walking and stance selection, or a held-out R1 evaluation. The full repository suite remains a milestone requirement.

## F. Passed tests and experiments

The complete fast suite passed 990 tests with 2 skips. The focused G1 grasp, hands, posture, supervisor and isolation run passed 90 tests. Both new whole-body preview regressions passed on the final implementation. Ruff lint and format checks passed across the repository. The complete slow suite remains pending with the R1 milestone.

## G. Failed tests and experiments

The prescribed wrist rejected every candidate, including the geometries with live passes. The isolated actuated stage alone dropped two formerly passing cubes. Holding the loaded wrist orientation fixed preserved the same pass/fail pattern as static ranking. The initial whole-body experiment reused the evaluation noise seed and was superseded by the independent replay.

## H. Improvements

The new selector reasons over observed geometry and predicts contact retention under load, with a whole-body check for reach and balance effects. Across all nine matched live episodes it improved verified holds from four to six.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
The accepted opt-in selector retained all four previous live passes in the matched comparison. The stage alone lost two prior passes. The whole-body check adds a second simulation to each attempt. Runtime across larger distributions and hardware transfer were not measured.

## J. Comparison against baseline

Static ranking passed 4/6 cubes and 0/3 additional geometries. Stage-only selection passed 3/6 and 1/3. The whole-body preview with independent noise passed 5/6 and 1/3. The fixed loaded orientation passed 4/6 and 0/3. See results/g1_r1_rehearsal_dev_20261110/summary.json for per-case measurements.

## K. Research integrity and assumptions

Classification: task-family heuristic. The stage uses the existing Dex3 model and a force-limited actuator for vertical motion. The preview uses labelled oracle-assisted fixture geometry and odometry, a shared 0.10 kg mass and the fixture friction prior. It uses a deterministic noise stream distinct from evaluation and the same search rule across all fixtures. Predicted results only select a candidate. The independent live evaluator scores contact and lift without attachments or state writes during execution. These are matched development results.

## L. Cloud resource usage

None. All work used local CPU with at most three concurrent runs, including tests.

## M. Remaining work

Improve the grip for the remaining small and curved cases, extend geometry and dynamics coverage, verify a navigation-to-grasp stance interface, then freeze the full R1 development and held-out gates before advancing to R2.

## N. Final conclusion

A full-robot preview made the contact rehearsal useful for selection in matched fixed-stance development. The broader R1 milestone and its independent validation remain open.
