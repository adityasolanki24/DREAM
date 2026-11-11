# DREAM Research / Development Report: R1 bottle grasp failure attribution and recovery

## Task identification
- Task title: R1 bottle grasp failure attribution and recovery
- Task ID: 2026-11-11-001
- Date and time: 2026-11-11T11:35:39+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61
- Final commit: b9fa7a7886c4bdcf49b1f8554f4ab86608af20ba
- Task status: partially completed

## A. Objective

Attribute the bottle drop despite continued tactile pressure, correct failure recognition, and test whether orientation coverage in the candidate set yields a physically retained grasp.

## B. Starting state

The stricter R1 replay passed 8/9 matched cases, both earlier geometry shifts and both mugs, while two bottles and the sphere failed. All 107 focused tests passed. Compound fixtures, explicit friction and sensed-versus-evaluated outcomes were implemented.

## C. Work performed

### Implementation
Require the observed object geometry to remain at both opposing contact surfaces in the proprioceptive hand model. A sustained geometric gap stops execution even when tactile pressure persists. After the first candidate predictions fail, rehearse distinct wrist orientations at each depth and thumb setting, exclude previously tried ranks and check up to three further proposals with the whole-body controller.

### Evaluation
Reproduce the first bottle's failed candidate and a shifted bottle in the independent planning scene. Add a physics regression for the pressure-after-drop failure before correcting it. Verify the earlier positive controls and compare live bottle outcomes with the same configurations.

### Runs
Run records: see runs/ and the daily index
See runs/ for the independent predicted-hold audit and the live orientation-coverage comparison. Evidence includes source snapshots, original configurations and full traces.

## D. What was achieved

The lighter bottle passed automatic proximal-wrap selection and independent complete-object clearance for three seconds. Its minimum clearance was about 0.123 m. The pressure-after-drop regression now stops on a sustained gap between the observed object and the inferred contact surfaces. The second recorded bottle pose was also recognized as lost contact despite tactile pressure.

## E. What was not achieved

The original sphere and heavier bottle remain failed executions. Walking into a reachability-selected stance, the unified humanoid toolbox and the complete R1 development and held-out gates remain open.

## F. Passed tests and experiments

All 109 focused grasp, hands, fixture, posture, supervisor and observation-isolation tests passed on the final selector. Ruff lint and formatting passed. The physics regressions cover object-contact loss despite tactile pressure and automatic proximal-wrap selection. Complete fast and slow suite acceptance remains pending for the R1 milestone.

## G. Failed tests and experiments

The original drop regression failed before the correction because the controller returned no stop while finger pressure persisted. More orientation coverage did not recover a live bottle hold. Conservative payload and friction bounds did not recover either bottle. Distal hooking failed all three live trials and was not adopted.

## H. Improvements

The grasp chooser now tests a physically different closure after pinch predictions fail. The same geometry, full-body prediction and verification rules apply across shapes. Object-contact consistency combines sensed pose, proprioception and pressure.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
All twelve earlier successful grasps passed the final replay. The sensed result matched independent evaluation on all seventeen episodes. Full-suite regressions and the end-to-end compute cost over the R1 distribution were not measured.

## J. Comparison against baseline

The same fifteen development cases improved from 12/15 to 13/15. The lighter bottle was recovered. A new 0.12 kg bottle passed with at least 0.113 m of complete-object clearance, and a new 0.22 kg bottle failed. The final total was 14/17, including the two new cases. The larger payloads and the sphere remain failures.

## K. Research integrity and assumptions

The contact-loss guard is a general algorithmic correction. The wrap is a deterministic task-family grasp template within Unitree joint limits. Geometry and retention predictions select it. Planning materials are declared separately from the unknown execution material, with an independent noise stream. Oracle-assisted object pose and geometry remain in use. Rejected search and closure variants retain their source snapshots and negative results.

## L. Cloud resource usage

None. Local CPU with at most three concurrent test or simulation processes.

## M. Remaining work

Integrate the existing stance planner and walking controller with one humanoid interface and toolbox. Retain the sphere and payload failures in the development distribution, then freeze and run the full 50-episode development and 60-episode held-out gates. Complete repository-wide acceptance testing at that milestone.

## N. Final conclusion

A shared wrap family supplied a new physically verified bottle grasp, while sensor fusion corrected pressure-after-drop failure recognition. The complete R1 objective remains active.

## Amendments

### 2026-11-11T12:06:55+11:00

The orientation-diverse search did not recover a live bottle hold and was removed. The current selector keeps the original pinch predictions, then tests a proximal-wrap closure family using the same geometry, material assumptions and full-body verification. The pressure-after-drop detector remains active. Explicit material priors support reproducible model-bound experiments. Automatic family selection and independent bottle transfer are still being validated.
