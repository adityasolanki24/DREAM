# DREAM Research / Development Report: Validate the humanoid grasp across the R1 development cohort

## Task identification
- Task title: Validate the humanoid grasp across the R1 development cohort
- Task ID: 2026-11-27-001
- Date and time: 2026-11-27T03:13:08+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53
- Final commit: fc3fa22
- Task status: partially completed

## A. Objective

Continue the adopted R1 stand-and-grasp capability without benchmark-specific action branches. Validate the new live humanoid tool on the requested development distribution, fix any reproduced correctness failure, and preserve the 50-development and 60-held-out acceptance gates.

## B. Starting state

Previous goal work made concrete progress by implementing the live Humanoid interface, tool dispatch, arm and hand checks, camera capture, and a 50-case development generator. Those edits were uncommitted at cf2851d. On resumption, the original process handle was absent and one walking case plus the campaign aggregate contained NUL bytes. Three matched case records and the camera check remained intact. Broad R1 reliability was unmeasured.

## C. Work performed

### Implementation
Connected the continuous grasp to the shared Humanoid protocol, G1Humanoid adapter, and HumanoidToolbox. Added conservative complete-object clearance from tracked pose and enclosing geometry, a bounded hold controller, and an independent final uninterrupted 3 s scoring window. Forecast worlds now use declared material priors in both fixed and walking modes. Restricted support-contact exceptions to the working hand in both IK constraints and clearance evaluation. Added the stratified 50-case development runner, source hashes, atomic JSON outputs, and regression tests. These are general algorithmic corrections and engineering integration.

### Evaluation
Recovered the corrupted front walking case in a new directory. Ran the five-family smoke, reproduced the false success, tested the initial clearance correction, reproduced two resulting physical regressions, recovered their holds with bounded feedback, and tested contact confirmation and material isolation. A two-sided geometry audit reproduced the unused-hand collision defect before its regression tests and fix. The final targeted approach comparison completed five episodes. Full fast validation passed. The full suite and the 50-case campaign were deliberately interrupted when regressions first appeared, so neither establishes a completed gate.

### Runs
Run records: see runs/ and the daily index
All runs are registered under runs/. They cover cohort-smoke, front-recovery, corrected-smoke, full-regression, development-50, clearance-regressions, clearance-feedback, head/contact confirmation, unused-hand-audit, and active-hand-clearance. The interrupted runs retain their logs and partial outputs. Per-case configurations, source snapshots or reconstruction notes, code revisions, model checksums, versions, and evidence hashes are linked in their records.

## D. What was achieved

The humanoid tool executes the real continuous episode. The centre-only false-success predicate is replaced by a complete observed surface bound. A recovered grip needs a new uninterrupted 3 s hold, checked independently. The two introduced hold regressions were physically reproduced and recovered. Material-isolation tests pass in both modes. The unused-hand collision mask is reproduced and corrected symmetrically. The final walking control remains a verified hold, and the previously blocked sphere reaches closure.

## E. What was not achieved

R1 is incomplete. The complete 50-episode development gate was not measured. The 60-episode held-out gate, shifted shapes and two unseen families have not run. In the final approach comparison, cylinder and mug fail before closure, sphere fails opposing contact, and a fresh box loses its grip. Large box 1000 remains a failed challenge case. Complete slow-suite validation and full replay of affected historical fixed-stance results remain pending.

## F. Passed tests and experiments

Final fast suite on the runtime source passed 1017 tests with two skipped and 91 slow tests deselected. The skips concern Isaac USD collection and an unavailable upstream vlm_orchestrator comparison. Ruff check, format --check, diff --check, and isolation passed. Regression tests first failed for the absent surface helper, absent uninterrupted-hold verifier, fixed-mode material isolation, and both unused-hand collision roles, then passed after correction. The recovered front case and multiple paired hold controls passed independent physics scoring. The final five-case approach diagnostic retained the walking control with zero table-contact samples in every walk.

## G. Failed tests and experiments

The initial five-family smoke had one verified bottle hold and one false sensed box success. Its minimum complete box clearance was 0.0942534537 m. A fixed extra surface margin recovered that box but lost the shift-in and walking holds. Both losses were reproduced as physical failures. Bounded feedback recovered them. Large box 1000 still fails later contact/clearance verification. The final approach comparison has four failed episodes and one verified hold. The full suite was interrupted after two failures in its earlier source variant. The 50-case campaign was stopped with six completed outputs. No complete gate is inferred from either partial run.

## H. Improvements

Sensed success now corresponds to the entire object's observed lower bound. The scorer cannot combine hold samples across a correction or a sampling gap. Forecast material assumptions are separated from execution values. Planner collision metrics and constraints keep the unused hand clear of support surfaces. Runtime tooling and a reproducible broad development cohort are available.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
The initial extra-lift correction introduced two verified regressions, which are preserved and were recovered by the later controller. A complete current-tree slow-suite result is unmeasured. Historical fixed-stance trajectory forecasts used execution material values and require a full affected replay after isolation. The final control walking grasp is retained. Other old physical comparisons cannot be declared current-tree validated from the fast suite alone. The large-box challenge is accurately reported as failed.

## J. Comparison against baseline

Compare each code stage separately in results/g1_r1_humanoid_dev_20261127/summary.json. The original box false success is removed. The initial fixed-margin candidate produced a genuine box hold but lost two other holds. Later bounded feedback restores those two, retains bottle and rotated controls, and leaves the large box as a failed challenge. The working-hand exception corrects the static collision metric and advances the sphere from blocked approach to closure, while cylinder and mug still fail and the checked walking hold is preserved. These targeted case sets do not estimate broad success or establish generalisation.

## K. Research integrity and assumptions

All current results are development evidence. No held-out outcome guided changes. The 35-of-50 development target and 0.10 m, 3 s, hand-only physical gate are preserved. The cohort retains five families, widths 0.04-0.08 m, masses 0.05-0.30 kg, table tops 0.70-0.90 m, and declared sliding friction 0.50-1.00. Friction is an assumed test distribution. Object geometry, pose, table geometry and noisy odometry remain oracle-assisted. Head camera capture is validated separately from tracking. Default mass forecasts use a common 0.10 kg prior. The fixed-stance material leak is corrected without changing execution materials. Negative cases and interrupted outputs remain intact. The confirmation source reconstruction is explicitly labelled in its amendment. Robot motion is physical, with equality constraints refused.

## L. Cloud resource usage

No cloud jobs, resources, paid API calls or cloud charges were incurred. Work used local CPU, at most two simulation campaign workers with one test or static-audit worker. Existing cached policies were reused. Camera capture used local OpenGL.

## M. Remaining work

Continue the full original R1 scope. Diagnose the remaining joint stance and approach failures and unopposed/slipping finger contacts. Revalidate affected fixed and walking holds. Run the complete fast and full suites on the final tree. Complete 50 development episodes and meet at least 35 verified holds before freezing. Then define and execute the 60-case held-out gate with size/mass shifts and two new shape families, and satisfy the remaining R1 recording and reporting requirements. The active goal remains open.

## N. Final conclusion

This is a validated development checkpoint for humanoid tool integration, truthful clearance verification, material isolation and a planner collision correction. R1 acceptance is still pending. The remaining failures are preserved and attributed to their observed stages without claiming broad success or autonomous learning.

## Amendments

### 2026-11-27T04:29:02+11:00

The head-camera execution belongs to previous task 2026-11-11-003. The confirmation run in this task checks contact and hold control. Code checkpoint fc3fa22 contains the tested runtime. The full R1 goal remains active, with the 50-development, complete slow-suite and held-out gates pending.
