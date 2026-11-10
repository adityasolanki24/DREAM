# DREAM Research / Development Report: Start the adopted G1 roadmap

## Task identification
- Task title: Start the adopted G1 roadmap
- Task ID: 2026-11-10-003
- Date and time: 2026-11-10T10:18:44+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: claude/isaac-semantic-research-20261109
- Starting commit: efd6872f4c375f948244e04c5beb71eade626925
- Final commit: efd6872f4c375f948244e04c5beb71eade626925
- Task status: partially completed

## A. Objective

Start R0 and the first diagnostic slice of R1 under the adopted G1 roadmap. Test whether a free-standing G1 can reach, close on and lift a nominal box from a fixed stance.

## B. Starting state

The primary checkout was at efd6872 on claude/isaac-semantic-research-20261109, with an uncommitted audit, separate adoption task and presentation work. The whole-body branch was at 8f47475, with W6 development gates failed. Its local CPU WBC weights and Python environment were already available.

## C. Work performed

### Implementation

Created the isolated codex/g1-integration worktree. A merge preflight found conflicts in README, the archive manifest, WBC, the assisted-reach rename, the Isaac runner and the isolation test. Ported the required model, name-based actuator bridge, tactile simulation, bounded WBC posture channels, planner and supervisor selectively from the whole-body branch, along with their tests and envelope inputs. Kept the existing Isaac runner using the new posture interface. Documented the adopted roadmap and evidence rules in AGENTS.md and added a scorecard in README.md. Added a narrow fixed-stance grasp diagnostic with an evaluator-only contact and lift trace.

### Evaluation

Used the existing cached NVIDIA policies on local CPU with MuJoCo 3.16.0. The runner accepts oracle-assisted fixture inputs for the table, the starting object pose and localization. The hands act through position servos and simulated contact. The grasp outcome is evaluated independently against object lift and hand-only contact. No equality constraints are present.

### Runs

Four development executions, registered in runs/. The initial and closer fixtures remain reproducible alongside the tucked fixture. Example:

```powershell
python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_tucked.json --out C:/Users/24adi/DREAM-r1-evidence/tucked-repeat
```

## D. What was achieved

The integration branch has an opt-in Dex3 control path and a reproducible tabletop runner. The nominal stance was rejected by the planner. A closer stance caused box contact during settling, reproduced at 39.17 N after 0.10 s. The arms-down setup cleared the early box contact, settled, planned and began the reach. It stopped after 1.60 s on the right wrist pitch torque guard, with hand-hip contact in the evaluator trace.

## E. What was not achieved

R0 is in progress. This checkpoint remains uncommitted because the imported slow test and repository-wide lint are red. The prior semantic research task and separate adoption record remain open. No trial reached finger closure, lift or independent grasp verification. No video was captured. There was no R1 claim or held-out evaluation.

## F. Passed tests and experiments

Focused fast tests including the isolation guard passed (104 passed, 3 deselected). The first focused run also passed the physical Dex3 finger sweep under the pretrained policy and the legacy locomotion trajectory comparison. The repository fast suite passed (990 passed, 2 skipped, 71 deselected). Ruff lint and formatting pass for the changed Python files.

## G. Failed tests and experiments

The slow monitored-reach test failed on both the source branch and the integration tree on this host, with the same rollback result: hand error rose from 24.24 to 38.08 mm. Repository-wide lint and format checks are already red on the base; the dream/experiments/tests/results comparison reported the same 120 lint errors and 8 unformatted files before and after the port. The complete slow suite was not run.

## H. Improvements

The trials established a reproducible distinction between planner refusal, box contact during initial settling, and an execution-time wrist torque stop. No task success improvement was measured.

## I. Regressions and dis-improvements

No change was detected by the legacy trajectory equality check. The reach rollback failure reproduced on the source branch. Other behavioural regressions were not measured.

## J. Comparison against baseline

The closer fixture changed the start x position by 0.08 m only. Its initial contact made a reach-planning comparison impossible. The next trial reproduced that stop with contact identity logging. The final trial kept the closer scene and set the initial arm pose to the existing arms-down reference. It reached execution, with no successful grasp in either configuration.

## K. Research integrity and assumptions

The trials use a single development seed, a known box and a nominal tabletop. The arms-down reset is a fixture change, not an algorithmic improvement. The evaluator logs contacted body identity but the controller sees only its touch sensors and robot state. The wrist result is a requested-torque ratio, not a measured applied torque. The coincidence with hand-hip contact does not alone establish causality. Runs use the dirty integration tree at the recorded base revision; run records include config hashes and trace checksums.

## L. Cloud resource usage

None. All work ran locally on CPU using cached weights.

## M. Remaining work

Reproduce the wrist stop with a trace of requested and applied torque, inspect the sweep of the articulated hand against the hip, then test a collision-checked pre-grasp transition from a clear stance. Keep the original fixtures as challenge cases and preserve the rollback failure for separate work. Complete integration validation before promoting the branch.

## N. Final conclusion

The roadmap is active on an isolated integration branch. R1 is at monitored reach bring-up, with the next investigation narrowed to the arm trajectory and wrist load before closure.
