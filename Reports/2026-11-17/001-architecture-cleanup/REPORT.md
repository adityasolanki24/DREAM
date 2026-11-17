# DREAM Research / Development Report: Architectural cleanup, simplification and stabilisation

## Task identification
- Task title: Architectural cleanup, simplification and stabilisation
- Task ID: 2026-11-17-001
- Date and time: 2026-11-17T13:44:42+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: claude/architecture-cleanup-20261117
- Starting commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53
- Final commit: 7f92ccc
- Task status: completed

## A. Objective

The owner asked for an evidence-driven architectural cleanup that turns DREAM into a smaller, coherent, understandable and reproducible research codebase, without destroying valuable work.

The work had five parts:
- inspect the repository and establish a baseline;
- write an inventory with evidence;
- define the architecture around the core pipeline, with the Unitree G1 in MuJoCo as the primary embodiment and the Panda and discovery work preserved;
- refactor incrementally on a dedicated branch with small, verified commits;
- improve README.md so a developer can answer where execution begins, how goals are interpreted and verified, what is oracle-provided and what is experimental.

Constraints:
- the original goal stays immutable through planning and verification;
- historical results, frozen configs and seeds are left untouched;
- nothing is pushed or force-pushed, and history is not rewritten;
- no cloud use, and no irreversible bulk deletions.

## B. Starting state

- **Base:** `codex/g1-integration` at `cf2851d` (11 Nov 2026), the latest pushed trunk. It contains main (7 Nov closeout), the semantic-search branches and the R0/R1 G1 work under the adopted R0-R7 roadmap. The cleanup branch `claude/architecture-cleanup-20261117` was created from it, in its own worktree, C:/Users/24adi/DREAM-cleanup.
- **Baseline at cf2851d:**
  - ruff check clean; ruff format clean (567 files);
  - tests/test_isolation.py 38 passed;
  - fast suite 994 passed, 2 skipped (pxr and vlm_orchestrator not installed);
  - slow suite 86 passed.
  - Correction: a progress note first said "1 skipped". That count was misread from pytest's progress dots; rerunning the same files at cf2851d showed both skips.
- **Preserved, untouched:**
  - uncommitted R1 work in the worktree C:/Users/24adi/DREAM-g1-integration (dream/types.py, experiments/g1_grasp_bringup.py, new dream/sim/g1_humanoid.py and dream/tools/humanoid.py; last modified 11 Nov);
  - untracked 10 Nov report folders in the primary checkout.
- **Failures the owner listed, all recorded in README and results/:**
  - W6 whole-body failed its gates;
  - Qwen request-to-goal substitution caused false claims (7 Nov);
  - room search failed on fresh seeds;
  - camera-based compartment opening reached 11/20, against 27/30 with oracle assistance;
  - MolmoAct2 never completed a task.
- **Cloud:** a read-only check found no DREAM cloud resources.

## C. Work performed

### Implementation
Twelve commits on `claude/architecture-cleanup-20261117` (cf2851d..HEAD):

1. `fb80eab` Move the `Budgeted` planner wrapper from experiments/g1_explore_room.py into dream/runtime/budget.py. Production runners no longer import from an experiment, and the old import path still works.
2. `91d22dc` Remove the unreachable W6-era contact-feasibility gate, dream/robot/g1_rollout.py and g1_correction.py (543 lines; only a test used them).
   - The restore command is `git checkout cf2851d -- dream/robot/g1_rollout.py dream/robot/g1_correction.py`.
   - The slow posture test now executes the plan directly.
3. `e5e7447` Declare mink 1.3.0 and daqp 0.10.3 as core dependencies: g1_posture imports mink at module top, so a clean install could not import it. The Pillow import in camera_stream is now lazy.
4. `b74b58c` Keep one angle-wrap helper, dream.navigation.follow.wrap. Four copies had identical bodies, and one unused copy was removed.
5. `584395c` Rename two colliding class names: dream.memory.home.Belief to ObjectBelief, and dream.perception.world.Track to WorldTrack.
6. `92162f9` Move 11 unused experiment scripts to experiments/archive/ with `git mv` (nothing deleted), and label the T1 modules EXPERIMENTAL.
7. `9191c6b` and `a1e3297` Move phone_session.py back to experiments/. It is the walking-G1 phone-page entry point, which the archive criteria missed. The second commit fixes its path.
8. `90f65a6` FUNCTIONAL:
   - `TaskAgent.run` requires the original-goal record (`spec`); without it the call raises TypeError.
   - A new `spec_from_goal` covers programmatic typed goals.
   - The language, phone and scheduler paths keep the request text in the record.
   - `interpret_mobile_goal` requires `spec`.
   - The unguarded 7 Nov interpretation and `UngatedTaskAgent` moved to experiments/nov7_baseline.py, used only by the measured comparison arms (Isaac `nov7` mode, the `abl_substitution_ungated` ablation).
   - About 47 callers were updated.
   - `VLMAgent` is documented as the legacy tabletop loop.
9. `0d0511a` README front restructured: Start here with verified commands, How DREAM works, Where things happen, Oracle versus estimated, Capability status, Module map and G1 roadmap. AGENTS.md got the layout lines and invariant 8 (TaskAgent.run requires the record).
10. `7f92ccc` `GoalSpecification.to_dict` returns JSON-native data. This fixes the slow-test regression the record caused (see G).

### Evaluation
- **Inventory** (inventory/): three read-only code audits (core pipeline, embodiments and perception, experiments/tests/dependencies) and a static import graph, import_graph.json, covering 107 dream modules and 140 experiment scripts.
- **Equivalence checks of the functional change** (90f65a6), each run once in a detached worktree at 9191c6b and once after:
  - Panda pick/place: byte-identical once the new spec field and wall time are scrubbed;
  - one semantic-search development episode: identical, apart from a random sighting uuid;
  - one G1 room-search episode: identical, 17 events.
  - All three scripts put their own checkout first on sys.path, so each run imported its own tree.
- **Start-here commands** were each run once in short form: reports check, isolation test, run_pick_place, run_tasks, a g1_explore_room smoke, a phone_session start and stop, a semantic-search smoke, and the g1_grasp_bringup walking grasp.
- **Full checks:** run at a1e3297, then after the fix, on the final tree (see F).

### Runs
Run records: see runs/ and the daily index
No experimental runs. This was a refactor: the checks are tests and short smoke commands, recorded in C, F and G.

## D. What was achieved

- **The original-goal guard is universal.**
  - Before: about 47 `TaskAgent.run` call sites ran without the record, so the guard was off on them.
  - Now: `run` refuses without the record (TaskAgent.run raises TypeError), and nothing under dream/ has an unguarded path.
  - The unsafe 7 Nov behaviour exists only in experiments/nov7_baseline.py, for the measured arms.
  - New test: in the live language path, "walk to the refrigerator" proposed as GoToPlace('entrance') ends as goal_substitution_rejected.
- **dream/ is smaller:** 594 lines deleted and 182 added (net -412), including 543 lines of dead W6-era code.
- **Clearer interfaces:**
  - production code no longer imports from experiments/;
  - a clean install can import the G1 posture stack (mink and daqp declared);
  - one angle-wrap helper instead of four;
  - no duplicate class names for Belief and Track.
- **Clearer entry points:** 11 unused scripts are isolated in experiments/archive/, and the T1 chain is labelled experimental.
- **No behaviour change** for typed goals in the three workflows checked.
- **README front:** a 244-line orientation, with commands that were run on 17 Nov, the eight-step pipeline mapped to modules, oracle labels per pipeline and one capability table with evidence paths.

## E. What was not achieved

- **Ground-truth readers not moved out of dream/sim/home.py** (planned C5). The generator itself calls them (verify_scenario, GroundTruthView, flood), so moving them to dream/eval would create a dream.sim -> dream.eval import cycle. They stay where they are.
- **Isaac FixedJoint assisted grasp kept.** experiments/isaac_task.py and dream/robot/g1_assisted_demo.py are a live path (isaac_g1_kitchen --mode task). It is labelled ASSISTED, and the README marks it as not a capability.
- **T1 chain not moved.** Cloud scripts and tests reference its paths, so it is only labelled.
- **Not attempted, because each would change behaviour or a frozen baseline:**
  - merging Toolbox and MobileToolbox (duplicated call, cancel and read-only logic);
  - making SemanticSearchPlanner reuse RoomSearchPlanner's gain frontier;
  - removing the classic path follower (frozen semantic-search results use it);
  - consolidating helpers copied between experiment scripts (those scripts produced frozen results);
  - reorganising the dream/tools vs dream/runtime naming.
- **VLMAgent still bypasses the goal record.** It keeps its expected_goal check and is documented as legacy.
- **Default outputs overwrite committed results.** run_pick_place and run_tasks write into results/ by default; the README warns, but the defaults were not changed.
- **Optional imports still undeclared** in pyproject: requests, truststore, spark_dsg, jwt, rosbags and websockets.
- **Not rerun:** web and Android checks (not affected by this cleanup), and the Isaac and cloud commands (no cloud use was authorised).

## F. Passed tests and experiments

Final tree 7f92ccc (owner laptop, CPU, shared venv, 17 Nov 15:48-16:26 AEDT):
- ruff check: all checks passed. ruff format --check: 572 files already formatted.
- tests/test_isolation.py: 38 passed (baseline 38). It covers the decision-package import rules, now including the agent-base and string-import patterns.
- Fast suite (pytest -m 'not slow'): 1010 passed, 2 skipped (pxr and vlm_orchestrator not installed). Baseline: 994 passed, 2 skipped. The difference is 17 new tests (the spec property test over the 9 goal types, TypeError without the record, the ungated agent confined to experiments/, the language-path refusal, the JSON round-trip regression test, test_budget) minus 1 removed test (the deleted correction gate).
- Slow suite (pytest -m slow): 86 passed, 1 skipped (the same pxr module). Baseline: 86 passed. The slow posture test now executes the plan directly and passed against the real controller.
- Equivalence of the functional change: 3 of 3 workflows identical (see C).
- Start-here commands: 8 run successfully on 17 Nov (README, Start here table). The fixed-stance grasp config refuses with 'no verified trajectory', which the README states.
- Not run: web (npx tsc, npm test, build) and Android (gradle). Neither web/ nor android/ was changed by this cleanup. No cloud or Isaac commands were run.

## G. Failed tests and experiments

1. **Slow-test regression, introduced by 90f65a6.**
   - Test: tests/test_cloud.py::test_cloud_physical_harness_reports_an_independently_scored_outcome failed at a1e3297.
   - Expected: a saved episode.json reloads equal to the in-memory result.
   - Actual: the record's tuple fields came back as lists.
   - Root cause (known): GoalSpecification.to_dict returned dataclass tuples.
   - Fix: 7f92ccc, with a new regression test that failed before and passes after.
   - Retest: the failing test passed alone, and the full suites were rerun on the final tree (F).
2. **Misclassified script.** The archive step moved phone_session.py, a user-facing entry point no file names. Detected in review and moved back (9191c6b); a1e3297 fixes its path, which 9191c6b had left at the archive version.
3. **Baseline skip count misread** (1 instead of 2), corrected in B.

## H. Improvements

- **Unguarded TaskAgent paths:** about 47 before, 0 now (TypeError without the record).
- **dream/ size:** net -412 lines.
- **Dead G1 modules:** 2 removed (543 lines).
- **Production imports from experiments/:** Budgeted removed (the main known case).
- **Undeclared core dependencies:** mink and daqp now declared.
- **Duplicate wrap helpers:** 4 reduced to 1.
- **Duplicate class names:** 2 pairs reduced to 0.
- **Top-level experiments scripts:** 140 to 130 (11 archived, 1 baseline module added).
- **Tests:** fast suite 994 to 1010 passing, as the new guard and regression tests were added.
- **README:** a reader-oriented front with commands verified on 17 Nov.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
<!-- If a metric was not measured, write "Not measured", not "No regression". -->
- **Stricter live Qwen path (deliberate):** experiments/phase1_interface.py now refuses a request the grammar cannot ground after three model proposals. Before, any model-chosen goal ran.
- **Episode format change:** episode and goal events now always contain a `spec` record, so consumers of episodes.jsonl see a new key. Old result files are unchanged.
- **Paths and names changed:**
  - the 11 archived scripts now live under experiments/archive/, so their old commands need the new path;
  - external code importing the old names memory.home.Belief and perception.world.Track breaks.
- **Possible merge conflicts** for the uncommitted R1 work in DREAM-g1-integration where it touches the same files. Not measured: that work was not touched.
- **README is longer** overall: 846 to 936 lines. The new front was added and the evidence record kept.
- **Runtime performance:** Not measured.

## J. Comparison against baseline

| Metric | cf2851d (before) | final (after) | Interpretation |
|---|---|---|---|
| TaskAgent.run without the goal record | allowed (about 47 call sites) | TypeError | guard universal |
| Unguarded 7 Nov interpretation in dream/ | yes (interpret_mobile_goal spec=None) | no (experiments/nov7_baseline.py only) | isolated |
| dream/ lines | baseline | net -412 | smaller |
| ruff check / format | clean / clean | clean / clean | unchanged |
| isolation tests | 38 passed | 38 passed | unchanged |
| fast suite | 994 passed, 2 skipped | see F | more tests, same skips |
| slow suite | 86 passed | see F | see F |
| Equivalence (3 workflows) | reference | identical apart from the spec field | no behaviour change for typed goals |

Same laptop, the same shared virtual environment and the same MuJoCo build for both columns.

## K. Research integrity and assumptions

- **Nothing in the evidence changed:** no results/, frozen config, seed, threshold, scoring rule or success criterion was edited.
- **Frozen evaluations still reproduce** from their recorded commits. The measured comparison arms (nov7, the ungated ablation) still run their unguarded behaviour, now from experiments/nov7_baseline.py.
- **No behaviour change for typed goals,** checked by running before and after.
- **The functional change is a Rule 3 General algorithmic correction.** The original failure (the 7 Nov substitutions) and independent cases (H1 scenario L, fault injection, the Isaac re-test) already validate the guard; this change makes it apply on every path.
- **Simulation only, and oracle assistance unchanged.** The new README states per pipeline what is oracle-assisted.
- **Git history intact:** no history was rewritten, nothing was pushed, and no branch or tag was deleted.

## L. Cloud resource usage

Not applicable: no cloud resources were used. A read-only listing on 17 Nov found no DREAM VM, disk, bucket, network or service account.

## M. Remaining work

Technical debt, ranked by its effect on the research:

1. **G1 manipulation is not an agent capability.** The R1 grasp stack (g1_posture, wholebody, g1_supervisor, g1_grasp_geometry) is reachable only from experiments/g1_grasp_bringup.py (about 1150 lines). There is no typed tool, and the object pose is oracle-assisted. This blocks R2/R3 and any navigation-to-contact baseline.
2. **Oracle assistance limits every result.** Perception and odometry are oracle-assisted in most pipelines: the MuJoCo semantic sensor is true segmentation with independent per-frame errors, and odometry is true increments plus noise. Results are therefore upper bounds.
3. **Duplicated execution logic.** Toolbox and MobileToolbox duplicate call, cancel and read-only logic, and SemanticSearchPlanner re-implements the gain-frontier baseline (a baseline-fairness risk).
4. **Isaac isolation gap.** The Isaac G1MobileBase exposes .env and .controller to the agent.
5. **Size of experiments/.** It holds about 130 scripts and 30k lines, with copy-pasted helpers and about 12 overlapping summarisers. The Panda runners write into committed results by default.
6. **Ground truth shares a module with the generator.** The truth readers stay in dream/sim/home.py (generator coupling).
7. **Undeclared optional dependencies** (requests, truststore, spark_dsg, jwt, rosbags, websockets), and a torch pin conflict between the loader extra and the cloud lock.
8. **Editable install points at another checkout.** The shared editable install resolves to the OneDrive checkout. The README tells users to run scripts as modules from the repository root.
9. **Uncommitted R1 work in DREAM-g1-integration** must be merged onto this branch by its owner.

## N. Final conclusion

**Completed as a cleanup. No new capability is claimed.**

What it did:
- Closed the most important architectural gap: the original-goal record is now mandatory on every TaskAgent path, and the unsafe 7 Nov behaviour survives only as a measured baseline.
- Removed dead code and an experiment-to-production import, declared the real dependencies, consolidated duplicates, isolated unused scripts and rebuilt the README front around what the code does.
- All checks pass on the final tree, and three workflows behave identically before and after.

Research readiness for language-guided navigation and whole-body interaction on the G1:
- **Navigation side, ready.** Goal intake and verification, the WBC walking G1 in MuJoCo, occupancy mapping, Dijkstra/RPP navigation, room and object search are a usable, tested foundation with honest oracle labels.
- **Interaction side, not ready.** The R1 grasp exists only as a development script with an oracle object pose, W6 failed, and no agent tool can command the arms or hands.

The minimum work for a reproducible G1 navigation-to-contact baseline:
1. Expose the R1 walk-to-stance and contact steps as one typed mobile tool behind the existing safety supervisor, labelled oracle-assisted.
2. Add an independent, evaluation-side contact verifier (the correct object touched, contact force above a threshold, no fall).
3. Freeze a small dev and held-out scenario set (start at least 3 m away, varied table pose and object) with seeds and gates.
4. Run it from TaskAgent with a typed goal and its record.
5. Then replace the oracle object pose with the depth-based estimate already used for search.

Ready to merge into codex/g1-integration once the owner has reviewed it. The uncommitted R1 work in DREAM-g1-integration should be committed or rebased first. Nothing has been pushed.
