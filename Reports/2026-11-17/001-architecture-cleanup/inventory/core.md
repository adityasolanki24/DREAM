<!-- Read-only code audit of cf2851d by an audit agent on 17 Nov 2026; spot-checked by the task owner against the import graph. Nothing was run. -->

# Core inventory: dream/runtime, tools, agent, memory, eval, types, session
Read-only audit of C:\Users\24adi\DREAM-cleanup at cf2851d. Evidence is file:line from source plus import_graph.json.
Paths are relative to the repo root. "E/T" = number of experiments/tests importers.

## 0. Headline findings
1. There is no single pipeline. There is ONE reusable loop, `TaskAgent.run` (dream/runtime/agent.py:265), and ONE separate
   model-driven loop, `VLMAgent.run` (dream/runtime/vlm.py:214) that bypasses TaskAgent, `Goal` and `verify_goal` planning.
2. The immutable `GoalSpecification` (dream/tools/intent.py:83) is consumed by `TaskAgent.run(spec=...)` (agent.py:271,
   check at 309 and 357) and by `interpret_mobile_goal(spec=...)` (mobile_model.py:117). Only TWO runners pass it:
   experiments/isaac_g1_live.py:512 and experiments/semantic_search_eval.py:391 (and there only when `arm.gate` is true).
   Everything else (session.py:218, scheduler/phone_session.py:65, phase1_interface.py:471, all g1_* demos, run_tasks, run_vlm)
   calls `run(goal)` with `spec=None`, so the substitution guard is OFF. README line 116 "closed in the tested paths" is
   accurate only in that narrow sense. This is the single biggest architectural gap.
3. Duplicated code: `experiments/p6_frozen_mobile_model.py` is a verbatim frozen copy of an OLD dream/runtime/mobile_model.py
   (intentional, config says so; needed by tests/test_p6_frozen_replay.py). Keep, but it is a second `MobileModelPlanner`.
4. `dream/tools/` is mis-named for its content: it holds goals, a grammar, the verifier dispatch, two robot toolboxes and a
   discovery module. `dream/runtime/` mixes planners, VLM backends, an HTTP server, a media publisher and OAuth helpers.

## 1. The pipeline as the code implements it
Entry points (every one found):
 - CLI scripts calling `TaskAgent(...).run(goal)` directly: ~22 files in experiments/ (grep list: run_tasks.py:129,
   run_vision.py:80, run_compartment.py:79, g1_explore_room.py:284, g1_navigation_eval.py:120, g1_memory_demo.py:116,
   g1_relocalise_demo.py:109, g1_interruption_demo.py:110, isaac_g1_kitchen.py:492, command_scenario.py:86,
   mobile_recovery_validation.py:80, record_rgbd_sequence.py:137, compartment_experience.py:62, g1_explore_unknown.py:205, ...).
 - `LiveSession.submit` (dream/session.py:128) -> `_run_goal` (session.py:206) -> `self.agent.run(goal, on_event)` (218). No spec.
 - Phone/web server: experiments/phone_session.py:65 builds `TaskScheduler(store, run_task, parse_goal)` (scheduler.py:65)
   -> `PhoneAPI` (phone_api.py:154) -> `TaskAgent(...).run(task.goal, history, preempt)`. Grammar-only, no spec, planner is
   `RoomSearchPlanner("gain")` or `PlacePlanner`, never a model.
 - Model entry: experiments/phase1_interface.py (`interpret_mobile_goal` without spec at :202-213, `NaturalLanguageGoal` at :53,
   run at :471) and experiments/isaac_g1_live.py (spec path at :469-512).
 - Semantic search eval: experiments/semantic_search_eval.py:345 `ground_request` -> `bind_request` (:346) -> `agent.run(spec=)` (:391).
 - Model-in-the-loop on the tabletop: `VLMAgent.run(instruction, expected_goal=)` from run_vlm.py:84, cloud_qwen_pilot.py:168, g1_explore_unknown.py.
 - Eval runners in dream/eval/ are scorers/episode helpers, not goal entry points; dream/eval/episode.py runs discovery
   episodes (gantry), not task goals.

Chain (typed-goal path, MuJoCo/Isaac mobile G1 or Panda):
 1. Text/dict in. Interpretation (three alternatives, not one):
    a. Grammar: `tools/goals.parse_goal` (goals.py:110). Accepts only inspect cabinet, goto <place>, in/on <obj> <target>.
    b. Grammar + model: `tools/intent.ground_request` (intent.py:174) builds a `GoalSpecification` (:83); if ungrounded
       `runtime/mobile_model.propose_grounding` (mobile_model.py:460) asks a VLM, validated by `check_anchoring` (intent.py:249).
       Then `bind_request(spec, caps)` (intent.py:357) gives a typed `Goal`; `assess_support` (:224) refuses unsupported.
    c. Model-only: `interpret_mobile_goal` (mobile_model.py:117). With spec it loops `check_binding`; with spec=None it is
       the kept "7 Nov 2026 behaviour" (docstring :123), i.e. the known-bad comparison arm lives in production code.
 2. Original-goal record: `GoalSpecification` frozen dataclass with `digest` (intent.py:106). Held only if the caller passes it.
 3. `TaskAgent.run` (agent.py:265): refuse before acting if `check_binding` fails (:309); then per step (:320-):
    observe via `tools.call("observe_scene")`, `call("verify_goal")`, `call("get_robot_state")` (:339-343).
 4. Observation/state estimation: lives BELOW the agent, in the toolbox:
    - Panda: `Toolbox._observe_scene` (toolbox.py:192) over `dream.perception.vision.VisionPerception` / oracle tracker.
    - Mobile: `MobileToolbox._observe_scene` (tools/mobile.py:296) over `SpatialMemory` + occupancy grid + `base.odometry()`
      (oracle-assisted odometry per AGENTS.md section 2 item 5); semantic variant `SemanticWorld` (dream/semantics/world.py:23).
 5. Action choice: `self.planner.next_action(goal, scene, verdict, history)` (agent.py:350), one of the classes in section 2.
 6. Finish handling: `_typed_outcome` (agent.py:~399) forbids success from an exploration-point goal; with spec the success
    needs `check_binding` AND `verdict["satisfied"]` (agent.py:357-370), else `goal_substitution_rejected`.
 7. Execution: `Toolbox.call` (toolbox.py:168) validates schema (`_validate` :560), preconditions, cancel; then skill or
    mobile primitive on the robot adapter (dream/skills, dream/robot, dream/sim). Controllers are below the isolation boundary.
 8. Verification (all independent of the planner): `Toolbox._verify_goal` (toolbox.py:319, uses `tools.goals.verify` :172),
    `MobileToolbox._verify_goal` (mobile.py:367, inspect/goto/find only), `dream/semantics/verify.py` (track purity, views),
    `Belief.fit_check` (dream/agent/belief.py) and `agent.operate.constraint_deviation` (operate.py:68) for discovery.
 9. Failure / recovery / refusal: `UnsupportedGoal` goal (goals.py:94) refused by planners (e.g. mobile_model.py:296);
    `RuleBasedPlanner` retries via `_failures` (agent.py:199); TERMINAL_OUTCOMES typed in intent.py:~71; protective stop
    (agent.py:344), cancel (:322), preempt/suspend (:326, scheduler.py), step budget (:416). `Budgeted` wrapper lives in an
    experiment (experiments/g1_explore_room.py), not the package, yet phone_session.py and isaac_g1_live.py import it from there.
Discovery chain (Panda compartment): TaskAgent -> RuleBasedPlanner -> `investigate_opening` tool (tools/discovery.py) ->
`agent.belief.Belief` + `agent.exploration.EIGSelector` + `agent.model` -> `agent.operate.plan_opening` -> skills/articulation.
Separate chain (VLMAgent): instruction -> VLMBackend (`QwenTransformersBackend`/`OpenAICompatibleBackend` qwen.py:36/133,
`ChatGPTPlanBackend` openai_plan.py:100) -> `parse_tool_call` (vlm.py:136) -> `Toolbox.call`; completion accepted only if
`expected_goal` verifies (vlm.py module docstring). It does not use GoalSpecification at all.

## 2. Planners and loops
| Class | Where | Purpose | Run from | Tests | results/ evidence |
|---|---|---|---|---|---|
| TaskAgent | runtime/agent.py:259 | the loop (E 24, T 16) | everywhere | test_tools_runtime, test_scheduler, test_intent ... | all |
| RuleBasedPlanner | agent.py:72 | Panda tabletop pick/place/open, scripted recovery | run_tasks, run_vision, run_compartment, run_vlm | tests/test_tools_runtime.py | results/tasks, pick_place (Panda oracle 50/50, camera 39/40), compartment (27/30 oracle) |
| ExplorationPlanner | runtime/exploration.py:11 | red-sphere/blue-target G1 object task with bounded probes | g1_explore_unknown.py | test_g1_exploration, test_g1_release | results/g1_exploration_dev1-6, g1_exploration_heldout_* (dev/held-out, see README) |
| RoomSearchPlanner (gain/nearest) | runtime/search.py:33 | frontier exploration then inspect cabinet | 10 experiments incl. phone_session | test_mobile_exploration, test_room_instrumentation | results/g1_explore_room*, g1_navigation_*; README "Room search, frozen fresh seeds: FAIL" |
| PlacePlanner | search.py:109 | goto remembered place | command_scenario, phone_session, interruption, phase1 | NONE | g1_interruption_demo (demo only) |
| MemoryGuidedPlanner | runtime/memory_search.py:43 | wraps RoomSearchPlanner, checks remembered spots first | g1_memory_demo, g1_relocalise_demo, demo_home_memory, command_scenario | test_memory_search | results/g1_memory_demo (README: PARTIAL; worse when object moved) |
| SemanticSearchPlanner | runtime/semantic_search.py:88 | utility frontier+semantic search arms (utility, frontier_gain, frontier_nearest, ablations) | semantic_search_eval, isaac_semantic | test_semantic_search_integration, test_semantic_diagnostics | results/semantic_search_heldout, semantic_search_dev, isaac_semantic_* (held-out G0 FAILED: 1 false claim) |
| MobileModelPlanner | runtime/mobile_model.py:286 | VLM chooses one of admissible actions per step | isaac_g1_live, phase1_interface, vlm_compare*, audit_mobile_inspection | test_mobile_model, test_closeout_defects, test_isaac_live_replay | results/isaac_qwen_*, p6_qwen_frozen11_closeout (FAIL, 3 false claims on frozen matrix) |
| p6 MobileModelPlanner | experiments/p6_frozen_mobile_model.py:59 | frozen copy for replay | vlm_compare_p6_qwen | test_p6_frozen_replay | p6_qwen_frozen11_closeout |
| VLMAgent | runtime/vlm.py:214 | tabletop model-chosen tool loop (own loop, not a Planner) | run_vlm, cloud_qwen_pilot, g1_explore_unknown | test_vlm, test_g1 | results/phase1_vlm_compare, qwen_cloud_pilot_dev |
| MolmoBackend | runtime/molmo.py:115 | decode MolmoAct2 chunks, "no direct dispatch" | cloud_molmo_closedloop, molmo_openloop | NONE for MolmoBackend (decode_actions tested elsewhere) | results/molmo_* (README: commands tracked 150/150, never reached the object: verified negative) |
| SubstitutingPlanner | experiments/semantic_search_eval.py:90 | fault injection | semantic_search_eval | - | README: 7/40 false claims without spec, 0 with |
| ChatGPTPlanBackend | runtime/openai_plan.py:100 | ChatGPT-plan provider behind VLMBackend | phase1_interface, vlm_compare | test_openai_plan_resume, test_phase1_model_choice | results/phase1_vlm_compare (6/11 vs Qwen 5/11, descriptive) |

Duplicates (same job):
 - `VLMAgent` vs `TaskAgent`+`MobileModelPlanner`: both loop "observe, ask model for one tool call, execute, verify". They
   have different message formats, different completion gates (`expected_goal` vs `spec`+`verify_goal`), different step
   events (VLMStep vs Step). Real duplication of the model-in-the-loop concept. Not safe to merge (changes prompts, so results).
 - `RoomSearchPlanner("gain")` is re-implemented inside `SemanticSearchPlanner` as arm `frontier_gain` (semantic_search.py
   docstring: "Baseline A ... dream/runtime/search.py"). It does not import search.py (imports list for semantic_search shows
   only runtime.agent), so the baseline is re-coded, not reused. Fair-baseline risk (AGENTS.md rule 4), not just duplication.
 - `agent/exploration.py` ("EIGSelector") vs `runtime/exploration.py` ("ExplorationPlanner"): same word, unrelated jobs
   (probe selection for gantry/Panda discovery vs G1 sphere-pushing task planner). Naming collision.
 - Panda `Toolbox` (toolbox.py:148) vs `MobileToolbox` (mobile.py:149) vs `DexterousToolbox` (dexterous.py:34) vs
   `ExplorationToolbox` (object_exploration.py:66) vs `ExperienceToolbox` (memory/experience.py:54): five toolbox classes
   re-defining `call`, `_observe_scene`, `_verify_goal`; `MobileToolbox` is not a `Toolbox` subclass (class line 149) yet
   `TaskAgent` is typed to `Toolbox`.

## 3. Wrappers, protocols, registries, redundant state
 - Planner Protocol (agent.py:62): ~8 implementations; legitimate.
 - VLMBackend Protocol (vlm.py:81): 3 real implementations + test doubles; legitimate.
 - `Embodiment`, `Perception`, `MobileBase`, `DexterousEmbodiment` Protocols (types.py:177,221,254,207): `DexterousEmbodiment` has
   no references outside types.py in the first grep (only `DexterousState`/dexterous.py); check before deleting (likely unused Protocol).
   `CompliantMode` is a dataclass used by sim/tabletop_env and skills/articulation, fine.
 - Forwarding wrappers: `MemoryGuidedPlanner` wraps a planner (genuine added behaviour). `Budgeted` (experiments) wraps a planner
   to add a time budget, forwards the rest. `_CancellableRobot` (toolbox.py:543) forwards robot calls with a cancel check.
   `AgentBase` (semantic_search_eval.py:124) forwards five methods by design (isolation). None purely redundant.
 - Registries: tool specs are literal lists (toolbox.py ~:94, mobile.py ~:94), `VLMAgent.schemas = tool_schemas(toolbox.specs)`
   (vlm.py:231) derives from them. Two parallel copies of the `verify_goal` spec (toolbox.py:94, mobile.py:94). No factory sprawl.
 - Redundant world/memory state (all separate classes for "what the robot believes about objects"):
   `perception.world.WorldModel` (Panda tracks), `perception.spatial.SpatialMemory` (mobile candidates+grid),
   `perception.semantic.SemanticMap` (+ its own `Track` class, semantic.py:81 vs world.py:42), `semantics.world.SemanticWorld`,
   `memory.home.HomeMemory` (sqlite beliefs/places, home.py:75) with `memory.scene_graph` view and `semantics.graph`,
   and `agent.belief.Belief` + `memory.home.Belief` (two classes named Belief, home.py:60 and belief.py:391) for different things;
   `agent.movability.MovabilityBelief` (third belief). Two `Track` classes. Suggest renaming (home.Belief -> RememberedSighting,
   perception.semantic.Track -> SemanticTrack): refactor only, no behaviour change.
 - `session.py` is `LiveSession` for the Panda tabletop only (imports sim.tabletop_env, tools.toolbox); not the G1/phone path.
   Name suggests a general session; it is the Panda demo shell.

## 4. Unused / unreachable
Per import graph (E=experiments importers, T=tests):
 - `dream/runtime/volo_contract.py` (400 lines): E 0, T 2, imported by nothing in dream/. Purpose: fail-closed, non-actuating
   description/validation of NVlabs VoLoAgent messages (module docstring :1-13). It imports `tools.toolbox.ToolResult` only.
   Nothing reachable at runtime uses it. README: "VoLoAgent | Interface contract and mock; no actuation | NOT EVALUATED".
   It is evidence of an integration assessment, not a component. Status (e)/(c). Move to experiments/ or tests/support if the
   test is kept; deleting loses the phase1_volo_evidence_20261028 reproducibility (AGENTS.md rule 7) so do NOT delete.
 - `dream/eval/truth.py` (33 lines): `spec_to_hypothesis`, `compartment_to_hypothesis` convert simulator specs to `agent.model.Hypotheses`
   for scoring/validation. Used by tests/test_belief.py-style tests only (T 2, E 0, D 0). Its own docstring says "Never import from
   dream.agent" but it imports `dream.agent.model` (truth.py:7); the isolation test passes because it checks the other direction.
   Cosmetic; keep as a test helper, move to tests/ if desired.
 - `dream/robot/g1_rollout.py` (341 lines): imported by tests only (test_g1_posture.py:186) and dream/robot/g1_posture.py docstring
   (g1_posture.py:14). Graph shows E 0, T 1, D 0: nothing reachable calls `verify_top_k`. It is the "rollout gate" for the whole-body
   posture planner from the R1 plan; unused by any executed pipeline. Status (b)/(e): development scaffolding for R1 only.
 - `dream/agent/operate.py` (87 lines): E 0, T 0 directly, reached only through tools.discovery and eval.episode. OK (reachable).
 - `dream/eval/metrics.py`: T 0, reached through eval.episode. OK. `dream/eval/plots.py`: E 2, T 0 (untested). `dream/eval/vision.py`: E 1.
 - `dream/tools/dexterous.py` (E 0, T 1) and `object_exploration.py` (E 1: g1_explore_unknown, T 5): reachable only via the G1 Dex3
   explore experiment; README: Dex3 actuator path is opt-in, W6 FAILED. Status (d)/(b).
 - `dream/memory/experience.py` (E 1: compartment_experience, T 0): the only "experience memory" class; README states
   "Experience memory: planned"; this is a prototype hook (AGENTS.md section 5 item F) with no test. Status (c).
 - `dream/runtime/openai_plan.py` + `siwc_auth.py`: reachable from phase1/vlm_compare only; siwc_auth (E 2, T 4) OAuth helper.
 - `dream/runtime/media_runtime.py` (E 2, T 1) and `camera_stream.py` (E 4, T 2): optional streaming publishers, not part of the control loop.
 - Untested planner: `PlacePlanner` (no test), `MolmoBackend` (no direct test), `eval.plots`, `memory.experience`.
 - No file in scope is imported by nothing at all in any category.

## 5. Broken / unclear interfaces
 - `TaskAgent.run(goal, on_event, history, preempt, spec)`: `spec=None` silently disables the substitution guard
   (agent.py:265-290). A mode switch hidden in an optional argument; production phone path never sets it. FUNCTIONAL CHANGE to fix.
 - `interpret_mobile_goal(..., spec=None)` (mobile_model.py:117): spec=None runs a different, known-defective algorithm. Two
   unrelated behaviours behind one flag; retained intentionally as comparison arm.
 - `finish` is a nested closure re-defined inside `run` when spec is given (agent.py:~296) with a different signature default;
   `Episode.spec` set after construction. Works, hard to read.
 - `Planner.next_action` returns `Action` (ToolCall | Finish) but `MemoryGuidedPlanner`, `ExplorationPlanner` etc. omit return annotations.
 - Toolbox duck-typing: `TaskAgent` calls `getattr(self.tools, "hold_for_reasoning", None)` (agent.py:321); implemented in
   robot/g1.py:226 and object_exploration.py:103 only. VLMAgent calls a different fallback (`robot.hold`, vlm.py:270).
 - Two goal vocabularies: `Goal` union (goals.py:100: InRegion, OnTop, Inspect, GoToPlace, GoToObserved, FindObject, UnsupportedGoal)
   and `NaturalLanguageGoal` (goals.py:104) which is NOT in the union and only accepted by experiments/phase1 and isaac_g1_live.
   `parse_goal` cannot produce GoToObserved/FindObject; those only come from `bind_request`.
 - `tools/goals.parse_goal` hard-codes "cabinet" as the only inspect target (goals.py:~137) and the `verify` dispatcher lives in
   goals.py (the verdict) while MobileToolbox._verify_goal reimplements the dispatch: two verifier entry points with different coverage.
 - Name mismatches: `dream/agent/` (renamed `discovery` per AGENTS.md section 5), `dream/runtime/exploration.py` (not exploration:
   a sphere-task planner), `dream/session.py` (Panda-only), `dream/tools/mobile.py` (797 lines: toolbox + spec list + frontier helpers).
 - Isolation: AGENTS.md rule 1 says decision code must not import dream.sim; `session.py` imports `dream.sim.*` (graph) but it is
   declared "wiring, like dream/eval" (AGENTS.md section 14), consistent.

## 6. Status per component
(a) Verified end to end with evidence:
 - Panda pick-and-place via TaskAgent + RuleBasedPlanner: results/pick_place, results/tasks (oracle 50/50, camera 39/40, PASS).
 - Compartment opening, oracle-assisted: results/compartment* (27/30, 27/30, 30/30, 0 false claims, PASS).
 - Active Physical Discovery (agent/belief, exploration, model, operate): README gantry results; eval.episode/metrics.
 - TaskScheduler + PhoneAPI plumbing: tests/test_scheduler.py, test_phone_api.py; camera loopback PASS (README).
(b) Narrow simulation/oracle assumptions:
 - RoomSearchPlanner/MemoryGuidedPlanner: oracle-assisted odometry, procedural rooms; frozen fresh seeds FAIL (15/20, 4/10, 10/10).
 - Camera compartment opening 11/20 (FAIL) uses the camera but the validated results still use the oracle handle tracker.
 - Substitution guard: MuJoCo held-out 7/40 -> 0/40 (fault injection); Isaac 0/9 vs 4/6 but gates incomplete (l_island never loaded).
 - Home memory: recall 5/5 but worse when an object moved (PARTIAL).
(c) Partial:
 - SemanticSearchPlanner: held-out 83/87, 31/32, 14/15 but G0 (zero false claims) FAILED with one shower-as-bathtub claim.
 - HomeMemory/experience.py/scene_graph.py; MobileToolbox.verify (inspect/goto/find only).
 - VoLo contract (non-actuating).
(d) Failed integration:
 - MolmoAct2 (runtime/molmo.py): tracks 150/150 but never reaches the object or closes the gripper (README "Verified negative").
 - Qwen goal substitution (MobileModelPlanner spec=None arm): 3 false claims on the frozen matrix, T3 1/3, T4 2/3 (README FAIL);
   fixed only in spec-gated paths.
 - W6 whole-body (tools/dexterous, object_exploration, g1_rollout): off-band 0.667 (0.70), refusal 0.864 (0.90), not merged.
 - GR00T: open-loop inference only, BLOCKED (not in this scope).
(e) Unused: volo_contract (tests only), eval.truth (tests only), g1_rollout (tests only), DexterousEmbodiment (probably),
 hosted OpenAICompatibleBackend (qwen.py docstring says unvalidated).

## 7. Cleanup suggestions (minimal; F = functional change, R = refactor, no result change)
 1. F: make the original request mandatory. Add `spec` creation to scheduler/phone path (`ground_request` + `bind_request`) or at
    least make `TaskAgent.run` warn/raise when `spec is None` in model-driven runs. Changes accepted-success behaviour; must be
    rerun/reported (AGENTS.md rule 7). Highest value.
 2. R: move `Budgeted` from experiments/g1_explore_room.py into dream/runtime so phone_session.py and isaac_g1_live.py stop
    importing a planner wrapper from an experiment. No behaviour change.
 3. R: delete duplicate `verify_goal` spec literal by sharing it between toolbox.py:94 and mobile.py:94 (check text identical first).
 4. F (if done): make `semantic_search` `frontier_gain` arm call `RoomSearchPlanner("gain")` instead of re-implementing it. Could
    change baseline numbers; first diff the two on seeds and prove identical decisions, otherwise report as baseline change.
 5. R: rename `runtime/exploration.py` to something like `sphere_task_planner.py` and later `agent/` to `discovery/` as AGENTS.md
    already plans; update test_isolation DECISION_PACKAGES in the same commit.
 6. R: move `volo_contract.py`, `eval/truth.py` and (if no pipeline needs it) `robot/g1_rollout.py` to a clearly labelled
    `dream/eval`/`dream/audit` home or keep with an "audit only" docstring line. Do not delete (rule 7, evidence results exist).
 7. R: rename `memory.home.Belief` and `perception.semantic.Track` to remove name collisions; update imports in a single commit.
 8. R: delete `DexterousEmbodiment` Protocol only after re-grep (`grep -rn DexterousEmbodiment` outside types.py returned nothing in
    dream/ in this audit; not checked in experiments beyond that grep).
 9. R: add a base `class MobileToolbox(Toolbox)` or document why it is not (it duplicates `call`/cancel/READ_ONLY logic).
    Larger than a rename; do last, with the full test suite.
10. R: add a one-paragraph "which loop is canonical" note to README (TaskAgent is canonical; VLMAgent legacy Panda/tabletop-only).

## 8. Gaps and uncertainties in this audit
 - I did not read dream/tools/toolbox.py or mobile.py bodies line by line; claims about duplicated `call`/cancel logic come from
   class/def listings and the import graph, not a diff.
 - Test names are from grep -l, not from running the suite; "NONE" means no file under tests/ mentions the symbol.
 - Which results/ directory backs which gate is taken from README status rows and directory names, not from opening each summary.json.
