<!-- Read-only code audit of cf2851d by an audit agent on 17 Nov 2026; spot-checked by the task owner against the import graph. Nothing was run. -->

# DREAM-cleanup audit: experiments, cloud, tests, deps, README reproduction
Tree: C:\Users\24adi\DREAM-cleanup, branch claude/architecture-cleanup-20261117 @ cf2851d. Read-only. Method: AST import check, git grep of results/ and Reports/ (metadata.json, cloud_job.json, job.json, summary.json, logs), configs/tests cross-reference, argparse comparison. The 55-script "unreferenced" list from import_graph.json is largely explained by scripts being run from the command line (see table).

Confidence tags: "name" = only matched by results directory name; "probable" = inferred; "unknown" = could not determine. No script was run.

## 0. Key findings
- No broken imports found: every `dream.*`, `cloud.*`, `experiments.*` import in experiments/, cloud/, tests/ resolves (AST check incl. lazy imports, symbols grep-checked). import_graph flags (`from experiments import x`) are namespace-package artefacts, not breakage. experiments/ has no __init__.py (namespace package).
- Stale pointers only: dream/robot/g1_posture.py:530 docstring names experiments/g1_reach_h3_diagnosis.py (does not exist). tests/test_isaac_g1.py builds a fake experiments/hello.py in a tarfile (intentional).
- results/*/metadata.json in 3 dirs (g1_adaptation_dev, g1_autonomy_dev, g1_autonomy_recovery freeze.json) embed a source snapshot (results/<dir>/source/ or source_snapshot.tar.gz) listing ~35 scripts. These hits are NOT evidence a script produced results; I excluded them as evidence. Likewise `validation.json` in g1_cloud_preparation / groot_openloop_preparation / groot_retry_validation / g1_integration_dev list the whole experiments set (gate commands), weak evidence.
- Real run-command evidence exists only for: results/*/cloud_job.json (args), results/research_closeout_20261106/inventory.json, Reports/2026-11-09/001 and 2026-11-10/010,011 metadata.json "command", results/semantic_*/summary.json. Reports has only 4 dated dirs (2026-11-09,10,11,17), 165 metadata.json under Reports, so all older (Oct 2026) results predate the Reports system and their commands are mostly unrecorded.
- Class A requires "validated, README-cited": almost nothing is validated in the README sense beyond the semantic-search/verifier/room/Qwen chains (frozen configs). Older Panda/G1 results are cited only through README tables not by command.

## 1. Classification of all 140 experiments/*.py
Classes: A entry point of validated README-cited workflow (or shared module those workflows need); B runner/summariser that produced a committed results dir; C one-off diagnostic/probe/audit with committed output; D one-off, no committed output found; E? broken or superseded (none verified broken; E? = probable superseded, unverified). Last column: test files that import/name it ("-" = none).
Counts: A=22 B=69 C=21 D=25 E?=3 (total 140). Non-.py files in experiments/: camera_latency_driver.cjs, camera_latency_probe.js, camera_latency_raw.html, camera_remote_driver.cjs, camera_remote_page.html (drivers of camera_latency_measure and camera_remote_measure; class B with them), data/, p6_inputs/, configs/.

| script | class | evidence | used by tests |
|---|---|---|---|
| analyse_molmo_failure_mode | C | results/molmo_failure_mode_closeout/summary.json names it | - |
| analyse_molmo_multichunk | B | results/molmo_multichunk_closeout (frozen H4 rule) | molmo_multichunk_rule |
| audit_foundation_models | B | results/foundation_model_audit*; configs main/vision | - |
| audit_g1_terminal | D | no committed output or reference found | - |
| audit_groot_actions | C | offline audit of saved decoded predictions; groot_* dirs | groot_audit |
| audit_mobile_inspection | C | results/mobile_inspection_semantics (probable) | mobile_recovery |
| audit_t1_run | C | summary of a retrieved T1 run; t1_* result dirs (name only) | - |
| audit_vla_sources | C | results/vla_source_audit; config vla_sources.json | - |
| calibrate_adequacy | B | results/adequacy_calibration; imported by calibrate_panda_adequacy | - |
| calibrate_handle_measurement | B | results/handle_measurement_calibration | - |
| calibrate_panda_adequacy | B | results/panda_adequacy_calibration | - |
| camera_delivery_benchmark | E? | old PNG vs JPEG cost comparison; no output; probably superseded by camera_latency_measure (unverified) | - |
| camera_latency_measure | B | results/camera_delivery_closeout_20261106 (latency*.csv); inventory.json; ruff-exempt | - |
| camera_media_vm_probe | B | results/camera_delivery_cloud_20261107 (inventory.json) | - |
| camera_noninterference | B | results/camera_delivery_closeout_20261106/noninterference.json (probable; name match) | - |
| camera_remote_measure | B | results/camera_delivery_cloud_20261107; inventory.json | camera_remote_measure |
| capture_groot_input | B | results/groot_openloop_preparation/validation.json lists it | groot_openloop |
| cloud_bootstrap | A | README H4 path; results/semantic_search_heldout/cloud_job.json | - |
| cloud_g1_qwen | B | results/g1_qwen_cuda_dev (name); g1_cloud_preparation/validation.json; config cloud_g1_qwen.json | g1_cloud,g1_snapshot_gate |
| cloud_groot_openloop | B | results/groot_cuda_openloop, groot_dispatch_failure/vm.log, groot_first_cuda_failure | groot_assets,groot_openloop |
| cloud_molmo_closedloop | B | results/molmo_cloud_feedback_dev, molmo_multichunk_closeout (frozen config) | molmo_feedback,molmo_multichunk |
| cloud_molmo_openloop | B | results/molmo_cloud_openloop | molmo |
| cloud_qwen_pilot | B | results/qwen_cloud_pilot_dev, cloud_pilot_validation, cloud_preflight meta.json | cloud |
| code_version | A | shared helper imported by frozen runners (g1_explore_room, detector_compare, nested_replay) | code_version |
| command_scenario | D | used only by g1_command_demo/isaac_g1_kitchen | - |
| compare_compartment_arms | D | per-seed vision vs oracle table; no output found | - |
| compare_g1_rendering | C | results/g1_release_development_final/support metadata | g1_platform_audit |
| compare_room_knowledge | A | README Repro -> results/room_knowledge_comparison/comparison.json | - |
| compare_room_regression | B | results/g1_explore_room_regression_closeout | room_regression_config |
| compartment_experience | B | results/compartment_experience(_smoke); inventory.json | - |
| compose_demo_video | D | no output evidence; pairs with isaac_replay_video | - |
| demo | C | README/AGENTS interactive demo; results/demo (name match only) | isaac_g1 |
| demo_compartment | D | recorded demo; no dedicated results dir (unknown) | - |
| demo_home_memory | D | scripted DB for interface development; no output found | - |
| demo_vision | B | results/demo_vision | - |
| detector_compare | B | results/detector_compare_stub_dev; Reports 2026-11-09 | detector_compare |
| diagnose_search_episode | A | README Repro -> H2_false_claim_diagnosis.json | - |
| extract_household_knowledge | A | README: produced the frozen ConceptNet extract (ruff exemption) | - |
| fetch_wbc_policy | D | tooling (downloads weights) | g1_wbc |
| g1_autonomy_report | B | results/g1_autonomy_cuda_frozen/recovery; config g1_autonomy_frozen | g1_autonomy_report |
| g1_command_demo | B | results/g1_command_demo_mujoco (name) | - |
| g1_contact_resolution | B | results/g1_contact_resolution_dev | - |
| g1_dataset | B | LeRobot export; results/g1_autonomy_*, g1_integration_dev | g1_dataset |
| g1_dexterous | B | results/g1_dex3_dev(_v2,_v3), g1_dex3_challenge, g1_dex3_checkpoint | g1 |
| g1_explore_room | A | README room search; results/g1_explore_room_*closeout*, postfix_closeout (frozen configs) | phase1_model_choice,room_instrumentation,room_postfix_config |
| g1_explore_unknown | B | results/g1_exploration_dev1-6, validation; config g1_explore_dev.json | g1_cloud,g1_inference_recording |
| g1_failure_audit | C | offline audit; g1_release_development_* metadata | - |
| g1_grasp_bringup | B | dev diagnostic; Reports 2026-11-10/010,011 metadata commands; results/g1_r1_*_dev_*; README lists them (R1 dev, not a validated claim) | g1_grasp_bringup |
| g1_interruption_demo | B | results/g1_interruption_demo(_v2,_video) cloud_job.json | scheduler |
| g1_locomotion | B | results/g1_locomotion (config g1_locomotion.json) | - |
| g1_memory_demo | B | results/g1_memory_demo(_video) cloud_job.json | - |
| g1_modality | B | module for g1_validate_loader/dataset; results/g1_contract_resolution | - |
| g1_navigation_eval | B | results/g1_navigation_dev(_v2), heldout(_v2) cloud_job.json args | navigation_eval,rpp |
| g1_recording | B | imported by g1_dataset/g1_dexterous | g1_dataset,g1_recording |
| g1_relocalise_demo | B | only results/g1_relocalise_demo_smoke committed; full config has no result | - |
| g1_replay_decisions | C | replays saved model text; g1_autonomy_dev/release metadata | - |
| g1_skill_collection | B | g1_autonomy_recovery/release dev dirs (name) | g1_skill_collection |
| g1_validate_loader | C | results/g1_integration_dev (probable); config g1_loader_sources.json | - |
| g1_wbc_envelope | B | results/g1_wbc_envelope(_v2); read by test_g1_supervisor | - |
| groot_assets | B | results/groot_retry_validation; groot_openloop_preparation | groot_assets |
| groot_contract | B | results/groot_contract_audit/report.json,validation.json | groot_contract |
| groot_openloop | B | guest side of cloud_groot_openloop; groot_* result dirs | groot_openloop |
| inspect_molmo_actions | C | results/molmo_contract_audit / observation_* (probable) | - |
| inspect_molmo_feedback | C | results/molmo_observation_dev/confirmed (probable) | - |
| inspect_t1_residuals | C | results/t1_stand_*, t1_validation_checkpoint (probable) | t1_residuals |
| inspect_t1_stand | C | results/t1_stand_*_local, t1_stand_state_failure (probable) | - |
| isaac_asset_validation | B | Reports 2026-11-09 metadata; config isaac_assets.json; result dir unknown | isaac_asset_validation,isaac_home_usd,isaac_marker_check |
| isaac_capture_manifest | B | manifest for results/isaac_perception_capture (name) | isaac_capture_manifest |
| isaac_g1_heldout | D | 12 held-out kitchen episodes wrapper; no results found | - |
| isaac_g1_kitchen | B | results/isaac_g1_kitchen_bringup, _explore_dev cloud_job.json | isaac_g1,isaac_live |
| isaac_g1_live | A | module behind isaac_g1_live_eval; results/isaac_qwen_live_20261101/job.json | isaac_live,isaac_live_eval,isaac_semantic |
| isaac_g1_live_eval | A | README Repro; results/isaac_qwen_closeout_20261107/job1_matrix.json,summary.json | isaac_live_eval |
| isaac_home_compose | B | library for isaac_home_runs | isaac_asset_validation,isaac_home_compose,isaac_home_exclusions |
| isaac_home_refusals | D | table of composer refusals; no output found | - |
| isaac_home_runs | B | results/isaac_home_dev; Reports 2026-11-09 metadata | isaac_home_runs |
| isaac_kitchen_repair | B | results/isaac_kitchen_repair | isaac_kitchen_repair |
| isaac_replay_video | D | no output evidence | - |
| isaac_scene_probe | C | results/isaac_semantic_dev/job3.json,job4.json | - |
| isaac_semantic | A | module for isaac_semantic_eval/isaac_g1_live; README mention | isaac_semantic |
| isaac_semantic_dev | A | README; results/isaac_semantic_dev job.json, error.txt (development status) | isaac_semantic_dev |
| isaac_semantic_eval | A | README; results/isaac_semantic_eval; frozen config isaac_semantic_eval.json | isaac_semantic |
| isaac_t1_walk | B | results/t1_* dirs; cloud/isaac_t1_job.sh; config t1_walk.json | t1_packaging |
| isaac_task | D | library for g1_grasp_bringup/isaac_g1_kitchen | - |
| mobile_recording | D | recording helper imported by 4 demo scripts | - |
| mobile_recovery_validation | B | results/mobile_recovery_20261031, _closeout_20261106 (probable; no script hit) | - |
| molmo_openloop | B | guest side of molmo cloud runs; results/molmo_cloud_openloop | - |
| nested_replay | B | results/nested_association_dev; Reports 2026-11-09 | nested_replay |
| openai_plan_login | D | auth tool; no output | phase1_model_choice |
| openai_plan_refresh | D | auth tool; no output | openai_plan_refresh |
| p6_frozen_mobile_model | B | model for p6 frozen replay; frozen config | p6_frozen_replay |
| phase1_interface | B | results/phase1_interface receipts | isaac_live_eval,mobile_model,phase1_model_choice |
| phone_session | D | live-session CLI; no output | - |
| plot_g1_qwen | B | plots for g1_qwen results | - |
| preflight_groot_tokenizer | C | results/groot_retry_validation | - |
| prepare_g1_usd | D | tooling (USD conversion) | - |
| provenance | A | shared helper; README mention; imported by box_detectors, joint, cloud/launch | - |
| qwen_room_priors | A | README; results/qwen_room_priors/cloud_job.json, priors.json | isaac_semantic_dev,knowledge_sources |
| qwen_tool_server | A | infra for isaac_g1_live_eval; named in isaac_qwen_closeout summary.json | - |
| record_rgbd_sequence | D | SLAM input; no results | - |
| record_volo_messages | C | probably produced tests/fixtures/volo_upstream_messages.json (unverified) | volo_contract |
| render_homes | D | plots procedural homes; no results dir | - |
| replay | D | renders results/sessions/<t>/trajectory.json to GIF; no committed sessions dir | - |
| replay_molmo_prefix | B | results/molmo_recorded_prefix_dev/confirmed/validated | - |
| reports | A | AGENTS.md s15 mandatory Reports tool; test_reports | reports |
| resolve_g1_contract | C | results/g1_contract_resolution | - |
| rtabmap_eval | D | no results | - |
| run_benchmark | B | results/main, main_geometric_prior (config main.json) | - |
| run_compartment | B | results/compartment*; summary.json; frozen closeout configs | - |
| run_pick_place | B | results/pick_place | - |
| run_tasks | B | results/tasks (config tasks.json) | isaac_live |
| run_vision | B | results/vision (config vision.json) | - |
| run_vlm | D | config vlm.json exists; no results/vlm dir found; unknown whether output was kept | - |
| run_volo_passthrough_smoke | C | inventory.json item (smoke); no results dir | - |
| semantic_exploration_compare | D | pairs with DRAFT config semantic_exploration_k2.json; no output | - |
| semantic_knowledge_gates | B | results/semantic_knowledge_xl/summary.json names it | - |
| semantic_search_diagnosis | D | task 2026-11-09 diagnosis; no results dir; unknown if output kept in Reports | - |
| semantic_search_eval | A | README Repro: H1-H4,V1 -> results/semantic_search_heldout, _verifier_v2; cloud_job.json records it | agent_base_isolation,knowledge_sources,semantic_search_integration |
| semantic_search_gates | A | README Repro -> results/semantic_search_heldout/gates.json | - |
| session | A | README/AGENTS interactive session CLI; no results dir | - |
| stress_shoulder_singularity | C | results/shoulder_singularity_stress | - |
| summarise | B | generic recompute of results/<name>/summary.json | eval |
| summarise_closeout | B | closeout gate summaries; frozen closeout configs | summarise_closeout |
| summarise_isaac_postfix | A | README Repro; inventory.json; results/isaac_qwen_postfix_closeout | - |
| summarise_isaac_semantic | E? | Isaac semantic merge; no hits; overlaps summarise_isaac_postfix/summarize_isaac_live (unverified) | - |
| summarise_room_postfix | A | README Repro; named in research_closeout_20261106/inventory.json | - |
| summarise_semantic_search | A | README Repro; summary.json in semantic_search_heldout/verifier_v2/semantic_knowledge_xl | - |
| summarize_g1_exploration | B | results/g1_exploration_validation (probable) | - |
| summarize_g1_qwen | B | results/g1_qwen_cuda_dev (probable) | - |
| summarize_g1_reliability | B | results/g1_release_reliability_before, g1_release_development_* | - |
| summarize_isaac_live | E? | per-task table for live Isaac sessions; no hits; overlaps summarise_isaac_postfix (unverified) | - |
| summarize_vlm_comparison | B | summarises phase1_vlm_compare/p6 | summarize_vlm_comparison |
| t1_recording | B | helper for isaac_t1_walk recordings | t1_recording |
| tum_to_rosbag | D | no results | - |
| validate_adequacy | B | results/fresh_adequacy_validation | adequacy |
| validate_joint_controller | B | results/joint_controller_dev/confirmed, joint_hold_failure_dev; config joint_validation | - |
| verifier_v2_gates | A | README Repro -> semantic_search_verifier_v2/gates.json (summary.json names it) | - |
| verify_cloud_recovery | C | g1_release_development_* metadata; tmp_path unit test | cloud_recovery_integrity |
| vlm_compare | B | results/phase1_vlm_compare, phase1_gpt_*, p6_qwen_frozen11_closeout run_*.json | p6_frozen_replay,vlm_compare_cloud |
| vlm_compare_frames | D | renders fixed set for phase1_vlm_compare; no direct record (unknown) | - |
| vlm_compare_p6_qwen | B | results/p6_qwen_frozen11_closeout; inventory.json | p6_frozen_replay |

## 2. Duplicated runners and copy-pasted helpers
Identical function bodies (AST-equal, >250 chars) across experiments/*.py:
- file_hash: cloud_g1_qwen, groot_assets
- write: cloud_molmo_closedloop, molmo_openloop
- iou: detector_compare, isaac_home_runs, isaac_semantic_dev
- mask_box: isaac_home_runs, isaac_semantic_dev
- digest: g1_recording, groot_contract, isaac_home_runs, prepare_g1_usd
- load_manifest: isaac_home_compose, reports
Near-duplicate helpers: `git_sha` re-implemented in camera_latency_measure, g1_wbc_envelope, run_benchmark although experiments/code_version.py (code_sha, archive-aware) exists; `wilson` interval in g1_wbc_envelope and summarise_semantic_search; `_job` in 5 Panda adequacy scripts (calibrate_adequacy, calibrate_handle_measurement, calibrate_panda_adequacy, run_benchmark, validate_adequacy); `episode` and `summarise` functions in about 10 runners each (g1_explore_room, g1_navigation_eval, g1_interruption_demo, g1_memory_demo, run_compartment, run_pick_place, isaac_g1_heldout, compartment_experience, ...).
Overlapping jobs (candidates to merge; not compared byte-level):
- Summarisers: summarise (generic episodes.jsonl), summarise_closeout, summarise_room_postfix, summarise_isaac_postfix, summarise_isaac_semantic, summarise_semantic_search, summarize_g1_exploration, summarize_g1_qwen, summarize_g1_reliability, summarize_isaac_live, summarize_vlm_comparison, g1_autonomy_report. Both spellings summarise/summarize coexist. Gate scripts semantic_knowledge_gates, semantic_search_gates, verifier_v2_gates share one shape (read frozen config gates, evaluate, write gates.json).
- Paired comparison: compare_compartment_arms, compare_room_regression, semantic_exploration_compare, compare_g1_rendering, vlm_compare.
- Isaac live/Qwen summarising: summarise_isaac_semantic, summarize_isaac_live, summarise_isaac_postfix (E? candidates).
- Offline audits of saved evidence ("read-only, no physics" pattern): audit_g1_terminal, g1_failure_audit, inspect_t1_stand, inspect_t1_residuals, audit_t1_run, inspect_molmo_actions, inspect_molmo_feedback, audit_groot_actions, audit_mobile_inspection.
- Demos: demo, demo_compartment, demo_vision, demo_home_memory, g1_command_demo, g1_memory_demo, g1_relocalise_demo (docstring "Tests A/B again", supersedes g1_memory_demo), g1_interruption_demo, phone_session, session. Recorders: mobile_recording, g1_recording, t1_recording, record_rgbd_sequence.
- Camera delivery: camera_delivery_benchmark (PNG vs JPEG, no output) vs camera_latency_measure + camera_remote_measure + camera_media_vm_probe + camera_noninterference.
- Config-per-variant: 14 g1_grasp_bringup_*.json for one 1148-line script; 14 compartment_*.json for run_compartment.
- Cloud pilots: cloud_qwen_pilot, cloud_g1_qwen, cloud_groot_openloop, cloud_molmo_openloop, cloud_molmo_closedloop plus guest side molmo_openloop, groot_openloop, all via cloud/launch.py (Oct 2026, historical).

## 3. experiments/configs/*.json (106 files)
Key `status` present:
- frozen: compartment_oracle_reference_closeout, compartment_vision_closeout, g1_explore_room_bisect_closeout, g1_explore_room_closeout, g1_explore_room_postfix_closeout, g1_explore_room_regression_closeout, isaac_g1_qwen_closeout, isaac_semantic_eval, molmo_multichunk_closeout, p6_qwen_frozen11_closeout, semantic_knowledge_xl (frozen_commit 1f06db2...), semantic_search_heldout (frozen_commit is prose, "parent of the commit that sets this field"; README: 9c19328), semantic_search_verifier_v2 (frozen_commit 2db3904...). README: frozen_commit values are pre-rewrite IDs, map in results/commit_map_20261109.json.
- development: isaac_home_dev, semantic_exploration_dev, semantic_search_dev.
- DRAFT: semantic_exploration_k2 (status "DRAFT (not frozen; for owner review)", frozen_commit None). Read only by semantic_exploration_compare, which has no committed result.
Informally "frozen" in text but no `status` key (not machine checkable): g1_autonomy_frozen, g1_navigation_heldout(_v2), g1_wbc_envelope_v2, g1_dex3_challenge, g1_dex3_dev, g1_interruption_demo_v2/_video, g1_memory_demo_video, g1_relocalise_demo(+_smoke), isaac_g1_kitchen, compartment_experience, compartment_shift, g1_release_reliability_dev.
Cloud job configs (cloud_*.json, 13): read by cloud/launch.py and cloud/isaac_session.py (cloud_isaac_g1_session). cloud_preflight.json: no reader found (purpose "read-only readiness check"); results/cloud_preflight exists.
Source or pin lists, not experiments: vla_sources, molmo_contract_sources, navigation_sources, integration_sources, groot_action_contract, isaac_asset_candidates, isaac_assets, isaac_asset_fronts, g1_loader_sources, wbc_policy, media_runtime.
Not named in any .py/.sh/.cjs under experiments, cloud, dream, tests. Configs are passed by path on the command line, so "unread" is not proof of staleness; those with a results/<name>/ dir are historical evidence:
- compartment_challenge_{armature,bar,deep,gravity_lid}, compartment_oracle_reference_closeout, compartment_out_of_class, compartment_regression_vision_fixes, compartment_shift, compartment_smoke, compartment_vision_continuation_{dev,smoke}, compartment_vision_dev, g1_adaptation_dev, g1_autonomy_dev, g1_dex3_challenge, g1_explore_heldout_eig, g1_explore_heldout_fixed_short, g1_explore_room_{bisect_closeout,dev,smoke}, g1_interruption_demo_{v2,video}, g1_locomotion_smoke, g1_memory_demo_video, g1_navigation_{dev_v2,heldout,heldout_v2,smoke}, g1_release_reliability_dev, g1_relocalise_demo_smoke, isaac_asset_fronts, main_geometric_prior, molmo_contract_sources, p6_qwen_frozen11_closeout, semantic_exploration_dev, tasks_smoke, vision_smoke, cloud_preflight, and 9 of the 14 g1_grasp_bringup_* variants (only 5 named by tests/test_g1_grasp_bringup.py).
- Strongest "unused" candidates (no reader and no results dir by that name): g1_explore_heldout_eig, g1_explore_heldout_fixed_short (README results are named g1_exploration_heldout_*), g1_release_reliability_dev, g1_navigation_smoke, g1_locomotion_smoke, compartment_smoke, tasks_smoke, vision_smoke, g1_explore_room_smoke, and the grasp variants whose results may be in results/g1_r1_*_dev_* (unknown).

## 4. cloud/
Live tooling (README Reproduction, imported by experiments, tested):
- isaac_session.py (489 lines; start/submit/stop/job; argparse matches README), isaac_session_guest.py (guest runner), isaac_g1_session.sh (Isaac Sim 6.2 cold install, sourced by vm_startup.sh), tailnet.py (imported by experiments/isaac_g1_live; test_cloud_tailnet), isaac62.lock.txt + isaac.constraints.txt (Isaac 6.2 pins), semantic.requirements.lock.txt (hash-pinned scipy for H4 via cloud_bootstrap).
- launch.py (1122 lines): older capped single-job GPU launcher; hard-codes the cloud_*.json configs and cloud_* scripts; imported by isaac_session, audit_t1_run, inspect_t1_stand, verify_cloud_recovery, tests/test_cloud*.py. Its own pilot workflows are historical but it is imported by live code.
- vm_startup.sh (148 lines): VM boot script used by launch.py and the Isaac session; live plumbing.
Historical (Oct 2026 pilots): isaac_t1_job.sh (T1 walking job, "candidate independent Isaac environment"), requirements.in + lock (Qwen torch 2.8 stack), molmo.requirements.in + lock, groot.requirements.in + lock (GR00T upstream 944fe70, non-commercial checkpoint). About 5.5k generated lockfile lines for these; results kept in results/{qwen_cloud_pilot_dev,molmo_cloud_*,groot_*}.
Uncertain: build_g1_runtime_image.py + cached_runtime.py (README cites immutable image 4812387390768042548; configs g1_runtime_build_demo.json, cloud_g1_runtime_build.json; imported only by each other and tests/test_cached_runtime). Current reuse unknown.

## 5. tests/ (about 125 files, 120 test_*.py, 1 conftest, fixtures)
Markers: pytest.mark.slow in 20 files (test_compartment 5, test_semantic_search_integration 4, test_panda 3, test_vision 3, rest 1-2 each). pyproject: addopts -q, testpaths tests, one marker. README (not rerun): fast 772 passed 1 skipped, slow 61, isolation 19.
Categories (judgement from imports and content):
- Contract and isolation guards (fast): test_isolation, test_agent_base_isolation, test_verifier_purity, test_closeout_defects, test_wholebody_height_contract, test_volo_contract, test_groot_contract, test_g1_goal_contract, test_semantic_scoring, test_task_scoring.
- Unit tests of dream/ packages (fast, majority, about 85 files): belief, model, joint, panda, kinematic_base, navigation, rpp, semantics*, scene_graph, memory_search, home*, phone_api, scheduler, tools_runtime, vlm, vision, ...
- Slow MuJoCo simulation tests: the marked files above (compartment, semantic_search_integration, panda, vision, session, eval, home, mobile_exploration, g1_*, tools_runtime, vlm, semantic_sensor, home_xl, g1_wbc, cloud 1).
- Tests that cover only experiments/ scripts (no dream code): test_camera_remote_measure, test_cloud_recovery_integrity, test_code_version, test_detector_compare, test_g1_autonomy_report, test_g1_platform_audit, test_isaac_asset_validation, test_isaac_capture_manifest, test_isaac_home_{compose,exclusions,runs,scale,usd}, test_isaac_kitchen_repair, test_isaac_marker_check, test_molmo_multichunk_rule, test_navigation_eval, test_nested_replay, test_openai_plan_{auth,refresh,resume}, test_p6_frozen_replay, test_room_{postfix_config,regression_config,scene_failure}, test_summarise_closeout, test_summarize_vlm_comparison, test_t1_{packaging,recording,residuals}, test_g1_skill_collection, test_g1_snapshot_gate. These would be deleted with their scripts.
- Thin or smoke-like (1-2 tests): test_g1_goal_contract (real contract), test_code_version (env-var path only, 8 lines), test_room_scene_failure (real regression), test_g1_platform_audit (one finite-depth assertion), test_g1_regrasp (mock-heavy state-flag asserts), test_cloud_telemetry (1 test), test_isaac_capture_manifest, test_g1_snapshot_gate. test_room_postfix_config only checks that a committed frozen config declares expected arms/seeds (config integrity, not behaviour). No test function file has zero assertions by regex count (every file has at least 1).
- Skips and why: test_cloud_telemetry (pytest.skip "Bash unavailable"); test_detector_compare (skipif dataset experiments/data/detector_dev absent); test_g1_grasp_bringup, test_g1_posture, test_g1_supervisor (importorskip mink, daqp, onnxruntime; mink and daqp are in no declared extra); test_g1_hands, test_g1_wbc, test_g1_posture (skipif WBC weights not fetched, see experiments/fetch_wbc_policy.py); test_g1_posture, test_g1_supervisor (results/g1_wbc_envelope/envelope.json absent); test_intent (frozen Qwen evidence absent); test_volo_contract (importorskip vlm_orchestrator, external). On a clean install of the declared extras the posture/supervisor/grasp stack is skipped; whether that is the README "1 skipped" is unverified.
- The Isaac tests (test_isaac_*, about 15 files) run on CPU via mocks; isaacsim/omni/pxr are absent from the core env, so they test the pure-python parts only.

## 6. Dependencies
pyproject.toml: requires-python >=3.10. Core: mujoco==3.16.0, numpy>=1.26, matplotlib>=3.8, scipy>=1.11. Extras: dev (pytest>=8, ruff>=0.6), demo (pillow==13.0.0), dataset (pyarrow==22.0.0), loader (pandas==2.3.3, pyarrow==22.0.0, torch==2.9.0, torchcodec==0.8.1), locomotion (onnxruntime==1.23.2). No root requirements or lock file; locks exist only in cloud/. Packages found: dream* only.
Declared vs imported:
- mujoco, numpy, matplotlib (dream/eval/plots.py), scipy (dream/semantics, belief, rooms): imported. pytest imported by tests; ruff is a tool.
- pillow (extra demo): imported at module top in dream/runtime/camera_stream.py (also lazily in box_detectors, molmo, qwen, openai_plan; tests test_camera_stream, test_box_detectors). Optional extra but needed for core runtime import of camera_stream.
- pyarrow: experiments/g1_dataset.py, resolve_g1_contract.py. pandas: only experiments/g1_validate_loader.py. onnxruntime: lazy in dream/robot/g1_wbc.py, t1.py, experiments/g1_wbc_envelope.py, test_g1_hands. All used.
- torch: imported by experiments (cloud_*, g1_validate_loader) and lazily by dream (box_detectors, molmo, qwen); the loader pin (2.9.0) conflicts with cloud qwen pin 2.8.0 (pyproject comment admits "incompatible").
- torchcodec: declared but never imported by name; only mentioned in g1_validate_loader docstring (needed implicitly by the NVIDIA loader).
Imported but NOT declared in pyproject:
- mink (dream/robot/g1_posture.py top-level import, which g1_correction, g1_rollout, g1_supervisor and g1_grasp_bringup import) and daqp (tests importorskip): a clean install cannot import dream.robot.g1_posture.
- requests, truststore (dream/runtime/siwc_auth.py, lazy), jwt (experiments/openai_plan_login.py).
- spark_dsg (dream/memory/scene_graph.py, lazy _dsg()); test handling unverified.
- huggingface_hub, transformers, accelerate (cloud scripts; dream box_detectors/molmo/qwen lazily): only in cloud/*.requirements.in.
- isaacsim, omni, pxr, carb, mujoco_usd_converter: Isaac environment (cloud/isaac62.lock.txt, isaac.constraints.txt).
- gr00t, rosbags, websockets, packaging (cloud/cached_runtime.py), vlm_orchestrator (external VoLoAgent): undeclared.
AGENTS.md section 9 says pin versions in pyproject.toml; mink, daqp, requests, truststore, spark_dsg, rosbags, websockets, jwt are used and not pinned there.

## 7. README "Reproduction" commands vs the tree
Checked against argparse; nothing was run.
- pip install -e ".[dev,demo,locomotion]": extras exist (mink/daqp missing, see 6). pytest isolation/fast/slow: tests/test_isolation.py and marker exist. web/package.json and android/gradlew exist (npm/gradle commands not run).
- cloud/isaac_session.py start/submit/stop: options (--project, --allowance-usd, --max-minutes, --yes, --manifest, --env isaac|dream, --timeout-s) all exist. OK.
- submit ... experiments/isaac_g1_live_eval.py --out "{out}/run" --budget-s 9300: both args exist. OK.
- submit ... experiments/g1_explore_room.py experiments/configs/g1_explore_room_postfix_closeout.json --arm R3 --renderer osmesa --workers 7 --out ...: config exists (frozen, arms R1,R2_fresh,R2_hist,R3); --arm, --renderer (egl|osmesa|glfw), --workers, --out exist. OK.
- summarise_room_postfix.py / summarise_isaac_postfix.py: exist; the first takes an optional positional root via sys.argv (no argparse; README gives no args, default results/). "Each inventory item has its own command" refers to results/research_closeout_20261106/inventory.json, which exists.
- semantic_search_eval.py (H1 main, H1 ablations, H2, H3): args (config, --split, --arms, --base, --workers, --out, --limit) exist; all 12 arm names used exist in semantic_search_heldout.json; splits H1-H4 defined. OK.
- summarise_semantic_search.py (dirs, --reference, --budget, --out), semantic_search_gates.py (root, --budget, --out), compare_room_knowledge.py (--qwen, --out; results/qwen_room_priors/priors.json exists): OK.
- H4 via experiments/cloud_bootstrap.py + cloud/semantic.requirements.lock.txt: both exist. OK.
- V1: semantic_search_verifier_v2.json has splits V1, dev_smoke, arms proposed, proposed_v2; verifier_v2_gates.py args (dir, --config, --candidate, --reference, --margin, --out) exist. OK.
- diagnose_search_episode.py: --split --scenario --target --seed --code-root --focus-tracks --out all exist; output H2_false_claim_diagnosis.json is committed. OK.
- Unverifiable offline: checkout of commits 9c19328, 1e11889, 6d680b4 (git cat-file not run; README says results use pre-rewrite IDs, map in results/commit_map_20261109.json); image id 4812387390768042548; project nerva-adityapersonal.
- Stale: "Final verification, on this branch at 6d680b4: 772 passed..." describes another tree, so the counts are not evidence for cf2851d; "Android unchanged since 7 November, not rerun".
- Gaps (cited results with no reproduction command): R1 grasp dev (results/g1_r1_*), camera delivery, Panda/tabletop (run_benchmark, run_compartment, run_tasks, run_vision), Molmo/GR00T/T1 pilots, WBC envelope, compartment closeouts, navigation/memory/interruption demos.
- No Reproduction command points at a missing script, config or argument.
