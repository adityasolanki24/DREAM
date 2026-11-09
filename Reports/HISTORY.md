# Historical record before the Reports system

Verified results only; not reconstructed run logs. Each row names a result directory that the README cites and the outcome the README states for it. Dates come from the directory name or the README ("8-9 November" for the November semantic-search work). Commit IDs are those the README prints; IDs recorded inside `results/` are pre-rewrite IDs (see `results/commit_map_20261109.json`). Nothing here has a task or run record; new work is recorded under the dated directories.

| Date | Topic | Results path | Frozen commit (README) | Outcome |
|---|---|---|---|---|
| 2026-11-01 | Qwen driving the Isaac G1, live | `results/isaac_qwen_live_20261101` | - | Kept as an unsuccessful outcome, not rescored |
| 2026-11-06 | Research closeout (inventory of all evaluations) | `results/research_closeout_20261106` | - | Inventory of the closeout evaluations and the room fix |
| 2026-11-06 | Room search, fresh seeds | `results/g1_explore_room_closeout` | - | Failed frozen gates (15/20, 4/10); kept with `failed_attempts/` |
| 2026-11-06 | Room search, local partial run | `results/g1_explore_room_closeout_local_partial` | - | Local hang after 2/60, cause not established |
| 2026-11-06 | Compartment opening from camera perception | `results/compartment_vision_closeout` | - | Failed (11/20) |
| 2026-11-06 | Compartment out-of-class cases | `results/compartment_out_of_class` | - | Kept as an unsuccessful outcome, not rescored |
| 2026-11-06 | MolmoAct2 multi-chunk analysis | `results/molmo_multichunk_closeout` | - | Did not complete the task (0/1) |
| 2026-11-06/07 | Camera delivery over the remote link | `results/camera_delivery_closeout_20261106`, `results/camera_delivery_cloud_20261107` | - | Loopback and Isaac-VM measurements; 10 fps over the remote link not demonstrated |
| 2026-11-07 | Qwen on the Isaac G1, frozen evaluation | `results/isaac_qwen_closeout_20261107` | - | Failed its gates: goal substitution, false success claims |
| - | Isaac live idle failure | `results/isaac_live_idle_failure` | - | Kept as an unsuccessful outcome, not rescored |
| 2026-11-08/09 | Isaac kitchen scene probe (5 kitchens) | `results/isaac_semantic_dev/scene_probe_5_kitchens.json` | - | Kitchens lack the refrigerator and oven (404 assets) |
| 2026-11-08/09 | Semantic object search, held-out H1-H4 | `results/semantic_search_heldout` | `9c19328` | 83/87 H1 homes found, one false claim in 216 episodes (H2 bathtub, seed 6001); gate G0 failed, G3 not met, G8 passed |
| 2026-11-08/09 | Room knowledge source comparison | `results/room_knowledge_comparison`, `results/qwen_room_priors` | - | ConceptNet never recognised a laundry or storage room; Qwen raw answers agree with the tables in 23 vs 17 of 27 classes |
| 2026-11-08/09 | Isaac held-out re-test of goal consistency | `results/isaac_semantic_eval` | `c4100e2` | Old path substituted the goal in 4 of 6 cases, new path 0/9; gates incomplete (fourth kitchen never loaded) |
| 2026-11-09 | Verifier v2 (label-purity check), fresh seeds | `results/semantic_search_verifier_v2` | `acea9d3` (config), code from `1e11889` | 384 episodes: 0 false claims in 192 against 1 for the frozen rule; 94/96 found against 93; now the verifier default (`6d680b4`) |
| 2026-11-09 | Commit-ID rewrite map | `results/commit_map_20261109.json` | - | Maps each pre-rewrite commit ID to its new one |
