# Run report: Isaac asset validation re-run after the un-instancing fix, plus new classes (cloud session 1, job 3; registered after it ran)

- Run ID: 2026-11-09-001/20261109-153008-asset-validation-recheck
- Parent task: 2026-11-09-001
- Started: 2026-11-09T15:30:08+11:00
- Ended: 2026-11-09T15:31:55+11:00
- Git commit: 4d17106b367c434dc6f451ae1978e8cb666929df (dirty)
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (714054c432ed93c87ef3ccc517f53314a101e64105dd7d3e023928c86991a353)
- Seeds: n/a (asset validation)
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_asset_validation.py --stop-after-pass --budget-s 780 --classes armchair,toilet,... (all 29 classes)`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 29
- Cases completed: 29

## Purpose

Re-run the validator after the un-instancing fix, on the classes job 2 did not reach (armchair onward) and on those it had failed for the instancing defect, stopping each class at its first validated candidate. Registered after the job ran (it was submitted without a separate run record; reported here rather than hidden).

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Completed: 29 candidates attempted, 29 recorded, in about 13 minutes. Validated (first passing candidate per class): armchair, toilet, washbasin, shower, workbench, storage_shelf, washing_machine, filing_cabinet, dishwasher, kettle, tv, plant, refrigerator (Residential Refrigerator_A01, the file the NVIDIA kitchens referenced under a wrong name), countertop, sink, kitchen_cabinet, dining_table, bed, wardrobe, nightstand, desk, office_chair, bookshelf, sofa, coffee_table; together with stove, chair and microwave from job 2. Failed: tool_rack (both candidates implausibly sized: 0.58 x 0.94 x 1.04 m and 1.29 x 0.53 x 1.27 m against nominal 0.15 x 1.2 x 1.8), and no candidate exists on the server for bathtub, dryer, chest_freezer and tv_stand. Substitutes, not the real class: bed is a hospital bed, sofa and armchair are hospital furniture (a patient chair), shower is an emergency eyewash-and-shower station, wardrobe is a locker unit, dining_table is an industrial metal table, bookshelf is an industrial shelf. Scene inventory (Environments) still not run. Snapshots of this job reuse the stale-frame behaviour and are not used to read orientations.

## Regressions observed

None.

## Interruptions and failures

None. Process exit 0. Shortfalls: the scene inventory and the real-orientation read remain to do; snapshots have the stale-frame defect fixed after this job.

## Evidence

Job 3 output under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/3/out/val; records in results/isaac_env_validation/assets.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/3/out/val (dir, sha256 46a22f22963818acbf7ebb54f0bceeaa4ca44e878ee44f3c5d39ade1fddcaad3)
- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/3/out/val (dir, sha256 46a22f22963818acbf7ebb54f0bceeaa4ca44e878ee44f3c5d39ade1fddcaad3)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b after a-zone stockout)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-141110-8b500f
- job_ids: 3
- max_lifetime_min: 75
- estimated_runtime_min: 15
- actual_runtime_min: 13
- estimated_cost_usd: 4.97
- billed_cost_usd: not read: no billing data was consulted; the figure above is an estimate, not a charge
- checkpoint_status: per-candidate JSON written before the next candidate started
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/<n>/out (outside Git and OneDrive); compact per-candidate records in results/isaac_env_validation/assets
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified: no DREAM VM, disk, address, router, NAT, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
