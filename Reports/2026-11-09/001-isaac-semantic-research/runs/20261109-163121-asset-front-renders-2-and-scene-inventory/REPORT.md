# Run report: Front-face renders of six more assets and ready-made scene inventory (cloud session 2, job 3)

- Run ID: 2026-11-09-001/20261109-163121-asset-front-renders-2-and-scene-inventory
- Parent task: 2026-11-09-001
- Started: 2026-11-09T16:31:21+11:00
- Ended: 2026-11-09T18:59:06+11:00
- Git commit: 515364e1620ce367f84ea94f77db3f7bc8f429b2
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (714054c432ed93c87ef3ccc517f53314a101e64105dd7d3e023928c86991a353)
- Seeds: n/a
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_asset_validation.py --indexes armchair:0,toilet:0,washbasin:0,workbench:0,washing_machine:0,filing_cabinet:0 --environments --budget-s 600`
- Exit status: 124
- Completed (yes/no/partial): no (timed_out)
- Cases attempted: 7
- Cases completed: 7

## Purpose

Render the remaining chosen assets of the front-face reading and inventory four ready-made scenes (Office, Hospital, Simple_Warehouse, Simple_Room). Submitted behind the crashed kitchen job.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Timed out (exit 124 at the 1500 s job limit). Six assets rendered (armchair, toilet, washbasin, workbench, washing_machine, filing_cabinet). Scene inventory: only Environments/Office/office.usd was inventoried: it resolves, opens with 0 unresolved dependencies, has 4664 prims, 140 meshes and 1 collider, carries no semantic labels (neither the primvar nor SemanticsAPI), and took 851 s to open. Hospital, Simple_Warehouse and Simple_Room were not reached. So no ready-made scene is established as a labelled out-of-distribution environment.

## Regressions observed

None.

## Interruptions and failures

Job limit reached while opening the Office scene; the remaining scenes were not inventoried.

## Evidence

results/isaac_env_validation/environments.json (Office only); renders not committed.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/3/out/val (dir, sha256 38c3c4697fe7849626fa9ebbcf9efe9b3dc1e4363cb9652f54303a5c3f200932)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-160004-5f4423
- job_ids: 3
- max_lifetime_min: 170
- estimated_runtime_min: 15
- actual_runtime_min: 25
- estimated_cost_usd: 7.4
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-candidate / per-item JSON and frames written as produced
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/<n>/out (outside Git and OneDrive); compact copies in results/isaac_kitchen_repair, results/isaac_env_validation, results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified independently after stop: no DREAM VM, disk, address, router, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
