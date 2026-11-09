# Run report: Isaac asset validation: every candidate asset class, plus ready-made scene inventory (cloud session 1, job 2)

- Run ID: 2026-11-09-001/20261109-144157-asset-validation
- Parent task: 2026-11-09-001
- Started: 2026-11-09T14:41:57+11:00
- Ended: 2026-11-09T15:31:46+11:00
- Git commit: ca9024892a3fe6bbebafb64372ebf4856455afd0
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (714054c432ed93c87ef3ccc517f53314a101e64105dd7d3e023928c86991a353)
- Seeds: n/a (asset validation)
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_asset_validation.py --stop-after-pass --budget-s 1100 --environments`
- Exit status: 0
- Completed (yes/no/partial): partial (partial)
- Cases attempted: 38
- Cases completed: 7

## Purpose

Validate every candidate USD asset (about 85 candidates for 33 classes: the 29 generator classes plus dishwasher, kettle, tv, plant) on the Isaac 6.2 asset server and inventory ready-made scenes (Office, Hospital, Simple_Warehouse, Simple_Room), using the validator after its first fix.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Partial. 38 candidates attempted in the 1100 s budget (refrigerator through armchair); the budget stopped the job before the remaining classes and before the scene inventory, which therefore did not run. 6 candidates passed (the ForgeCAD refrigerator, vanity sink, range oven, base cabinet, dining chair and microwave), 31 failed only through the instancing defect described below,; the remaining failures were implausible sizes (listed below). Geometry results were sound: every candidate that was attempted resolved (no 404), composed and had a finite non-empty bound; the only geometry failures were implausible sizes (countertop Palatial D00215, 0.98 x 0.98 x 0.09 m; Storage_Shelf_Basic A01 and C01 as a bookshelf). The physics step failed for every instanced asset (SimReady Residential, Hospital and Industrial) with a USD edit-validation error (opinions cannot be authored inside an instance); the ForgeCAD and Palatial assets, which are not instanced, passed. Those 31 records are a validator defect, not a property of the assets, and were dropped from the committed records and re-run in the next job. Snapshots were rendered but several views repeat the first frame (the camera move had not been rendered).

## Regressions observed

None (new tool).

## Interruptions and failures

Not interrupted; the job exited 0 at its budget. Defects found: instancing (fixed in the next commit: un-instance before overriding), stale snapshot frames (fixed later), scene inventory not run (budget).

## Evidence

Job 2 output under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/2/out/val; non-superseded records in results/isaac_env_validation/assets.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/2/out/val (dir, sha256 9a1ec520ae99127c8816e7efaf2efebfc9fa7a246c9afa9c9f1e1b472b4cd747)
- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/2/out/val (dir, sha256 9a1ec520ae99127c8816e7efaf2efebfc9fa7a246c9afa9c9f1e1b472b4cd747)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b after a-zone stockout)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-141110-8b500f
- job_ids: 2
- max_lifetime_min: 75
- estimated_runtime_min: 30
- actual_runtime_min: 19
- estimated_cost_usd: 4.97
- billed_cost_usd: not read: no billing data was consulted; the figure above is an estimate, not a charge
- checkpoint_status: per-candidate JSON written before the next candidate started
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/<n>/out (outside Git and OneDrive); compact per-candidate records in results/isaac_env_validation/assets
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified: no DREAM VM, disk, address, router, NAT, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
