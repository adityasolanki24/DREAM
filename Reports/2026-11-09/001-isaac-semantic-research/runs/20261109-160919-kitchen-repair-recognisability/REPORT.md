# Run report: NVIDIA kitchens with the refrigerator reference repaired: load check, recognisability and l_shape capture (cloud session 2, job 2)

- Run ID: 2026-11-09-001/20261109-160919-kitchen-repair-recognisability
- Parent task: 2026-11-09-001
- Started: 2026-11-09T16:09:19+11:00
- Ended: 2026-11-09T18:58:55+11:00
- Git commit: da8498c8fb8e68c18e7ec6f2b06970ae24ffa343
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_home_dev.json (c0cb18470632dd973475c47562c0fec8fe7b71b483c5a75ccc44423d84702064)
- Seeds: kitchen start seed 1 (development kitchen l_shape; others load and recognisability only)
- Model and checkpoint: Qwen/Qwen3-VL-4B-Instruct ebb281ec
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: Unitree G1
- Robot controller: NVIDIA Decoupled WBC
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_home_runs.py --no-homes --kitchens l_shape,g_shape,l_island,peninsula,u_shape --kitchen-repair --budget-s 2400`
- Exit status: 139
- Completed (yes/no/partial): no (crashed)
- Cases attempted: 3
- Cases completed: 2

## Purpose

Load NVIDIA's replicator kitchens with the broken refrigerator reference remapped (experiments/isaac_kitchen_repair.py), check the G1 stands, and measure Qwen3-VL recognisability of kitchen classes from reset poses 1.5 to 3 m away; capture frames in the development kitchen l_shape. Held-out kitchens: load and recognisability only, no exploration.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Crashed (exit 139) after two of the five kitchens. REPAIR WORKS FOR THE REFRIGERATOR: in l_shape and g_shape the remapped refrigerator has a finite bound (about 0.67 x 1.23 x 1.52 m) and 13000 to 31000 segmentation pixels in the views, the G1 stands in both, and no PhysX crash occurred. The oven reference has no asset to remap to (no oven exists under SimReady/Residential/Kitchen/Appliances) and stays unrepaired, so the stove is absent in these kitchens. l_island segfaulted in PhysX at reset (after robot_contract_checked, before world_reset) exactly as in the frozen Isaac evaluation, so it did not load; peninsula and u_shape were not reached. Recognisability on the repaired kitchens (3 views per class, development evidence): l_shape refrigerator 0 of 3 views with a same-class detection, kitchen_cabinet 0 of 1, sink 0 of 2 (countertop views had too few visible pixels to score); g_shape countertop 2 of 3 detections (2 at IoU 0.5), kitchen_cabinet 3 of 3 detections (none at IoU 0.5). Exploration in l_shape ended at once (the first observation had no frontier), so there is almost no capture from this job (1 frame).

## Regressions observed

None.

## Interruptions and failures

Process killed by a segfault inside the PhysX plugin while loading l_island. Partial results of l_shape and g_shape kept. Fixed afterwards: look around before exploring.

## Evidence

Compact copies: results/isaac_kitchen_repair/job2_kitchen_*_repaired.json; capture manifest results/isaac_perception_capture/manifest_job2_kitchens.json.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/2/out/runs (dir, sha256 a82a0d1601ae52dd35daa83c88f9306e0892a9deea3334e0695740d7138c44c7)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-160004-5f4423
- job_ids: 2
- max_lifetime_min: 170
- estimated_runtime_min: 45
- actual_runtime_min: 12
- estimated_cost_usd: 7.4
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-candidate / per-item JSON and frames written as produced
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/<n>/out (outside Git and OneDrive); compact copies in results/isaac_kitchen_repair, results/isaac_env_validation, results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified independently after stop: no DREAM VM, disk, address, router, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
