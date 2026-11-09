# Run report: Kitchens re-run with the non-finite-collider repair: l_shape capture, l_island load test, peninsula, u_shape recognisability (cloud session 2, job 4)

- Run ID: 2026-11-09-001/20261109-164552-kitchen-repair-rerun
- Parent task: 2026-11-09-001
- Started: 2026-11-09T16:45:52+11:00
- Ended: 2026-11-09T18:59:06+11:00
- Git commit: 32ad5a70e4aa50f399aeb490b0b196c3b184336e
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_home_dev.json (c0cb18470632dd973475c47562c0fec8fe7b71b483c5a75ccc44423d84702064)
- Seeds: kitchen start seed 1 (development kitchen l_shape; others load and recognisability only)
- Model and checkpoint: Qwen/Qwen3-VL-4B-Instruct ebb281ec
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: Unitree G1
- Robot controller: NVIDIA Decoupled WBC
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_home_runs.py --no-homes --kitchens l_shape,l_island,peninsula,u_shape --kitchen-repair --budget-s 1900`
- Exit status: 139
- Completed (yes/no/partial): no (crashed)
- Cases attempted: 2
- Cases completed: 1

## Purpose

Re-run the kitchens with the extra repair (collider and rigid body switched off on prims with non-finite bounds) and with a look-around before exploring: l_shape capture, l_island load test, peninsula and u_shape recognisability.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Crashed (exit 139) at l_island, after l_shape completed. l_shape: stands, explored 14.3 m2 over 4.8 m of path in two go_to calls (both succeeded), 39 frames captured with RGB, depth, intrinsics, odometry, true camera pose and instance segmentation ground truth, Qwen3-VL run at the two stops (2 calls); recognisability refrigerator 1 of 3 views with a same-class detection (1 at IoU 0.5), kitchen_cabinet 1 of 1 (none at IoU 0.5), sink 0 of 2. The refrigerator result differs from job 2 (0 of 3) with the same code path and poses, so the per-view figures are noisy and each is 3 views. The new repair found no prim with a non-finite bound that it could switch off (empty list), and l_island still segfaulted in PhysX at reset: the cause is not the placeholder prims we can see. DOES l_island LOAD AFTER THE REPAIR: NO. peninsula and u_shape were not run.

## Regressions observed

None.

## Interruptions and failures

Segfault in the PhysX plugin loading l_island (same stage as before: after robot_contract_checked, before world_reset), with Illegal BroadPhaseUpdateData non-finite bounds errors in the log (164 lines).

## Evidence

results/isaac_kitchen_repair/job4_kitchen_l_shape_repaired.json, l_island_crash_excerpt.txt; capture manifest results/isaac_perception_capture/manifest_job4_l_shape.json (39 frames, 9.9 MB raw outside Git).

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/4/out/runs (dir, sha256 f6c3c494bca98d3befb8184dc0d954bb348619693cbcf246eeefe87428800be7)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-160004-5f4423
- job_ids: 4
- max_lifetime_min: 170
- estimated_runtime_min: 30
- actual_runtime_min: 13
- estimated_cost_usd: 7.4
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-candidate / per-item JSON and frames written as produced
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/<n>/out (outside Git and OneDrive); compact copies in results/isaac_kitchen_repair, results/isaac_env_validation, results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified independently after stop: no DREAM VM, disk, address, router, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
