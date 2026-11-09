# Run report: Session 4 job 2: navigation across rooms and perception capture in the four dev homes (320x240 and 640x480)

- Run ID: 2026-11-09-001/20261109-235333-s4-home-capture
- Parent task: 2026-11-09-001
- Started: 2026-11-09T23:53:33+11:00
- Ended: 2026-11-10T04:44:57+11:00
- Git commit: 3f9963b73cd3405c47cdfcce8f961804153bfe21 (dirty)
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_home_dev.json (1ffe0bc74057d081935aee1ad39788ead3c9fec427972cf4c859f033f8f792f9)
- Seeds: home generator seeds 100 B, 101 D, 102 E, 103 F (development)
- Model and checkpoint: Qwen/Qwen3-VL-4B-Instruct ebb281ec (recognisability only)
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: Unitree G1
- Robot controller: NVIDIA Decoupled WBC
- Hardware: GCP g2-standard-8 L4
- Command: `isaac_home_runs.py --homes 100,101,102,103 --budget-s 3700`
- Exit status: 0
- Completed (yes/no/partial): partial (partial)
- Cases attempted: 4
- Cases completed: 1

## Purpose

Navigation across rooms and perception capture (320x240 and 640x480) in the four dev homes.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Partial: 1 of 4 homes ran, and it settles the cause of the failures. The three homes whose placed assets carried PhysicsScene prims (100 B: 3 extra scenes, 102 E: 3, 103 F: 3) all failed to stand (pelvis tilt 118, 113 and 51 degrees; 102 E and 103 F show 0.25 m or NaN height). Home 101 D sofa, whose assets carry none (only /physicsScene on the stage), stood, explored with the existing mobile tools (rpp navigator, frontier go_to) for 10.2 m, visited rooms 0 and 1 (true rooms; two rooms), made 5 go_to calls (all succeeded), did not fall, and mapped 37 m2. Capture: 67 frames at 320x240 and 67 at 640x480 from identical poses (71.4 MB), each with odometry, true pose and camera, instance-segmentation ground truth with instance ids and classes at both resolutions (2 visible placed instances in the first frame). Recognisability from 3 views per class (320x240, development evidence): toilet 3/3 detections (2 at IoU 0.5), coffee_table 2/2, bed 1/1, desk 1/1, bookshelf 1/3, sofa 0/3, nightstand 0/3, washbasin 0/3, wardrobe 0/1. The causal claim rests on 4 homes (3 with extra scenes failed, 1 without stood): strong but not an intervention; the deactivation fix is committed (8486603) and offline-tested but has not run in Isaac.

## Regressions observed

None.

## Interruptions and failures

Three of four homes ended at the standing check. Driver notes: the four jobs were submitted back to back by a script whose submit processes for jobs 3 and 4 gave up waiting (their wait deadline counts from submission, the jobs queued behind jobs 1 and 2), so jobs 3 and 4 have no per-job output folder; the jobs themselves ran (exit 0) and their outputs were retrieved by `stop` into the session's out/jobs/3 and out/jobs/4. An earlier version of the driver issued `stop` by mistake 5 minutes after the start (a VM-presence check returned empty); I killed it and removed the STOP marker before the guest started any job, so no job was lost.

## Evidence

results/isaac_home_dev/s4_job2_*.json; capture manifest results/isaac_perception_capture/manifest_s4_home_101D.json.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/2/runs (dir, sha256 b821f18fef25c76b22e0cd1d885bdfa63efe397f43f4dae2b420e3d91124c5c4)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (zone a)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-235410-1d7369
- job_ids: 2
- max_lifetime_min: 140
- estimated_runtime_min: 62
- actual_runtime_min: 12
- estimated_cost_usd: 4.79
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-item JSON and per-frame files written as produced
- evidence_location: %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/<n> (outside Git and OneDrive); compact copies in results/isaac_home_dev, results/isaac_env_validation/closeup, manifests in results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified by `stop` before bucket removal
- cleanup_status: verified by the coordinator: no DREAM VM, disk, bucket, network, router or service account; cleanup_errors empty
- resources_remaining: runtime image only
