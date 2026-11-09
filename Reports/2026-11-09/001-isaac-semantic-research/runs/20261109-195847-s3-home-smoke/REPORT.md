# Run report: Session 3 job 2: one composed dev home end to end (home 100 B refrigerator)

- Run ID: 2026-11-09-001/20261109-195847-s3-home-smoke
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:58:47+11:00
- Ended: 2026-11-09T23:48:28+11:00
- Git commit: 5ea383b92323486c30ca7918ee5ba7706d86b182 (dirty)
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_home_dev.json (04951c1d30a6406e5b69e6c9f5ea8414bcd74d2b508a8b355e2c1e5eb4fd36a5); experiments/configs/isaac_assets.json (2c3c301a36f05e4bd30c7b4696652058215faa3815e2f56af9698ab5d481f08d)
- Seeds: home generator seed 100 (development), scenario B, target refrigerator
- Model and checkpoint: Qwen/Qwen3-VL-4B-Instruct ebb281ec (recognisability only)
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: Unitree G1
- Robot controller: NVIDIA Decoupled WBC
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_home_runs.py --homes 100 --budget-s 900`
- Exit status: 0
- Completed (yes/no/partial): no (failed)
- Cases attempted: 1
- Cases completed: 0

## Purpose

Run one composed dev home end to end in Isaac: the stage builds, the G1 spawns and stands, navigation crosses at least two rooms, RGB-D renders.

Run one composed dev home end to end in Isaac: the stage builds, the G1 spawns and stands, navigation crosses at least two rooms, RGB-D renders.

## Results

Failed at the standing check. The stage built and loaded: 284 static overrides applied, 14 of 14 placed instances have a finite semantic bound, no PhysX non-finite-bound error appeared (the shower and tool_rack were omitted by the exclusion list; the refrigerator and a bookshelf were placed at native size). The G1 did not stand: after 3 s the pelvis tilt was 69 degrees and the torso height read 39.8 m, i.e. the robot was thrown upward. The generator start pose is at least 0.69 m from every placed footprint and 0.6 m from every wall (checked on the CPU), so a start inside furniture is not the explanation. Hypotheses, not findings: an interpenetration with a floor-slab edge or the invisible bounding-box colliders at spawn, or a second physics scene. No navigation, no capture, no recognisability.

<!-- Passed and failed cases; quantitative metrics. -->
Failed at the standing check. The stage built and loaded: 284 static overrides applied, 14 of 14 placed instances have a finite semantic bound, no PhysX non-finite-bound error appeared (the shower and tool_rack were omitted by the exclusion list; the refrigerator and a bookshelf were placed at native size). The G1 did not stand: after 3 s the pelvis tilt was 69 degrees and the torso height read 39.8 m, i.e. the robot was thrown upward. The generator start pose is at least 0.69 m from every placed footprint and 0.6 m from every wall (checked on the CPU), so a start inside furniture is not the explanation. Hypotheses, not findings: an interpenetration with a floor-slab edge or the invisible bounding-box colliders at spawn, or a second physics scene. No navigation, no capture, no recognisability.

## Regressions observed

None.

None.

## Interruptions and failures

The item stopped at the standing check by design (the runner returns before exploring). The job exited 0 after 2 minutes. The rest of the session was not used: the VM idled until it self-deleted because the agent was cut off by a usage limit (wasted time, about 2.4 hours of the 175-minute lifetime after the job ended at 09:02Z; the coordinator stopped the session).

The item stopped at the standing check by design (the runner returns before exploring). The job exited 0 after 2 minutes. The rest of the session was not used: the VM idled until it self-deleted because the agent was cut off by a usage limit (wasted time, about 2.4 hours of the 175-minute lifetime after the job ended at 09:02Z; the coordinator stopped the session).

## Evidence

results/isaac_home_dev/nominal_100_B_refrigerator.json and its scene truth.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-192221-327e9a/jobs/2/out/runs (dir, sha256 ec0283efc27d45358a8720a69f83ca510060da9887484277f523bc2fc0c91c92)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone c)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-192221-327e9a
- job_ids: 2
- max_lifetime_min: 175
- estimated_runtime_min: 20
- actual_runtime_min: 2
- estimated_cost_usd: 7.53
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-asset / per-item JSON written as produced
- evidence_location: %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-192221-327e9a/jobs/<n>/out (outside Git and OneDrive); compact copies in results/isaac_env_validation/assets and results/isaac_home_dev
- upload_status: outputs retrieved and SHA-256 verified by `stop` (run by the coordinator) before bucket removal
- cleanup_status: verified independently by the coordinator: no DREAM VM, disk, address, router, network, bucket or service account; runtime image kept
- resources_remaining: runtime image only
