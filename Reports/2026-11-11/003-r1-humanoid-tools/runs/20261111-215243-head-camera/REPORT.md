# Run report: Capture the humanoid head camera before and after supervised walking

- Run ID: 2026-11-11-003/20261111-215243-head-camera
- Parent task: 2026-11-11-003
- Started: 2026-11-11T21:52:43+11:00
- Ended: 2026-11-27T03:14:19+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: No config file. Seed and setup are explicit in the checksummed run.py.
- Seeds: 43
- Model and checkpoint: Cached NVIDIA Decoupled WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: MuJoCo native OpenGL
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: NVIDIA Decoupled WBC, cached ONNX
- Hardware: Local Windows CPU and local OpenGL
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-camera-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Verify RGB-D capture through the humanoid toolbox on the same free-standing physics owner before and after walking.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Captured 240 by 320 RGB and depth frames. Finite depth fraction was 0.6845182292. Camera calibration translated 0.1156231429 m during 1.2 s of supervised walking.

## Regressions observed

No camera baseline comparison was measured. The requested capture contract passed.

## Interruptions and failures

The saved camera result and prior process exit establish completion. Administrative closeout occurred at resumption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-camera-001 (dir, sha256 163ff6fd88ab00634b16dbc08c8d9e5886a1c77eb40a8b81a4fc72a5434506a1)
