# Run report: Check whether lower and steeper grasps support the sphere

- Run ID: 2026-11-10-011/20261110-225750-sphere-wrist-envelope
- Parent task: 2026-11-10-011
- Started: 2026-11-10T22:57:50+11:00
- Ended: 2026-11-10T22:58:13+11:00
- Git commit: c04ffd81e7a2adffeb84f65a47e64afee342467e (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_sphere.json (7765331ea21723f3cf865a23638312077868948214294e28a2c429ecd078ac14)
- Seeds: 43
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Dex3 rehearsal stage
- Robot controller: Force-limited vertical stage
- Hardware: local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-sphere-wrist-envelope-001/screen.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check the hypothesis that existing wrist tilt and height bands excluded a sphere hold. Widen roll to plus or minus 30 degrees, pitch from minus 15 to 60 degrees, and lower the grip target to the object midline where clearance permits.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
119 feasible poses at each of three thumb offsets produced zero verified stage holds.

## Regressions observed

Live performance was not measured.

## Interruptions and failures

The run completed. No candidate retained the sphere under lift in the separate stage.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-sphere-wrist-envelope-001 (dir, sha256 b74300eaebcf89ba6b167ed544dadfde7cae23896740e41da2a530f78f4c593b)
