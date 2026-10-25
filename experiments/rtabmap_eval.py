"""RTAB-Map on the walking G1's own sensors, scored against ground truth it never sees.

    python experiments/rtabmap_eval.py --seed 3 --out <dir>      # needs Docker (cloud VM)

1. Record the G1 search-and-return sequence (experiments/record_rgbd_sequence.py) in MuJoCo.
2. Convert it to a ROS 2 bag (experiments/tum_to_rosbag.py): RGB-D, camera info, IMU, TF.
3. Replay it through the official rtabmap_ros container (introlab3it/rtabmap_ros, pinned by
   digest at run time) with RGB-D visual odometry and loop closure; export the odometry and
   optimised-graph trajectories with rtabmap-report --poses_raw.
4. Score: absolute trajectory error of the base after a rigid (rotation + translation)
   alignment, for RTAB-Map odometry, RTAB-Map SLAM and DREAM's ORACLE-ASSISTED odometry,
   on the same timestamps; plus how much of the sequence RTAB-Map tracked.
Ground truth (groundtruth_base.txt) is read only in step 4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMAGE = "introlab3it/rtabmap_ros:jazzy-latest"

RUN = r"""
set -e
source /opt/ros/jazzy/setup.bash
cd /data
ros2 launch rtabmap_launch rtabmap.launch.py rtabmap_viz:=false use_sim_time:=true \
  frame_id:=base_link rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_raw camera_info_topic:=/camera/color/camera_info \
  approx_sync:=false visual_odometry:=true database_path:=/data/rtabmap.db \
  rtabmap_args:="--delete_db_on_start" > /data/launch.log 2>&1 &
LAUNCH=$!
sleep 15
ros2 bag play /data/bag --clock 100 > /data/play.log 2>&1
sleep 10
kill -INT $LAUNCH || true
wait $LAUNCH || true
rtabmap-report --poses_raw /data/rtabmap.db > /data/report.log 2>&1 || true
ls -la /data > /data/listing.txt
"""


def load_tum(path: Path) -> dict[float, np.ndarray]:
    out = {}
    for ln in path.read_text().splitlines():
        if ln and not ln.startswith("#"):
            v = [float(x) for x in ln.split()]
            out[round(v[0], 3)] = np.array(v[1:4])
    return out


def ate(est: dict, gt: dict, offset: float = 0.0) -> dict | None:
    """RMSE of positions after the best rigid alignment (Umeyama without scale)."""
    keys = [k for k in est if round(k - offset, 3) in gt]
    if len(keys) < 10:
        return None
    a = np.array([est[k] for k in keys])
    b = np.array([gt[round(k - offset, 3)] for k in keys])
    ma, mb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b - mb).T @ (a - ma))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    err = np.linalg.norm((a - ma) @ r.T + mb - b, axis=1)
    return {"n": len(keys), "rmse_m": float(np.sqrt((err**2).mean())), "max_m": float(err.max()),
            "path_m": float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum())}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    seq, data = out / "sequence", out / "rtabmap"
    data.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    t0 = time.time()
    subprocess.run([py, str(ROOT / "experiments/record_rgbd_sequence.py"), "--seed",
                    str(args.seed), "--out", str(seq)], check=True)  # fmt: skip
    try:
        import rosbags  # noqa: F401
    except ImportError:  # sessions started before it joined the DREAM environment
        subprocess.run(["uv", "pip", "install", "--python", py, "rosbags==0.11.9"], check=True)
    from experiments.tum_to_rosbag import STAMP_OFFSET, convert

    counts = convert(seq, data / "bag")
    if shutil.which("docker") is None:  # the GCP image has none (session 4); ephemeral root VM
        subprocess.run(["apt-get", "install", "-y", "-q", "docker.io"], check=True)
        subprocess.run(["systemctl", "start", "docker"], check=True)
    subprocess.run(["docker", "pull", IMAGE], check=True)
    digest = subprocess.run(["docker", "inspect", "--format", "{{index .RepoDigests 0}}", IMAGE],
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    (data / "run.sh").write_text(RUN)
    t1 = time.time()
    command = ["docker", "run", "--rm", "-v", f"{data}:/data", IMAGE, "bash", "/data/run.sh"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    (data / "docker.log").write_text(proc.stdout + proc.stderr)
    gt = load_tum(seq / "groundtruth_base.txt")
    result = {"seed": args.seed, "image": digest, "bag_counts": counts,
              "docker_exit": proc.returncode, "rtabmap_wall_s": time.time() - t1,
              "total_wall_s": time.time() - t0}  # fmt: skip
    for name in ("odom", "slam"):
        f = data / f"rtabmap_{name}.txt"
        est = load_tum(f) if f.is_file() else {}
        # rtabmap stamps carry the bag's +1 s offset
        result[f"rtabmap_{name}"] = ate(est, gt, STAMP_OFFSET) if est else None
        result[f"rtabmap_{name}_poses"] = len(est)
    odo = {}
    for ln in (seq / "odometry.txt").read_text().splitlines():
        if ln and not ln.startswith("#"):
            t, x, y, _ = map(float, ln.split())
            odo[round(t, 3)] = np.array([x, y, 0.0])
    gt2 = {k: np.array([v[0], v[1], 0.0]) for k, v in gt.items()}
    result["oracle_assisted_odometry_xy"] = ate({k: v for k, v in odo.items() if k in gt2}, gt2)
    result["frames"] = len(gt)
    (out / "rtabmap_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if os.environ.get("DREAM_STRICT") and proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
