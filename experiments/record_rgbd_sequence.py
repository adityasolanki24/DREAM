"""Record the walking G1's own sensor stream in TUM RGB-D format, for SLAM backends.

    python experiments/record_rgbd_sequence.py --seed 0 --out <dir outside Git> [--hz 5]

The robot runs the Milestone B search (rpp navigator) in a procedural room and then walks
back to its start, so the sequence revisits its first view (a loop-closure opportunity).
Written while it runs:
- rgb/<t>.png, depth/<t>.png (uint16, depth x 5000, 0 = no return), rgb.txt, depth.txt,
  associations.txt: the head RGB-D camera exactly as perception receives it (noise included);
- imu.txt: pelvis gyro (rad/s, body frame) and accelerometer (m/s^2, specific force) at 50 Hz;
- odometry.txt: the ORACLE-ASSISTED odometry DREAM navigates with today (baseline);
- calibration.yaml: pinhole intrinsics (fx = fy, cx, cy) and the camera-to-pelvis transform;
- groundtruth.txt: the true optical-frame camera pose, TUM order. EVALUATION ONLY: no SLAM
  backend or DREAM component may read it.
Optical frame: x right, y down, z forward (MuJoCo cameras look along -z with y up).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.search import RoomSearchPlanner  # noqa: E402
from dream.sim.g1_mobile import HEAD_CAMERA, G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.goals import Inspect  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.g1_explore_room import Budgeted  # noqa: E402

TO_OPTICAL = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera axes -> optical axes
DEPTH_SCALE = 5000.0  # TUM convention


def quat_xyzw(r: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r).ravel())
    return np.array([q[1], q[2], q[3], q[0]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--budget", type=float, default=120.0)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(ROOT.resolve()):
        raise SystemExit("sequences stay outside Git")
    from PIL import Image

    for d in ("rgb", "depth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    room = sample_room(np.random.default_rng([args.seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([args.seed, 1])
    )
    env.reset(*room.start)
    cam = env._camera  # the robot's own head camera driver (renders sensor arrays only)
    every = max(1, round(1.0 / (args.hz * 0.02)))
    files = {k: open(out / f"{k}.txt", "w") for k in
             ("rgb", "depth", "associations", "groundtruth", "groundtruth_base", "imu",
              "odometry", "camera_in_base")}  # fmt: skip
    for k in ("rgb", "depth"):
        files[k].write(f"# {k} images: timestamp filename\n")
    files["groundtruth"].write("# EVALUATION ONLY: timestamp tx ty tz qx qy qz qw\n")
    files["groundtruth_base"].write("# EVALUATION ONLY, pelvis: timestamp tx ty tz qx qy qz qw\n")
    files["camera_in_base"].write(
        "# camera (optical) in pelvis, encoders: t tx ty tz qx qy qz qw\n"
    )
    files["imu"].write("# timestamp gx gy gz ax ay az (pelvis frame)\n")
    files["odometry"].write("# ORACLE-ASSISTED planar odometry: timestamp x y yaw\n")
    tick = {"n": 0}
    import mujoco

    # The MJCF's pelvis IMU. MuJoCo does not apply a sensor's noise attribute, so the
    # documented standard deviations (gyro 5e-4 rad/s, accelerometer 1e-2 m/s^2) are added.
    gyro_s, acc_s = (env.model.sensor(n) for n in ("imu-pelvis-angular-velocity",
                                                    "imu-pelvis-linear-acceleration"))  # fmt: skip
    imu_rng = np.random.default_rng([args.seed, 9])

    def on_tick():
        tick["n"] += 1
        t = env.time
        d = env.data
        gyro = d.sensordata[gyro_s.adr[0] : gyro_s.adr[0] + 3] + imu_rng.normal(0, 5e-4, 3)
        acc = d.sensordata[acc_s.adr[0] : acc_s.adr[0] + 3] + imu_rng.normal(0, 1e-2, 3)
        files["imu"].write(f"{t:.4f} " + " ".join(f"{x:.6f}" for x in (*gyro, *acc)) + "\n")
        o = env.odometry()
        files["odometry"].write(f"{t:.4f} {o[0]:.5f} {o[1]:.5f} {o[2]:.5f}\n")
        if tick["n"] % every:
            return
        frame = cam.capture(d)  # world-frame calibration; used for ground truth only
        name = f"{t:.4f}.png"
        Image.fromarray(frame.rgb).save(out / "rgb" / name)
        depth = np.nan_to_num(frame.depth, nan=0.0) * DEPTH_SCALE
        Image.fromarray(np.clip(depth, 0, 65535).astype(np.uint16)).save(out / "depth" / name)
        files["rgb"].write(f"{t:.4f} rgb/{name}\n")
        files["depth"].write(f"{t:.4f} depth/{name}\n")
        files["associations"].write(f"{t:.4f} rgb/{name} {t:.4f} depth/{name}\n")
        c = frame.calibration
        rot = c.rotation @ TO_OPTICAL
        q = quat_xyzw(rot)
        files["groundtruth"].write(
            f"{t:.4f} " + " ".join(f"{x:.6f}" for x in (*c.position, *q)) + "\n"
        )
        rp, pp = d.xmat[env.pelvis].reshape(3, 3), d.xpos[env.pelvis]
        files["groundtruth_base"].write(
            f"{t:.4f} " + " ".join(f"{x:.6f}" for x in (*pp, *quat_xyzw(rp))) + "\n"
        )
        # Robot kinematics (encoders), not ground truth: the camera moves with the waist.
        rc, pc = rp.T @ rot, rp.T @ (c.position - pp)
        files["camera_in_base"].write(
            f"{t:.4f} " + " ".join(f"{x:.6f}" for x in (*pc, *quat_xyzw(rc))) + "\n"
        )

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([args.seed, 2])), navigator="rpp")
    planner = Budgeted(RoomSearchPlanner("gain"), base, args.budget)
    ep = TaskAgent(tb, planner, max_steps=40).run(Inspect("cabinet"))
    home = tb.call("go_to", {"x": 0.0, "y": 0.0, "heading": 0.0})  # revisit the first view
    for f in files.values():
        f.close()
    d = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, d)
    world_cam = d.cam(HEAD_CAMERA.name)
    f = 0.5 * HEAD_CAMERA.height / np.tan(np.radians(HEAD_CAMERA.fovy_deg) / 2)
    pelvis = d.body("pelvis")
    rp = pelvis.xmat.reshape(3, 3)
    cam_in_pelvis_r = rp.T @ world_cam.xmat.reshape(3, 3) @ TO_OPTICAL
    cam_in_pelvis_p = rp.T @ (world_cam.xpos - pelvis.xpos)
    calib = {
        "width": HEAD_CAMERA.width, "height": HEAD_CAMERA.height, "fx": f, "fy": f,
        "cx": (HEAD_CAMERA.width - 1) / 2, "cy": (HEAD_CAMERA.height - 1) / 2,
        "depth_scale": DEPTH_SCALE, "max_range_m": 3.0,
        "camera_in_pelvis_at_reset": {"rotation": cam_in_pelvis_r.tolist(),
                                      "translation": cam_in_pelvis_p.tolist(),
                                      "note": "waist joints move it slightly while walking"},
        "imu": "pelvis gyro + accelerometer, 50 Hz", "mount": HEAD_CAMERA.mount_body,
        "episode": {"claimed": ep.success_claimed, "reason": ep.reason,
                    "return_home": home.status, "sim_s": env.time, "seed": args.seed},
    }  # fmt: skip
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))
    (out / "calibration.yaml").write_text(
        "%YAML:1.0\n---\n"
        f"camera_name: dream_g1_head\nimage_width: {calib['width']}\n"
        f"image_height: {calib['height']}\ncamera_matrix:\n  rows: 3\n  cols: 3\n"
        f"  data: [{f}, 0.0, {calib['cx']}, 0.0, {f}, {calib['cy']}, 0.0, 0.0, 1.0]\n"
        "distortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "distortion_model: plumb_bob\n"
    )
    print(json.dumps(calib["episode"]))


if __name__ == "__main__":
    main()
