"""Convert a recorded G1 sequence (experiments/record_rgbd_sequence.py) into a ROS 2 bag.

    python experiments/tum_to_rosbag.py <sequence dir> <bag dir>

Pure Python (rosbags 0.11.9, Apache-2.0), so SLAM backends that consume ROS 2 topics
(rtabmap_ros) can replay exactly what the robot sensed. Topics, all stamped with the
simulation time + 1 s (ROS time zero is reserved):
- /camera/color/image_raw   sensor_msgs/Image rgb8, frame camera_color_optical_frame
- /camera/depth/image_raw   sensor_msgs/Image 16UC1 (mm, 0 = no return), same frame
- /camera/color/camera_info sensor_msgs/CameraInfo (pinhole, no distortion)
- /imu/data                 sensor_msgs/Imu (gyro, accelerometer; no orientation), base_link
- /tf_static                base_link -> camera_color_optical_frame at reset, from encoders
Ground-truth files are never read here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STAMP_OFFSET = 1.0  # s
BASE, CAMERA = "base_link", "camera_color_optical_frame"


def _rows(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


def convert(seq: Path, bag: Path) -> dict:
    from PIL import Image
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_JAZZY)
    T = ts.types  # noqa: N806
    cal = json.loads((seq / "calibration.json").read_text())

    def header(t: float, frame: str):
        sec = int(t + STAMP_OFFSET)
        nsec = int(round((t + STAMP_OFFSET - sec) * 1e9))
        stamp = T["builtin_interfaces/msg/Time"](sec=sec, nanosec=min(nsec, 999_999_999))
        return T["std_msgs/msg/Header"](stamp=stamp, frame_id=frame), sec * 10**9 + nsec

    def quat(r: np.ndarray):
        w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
        if w > 1e-6:
            x, y, z = (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (
                r[1, 0] - r[0, 1]) / (4 * w)  # fmt: skip
        else:  # 180-degree rotations; not expected for this mount
            x, y, z = np.sqrt(np.maximum(0, 1 + 2 * np.diag(r) - np.trace(r))) / 2
        return T["geometry_msgs/msg/Quaternion"](x=float(x), y=float(y), z=float(z), w=float(w))

    counts = {"rgb": 0, "depth": 0, "imu": 0}
    with Writer(bag, version=9) as w:
        c_rgb = w.add_connection("/camera/color/image_raw", "sensor_msgs/msg/Image", typestore=ts)
        c_dep = w.add_connection("/camera/depth/image_raw", "sensor_msgs/msg/Image", typestore=ts)
        c_inf = w.add_connection("/camera/color/camera_info", "sensor_msgs/msg/CameraInfo",
                                 typestore=ts)  # fmt: skip
        c_imu = w.add_connection("/imu/data", "sensor_msgs/msg/Imu", typestore=ts)
        c_tf = w.add_connection("/tf_static", "tf2_msgs/msg/TFMessage", typestore=ts)
        ext = cal["camera_in_pelvis_at_reset"]
        hdr, ns = header(0.0, BASE)
        tf = T["geometry_msgs/msg/TransformStamped"](
            header=hdr, child_frame_id=CAMERA,
            transform=T["geometry_msgs/msg/Transform"](
                translation=T["geometry_msgs/msg/Vector3"](*map(float, ext["translation"])),
                rotation=quat(np.asarray(ext["rotation"]))))  # fmt: skip
        msg = T["tf2_msgs/msg/TFMessage"](transforms=[tf])
        w.write(c_tf, ns, ts.serialize_cdr(msg, "tf2_msgs/msg/TFMessage"))
        k = [cal["fx"], 0.0, cal["cx"], 0.0, cal["fy"], cal["cy"], 0.0, 0.0, 1.0]
        p = [cal["fx"], 0.0, cal["cx"], 0.0, 0.0, cal["fy"], cal["cy"], 0.0, 0.0, 0.0, 1.0, 0.0]
        for t_rgb, f_rgb, _, f_dep in _rows(seq / "associations.txt"):
            t = float(t_rgb)
            rgb = np.asarray(Image.open(seq / f_rgb).convert("RGB"), np.uint8)
            dep = np.asarray(Image.open(seq / f_dep), np.float64) / cal["depth_scale"]
            dep_mm = np.clip(np.round(dep * 1000), 0, 65535).astype(np.uint16)
            h, wd = rgb.shape[:2]
            hdr, ns = header(t, CAMERA)
            img = T["sensor_msgs/msg/Image"](header=hdr, height=h, width=wd, encoding="rgb8",
                                             is_bigendian=0, step=wd * 3,
                                             data=rgb.reshape(-1))  # fmt: skip
            w.write(c_rgb, ns, ts.serialize_cdr(img, "sensor_msgs/msg/Image"))
            dimg = T["sensor_msgs/msg/Image"](header=hdr, height=h, width=wd, encoding="16UC1",
                                              is_bigendian=0, step=wd * 2,
                                              data=dep_mm.view(np.uint8).reshape(-1))  # fmt: skip
            w.write(c_dep, ns, ts.serialize_cdr(dimg, "sensor_msgs/msg/Image"))
            roi = T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0,
                                                        do_rectify=False)  # fmt: skip
            info = T["sensor_msgs/msg/CameraInfo"](
                header=hdr, height=h, width=wd, distortion_model="plumb_bob",
                d=np.zeros(5), k=np.array(k), r=np.eye(3).reshape(-1), p=np.array(p),
                binning_x=0, binning_y=0, roi=roi)  # fmt: skip
            w.write(c_inf, ns, ts.serialize_cdr(info, "sensor_msgs/msg/CameraInfo"))
            counts["rgb"] += 1
            counts["depth"] += 1
        unknown = np.zeros(9)
        unknown[0] = -1.0  # covariance[0] = -1: orientation not provided
        for row in _rows(seq / "imu.txt"):
            t, g, a = float(row[0]), list(map(float, row[1:4])), list(map(float, row[4:7]))
            hdr, ns = header(t, BASE)
            v3 = T["geometry_msgs/msg/Vector3"]
            imu = T["sensor_msgs/msg/Imu"](
                header=hdr, orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0,
                                                                           w=1.0),
                orientation_covariance=unknown, angular_velocity=v3(*g),
                angular_velocity_covariance=np.eye(3).reshape(-1) * (5e-4) ** 2,
                linear_acceleration=v3(*a),
                linear_acceleration_covariance=np.eye(3).reshape(-1) * (1e-2) ** 2)  # fmt: skip
            w.write(c_imu, ns, ts.serialize_cdr(imu, "sensor_msgs/msg/Imu"))
            counts["imu"] += 1
    return counts


if __name__ == "__main__":
    print(json.dumps(convert(Path(sys.argv[1]), Path(sys.argv[2]))))
